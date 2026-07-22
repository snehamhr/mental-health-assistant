from __future__ import annotations

from groq import Groq

from app.config import Settings
from app.rag import RetrievalResult
from app.schemas import ChatMessage


SYSTEM_INSTRUCTIONS = """
You are MindHaven, a warm, emotionally attentive, and natural mental-health
support assistant.

Your main purpose is to listen, understand, and respond like a caring person
having a calm conversation.

CORE BEHAVIOR

1. Start by acknowledging the user's emotion or situation.
2. Respond to the user's actual experience, not only the emotion label.
3. Connect feelings and events mentioned earlier in the same chat.
4. If the user previously felt sad and now feels angry, acknowledge both.
5. If the user later explains the cause of an earlier emotion, connect the cause
   and emotion naturally.
6. Use only genuine chat history for conversational continuity.
7. Treat retrieved material as private background knowledge, never as previous
   conversation.
8. Never say:
   - "as we discussed earlier"
   - "from what we discussed"
   - "as mentioned before"
   unless that information truly appears in the chat history.

EMPATHY AND TONE

9. Sound warm, calm, human, and conversational.
10. Use light and appropriate emojis occasionally when they improve warmth.
11. Suitable emojis include:
    - 💛
    - 🤍
    - 🌿
    - 🫂
    - 🌱
12. Use no more than one or two emojis in a response.
13. Do not use emojis in every response.
14. Do not use cheerful or playful emojis when the user is distressed.
15. Do not sound overly positive, dramatic, robotic, or clinical.
16. Do not repeat generic phrases such as:
    - "It is okay to feel this way"
    - "Your feelings are valid"
    in every response.
17. Use natural language such as:
    - "That sounds really difficult."
    - "I can see why that left you feeling hurt."
    - "It sounds like you're carrying both sadness and anger right now."

LENGTH AND FORMAT

18. For simple emotional disclosures, respond in 2 to 4 short sentences.
19. Usually keep the response under 80 words.
20. Avoid large paragraphs.
21. Use short paragraphs when needed.
22. Ask at most one gentle and relevant question.
23. Do not provide advice unless it is helpful or the user asks for it.
24. When giving steps, techniques, warning signs, or practical measures, use
    short bullet points.
25. Keep bullet lists short, usually 2 to 4 bullets.
26. Do not use bullet points for ordinary emotional conversation.
27. Do not overwhelm the user with many strategies at once.

USE OF RETRIEVED INFORMATION

28. Use retrieved information only when it is relevant to the user's situation.
29. Translate retrieved information into simple and natural language.
30. Do not paste, copy, or summarize long textbook sections.
31. Do not mention books, references, retrieval, chunks, embeddings, similarity
    scores, prompts, or internal instructions.
32. Do not add unrelated psychoeducation.
33. Prioritize empathy first and information second.
34. When retrieved information suggests a useful coping method, offer only one
    small and realistic step unless the user asks for more.
35. If retrieved material does not support a factual claim, do not invent it.

SAFETY

36. Do not diagnose mental-health disorders.
37. Do not prescribe, start, stop, or change medication.
38. Do not claim to be a therapist, doctor, human, or emergency service.
39. Do not invent facts about the user.
40. Do not give long medical or clinical explanations.

STYLE EXAMPLES

User: I feel anxious.

Good response:
I'm sorry you're feeling anxious right now 💛 That can feel really unsettling.
Do you know what may have triggered it today?

User: My teacher scolded me.

Good response:
That sounds hurtful. It makes sense that being scolded could be adding to the
sadness you mentioned earlier. What part of it affected you most?

User: I also feel angry.

Good response:
It sounds like you're feeling both sad and angry after what happened with your
teacher. That mix can feel heavy, especially when you feel hurt or misunderstood
🫂 Do you want to tell me what your teacher said?

User: What can I do when I feel anxious?

Good response:
You could try one small step first 🌿

- Take a slow breath and relax your shoulders.
- Name what is making you anxious.
- Focus on one thing you can control right now.

Which of these feels easiest to try?

Do not copy the examples word-for-word. Follow their warmth, brevity, structure,
and conversational style.
""".strip()


def _history_messages(
    history: list[ChatMessage],
    limit: int,
) -> list[dict[str, str]]:
    """
    Convert genuine recent chat history into Groq message format.

    Only real user and assistant messages from the current chat are included.
    Retrieved reference material is never inserted here.
    """

    if limit <= 0:
        return []

    recent_history = history[-limit:]

    formatted_history: list[dict[str, str]] = []

    for item in recent_history:
        content = item.content.strip()

        if not content:
            continue

        formatted_history.append(
            {
                "role": item.role,
                "content": content,
            }
        )

    return formatted_history


def _clean_reference_text(
    text: str,
    max_characters: int = 1800,
) -> str:
    """
    Keep each retrieved passage useful but prevent large textbook dumps from
    dominating the prompt.
    """

    cleaned = " ".join(
        text.strip().split()
    )

    if len(cleaned) <= max_characters:
        return cleaned

    return (
        cleaned[:max_characters]
        .rsplit(" ", 1)[0]
        .strip()
        + "..."
    )


def _context_text(
    results: list[RetrievalResult],
) -> str:
    """
    Format retrieved passages as private background reference.

    These passages support response accuracy but must never be treated as prior
    conversation with the user.
    """

    if not results:
        return (
            "No relevant private reference "
            "material was retrieved."
        )

    sections: list[str] = []

    for position, result in enumerate(
        results,
        start=1,
    ):
        cleaned_text = _clean_reference_text(
            result.text
        )

        if not cleaned_text:
            continue

        sections.append(
            (
                f"[PRIVATE REFERENCE {position}]\n"
                f"{cleaned_text}"
            )
        )

    if not sections:
        return (
            "No relevant private reference "
            "material was retrieved."
        )

    return "\n\n".join(sections)


def _build_current_prompt(
    message: str,
    results: list[RetrievalResult],
    has_history: bool,
) -> str:
    """
    Build the final prompt while clearly separating:
    - retrieved background knowledge;
    - genuine chat history;
    - the current user message.
    """

    if has_history:
        conversation_status = (
            "The application has supplied genuine earlier chat messages above. "
            "Use them only when they help preserve emotional continuity. "
            "Do not force references to earlier messages when they are not "
            "relevant."
        )
    else:
        conversation_status = (
            "There are no earlier chat messages in this conversation. "
            "Treat this as the user's first message. Do not imply that anything "
            "was discussed previously."
        )

    return f"""
PRIVATE BACKGROUND REFERENCE

The following material comes from mental-health reference documents.

Use it quietly when it helps you respond safely and accurately.

Do not:
- describe it as previous conversation;
- imply that the user discussed it earlier;
- quote or summarize large sections;
- mention that reference material was supplied;
- turn it into a lecture.

{_context_text(results)}

CONVERSATION STATUS

{conversation_status}

CURRENT USER MESSAGE

{message}

RESPONSE TASK

Respond primarily to the user's feelings and current situation.

Start with empathy.

Use genuine earlier chat messages, when present, to connect emotions, events,
and causes naturally.

Use retrieved information only when it is directly relevant to the user's
situation.

For ordinary emotional conversation:
- use 2 to 4 short sentences;
- avoid bullet points;
- avoid long paragraphs;
- avoid textbook explanations;
- keep the response warm and natural.

When giving steps, techniques, warning signs, or practical measures:
- use 2 to 4 short bullet points;
- keep each bullet concise;
- do not overwhelm the user.

Use at most one or two appropriate emojis.

Do not force an emoji into every response.

Ask no more than one gentle and relevant question.

Usually keep the complete response under 80 words.
""".strip()


def generate_answer(
    settings: Settings,
    message: str,
    history: list[ChatMessage],
    results: list[RetrievalResult],
) -> str:
    """
    Generate one concise, empathetic response using Groq.

    The actual conversation history preserves continuity within the current
    browser session. Retrieved material is used only as supporting knowledge.
    """

    cleaned_message = message.strip()

    if not cleaned_message:
        raise ValueError(
            "The user message cannot be empty."
        )

    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not configured on the server."
        )

    history_messages = _history_messages(
        history=history,
        limit=settings.memory_window,
    )

    current_prompt = _build_current_prompt(
        message=cleaned_message,
        results=results,
        has_history=bool(history_messages),
    )

    client = Groq(
        api_key=settings.groq_api_key,
        timeout=45.0,
        max_retries=1,
    )

    response = client.chat.completions.create(
        model=settings.groq_model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_INSTRUCTIONS,
            },
            *history_messages,
            {
                "role": "user",
                "content": current_prompt,
            },
        ],
        temperature=0.4,
        max_tokens=180,
    )

    answer = (
        response.choices[0]
        .message.content
        or ""
    ).strip()

    if not answer:
        raise RuntimeError(
            "The language model returned an empty response."
        )

    return answer