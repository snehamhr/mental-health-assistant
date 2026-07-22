from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from groq import APIConnectionError, APIStatusError, RateLimitError

from app.config import BASE_DIR, get_settings
from app.llm import generate_answer
from app.rag import RAGEngine, RetrievalResult
from app.safety import crisis_response, is_crisis_message
from app.schemas import ChatRequest, ChatResponse, SourcePreview


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger("mindhaven")


settings = get_settings()

rag_engine = RAGEngine(
    settings=settings,
)

STATIC_DIR = (
    BASE_DIR
    / "app"
    / "static"
)


EMOTIONAL_SUPPORT_PATTERNS = [
    r"\bi feel\b",
    r"\bi am feeling\b",
    r"\bi'm feeling\b",
    r"\bi am sad\b",
    r"\bi'm sad\b",
    r"\bi feel sad\b",
    r"\bi am anxious\b",
    r"\bi'm anxious\b",
    r"\bi feel anxious\b",
    r"\bi am worried\b",
    r"\bi'm worried\b",
    r"\bi feel worried\b",
    r"\bi am angry\b",
    r"\bi'm angry\b",
    r"\bi feel angry\b",
    r"\bi am scared\b",
    r"\bi'm scared\b",
    r"\bi feel scared\b",
    r"\bi am nervous\b",
    r"\bi'm nervous\b",
    r"\bi feel nervous\b",
    r"\bi feel lonely\b",
    r"\bi am lonely\b",
    r"\bi'm lonely\b",
    r"\bi feel low\b",
    r"\bi feel upset\b",
    r"\bi feel hurt\b",
    r"\bi feel overwhelmed\b",
    r"\bi am overwhelmed\b",
    r"\bi'm overwhelmed\b",
    r"\bi feel stressed\b",
    r"\bi am stressed\b",
    r"\bi'm stressed\b",
    r"\bi can't stop thinking\b",
    r"\bi cannot stop thinking\b",
    r"\bi don't feel okay\b",
    r"\bi do not feel okay\b",
    r"\bi'm not okay\b",
    r"\bi am not okay\b",
    r"\bmy teacher\b",
    r"\bmy parents\b",
    r"\bmy family\b",
    r"\bmy friend\b",
    r"\bmy friends\b",
    r"\bmy partner\b",
    r"\bmy boyfriend\b",
    r"\bmy girlfriend\b",
    r"\bmy husband\b",
    r"\bmy wife\b",
    r"\bmy boss\b",
    r"\bmy manager\b",
    r"\bscolded me\b",
    r"\byelled at me\b",
    r"\bshouted at me\b",
    r"\bignored me\b",
    r"\brejected me\b",
    r"\bhurt me\b",
    r"\bmade me cry\b",
    r"\bi cried\b",
    r"\bi am crying\b",
    r"\bi'm crying\b",
    r"\bpanic\b",
    r"\bpanic attack\b",
    r"\banxiety\b",
    r"\bdepressed\b",
    r"\bdepression\b",
    r"\bstress\b",
    r"\bsadness\b",
    r"\bloneliness\b",
    r"\bgrief\b",
    r"\bheartbreak\b",
    r"\bbreakup\b",
    r"\btrauma\b",
    r"\btraumatic\b",
    r"\bmental health\b",
    r"\bemotional\b",
    r"\bemotions\b",
    r"\bfeelings\b",
    r"\bcoping\b",
    r"\btherapy\b",
    r"\btherapist\b",
    r"\bcounselor\b",
    r"\bcounselling\b",
    r"\bcounseling\b",
    r"\brelationship\b",
    r"\bself-esteem\b",
    r"\bself esteem\b",
    r"\bconfidence\b",
    r"\bsleep\b",
    r"\binsomnia\b",
    r"\bcan't sleep\b",
    r"\bcannot sleep\b",
]


OUT_OF_SCOPE_PATTERNS = [
    r"\bwrite code\b",
    r"\bpython code\b",
    r"\bjavascript\b",
    r"\bjava code\b",
    r"\bc\+\+\b",
    r"\bsql query\b",
    r"\bweather\b",
    r"\bstock price\b",
    r"\bfootball score\b",
    r"\bcricket score\b",
    r"\bcapital of\b",
    r"\btranslate\b",
    r"\brecipe\b",
    r"\bcook\b",
    r"\bmathematics\b",
    r"\bsolve this equation\b",
    r"\bcalculate\b",
    r"\bhistory of\b",
    r"\bwho is the president\b",
    r"\bmovie recommendation\b",
    r"\blaptop recommendation\b",
]


def normalize_text(
    value: str,
) -> str:
    """
    Normalize user text for routing checks.
    """

    return re.sub(
        r"\s+",
        " ",
        value.lower().strip(),
    )


def matches_any_pattern(
    text: str,
    patterns: list[str],
) -> bool:
    """
    Return True when text matches at least one pattern.
    """

    return any(
        re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )
        is not None
        for pattern in patterns
    )


def is_emotional_support_message(
    message: str,
    history: list,
) -> bool:
    """
    Detect messages that clearly belong to emotional or mental-health support.

    This allows short messages such as:
    - I feel sad
    - I feel angry
    - My teacher scolded me
    - I am anxious

    These should not be rejected only because their embedding similarity is low.
    """

    normalized_message = normalize_text(
        message
    )

    if matches_any_pattern(
        normalized_message,
        OUT_OF_SCOPE_PATTERNS,
    ):
        return False

    if matches_any_pattern(
        normalized_message,
        EMOTIONAL_SUPPORT_PATTERNS,
    ):
        return True

    recent_user_text = " ".join(
        item.content
        for item in history[-8:]
        if item.role == "user"
    )

    combined_text = normalize_text(
        f"{recent_user_text} {message}"
    )

    return matches_any_pattern(
        combined_text,
        EMOTIONAL_SUPPORT_PATTERNS,
    )


def build_retrieval_query(
    request: ChatRequest,
) -> str:
    """
    Build a context-aware retrieval query.

    Recent user messages are included so retrieval understands the developing
    emotional story instead of embedding only the latest short sentence.

    Example:

    Earlier:
        I feel sad.
        My teacher scolded me.

    Current:
        I also feel angry.

    Retrieval query:
        I feel sad.
        My teacher scolded me.
        I also feel angry.
    """

    recent_user_messages = [
        item.content.strip()
        for item in request.history
        if (
            item.role == "user"
            and item.content.strip()
        )
    ]

    recent_user_messages = (
        recent_user_messages[
            -settings.retrieval_user_turns:
        ]
    )

    recent_user_messages.append(
        request.message.strip()
    )

    return "\n".join(
        recent_user_messages
    )


def safe_source_previews(
    results: list[RetrievalResult],
) -> list[SourcePreview]:
    """
    Create short internal source previews.

    The frontend does not currently display these, but they are useful for
    development and debugging.
    """

    previews: list[SourcePreview] = []

    for result in results:
        cleaned_preview = re.sub(
            r"\s+",
            " ",
            result.text,
        ).strip()

        previews.append(
            SourcePreview(
                index=result.index,
                similarity=float(
                    result.similarity
                ),
                preview=cleaned_preview[:180],
            )
        )

    return previews


@asynccontextmanager
async def lifespan(
    _: FastAPI,
):
    """
    Load the RAG engine once during server startup.
    """

    try:
        rag_engine.load()

        logger.info(
            "RAG engine loaded successfully."
        )

    except Exception:
        logger.exception(
            "RAG engine could not load during startup. "
            "The health endpoint will report degraded status."
        )

    yield


app = FastAPI(
    title=settings.app_name,
    description=(
        "A private, session-based mental-health "
        "support RAG chatbot."
    ),
    version="3.0.0",
    lifespan=lifespan,
)


app.mount(
    "/static",
    StaticFiles(
        directory=STATIC_DIR,
    ),
    name="static",
)


@app.get(
    "/",
    include_in_schema=False,
)
def index() -> FileResponse:
    """
    Serve the chatbot website.
    """

    return FileResponse(
        STATIC_DIR
        / "index.html"
    )


@app.get(
    "/health",
)
def health() -> dict[str, object]:
    """
    Return a deployment health check.
    """

    return {
        "status": (
            "ok"
            if rag_engine.ready
            else "degraded"
        ),
        "rag_ready": rag_engine.ready,
        "groq_configured": bool(
            settings.groq_api_key
        ),
        "groq_model": (
            settings.groq_model
        ),
        "embedding_model": (
            settings.embedding_model
        ),
        "similarity_threshold": (
            settings.similarity_threshold
        ),
        "top_k_results": (
            settings.top_k_results
        ),
        "memory_window": (
            settings.memory_window
        ),
        "retrieval_user_turns": (
            settings.retrieval_user_turns
        ),
    }


@app.post(
    "/api/chat",
    response_model=ChatResponse,
)
def chat(
    request: ChatRequest,
) -> ChatResponse:
    """
    Process one chatbot message.

    Routing order:

    1. Validate the message.
    2. Run crisis-language safety routing.
    3. Retrieve supporting passages.
    4. Allow clearly emotional short messages even when similarity is modest.
    5. Reject genuinely unrelated requests without calling Groq.
    6. Generate a brief empathetic response using Groq and real chat history.

    No conversation is stored on the server.
    """

    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=422,
            detail=(
                "Message cannot be empty."
            ),
        )

    if (
        len(message)
        >
        settings.max_user_message_chars
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "Message is too long. "
                f"Please keep it under "
                f"{settings.max_user_message_chars:,} "
                "characters."
            ),
        )

    if is_crisis_message(
        message
    ):
        return ChatResponse(
            answer=crisis_response(),
            route="crisis",
            similarity=None,
            sources=[],
        )

    if not rag_engine.ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "The knowledge base is not ready. "
                "Confirm that "
                "data/combined_chunks.pkl and "
                "data/embeddings.npy are present."
            ),
        )

    retrieval_query = (
        build_retrieval_query(
            request
        )
    )

    try:
        results = (
            rag_engine.retrieve(
                query=retrieval_query,
                top_k=(
                    settings.top_k_results
                ),
            )
        )

    except Exception as exc:
        logger.exception(
            "Knowledge-base retrieval failed."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "The knowledge base is temporarily "
                "unavailable. Please try again."
            ),
        ) from exc

    best_score = (
        float(
            results[0].similarity
        )
        if results
        else 0.0
    )

    emotional_message = (
        is_emotional_support_message(
            message=message,
            history=request.history,
        )
    )

    passes_similarity_threshold = (
        best_score
        >=
        settings.similarity_threshold
    )

    logger.info(
        (
            "Chat routing | "
            "score=%.4f | "
            "threshold=%.4f | "
            "emotional=%s | "
            "history=%d"
        ),
        best_score,
        settings.similarity_threshold,
        emotional_message,
        len(request.history),
    )

    if (
        not passes_similarity_threshold
        and not emotional_message
    ):
        return ChatResponse(
            answer=(
                settings
                .out_of_context_message
            ),
            route="out_of_context",
            similarity=best_score,
            sources=[],
        )

    try:
        answer = generate_answer(
            settings=settings,
            message=message,
            history=request.history,
            results=results,
        )

    except RateLimitError as exc:
        logger.exception(
            "Groq rate limit reached."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "MindHaven is receiving many "
                "messages right now. Please wait "
                "a moment and try again."
            ),
        ) from exc

    except APIConnectionError as exc:
        logger.exception(
            "Could not connect to Groq."
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "The language service is "
                "temporarily unreachable. "
                "Please try again shortly."
            ),
        ) from exc

    except APIStatusError as exc:
        logger.exception(
            "Groq returned an API error."
        )

        status_code = getattr(
            exc,
            "status_code",
            502,
        )

        if status_code == 401:
            detail = (
                "The Groq API key is invalid or "
                "missing. Check GROQ_API_KEY in "
                "the .env file."
            )

        elif status_code == 429:
            detail = (
                "MindHaven is receiving many "
                "messages right now. Please wait "
                "a moment and try again."
            )

        else:
            detail = (
                "The language service could not "
                "generate a response right now."
            )

        raise HTTPException(
            status_code=(
                503
                if status_code == 429
                else 502
            ),
            detail=detail,
        ) from exc

    except RuntimeError as exc:
        logger.exception(
            "MindHaven configuration or generation error."
        )

        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Unexpected Groq response generation failure."
        )

        raise HTTPException(
            status_code=502,
            detail=(
                "MindHaven could not generate "
                "a response right now. "
                "Please try again."
            ),
        ) from exc

    return ChatResponse(
        answer=answer,
        route="rag",
        similarity=best_score,
        sources=safe_source_previews(
            results
        ),
    )