from __future__ import annotations

import re


CRISIS_PATTERNS = [
    r"\bi want to die\b",
    r"\bi want to kill myself\b",
    r"\bi'm going to kill myself\b",
    r"\bi am going to kill myself\b",
    r"\bi feel like killing myself\b",
    r"\bi feel like ending my life\b",
    r"\bi want to end my life\b",
    r"\bi don't want to live\b",
    r"\bi do not want to live\b",
    r"\bi can't go on\b",
    r"\bi cannot go on\b",
    r"\bthere is no reason to live\b",
    r"\blife is not worth living\b",
    r"\bi wish i were dead\b",
    r"\bi wish i was dead\b",
    r"\bi would be better off dead\b",
    r"\beveryone would be better without me\b",
    r"\bi am planning to kill myself\b",
    r"\bi have a plan to kill myself\b",
    r"\bi have a suicide plan\b",
    r"\bi am suicidal\b",
    r"\bi'm suicidal\b",
    r"\bsuicidal thoughts\b",
    r"\bsuicide\b",
    r"\bself[- ]?harm\b",
    r"\bi want to hurt myself\b",
    r"\bi feel like hurting myself\b",
    r"\bi am going to hurt myself\b",
    r"\bi'm going to hurt myself\b",
    r"\bi cut myself\b",
    r"\bi am cutting myself\b",
    r"\bi'm cutting myself\b",
    r"\bi took an overdose\b",
    r"\bi overdosed\b",
    r"\bi swallowed pills\b",
    r"\bi took too many pills\b",
    r"\bi have a weapon\b",
    r"\bi have a gun\b",
    r"\bi have a knife\b",
    r"\bi am in immediate danger\b",
    r"\bi'm in immediate danger\b",
    r"\bsomeone is going to hurt me\b",
    r"\bsomeone wants to kill me\b",
    r"\bi am being abused\b",
    r"\bi'm being abused\b",
    r"\bi am not safe\b",
    r"\bi'm not safe\b",
]


NEGATED_OR_NON_CURRENT_PATTERNS = [
    r"\bi do not want to kill myself\b",
    r"\bi don't want to kill myself\b",
    r"\bi am not suicidal\b",
    r"\bi'm not suicidal\b",
    r"\bi am not going to hurt myself\b",
    r"\bi'm not going to hurt myself\b",
    r"\bi would never hurt myself\b",
    r"\bi would never kill myself\b",
    r"\bi used to feel suicidal\b",
    r"\bi was suicidal in the past\b",
    r"\bsomeone else is suicidal\b",
    r"\bmy friend is suicidal\b",
    r"\bmy friend wants to die\b",
]


def normalize_text(
    value: str,
) -> str:
    """
    Normalize user text before crisis-language matching.
    """

    if not isinstance(
        value,
        str,
    ):
        return ""

    lowered = value.lower().strip()

    lowered = re.sub(
        r"[’‘`]",
        "'",
        lowered,
    )

    lowered = re.sub(
        r"\s+",
        " ",
        lowered,
    )

    return lowered


def matches_any_pattern(
    text: str,
    patterns: list[str],
) -> bool:
    """
    Return True when text matches at least one crisis-related pattern.
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


def is_crisis_message(
    message: str,
) -> bool:
    """
    Detect likely immediate self-harm, suicide, abuse, or danger messages.

    This is a conservative keyword-based safety layer. It runs before RAG and
    before the Groq API call.

    Clearly negated or historical phrases are excluded when possible, although
    this is not a complete clinical risk-assessment system.
    """

    normalized_message = normalize_text(
        message
    )

    if not normalized_message:
        return False

    if matches_any_pattern(
        normalized_message,
        NEGATED_OR_NON_CURRENT_PATTERNS,
    ):
        return False

    return matches_any_pattern(
        normalized_message,
        CRISIS_PATTERNS,
    )


def crisis_response() -> str:
    """
    Return a concise immediate-safety response.

    The response avoids diagnosis and does not rely on retrieved documents.
    """

    return (
        "I’m really glad you told me. Your immediate safety matters most right "
        "now. Please move away from anything you could use to hurt yourself and "
        "contact someone you trust who can stay with you.\n\n"
        "- Call your local emergency number if you may act soon or are in danger.\n"
        "- If you are in the U.S. or Canada, call or text 988.\n"
        "- If you are elsewhere, contact your local crisis line or emergency service.\n\n"
        "Can you tell me whether you are in immediate danger right now?"
    )