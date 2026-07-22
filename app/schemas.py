from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatMessage(BaseModel):
    """
    One message from the current browser-session conversation.

    Only user and assistant roles are accepted.
    """

    role: Literal[
        "user",
        "assistant",
    ]

    content: str = Field(
        min_length=1,
        max_length=12000,
    )

    @field_validator(
        "content",
        mode="before",
    )
    @classmethod
    def clean_message_content(
        cls,
        value: object,
    ) -> str:
        """
        Ensure message content is a non-empty cleaned string.
        """

        if not isinstance(
            value,
            str,
        ):
            raise ValueError(
                "Message content must be text."
            )

        cleaned = value.strip()

        if not cleaned:
            raise ValueError(
                "Message content cannot be empty."
            )

        return cleaned


class ChatRequest(BaseModel):
    """
    Request body sent from the frontend to `/api/chat`.

    `history` contains only the active browser-tab conversation.
    Nothing is loaded from or stored in a server-side database.
    """

    message: str = Field(
        min_length=1,
        max_length=12000,
    )

    history: list[
        ChatMessage
    ] = Field(
        default_factory=list,
        max_length=40,
    )

    @field_validator(
        "message",
        mode="before",
    )
    @classmethod
    def clean_current_message(
        cls,
        value: object,
    ) -> str:
        """
        Clean the current user message before routing.
        """

        if not isinstance(
            value,
            str,
        ):
            raise ValueError(
                "Message must be text."
            )

        cleaned = value.strip()

        if not cleaned:
            raise ValueError(
                "Message cannot be empty."
            )

        return cleaned


class SourcePreview(BaseModel):
    """
    Small internal preview of one retrieved knowledge-base result.

    The current frontend does not display these previews, but they are useful
    for testing and debugging the RAG pipeline.
    """

    index: int = Field(
        ge=0,
    )

    similarity: float = Field(
        ge=-1.0,
        le=1.0,
    )

    preview: str = Field(
        min_length=1,
        max_length=500,
    )


class ChatResponse(BaseModel):
    """
    Response returned by `/api/chat`.
    """

    answer: str = Field(
        min_length=1,
        max_length=12000,
    )

    route: Literal[
        "rag",
        "out_of_context",
        "crisis",
        "error",
    ]

    similarity: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
    )

    sources: list[
        SourcePreview
    ] = Field(
        default_factory=list,
    )

    @field_validator(
        "answer",
        mode="before",
    )
    @classmethod
    def clean_answer(
        cls,
        value: object,
    ) -> str:
        """
        Ensure the returned assistant answer is valid text.
        """

        if not isinstance(
            value,
            str,
        ):
            raise ValueError(
                "Answer must be text."
            )

        cleaned = value.strip()

        if not cleaned:
            raise ValueError(
                "Answer cannot be empty."
            )

        return cleaned