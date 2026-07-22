from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "MindPal"

    groq_api_key: str = Field(
        default="",
        repr=False,
    )

    hf_api_key: str = Field(
        default="",
        repr=False,
    )

    groq_model: str = "llama-3.1-8b-instant"

    embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    chunks_path: Path = (
        BASE_DIR
        / "data"
        / "combined_chunks.pkl"
    )

    embeddings_path: Path = (
        BASE_DIR
        / "data"
        / "embeddings.npy"
    )

    similarity_threshold: float = 0.55

    top_k_results: int = 4

    memory_window: int = 16

    retrieval_user_turns: int = 6

    max_user_message_chars: int = 4000

    out_of_context_message: str = (
        "I’m sorry, but I’m only able to help with "
        "mental health, emotions, stress, relationships, "
        "coping, and well-being."
    )

    @field_validator(
        "app_name",
        "groq_model",
        "embedding_model",
        mode="before",
    )
    @classmethod
    def clean_required_string(
        cls,
        value: object,
    ) -> str:
        if not isinstance(
            value,
            str,
        ):
            raise ValueError(
                "This configuration value must be a string."
            )

        cleaned = value.strip()

        if not cleaned:
            raise ValueError(
                "This configuration value cannot be empty."
            )

        return cleaned

    @field_validator(
        "groq_api_key",
        "hf_api_key",
        mode="before",
    )
    @classmethod
    def clean_secret_string(
        cls,
        value: object,
    ) -> str:
        if value is None:
            return ""

        if not isinstance(
            value,
            str,
        ):
            raise ValueError(
                "API keys must be strings."
            )

        return value.strip()

    @field_validator(
        "similarity_threshold",
    )
    @classmethod
    def validate_similarity_threshold(
        cls,
        value: float,
    ) -> float:
        numeric_value = float(
            value
        )

        if not -1.0 <= numeric_value <= 1.0:
            raise ValueError(
                (
                    "SIMILARITY_THRESHOLD must be "
                    "between -1.0 and 1.0."
                )
            )

        return numeric_value

    @field_validator(
        "top_k_results",
        "memory_window",
        "retrieval_user_turns",
        "max_user_message_chars",
    )
    @classmethod
    def validate_positive_integer(
        cls,
        value: int,
    ) -> int:
        numeric_value = int(
            value
        )

        if numeric_value <= 0:
            raise ValueError(
                (
                    "This configuration value "
                    "must be greater than zero."
                )
            )

        return numeric_value

    @field_validator(
        "chunks_path",
        "embeddings_path",
        mode="before",
    )
    @classmethod
    def resolve_project_path(
        cls,
        value: object,
    ) -> Path:
        path = Path(
            str(value)
        ).expanduser()

        if not path.is_absolute():
            path = (
                BASE_DIR
                / path
            )

        return path.resolve()


@lru_cache(
    maxsize=1,
)
def get_settings() -> Settings:
    return Settings()