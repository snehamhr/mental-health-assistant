from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import InferenceClient

from app.config import Settings


logger = logging.getLogger("mindpal")


@dataclass(frozen=True)
class RetrievalResult:
    index: int
    text: str
    similarity: float


class RAGEngine:
    def __init__(
        self,
        settings: Settings,
    ) -> None:
        self.settings = settings

        self.client: InferenceClient | None = None
        self.chunks: list[str] = []
        self.embeddings: np.ndarray | None = None

        self.ready = False


    def load(self) -> None:
        self.ready = False

        chunks_path = Path(
            self.settings.chunks_path
        )

        embeddings_path = Path(
            self.settings.embeddings_path
        )

        self._validate_artifact_paths(
            chunks_path=chunks_path,
            embeddings_path=embeddings_path,
        )

        hf_api_key = self._clean_text(
            self.settings.hf_api_key
        )

        if not hf_api_key:
            raise ValueError(
                "HF_API_KEY is missing."
            )

        logger.info(
            "Loading knowledge-base chunks from %s",
            chunks_path,
        )

        raw_chunks = self._load_chunks(
            chunks_path
        )

        logger.info(
            "Loading knowledge-base embeddings from %s",
            embeddings_path,
        )

        raw_embeddings = np.load(
            embeddings_path,
            allow_pickle=False,
        )

        chunks = self._normalize_chunks(
            raw_chunks
        )

        embeddings = self._prepare_embeddings(
            raw_embeddings
        )

        self._validate_artifacts(
            chunks=chunks,
            embeddings=embeddings,
        )

        client = InferenceClient(
            provider="hf-inference",
            api_key=hf_api_key,
        )

        self.chunks = chunks
        self.embeddings = embeddings
        self.client = client
        self.ready = True

        logger.info(
            (
                "RAG engine ready | chunks=%d | "
                "embedding_dimension=%d | "
                "remote_embedding_model=%s"
            ),
            len(self.chunks),
            int(self.embeddings.shape[1]),
            self._get_embedding_model_id(),
        )


    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        self._ensure_ready()

        cleaned_query = self._clean_text(
            query
        )

        if not cleaned_query:
            return []

        requested_top_k = (
            self.settings.top_k_results
            if top_k is None
            else top_k
        )

        safe_top_k = max(
            1,
            min(
                int(requested_top_k),
                len(self.chunks),
            ),
        )

        query_embedding = self._encode_query(
            cleaned_query
        )

        similarity_scores = (
            self.embeddings
            @ query_embedding
        )

        top_indices = self._top_indices(
            scores=similarity_scores,
            top_k=safe_top_k,
        )

        results: list[RetrievalResult] = []

        for index in top_indices:
            score = float(
                similarity_scores[index]
            )

            results.append(
                RetrievalResult(
                    index=int(index),
                    text=self.chunks[index],
                    similarity=score,
                )
            )

        return results


    def _get_embedding_model_id(
        self,
    ) -> str:
        model_name = self._clean_text(
            self.settings.embedding_model
        )

        if not model_name:
            raise ValueError(
                "EMBEDDING_MODEL is missing."
            )

        if "/" in model_name:
            return model_name

        return (
            "sentence-transformers/"
            f"{model_name}"
        )


    def _validate_artifact_paths(
        self,
        chunks_path: Path,
        embeddings_path: Path,
    ) -> None:
        missing_files: list[str] = []

        if not chunks_path.exists():
            missing_files.append(
                str(chunks_path)
            )

        if not embeddings_path.exists():
            missing_files.append(
                str(embeddings_path)
            )

        if missing_files:
            missing_text = "\n".join(
                f"- {path}"
                for path in missing_files
            )

            raise FileNotFoundError(
                (
                    "Required RAG artifact files were not found:\n"
                    f"{missing_text}"
                )
            )


    def _load_chunks(
        self,
        path: Path,
    ) -> Any:
        try:
            with path.open("rb") as file:
                return pickle.load(
                    file
                )

        except Exception as exc:
            raise RuntimeError(
                (
                    "Could not read the chunk file: "
                    f"{path}"
                )
            ) from exc


    def _normalize_chunks(
        self,
        raw_chunks: Any,
    ) -> list[str]:
        if hasattr(
            raw_chunks,
            "tolist",
        ):
            raw_chunks = raw_chunks.tolist()

        if not isinstance(
            raw_chunks,
            (list, tuple),
        ):
            raise TypeError(
                (
                    "combined_chunks.pkl must contain "
                    "a list or tuple of chunks."
                )
            )

        normalized_chunks: list[str] = []

        for position, item in enumerate(
            raw_chunks
        ):
            text = self._extract_chunk_text(
                item
            )

            cleaned_text = self._clean_text(
                text
            )

            if not cleaned_text:
                logger.warning(
                    (
                        "Skipping empty knowledge-base "
                        "chunk at index %d."
                    ),
                    position,
                )

                continue

            normalized_chunks.append(
                cleaned_text
            )

        if not normalized_chunks:
            raise ValueError(
                (
                    "The chunk file contains no usable "
                    "text passages."
                )
            )

        return normalized_chunks


    def _extract_chunk_text(
        self,
        item: Any,
    ) -> str:
        if isinstance(
            item,
            str,
        ):
            return item

        if isinstance(
            item,
            dict,
        ):
            candidate_keys = (
                "text",
                "content",
                "chunk",
                "page_content",
                "document",
            )

            for key in candidate_keys:
                value = item.get(
                    key
                )

                if isinstance(
                    value,
                    str,
                ):
                    return value

            raise ValueError(
                (
                    "A chunk dictionary does not contain "
                    "a recognized text field."
                )
            )

        page_content = getattr(
            item,
            "page_content",
            None,
        )

        if isinstance(
            page_content,
            str,
        ):
            return page_content

        text_attribute = getattr(
            item,
            "text",
            None,
        )

        if isinstance(
            text_attribute,
            str,
        ):
            return text_attribute

        raise TypeError(
            (
                "Unsupported chunk item type: "
                f"{type(item).__name__}"
            )
        )


    def _prepare_embeddings(
        self,
        raw_embeddings: np.ndarray,
    ) -> np.ndarray:
        embeddings = np.asarray(
            raw_embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise ValueError(
                (
                    "embeddings.npy must contain a "
                    "two-dimensional array."
                )
            )

        if embeddings.shape[0] == 0:
            raise ValueError(
                (
                    "embeddings.npy contains no "
                    "embedding rows."
                )
            )

        if embeddings.shape[1] == 0:
            raise ValueError(
                (
                    "embeddings.npy has an invalid "
                    "embedding dimension."
                )
            )

        if not np.isfinite(
            embeddings
        ).all():
            raise ValueError(
                (
                    "embeddings.npy contains NaN or "
                    "infinite values."
                )
            )

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        zero_norm_rows = (
            norms.squeeze(axis=1)
            <= 1e-12
        )

        if np.any(
            zero_norm_rows
        ):
            number_of_zero_rows = int(
                np.sum(
                    zero_norm_rows
                )
            )

            raise ValueError(
                (
                    "embeddings.npy contains "
                    f"{number_of_zero_rows} zero-length "
                    "embedding vectors."
                )
            )

        normalized_embeddings = (
            embeddings
            / norms
        )

        return np.ascontiguousarray(
            normalized_embeddings,
            dtype=np.float32,
        )


    def _validate_artifacts(
        self,
        chunks: list[str],
        embeddings: np.ndarray,
    ) -> None:
        chunk_count = len(
            chunks
        )

        embedding_count = int(
            embeddings.shape[0]
        )

        if (
            chunk_count
            != embedding_count
        ):
            raise ValueError(
                (
                    "Knowledge-base artifact mismatch: "
                    f"{chunk_count} text chunks but "
                    f"{embedding_count} embedding rows."
                )
            )


    def _encode_query(
        self,
        query: str,
    ) -> np.ndarray:
        if self.client is None:
            raise RuntimeError(
                (
                    "The Hugging Face inference "
                    "client is not initialized."
                )
            )

        model_id = self._get_embedding_model_id()

        try:
            encoded = self.client.feature_extraction(
                query,
                model=model_id,
                normalize=True,
                truncate=True,
            )

        except Exception as exc:
            logger.exception(
                (
                    "Hugging Face query embedding "
                    "request failed."
                )
            )

            raise RuntimeError(
                (
                    "Could not generate the query "
                    "embedding. Please try again."
                )
            ) from exc

        query_embedding = np.asarray(
            encoded,
            dtype=np.float32,
        )

        query_embedding = np.squeeze(
            query_embedding
        )

        if query_embedding.ndim != 1:
            raise ValueError(
                (
                    "The remote query embedding has "
                    f"an unexpected shape: "
                    f"{query_embedding.shape}"
                )
            )

        if not np.isfinite(
            query_embedding
        ).all():
            raise ValueError(
                (
                    "The query embedding contains NaN "
                    "or infinite values."
                )
            )

        norm = float(
            np.linalg.norm(
                query_embedding
            )
        )

        if norm <= 1e-12:
            raise ValueError(
                (
                    "The query produced a zero-length "
                    "embedding."
                )
            )

        query_embedding = (
            query_embedding
            / norm
        )

        if self.embeddings is None:
            raise RuntimeError(
                (
                    "The stored embedding matrix "
                    "is not loaded."
                )
            )

        expected_dimension = int(
            self.embeddings.shape[1]
        )

        actual_dimension = int(
            query_embedding.shape[0]
        )

        if (
            actual_dimension
            != expected_dimension
        ):
            raise ValueError(
                (
                    "Embedding dimension mismatch: "
                    f"remote model produced "
                    f"{actual_dimension}, but stored "
                    f"embeddings use "
                    f"{expected_dimension} dimensions."
                )
            )

        return np.ascontiguousarray(
            query_embedding,
            dtype=np.float32,
        )


    def _top_indices(
        self,
        scores: np.ndarray,
        top_k: int,
    ) -> np.ndarray:
        scores = np.asarray(
            scores,
            dtype=np.float32,
        ).reshape(-1)

        if scores.size == 0:
            return np.array(
                [],
                dtype=np.int64,
            )

        if top_k >= scores.size:
            return np.argsort(
                scores
            )[::-1]

        candidate_indices = np.argpartition(
            scores,
            -top_k,
        )[-top_k:]

        sorted_candidate_order = np.argsort(
            scores[candidate_indices]
        )[::-1]

        return candidate_indices[
            sorted_candidate_order
        ]


    def _clean_text(
        self,
        value: str,
    ) -> str:
        if not isinstance(
            value,
            str,
        ):
            return ""

        return " ".join(
            value.strip().split()
        )


    def _ensure_ready(
        self,
    ) -> None:
        if not self.ready:
            raise RuntimeError(
                (
                    "The RAG engine is not ready."
                )
            )

        if self.client is None:
            raise RuntimeError(
                (
                    "The Hugging Face inference "
                    "client is not initialized."
                )
            )

        if self.embeddings is None:
            raise RuntimeError(
                (
                    "The embedding matrix is not loaded."
                )
            )

        if not self.chunks:
            raise RuntimeError(
                (
                    "The knowledge-base chunks are not loaded."
                )
            )