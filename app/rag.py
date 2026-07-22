from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import Settings


logger = logging.getLogger("mindhaven")


@dataclass(frozen=True)
class RetrievalResult:
    """
    One retrieved knowledge-base passage.

    Attributes
    ----------
    index:
        Position of the passage inside the stored chunk list.

    text:
        Retrieved passage text.

    similarity:
        Cosine-similarity score between the user query and the passage.
    """

    index: int
    text: str
    similarity: float


class RAGEngine:
    """
    Lightweight CPU-based semantic retrieval engine.

    The engine loads:

    - `all-MiniLM-L6-v2` for query embeddings;
    - pre-generated text chunks from `combined_chunks.pkl`;
    - pre-generated embeddings from `embeddings.npy`.

    The large language model is not loaded locally. Only the small embedding
    model runs on the local CPU.
    """

    def __init__(
        self,
        settings: Settings,
    ) -> None:
        self.settings = settings

        self.model: SentenceTransformer | None = None
        self.chunks: list[str] = []
        self.embeddings: np.ndarray | None = None

        self.ready: bool = False


    def load(self) -> None:
        """
        Load the embedding model and RAG artifacts.

        This method is called once during FastAPI startup.
        """

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

        logger.info(
            "Loading SentenceTransformer model: %s",
            self.settings.embedding_model,
        )

        model = SentenceTransformer(
            self.settings.embedding_model,
            device="cpu",
        )

        self.chunks = chunks
        self.embeddings = embeddings
        self.model = model
        self.ready = True

        logger.info(
            (
                "RAG engine ready | chunks=%d | "
                "embedding_dimension=%d"
            ),
            len(self.chunks),
            int(self.embeddings.shape[1]),
        )


    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve the most relevant knowledge-base passages.

        Parameters
        ----------
        query:
            Current retrieval query. The application may combine several recent
            user messages so short follow-ups retain conversational context.

        top_k:
            Number of passages to return. Defaults to the configured value.

        Returns
        -------
        list[RetrievalResult]
            Results sorted from highest to lowest similarity.
        """

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


    def _validate_artifact_paths(
        self,
        chunks_path: Path,
        embeddings_path: Path,
    ) -> None:
        """
        Confirm that required RAG files exist before loading.
        """

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
                    f"{missing_text}\n\n"
                    "Place the two PDF books in source_books/ and run:\n"
                    "python scripts/build_knowledge_base.py"
                )
            )


    def _load_chunks(
        self,
        path: Path,
    ) -> Any:
        """
        Load the serialized chunk object.
        """

        try:
            with path.open("rb") as file:
                return pickle.load(file)

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
        """
        Convert supported chunk structures into a clean list of strings.

        Supported stored formats include:

        - list[str]
        - list[dict] where text is stored under keys such as:
          `text`, `content`, `chunk`, or `page_content`
        - pandas-like objects convertible with `.tolist()`
        """

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
        """
        Extract text from one stored chunk item.
        """

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
        """
        Validate and L2-normalize stored embeddings.

        With normalized vectors, cosine similarity is simply a matrix-vector
        dot product, which is fast on CPU.
        """

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
        """
        Confirm that every text chunk has exactly one embedding.
        """

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
                    f"{embedding_count} embedding rows.\n"
                    "Rebuild both files together using:\n"
                    "python scripts/build_knowledge_base.py"
                )
            )


    def _encode_query(
        self,
        query: str,
    ) -> np.ndarray:
        """
        Encode and normalize one retrieval query on CPU.
        """

        if self.model is None:
            raise RuntimeError(
                (
                    "The embedding model is not loaded."
                )
            )

        encoded = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        query_embedding = np.asarray(
            encoded,
            dtype=np.float32,
        ).reshape(-1)

        if query_embedding.ndim != 1:
            raise ValueError(
                (
                    "The query embedding has an "
                    "unexpected shape."
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
                    f"query model produced {actual_dimension}, "
                    f"but stored embeddings use "
                    f"{expected_dimension} dimensions.\n"
                    "Rebuild the knowledge base using the "
                    "same EMBEDDING_MODEL configured in .env."
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
        """
        Return indices of the highest similarity scores.

        `argpartition` avoids fully sorting the entire knowledge base when only
        a few top passages are needed.
        """

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
        """
        Normalize whitespace without changing the meaning of the text.
        """

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
        """
        Prevent retrieval before successful startup loading.
        """

        if not self.ready:
            raise RuntimeError(
                (
                    "The RAG engine is not ready."
                )
            )

        if self.model is None:
            raise RuntimeError(
                (
                    "The embedding model is not loaded."
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