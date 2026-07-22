from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path

import fitz
import numpy as np
from sentence_transformers import SentenceTransformer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("build_knowledge_base")


BASE_DIR = Path(__file__).resolve().parent.parent

SOURCE_BOOKS_DIR = BASE_DIR / "source_books"

DATA_DIR = BASE_DIR / "data"

CHUNKS_OUTPUT_PATH = DATA_DIR / "combined_chunks.pkl"

EMBEDDINGS_OUTPUT_PATH = DATA_DIR / "embeddings.npy"


EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 900

CHUNK_OVERLAP_TARGET = 150

BATCH_SIZE = 64

MIN_CHUNK_CHARACTERS = 120

MAX_CHUNK_CHARACTERS = 1050


BOOK_HEADER_PATTERNS = [
    re.compile(
        r"^\s*fundamentals\s+of\s+psychological\s+disorders\s*\d*\s*$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^\s*psychology\s+of\s+human\s+emotion\s*\d*\s*$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^\s*fundamentals\s+of\s+psychological\s+disorders\s*$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^\s*psychology\s+of\s+human\s+emotion\s*$",
        flags=re.IGNORECASE,
    ),
]


STANDALONE_PAGE_NUMBER_PATTERN = re.compile(
    r"^\s*(?:page\s*)?\d{1,4}\s*$",
    flags=re.IGNORECASE,
)


URL_PATTERN = re.compile(
    r"""
    (?:
        https?://[^\s]+
        |
        www\.[^\s]+
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)


LICENSE_LINE_PATTERNS = [
    re.compile(
        r"\bcreative\s+commons\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\blicensed\s+material\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\blicensed\s+rights\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\blicensor\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\badapter'?s\s+license\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bpublic\s+license\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bdownstream\s+recipients\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\boffer\s+from\s+the\s+licensor\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bno\s+downstream\s+restrictions\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\byou\s+may\s+not\s+offer\s+or\s+impose\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\blicense\s+conditions\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\battribution\s+sharealike\b",
        flags=re.IGNORECASE,
    ),
]


LOW_VALUE_CHUNK_PATTERNS = [
    re.compile(
        r"\bcreative\s+commons\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\blicensed\s+material\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bpublic\s+license\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bdownstream\s+recipients\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\btable\s+of\s+contents\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bcopyright\s+notice\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\ball\s+rights\s+reserved\b",
        flags=re.IGNORECASE,
    ),
]


REFERENCE_HEADING_PATTERN = re.compile(
    r"^\s*(references|bibliography|works cited)\s*$",
    flags=re.IGNORECASE,
)


INDEX_HEADING_PATTERN = re.compile(
    r"^\s*index\s*$",
    flags=re.IGNORECASE,
)


SENTENCE_SPLIT_PATTERN = re.compile(
    r"(?<=[.!?])\s+(?=(?:[A-Z0-9“\"']))"
)


def normalize_unicode_punctuation(
    text: str,
) -> str:
    """
    Normalize common Unicode punctuation without changing meaning.
    """

    replacements = {
        "\u00a0": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\ufeff": "",
    }

    normalized = text

    for old, new in replacements.items():
        normalized = normalized.replace(
            old,
            new,
        )

    return normalized


def should_remove_line(
    line: str,
) -> bool:
    """
    Return True for lines that are likely headers, footers, page numbers,
    URLs, email-only lines, or license text.
    """

    stripped = line.strip()

    if not stripped:
        return False

    if STANDALONE_PAGE_NUMBER_PATTERN.fullmatch(
        stripped
    ):
        return True

    if any(
        pattern.fullmatch(stripped)
        for pattern in BOOK_HEADER_PATTERNS
    ):
        return True

    if URL_PATTERN.fullmatch(
        stripped
    ):
        return True

    if EMAIL_PATTERN.fullmatch(
        stripped
    ):
        return True

    if any(
        pattern.search(stripped)
        for pattern in LICENSE_LINE_PATTERNS
    ):
        return True

    return False


def clean_extracted_text(
    text: str,
) -> str:
    """
    Clean PDF-extracted text while preserving useful paragraph structure.

    Removes:
    - null characters;
    - broken hyphenation across line breaks;
    - standalone page numbers;
    - repeated book headers and footers;
    - URLs and email addresses;
    - common license/front-matter lines;
    - excessive whitespace.
    """

    if not isinstance(
        text,
        str,
    ):
        return ""

    cleaned = normalize_unicode_punctuation(
        text
    )

    cleaned = cleaned.replace(
        "\x00",
        " ",
    )

    cleaned = re.sub(
        r"(?<=\w)-\s*\n\s*(?=\w)",
        "",
        cleaned,
    )

    cleaned_lines: list[str] = []

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()

        if not line:
            cleaned_lines.append("")
            continue

        if should_remove_line(
            line
        ):
            continue

        line = URL_PATTERN.sub(
            " ",
            line,
        )

        line = EMAIL_PATTERN.sub(
            " ",
            line,
        )

        line = re.sub(
            r"[ \t]+",
            " ",
            line,
        ).strip()

        if line:
            cleaned_lines.append(
                line
            )

    cleaned = "\n".join(
        cleaned_lines
    )

    cleaned = re.sub(
        r"[ \t]+",
        " ",
        cleaned,
    )

    cleaned = re.sub(
        r"\n[ \t]+",
        "\n",
        cleaned,
    )

    cleaned = re.sub(
        r"[ \t]+\n",
        "\n",
        cleaned,
    )

    cleaned = re.sub(
        r"\n{3,}",
        "\n\n",
        cleaned,
    )

    return cleaned.strip()


def extract_pdf_pages(
    pdf_path: Path,
) -> list[tuple[int, str]]:
    """
    Extract usable text from every page of one PDF.

    Returns:
        List of `(page_number, cleaned_page_text)` tuples.

    Page numbers are one-based.
    """

    logger.info(
        "Opening PDF: %s",
        pdf_path.name,
    )

    try:
        document = fitz.open(
            pdf_path
        )

    except Exception as exc:
        raise RuntimeError(
            (
                "Could not open PDF file: "
                f"{pdf_path}"
            )
        ) from exc

    extracted_pages: list[
        tuple[int, str]
    ] = []

    try:
        for page_index in range(
            document.page_count
        ):
            page = document.load_page(
                page_index
            )

            page_text = page.get_text(
                "text"
            )

            cleaned_text = clean_extracted_text(
                page_text
            )

            if not cleaned_text:
                logger.warning(
                    (
                        "No extractable useful text found in "
                        "%s page %d."
                    ),
                    pdf_path.name,
                    page_index + 1,
                )

                continue

            extracted_pages.append(
                (
                    page_index + 1,
                    cleaned_text,
                )
            )

    finally:
        document.close()

    logger.info(
        (
            "Extracted %d usable pages "
            "from %s."
        ),
        len(extracted_pages),
        pdf_path.name,
    )

    return extracted_pages


def looks_like_heading(
    text: str,
) -> bool:
    """
    Detect short heading-like lines.
    """

    stripped = text.strip()

    if not stripped:
        return False

    if len(stripped) > 120:
        return False

    if stripped.endswith(
        (
            ".",
            "?",
            "!",
            ",",
            ";",
            ":",
        )
    ):
        return False

    words = stripped.split()

    if len(words) > 14:
        return False

    uppercase_ratio = sum(
        character.isupper()
        for character in stripped
    ) / max(
        1,
        sum(
            character.isalpha()
            for character in stripped
        ),
    )

    title_case_ratio = sum(
        word[:1].isupper()
        for word in words
        if word
    ) / max(
        1,
        len(words),
    )

    return (
        uppercase_ratio >= 0.6
        or title_case_ratio >= 0.7
    )


def split_into_paragraphs(
    text: str,
) -> list[str]:
    """
    Split cleaned page text into paragraphs.

    Blank lines are preferred boundaries. Single newlines are joined unless
    they appear to represent a short heading.
    """

    raw_blocks = re.split(
        r"\n\s*\n+",
        text,
    )

    paragraphs: list[str] = []

    for block in raw_blocks:
        block_lines = [
            line.strip()
            for line in block.splitlines()
            if line.strip()
        ]

        if not block_lines:
            continue

        current_lines: list[str] = []

        for line in block_lines:
            if looks_like_heading(
                line
            ):
                if current_lines:
                    paragraph = " ".join(
                        current_lines
                    ).strip()

                    if paragraph:
                        paragraphs.append(
                            paragraph
                        )

                    current_lines = []

                paragraphs.append(
                    line
                )

            else:
                current_lines.append(
                    line
                )

        if current_lines:
            paragraph = " ".join(
                current_lines
            ).strip()

            if paragraph:
                paragraphs.append(
                    paragraph
                )

    normalized_paragraphs: list[str] = []

    for paragraph in paragraphs:
        cleaned = re.sub(
            r"\s+",
            " ",
            paragraph,
        ).strip()

        if cleaned:
            normalized_paragraphs.append(
                cleaned
            )

    return normalized_paragraphs


def split_long_text_at_sentence_boundaries(
    text: str,
    max_length: int,
    overlap_target: int,
) -> list[str]:
    """
    Split long text at sentence boundaries.

    Falls back to word boundaries when one sentence itself is too long.
    """

    cleaned_text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if not cleaned_text:
        return []

    if len(cleaned_text) <= max_length:
        return [
            cleaned_text
        ]

    sentences = [
        sentence.strip()
        for sentence in SENTENCE_SPLIT_PATTERN.split(
            cleaned_text
        )
        if sentence.strip()
    ]

    if len(sentences) <= 1:
        return split_long_text_at_word_boundaries(
            text=cleaned_text,
            max_length=max_length,
            overlap_target=overlap_target,
        )

    chunks: list[str] = []

    current_sentences: list[str] = []

    current_length = 0

    for sentence in sentences:
        sentence_length = len(
            sentence
        )

        if sentence_length > max_length:
            if current_sentences:
                chunks.append(
                    " ".join(
                        current_sentences
                    ).strip()
                )

                current_sentences = []
                current_length = 0

            chunks.extend(
                split_long_text_at_word_boundaries(
                    text=sentence,
                    max_length=max_length,
                    overlap_target=overlap_target,
                )
            )

            continue

        proposed_length = (
            current_length
            + sentence_length
            + (
                1
                if current_sentences
                else 0
            )
        )

        if (
            current_sentences
            and proposed_length > max_length
        ):
            completed_chunk = " ".join(
                current_sentences
            ).strip()

            chunks.append(
                completed_chunk
            )

            overlap_sentences = select_sentence_overlap(
                sentences=current_sentences,
                overlap_target=overlap_target,
            )

            current_sentences = list(
                overlap_sentences
            )

            current_length = len(
                " ".join(
                    current_sentences
                )
            )

        current_sentences.append(
            sentence
        )

        current_length = len(
            " ".join(
                current_sentences
            )
        )

    if current_sentences:
        chunks.append(
            " ".join(
                current_sentences
            ).strip()
        )

    return [
        chunk
        for chunk in chunks
        if len(chunk) >= MIN_CHUNK_CHARACTERS
    ]


def split_long_text_at_word_boundaries(
    text: str,
    max_length: int,
    overlap_target: int,
) -> list[str]:
    """
    Split unusually long text at word boundaries.

    This avoids cutting individual words.
    """

    words = text.split()

    if not words:
        return []

    chunks: list[str] = []

    start_index = 0

    while start_index < len(
        words
    ):
        current_words: list[str] = []

        current_length = 0

        end_index = start_index

        while end_index < len(
            words
        ):
            word = words[end_index]

            proposed_length = (
                current_length
                + len(word)
                + (
                    1
                    if current_words
                    else 0
                )
            )

            if (
                current_words
                and proposed_length > max_length
            ):
                break

            current_words.append(
                word
            )

            current_length = proposed_length

            end_index += 1

        chunk = " ".join(
            current_words
        ).strip()

        if (
            len(chunk)
            >= MIN_CHUNK_CHARACTERS
        ):
            chunks.append(
                chunk
            )

        if end_index >= len(
            words
        ):
            break

        overlap_words: list[str] = []

        overlap_length = 0

        overlap_index = end_index - 1

        while overlap_index >= start_index:
            word = words[
                overlap_index
            ]

            proposed_overlap_length = (
                overlap_length
                + len(word)
                + (
                    1
                    if overlap_words
                    else 0
                )
            )

            if (
                overlap_words
                and proposed_overlap_length
                > overlap_target
            ):
                break

            overlap_words.insert(
                0,
                word,
            )

            overlap_length = (
                proposed_overlap_length
            )

            overlap_index -= 1

        next_start_index = max(
            start_index + 1,
            end_index - len(
                overlap_words
            ),
        )

        start_index = (
            next_start_index
        )

    return chunks


def select_sentence_overlap(
    sentences: list[str],
    overlap_target: int,
) -> list[str]:
    """
    Select complete trailing sentences for overlap.

    This prevents the next chunk from starting with a clipped word or fragment.
    """

    selected: list[str] = []

    total_length = 0

    for sentence in reversed(
        sentences
    ):
        proposed_length = (
            total_length
            + len(sentence)
            + (
                1
                if selected
                else 0
            )
        )

        if (
            selected
            and proposed_length
            > overlap_target
        ):
            break

        selected.insert(
            0,
            sentence,
        )

        total_length = proposed_length

        if total_length >= overlap_target:
            break

    return selected


def select_paragraph_overlap(
    paragraphs: list[str],
    overlap_target: int,
) -> list[str]:
    """
    Select complete trailing paragraphs for overlap.

    Short paragraphs are retained intact instead of slicing raw characters.
    """

    selected: list[str] = []

    total_length = 0

    for paragraph in reversed(
        paragraphs
    ):
        proposed_length = (
            total_length
            + len(paragraph)
            + (
                1
                if selected
                else 0
            )
        )

        if (
            selected
            and proposed_length
            > overlap_target
        ):
            break

        selected.insert(
            0,
            paragraph,
        )

        total_length = proposed_length

        if total_length >= overlap_target:
            break

    return selected


def is_low_value_chunk(
    text: str,
) -> bool:
    """
    Detect chunks dominated by license text, navigation material, references,
    URLs, or other low-value content.
    """

    normalized = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    if len(
        normalized
    ) < MIN_CHUNK_CHARACTERS:
        return True

    if any(
        pattern.search(
            normalized
        )
        for pattern in LOW_VALUE_CHUNK_PATTERNS
    ):
        return True

    if REFERENCE_HEADING_PATTERN.fullmatch(
        normalized
    ):
        return True

    if INDEX_HEADING_PATTERN.fullmatch(
        normalized
    ):
        return True

    alphabetic_characters = sum(
        character.isalpha()
        for character in normalized
    )

    if alphabetic_characters < 70:
        return True

    words = re.findall(
        r"[A-Za-z][A-Za-z'-]+",
        normalized,
    )

    if len(words) < 20:
        return True

    url_count = len(
        URL_PATTERN.findall(
            normalized
        )
    )

    if url_count >= 2:
        return True

    numeric_characters = sum(
        character.isdigit()
        for character in normalized
    )

    if (
        numeric_characters
        / max(
            1,
            len(normalized),
        )
        > 0.35
    ):
        return True

    return False


def remove_near_duplicate_chunks(
    chunks: list[str],
) -> list[str]:
    """
    Remove exact normalized duplicates while preserving original order.
    """

    unique_chunks: list[str] = []

    seen: set[str] = set()

    for chunk in chunks:
        normalized = re.sub(
            r"\s+",
            " ",
            chunk.lower(),
        ).strip()

        if not normalized:
            continue

        if normalized in seen:
            continue

        seen.add(
            normalized
        )

        unique_chunks.append(
            chunk
        )

    return unique_chunks


def chunk_page_text(
    text: str,
) -> list[str]:
    """
    Build paragraph-aware chunks from one page.

    Overlap uses complete paragraphs or sentences rather than raw character
    slices, preventing chunks from starting mid-word or mid-sentence.
    """

    paragraphs = split_into_paragraphs(
        text
    )

    if not paragraphs:
        return []

    chunks: list[str] = []

    current_parts: list[str] = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(
            paragraph
        ) > MAX_CHUNK_CHARACTERS:
            if current_parts:
                combined = " ".join(
                    current_parts
                ).strip()

                if not is_low_value_chunk(
                    combined
                ):
                    chunks.append(
                        combined
                    )

                current_parts = []

            long_paragraph_chunks = (
                split_long_text_at_sentence_boundaries(
                    text=paragraph,
                    max_length=CHUNK_SIZE,
                    overlap_target=CHUNK_OVERLAP_TARGET,
                )
            )

            for long_chunk in long_paragraph_chunks:
                if not is_low_value_chunk(
                    long_chunk
                ):
                    chunks.append(
                        long_chunk
                    )

            continue

        proposed_parts = (
            current_parts
            + [
                paragraph
            ]
        )

        proposed_text = " ".join(
            proposed_parts
        ).strip()

        if (
            current_parts
            and len(
                proposed_text
            ) > CHUNK_SIZE
        ):
            completed_chunk = " ".join(
                current_parts
            ).strip()

            if not is_low_value_chunk(
                completed_chunk
            ):
                chunks.append(
                    completed_chunk
                )

            overlap_parts = select_paragraph_overlap(
                paragraphs=current_parts,
                overlap_target=CHUNK_OVERLAP_TARGET,
            )

            current_parts = list(
                overlap_parts
            )

            proposed_with_overlap = " ".join(
                current_parts
                + [
                    paragraph
                ]
            ).strip()

            if len(
                proposed_with_overlap
            ) > MAX_CHUNK_CHARACTERS:
                current_parts = [
                    paragraph
                ]
            else:
                current_parts.append(
                    paragraph
                )

        else:
            current_parts.append(
                paragraph
            )

    if current_parts:
        final_chunk = " ".join(
            current_parts
        ).strip()

        if not is_low_value_chunk(
            final_chunk
        ):
            chunks.append(
                final_chunk
            )

    return remove_near_duplicate_chunks(
        chunks
    )


def build_chunks_from_pdf(
    pdf_path: Path,
) -> list[str]:
    """
    Extract and chunk one PDF while retaining source metadata.

    Every chunk begins with:
        Source: filename
        Page: page number
        Chunk: chunk number
    """

    pages = extract_pdf_pages(
        pdf_path
    )

    pdf_chunks: list[str] = []

    skipped_low_value_chunks = 0

    for page_number, page_text in pages:
        page_chunks = chunk_page_text(
            page_text
        )

        if not page_chunks:
            logger.info(
                (
                    "No useful chunks retained from "
                    "%s page %d."
                ),
                pdf_path.name,
                page_number,
            )

            continue

        for chunk_number, chunk in enumerate(
            page_chunks,
            start=1,
        ):
            cleaned_chunk = re.sub(
                r"\s+",
                " ",
                chunk,
            ).strip()

            if is_low_value_chunk(
                cleaned_chunk
            ):
                skipped_low_value_chunks += 1
                continue

            formatted_chunk = (
                f"Source: {pdf_path.name}\n"
                f"Page: {page_number}\n"
                f"Chunk: {chunk_number}\n\n"
                f"{cleaned_chunk}"
            )

            pdf_chunks.append(
                formatted_chunk
            )

    pdf_chunks = remove_near_duplicate_chunks(
        pdf_chunks
    )

    logger.info(
        (
            "Created %d useful chunks from %s. "
            "Skipped %d low-value chunks."
        ),
        len(pdf_chunks),
        pdf_path.name,
        skipped_low_value_chunks,
    )

    return pdf_chunks


def find_source_pdfs() -> list[Path]:
    """
    Find all PDF files inside `source_books`.
    """

    SOURCE_BOOKS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    pdf_files = sorted(
        path
        for path in SOURCE_BOOKS_DIR.iterdir()
        if (
            path.is_file()
            and path.suffix.lower() == ".pdf"
        )
    )

    if not pdf_files:
        raise FileNotFoundError(
            (
                "No PDF books were found.\n\n"
                "Place your PDF books inside:\n"
                f"{SOURCE_BOOKS_DIR}\n\n"
                "Then run this script again."
            )
        )

    return pdf_files


def embedding_text_from_chunk(
    chunk: str,
) -> str:
    """
    Remove metadata lines before embedding.

    Metadata remains in the stored chunk for traceability but does not influence
    semantic similarity.
    """

    text = re.sub(
        r"^Source:\s*.*?$",
        " ",
        chunk,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    text = re.sub(
        r"^Page:\s*\d+\s*$",
        " ",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    text = re.sub(
        r"^Chunk:\s*\d+\s*$",
        " ",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def encode_chunks(
    chunks: list[str],
) -> np.ndarray:
    """
    Encode all chunks using a CPU-friendly SentenceTransformer model.

    Embeddings are normalized so cosine similarity can be calculated using a
    fast dot product.
    """

    if not chunks:
        raise ValueError(
            "No chunks are available to encode."
        )

    texts_to_encode = [
        embedding_text_from_chunk(
            chunk
        )
        for chunk in chunks
    ]

    if any(
        not text
        for text in texts_to_encode
    ):
        raise ValueError(
            (
                "One or more chunks became empty after "
                "metadata removal."
            )
        )

    logger.info(
        "Loading embedding model: %s",
        EMBEDDING_MODEL,
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL,
        device="cpu",
    )

    logger.info(
        (
            "Encoding %d chunks "
            "with batch size %d."
        ),
        len(texts_to_encode),
        BATCH_SIZE,
    )

    embeddings = model.encode(
        texts_to_encode,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError(
            (
                "Embedding model returned an "
                "unexpected array shape."
            )
        )

    if embeddings.shape[0] != len(
        chunks
    ):
        raise ValueError(
            (
                "The number of generated embeddings "
                "does not match the number of chunks."
            )
        )

    if not np.isfinite(
        embeddings
    ).all():
        raise ValueError(
            (
                "Generated embeddings contain NaN "
                "or infinite values."
            )
        )

    return embeddings


def save_artifacts(
    chunks: list[str],
    embeddings: np.ndarray,
) -> None:
    """
    Atomically save chunks and embeddings together.
    """

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_chunks_path = (
        DATA_DIR
        / "combined_chunks.tmp.pkl"
    )

    temporary_embeddings_path = (
        DATA_DIR
        / "embeddings.tmp.npy"
    )

    logger.info(
        "Saving chunks to %s",
        CHUNKS_OUTPUT_PATH,
    )

    with temporary_chunks_path.open(
        "wb"
    ) as file:
        pickle.dump(
            chunks,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    logger.info(
        "Saving embeddings to %s",
        EMBEDDINGS_OUTPUT_PATH,
    )

    with temporary_embeddings_path.open(
        "wb"
    ) as file:
        np.save(
            file,
            embeddings,
            allow_pickle=False,
        )

    temporary_chunks_path.replace(
        CHUNKS_OUTPUT_PATH
    )

    temporary_embeddings_path.replace(
        EMBEDDINGS_OUTPUT_PATH
    )


def validate_saved_artifacts() -> None:
    """
    Reload and validate generated artifacts.
    """

    with CHUNKS_OUTPUT_PATH.open(
        "rb"
    ) as file:
        saved_chunks = pickle.load(
            file
        )

    saved_embeddings = np.load(
        EMBEDDINGS_OUTPUT_PATH,
        allow_pickle=False,
    )

    if not isinstance(
        saved_chunks,
        list,
    ):
        raise TypeError(
            (
                "Saved chunks are not stored "
                "as a list."
            )
        )

    if saved_embeddings.ndim != 2:
        raise ValueError(
            (
                "Saved embeddings must be a "
                "two-dimensional array."
            )
        )

    if len(
        saved_chunks
    ) != saved_embeddings.shape[0]:
        raise ValueError(
            (
                "Saved artifact mismatch: "
                f"{len(saved_chunks)} chunks but "
                f"{saved_embeddings.shape[0]} "
                "embedding rows."
            )
        )

    if not np.isfinite(
        saved_embeddings
    ).all():
        raise ValueError(
            (
                "Saved embeddings contain NaN or "
                "infinite values."
            )
        )

    chunk_lengths = np.array(
        [
            len(
                embedding_text_from_chunk(
                    chunk
                )
            )
            for chunk in saved_chunks
        ],
        dtype=np.int64,
    )

    logger.info(
        (
            "Validation passed | "
            "chunks=%d | "
            "embedding_shape=%s | "
            "average_chunk_chars=%.1f | "
            "median_chunk_chars=%.1f"
        ),
        len(saved_chunks),
        tuple(
            saved_embeddings.shape
        ),
        float(
            chunk_lengths.mean()
        ),
        float(
            np.median(
                chunk_lengths
            )
        ),
    )


def print_sample_chunks(
    chunks: list[str],
    number_of_samples: int = 5,
) -> None:
    """
    Print a few deterministic sample chunks for manual inspection.
    """

    if not chunks:
        return

    sample_indices = np.linspace(
        0,
        len(chunks) - 1,
        num=min(
            number_of_samples,
            len(chunks),
        ),
        dtype=int,
    )

    print()
    print("=" * 80)
    print("SAMPLE CHUNKS")
    print("=" * 80)

    for sample_number, index in enumerate(
        sample_indices,
        start=1,
    ):
        print()
        print(
            f"Sample {sample_number} | "
            f"Chunk index {index}"
        )

        print("-" * 80)

        print(
            chunks[index][
                :1000
            ]
        )


def main() -> None:
    """
    Complete preprocessing pipeline:

    1. Find source PDFs.
    2. Extract and clean page text.
    3. Remove common noise and low-value content.
    4. Build paragraph- and sentence-aware overlapping chunks.
    5. Generate normalized embeddings.
    6. Save both artifacts.
    7. Reload and validate them.
    8. Print sample chunks for inspection.
    """

    logger.info(
        "Starting MindHaven knowledge-base build."
    )

    pdf_files = find_source_pdfs()

    logger.info(
        "Found %d PDF file(s).",
        len(pdf_files),
    )

    all_chunks: list[str] = []

    for pdf_path in pdf_files:
        pdf_chunks = build_chunks_from_pdf(
            pdf_path
        )

        all_chunks.extend(
            pdf_chunks
        )

    all_chunks = remove_near_duplicate_chunks(
        all_chunks
    )

    if not all_chunks:
        raise RuntimeError(
            (
                "No usable text chunks were created. "
                "The PDFs may contain only scanned images "
                "or the cleaning filters may be too strict."
            )
        )

    logger.info(
        "Total useful chunks created: %d",
        len(all_chunks),
    )

    embeddings = encode_chunks(
        all_chunks
    )

    save_artifacts(
        chunks=all_chunks,
        embeddings=embeddings,
    )

    validate_saved_artifacts()

    logger.info(
        "Knowledge-base build completed successfully."
    )

    print()
    print("Done.")

    print(
        f"Chunks saved to: "
        f"{CHUNKS_OUTPUT_PATH}"
    )

    print(
        f"Embeddings saved to: "
        f"{EMBEDDINGS_OUTPUT_PATH}"
    )

    print(
        f"Total chunks: "
        f"{len(all_chunks):,}"
    )

    print(
        f"Embedding shape: "
        f"{embeddings.shape}"
    )

    print_sample_chunks(
        chunks=all_chunks,
        number_of_samples=5,
    )


if __name__ == "__main__":
    main()