from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    metadata: dict
    index: int


def chunk_text(
        text: str,
        chunk_size: int = 512,
        overlap: int = 50,
        metadata: dict | None = None
) -> list[Chunk]:
    """
    Split text into overlapping chunks.

    Overlap prevents losing context at boundaries - if a relevant
    sentence gets split, it appears in both chunks.

    512 tokens is a common size because:
    - Embedding models have context limits (often 512)
    - Smaller chunks = more precise retrieval but less context
    - Larger chunks = more context but retrieval noise
    """
    if metadata is None:
        metadata = {}

    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        # Try to break at sentence boundary if we're mid-text
        if end < len(text):
            # Look for last sentence-ending punctuation
            for sep in ['. ', '.\n', '? ', '!\n', '\n\n']:
                last_sep = chunk_text.rfind(sep)
                if last_sep > chunk_size // 2:  # Only if we keep >50% of chunk
                    chunk_text = chunk_text[:last_sep + 1]
                    end = start + last_sep + 1
                    break

        chunks.append(Chunk(
            text=chunk_text.strip(),
            metadata={**metadata, 'chunk_index': index},
            index=index
        ))

        start = end - overlap
        index += 1

    return chunks