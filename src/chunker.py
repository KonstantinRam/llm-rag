from dataclasses import dataclass
from typing import Any

SENTENCE_BOUNDARIES = ('. ', '.\n', '? ', '?\n', '! ', '!\n', '\n\n')

@dataclass(frozen=True, slots = True)
class Chunk:
    text: str
    metadata: dict[str, Any]
    index: int


def chunk_text(
        text: str,
        chunk_size: int = 512,
        overlap: int = 50,
        metadata: dict | None = None
) -> list[Chunk]:
    """Split text into overlapping chunks.

    Args:
        text: The input text to chunk.
        chunk_size: Maximum characters per chunk.
        overlap:
        metadata:
    Returns:
        List of Chunk objects with metadata.



    """
    if metadata is None:
        metadata = {}

    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = start + chunk_size
        segment = text[start:end]

        # Try to break at sentence boundary if we're mid-text
        if end < len(text):
            # Look for last sentence-ending punctuation
            for sep in SENTENCE_BOUNDARIES:
                last_sep = segment.rfind(sep)
                if last_sep > chunk_size // 2:  # Only if we keep >50% of a chunk
                    segment = segment[:last_sep + 1]
                    end = start + last_sep + 1
                    break

        chunks.append(Chunk(
            text=segment.strip(),
            metadata={**metadata, 'chunk_index': index},
            index=index
        ))

        start = end - overlap
        index += 1

    return chunks