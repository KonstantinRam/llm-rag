from dataclasses import dataclass

import torch
import logging
from sentence_transformers import SentenceTransformer


from src.chunker import Chunk
"""
TODO: from .chunker import Chunk. Packaging
"""

logger = logging.getLogger(__name__)
class EmbedderError(Exception):
    pass


class ModelLoadError(EmbedderError):
    """Failed to load an embedding model."""
    pass

@dataclass
class EmbeddedChunk:
    chunk: Chunk
    embedding: list[float]

def _select_device() -> str:
    if torch.cuda.is_available():
        # Check if CUDA actually works, not just "available"
        try:
            torch.cuda.current_device()
            return "cuda"
        except RuntimeError as e:
            logger.warning(f"CUDA available but not functional: {e}")
            return "cpu"
    return "cpu"


class Embedder:
    device: str
    model: SentenceTransformer
    dimension: int


    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        all-MiniLM-L6-v2: 384 dimensions, fast, decent quality
        all-mpnet-base-v2: 768 dimensions, slower, better quality

        We use MiniLM; it's small, runs locally.
        """

        self.device = _select_device()
        self.model = self._load_model(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

        logger.info(f"Embedder ready on {self.device}, dimension={self.dimension}")

    def _load_model(self, model_name: str) -> SentenceTransformer:
        try:
            return SentenceTransformer(model_name, device=self.device)
        except torch.cuda.OutOfMemoryError:
            if self.device == "cuda":
                logger.warning("CUDA OOM during model load, retrying on CPU")
                self.device = "cpu"
                return SentenceTransformer(model_name, device="cpu")
            raise  # Already on CPU, nothing we can do
        except OSError as e:
            # Model isn't found locally or failed to download
            raise ModelLoadError(f"Could not load model '{model_name}'") from e

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        tokens = self.model.tokenizer(text)
        if len(tokens['input_ids']) > self.model.max_seq_length:
            logger.warning(f"Text truncated: {len(tokens['input_ids'])} tokens > {self.model.max_seq_length}")

        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

    def embed_chunks(self, chunks: list[Chunk], batch_size: int = 32) -> list[EmbeddedChunk]:
        texts = [c.text for c in chunks]
        embeddings = self.model.encode(texts, batch_size=batch_size, convert_to_tensor=False, show_progress_bar=True)
        return [EmbeddedChunk(c, e.tolist()) for c, e in zip(chunks, embeddings)]