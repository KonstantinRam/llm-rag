import torch
from sentence_transformers import SentenceTransformer

from src.chunker import Chunk


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        all-MiniLM-L6-v2: 384 dimensions, fast, decent quality
        all-mpnet-base-v2: 768 dimensions, slower, better quality

        For production RAG, you'd likely use:
        - OpenAI's text-embedding-3-small/large
        - Cohere's embed-v3
        - Or fine-tuned domain-specific models

        We use MiniLM because it's small, runs locally, and
        demonstrates the concepts without API costs.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()

        print(f"Embedder loaded on {self.device}, dimension={self.dimension}")

    def embed_text(self, text: str) -> list[float]:
        """Embed a single text string."""
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

    def embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        """
        Embed multiple chunks efficiently.

        Batching matters for GPU utilization - encoding one at a time
        underutilizes the parallel compute. SentenceTransformer handles
        batching internally.
        """
        texts = [chunk.text for chunk in chunks]
        embeddings = self.model.encode(texts, convert_to_tensor=False, show_progress_bar=True)
        return embeddings.tolist()