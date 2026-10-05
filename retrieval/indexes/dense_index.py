import numpy as np
from sentence_transformers import SentenceTransformer

from core.schemas import Chunk, SearchResult


class DenseRetriever:
    def __init__(self, chunks: list[Chunk], model_name: str = "all-MiniLM-L6-v2"):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        self.vectors = self.model.encode([c.text for c in chunks],
                                         normalize_embeddings=True)

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        q_vec = self.model.encode(query, normalize_embeddings=True)
        scores = self.vectors @ q_vec
        top = np.argsort(-scores)[:k]
        return [SearchResult(self.chunks[i], float(scores[i])) for i in top]
