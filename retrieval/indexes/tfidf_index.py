from core.schemas import Chunk, SearchResult
from representations.tfidf.tfidf import compute_idf, tfidf_vector
from representations.tfidf.tokenizer import tokenize
from representations.tfidf.similarity import cosine_similarity


class TfidfRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        chunks_tokens = [tokenize(c.text) for c in chunks]
        self.idf = compute_idf(chunks_tokens)
        self.vectors = [tfidf_vector(t, self.idf) for t in chunks_tokens]

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        q_vec = tfidf_vector(tokenize(query), self.idf)
        results = [SearchResult(c, cosine_similarity(q_vec, v))
                   for c, v in zip(self.chunks, self.vectors)]
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]
