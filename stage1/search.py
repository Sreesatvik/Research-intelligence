import math

from tokenize_count import tokenize
from tfidf import compute_idf, tfidf_vector


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine of the angle between two sparse vectors."""
    shared = set(a) & set(b)
    dot = sum(a[w] * b[w] for w in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class TfidfSearch:
    def __init__(self, docs: list[str]):
        self.docs = docs
        docs_tokens = [tokenize(d) for d in docs]
        self.idf = compute_idf(docs_tokens)
        self.doc_vectors = [tfidf_vector(t, self.idf) for t in docs_tokens]

    def search(self, query: str, k: int = 3) -> list[tuple[float, str]]:
        q_vec = tfidf_vector(tokenize(query), self.idf)
        scored = [(cosine_similarity(q_vec, v), d)
                  for v, d in zip(self.doc_vectors, self.docs)]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]


if __name__ == "__main__":
    docs = [
        "Browser agents can leak personal data when navigating websites.",
        "We prevent leakage of private information in web-navigating LLM assistants.",
        "Convolutional networks achieve strong results on image classification.",
        "Transformers use attention to model relationships between tokens.",
    ]
    engine = TfidfSearch(docs)
    for q in ["browser agents privacy", "attention in transformers"]:
        print(f"\nQuery: {q}")
        for score, doc in engine.search(q):
            print(f"  {score:.3f}  {doc}")
