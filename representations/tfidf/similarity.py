import math

from representations.tfidf.tokenizer import tokenize
from representations.tfidf.tfidf import compute_idf, tfidf_vector


def cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine of the angle between two sparse vectors."""
    shared = set(a) & set(b)
    dot = sum(a[w] * b[w] for w in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
