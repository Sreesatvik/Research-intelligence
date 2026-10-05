def unique_documents(ranked_doc_ids: list[str]) -> list[str]:
    """Drop repeated documents, keeping the rank of the first occurrence."""
    seen = []
    for doc_id in ranked_doc_ids:
        if doc_id not in seen:
            seen.append(doc_id)
    return seen


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the relevant documents that appear in the top k."""
    hits = len(set(ranked[:k]) & relevant)
    return hits / len(relevant)


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top k documents that are relevant."""
    hits = len(set(ranked[:k]) & relevant)
    return hits / k


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """1 / rank of the first relevant document, or 0 if none was retrieved."""
    for rank, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1 / rank
    return 0.0
