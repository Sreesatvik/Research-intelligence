import json
import time

from core.interfaces import Retriever
from evaluation.metrics import (precision_at_k, recall_at_k,
                                reciprocal_rank, unique_documents)


def evaluate(retriever: Retriever, eval_set: list[dict], k: int = 3) -> dict:
    """Run every query and average each metric over the eval set."""
    for item in eval_set:
        if not item["relevant"]:
            raise ValueError(f"query has no relevant docs: {item['query']}")
    recalls, precisions, rrs = [], [], []
    start = time.perf_counter()
    for item in eval_set:
        # retrieve extra chunks, since several chunks can share one document
        results = retriever.search(item["query"], k=k * 3)
        ranked = unique_documents([r.chunk.document_id for r in results])
        relevant = set(item["relevant"])
        recalls.append(recall_at_k(ranked, relevant, k))
        precisions.append(precision_at_k(ranked, relevant, k))
        rrs.append(reciprocal_rank(ranked, relevant))
    n = len(eval_set)
    return {
        f"recall@{k}": sum(recalls) / n,
        f"precision@{k}": sum(precisions) / n,
        "mrr": sum(rrs) / n,
        "ms/query": (time.perf_counter() - start) / n * 1000,
    }


def load_eval_set(path: str = "data/eval/queries.json") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
