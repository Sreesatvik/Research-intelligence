import json

from evaluation.run_eval import load_eval_set
from ingestion.loader import load_corpus
from retrieval.indexes.dense_index import DenseRetriever

dense = DenseRetriever(load_corpus("data/corpus_arxiv"))
answerable = [q["query"] for q in load_eval_set("data/eval/queries_arxiv.json")]
unanswerable = json.load(open("data/eval/unanswerable.json", encoding="utf-8"))


def top_score(q: str) -> float:
    return dense.search(q, k=1)[0].score


a = sorted(top_score(q) for q in answerable)
u = sorted(top_score(q) for q in unanswerable)
print("answerable   top scores: min %.2f  median %.2f  max %.2f" % (a[0], a[len(a)//2], a[-1]))
print("unanswerable top scores: min %.2f  median %.2f  max %.2f" % (u[0], u[len(u)//2], u[-1]))
print("\nunanswerable scores:", [round(x, 2) for x in u])
print("lowest answerable  :", [round(x, 2) for x in a[:5]])

print("\nthreshold | answerable kept | unanswerable refused")
for t in (0.10, 0.15, 0.20, 0.25, 0.30):
    kept = sum(s >= t for s in a) / len(a)
    refused = sum(s < t for s in u) / len(u)
    print(f"  {t:.2f}    |     {kept:5.0%}      |       {refused:5.0%}")
