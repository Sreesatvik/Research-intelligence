from evaluation.metrics import unique_documents
from evaluation.run_eval import load_eval_set
from ingestion.loader import load_corpus
from retrieval.indexes.dense_index import DenseRetriever

eval_set = load_eval_set("data/eval/queries_arxiv.json")
chunks = load_corpus("data/corpus_arxiv")
titles = {c.document_id: c.text[:60] for c in chunks if c.position == 0}
dense = DenseRetriever(chunks)

for item in eval_set:
    results = dense.search(item["query"], k=len(chunks))
    ranked = unique_documents([r.chunk.document_id for r in results])
    best = min(ranked.index(d) for d in item["relevant"]) + 1
    if best <= 1:
        continue
    print(f"\nQ: {item['query']}\n   wanted rank {best}: {item['relevant']}")
    for d in ranked[:3]:
        mark = "*" if d in item["relevant"] else " "
        print(f"   {mark} {d}  {titles[d]}")
