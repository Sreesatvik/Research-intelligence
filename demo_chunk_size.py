from evaluation.run_eval import evaluate, load_eval_set
from ingestion.loader import load_corpus
from retrieval.indexes.dense_index import DenseRetriever
from retrieval.indexes.tfidf_index import TfidfRetriever

eval_set = load_eval_set("data/eval/queries_arxiv.json")
print("chunk_size | chunks | engine  | recall@3 |  mrr")
for size in (30, 50, 100, 200, 1000):
    chunks = load_corpus("data/corpus_arxiv", chunk_size=size, overlap=size // 5)
    for name, cls in (("TF-IDF", TfidfRetriever), ("Dense", DenseRetriever)):
        s = evaluate(cls(chunks), eval_set, k=3)
        print(f"{size:10d} | {len(chunks):6d} | {name:7s} | {s['recall@3']:8.3f} | {s['mrr']:.3f}")
