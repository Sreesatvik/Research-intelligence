from evaluation.run_eval import evaluate, load_eval_set
from ingestion.loader import load_corpus
from retrieval.indexes.dense_index import DenseRetriever
from retrieval.indexes.tfidf_index import TfidfRetriever

eval_set = load_eval_set("data/eval/queries_arxiv.json")
chunks = load_corpus("data/corpus_arxiv", chunk_size=100, overlap=20)
print(f"{len(eval_set)} queries, {len(chunks)} chunks\n")

engines = {"TF-IDF": TfidfRetriever(chunks), "Dense": DenseRetriever(chunks)}
for k in (1, 3, 5):
    for name, engine in engines.items():
        s = evaluate(engine, eval_set, k=k)
        print(f"k={k} {name:7s}", {key: round(v, 3) for key, v in s.items()})
