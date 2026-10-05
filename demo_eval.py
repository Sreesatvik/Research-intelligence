from evaluation.run_eval import evaluate, load_eval_set
from ingestion.loader import load_corpus
from retrieval.indexes.dense_index import DenseRetriever
from retrieval.indexes.tfidf_index import TfidfRetriever

eval_set = load_eval_set()
chunks = load_corpus("data/corpus", chunk_size=100, overlap=20)

print(f"{len(eval_set)} queries, {len(chunks)} chunks\n")
for name, retriever in [("TF-IDF", TfidfRetriever(chunks)),
                        ("Dense", DenseRetriever(chunks))]:
    scores = evaluate(retriever, eval_set, k=3)
    print(name, {key: round(v, 3) for key, v in scores.items()})
