from evaluation.metrics import unique_documents
from evaluation.run_eval import evaluate, load_eval_set
from ingestion.loader import load_corpus
from retrieval.indexes.dense_index import DenseRetriever
from retrieval.indexes.tfidf_index import TfidfRetriever

eval_set = load_eval_set()
chunks = load_corpus("data/corpus")
engines = {"TF-IDF": TfidfRetriever(chunks), "Dense": DenseRetriever(chunks)}

for name, eng in engines.items():
    s = evaluate(eng, eval_set, k=1)
    print(name, "k=1", {k: round(v, 3) for k, v in s.items() if k != "ms/query"})

print("\nQueries where the top-1 document is WRONG:")
for item in eval_set:
    for name, eng in engines.items():
        top = unique_documents([r.chunk.document_id for r in eng.search(item["query"], 9)])[0]
        if top not in item["relevant"]:
            print(f"  [{name}] '{item['query']}' -> got {top}, wanted {item['relevant']}")
