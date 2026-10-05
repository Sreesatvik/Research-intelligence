from core.interfaces import Retriever
from ingestion.loader import load_corpus
from retrieval.indexes.tfidf_index import TfidfRetriever
from retrieval.indexes.dense_index import DenseRetriever


def show(name: str, retriever: Retriever, query: str, k: int = 3) -> None:
    print(f"\n[{name}] {query}")
    for r in retriever.search(query, k):
        print(f"  {r.score:.3f}  {r.chunk.chunk_id:28s} {r.chunk.text[:55]}...")


chunks = load_corpus("data/corpus")
engines = {"TF-IDF": TfidfRetriever(chunks), "Dense ": DenseRetriever(chunks)}
queries = [
    "how do agents protect user privacy",
    "how do I know if my search system is any good",
    "why does a model split rare words into pieces",
]
for q in queries:
    for name, engine in engines.items():  # same call, any engine
        show(name, engine, q)
