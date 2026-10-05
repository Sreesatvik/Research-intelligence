from generation.groq_llm import GroqLLM
from ingestion.loader import load_corpus
from rag.pipeline import RAG
from retrieval.indexes.dense_index import DenseRetriever

rag = RAG(DenseRetriever(load_corpus("data/corpus_arxiv")), GroqLLM(), k=3)
q = "Why are prompt injection attacks a problem for LLM applications?"
for temp in (0.0, 1.5):
    print(f"\n=== temperature {temp} ===")
    for _ in range(3):
        print("-", rag.ask(q, temperature=temp).answer.replace("\n", " ")[:230])
