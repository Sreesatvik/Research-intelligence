from generation.groq_llm import GroqLLM
from ingestion.loader import load_corpus
from rag.pipeline import RAG
from retrieval.indexes.dense_index import DenseRetriever

rag = RAG(DenseRetriever(load_corpus("data/corpus_arxiv")), GroqLLM(), k=3)

for q in ["Can a model decide by itself when to retrieve documents while writing?",
          "What is the capital of France?"]:
    a = rag.ask(q)
    print("\nQ:", q)
    print("A:", a.answer)
    print("cited:", a.cited_chunk_ids)
    print("shown:", [(r.chunk.chunk_id, round(r.score, 2)) for r in a.retrieved])
