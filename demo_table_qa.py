from generation.groq_llm import GroqLLM
from ingestion.loader import load_pdfs
from rag.pipeline import RAG
from retrieval.indexes.dense_index import DenseRetriever

rag = RAG(DenseRetriever(load_pdfs("data/pdfs")), GroqLLM(), k=4)
q = "In the FLARE paper, what exact-match score does single-time retrieval get on StrategyQA?"
out = rag.ask(q)
print("Q:", q)
for r in out.retrieved:
    c = r.chunk
    print(f"  {r.score:.2f} {c.chunk_id:18s} p{c.page} {c.content_type:5s} {c.text[:70]!r}")
print("\nANSWER:", out.answer)
