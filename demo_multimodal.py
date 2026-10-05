from generation.groq_llm import GroqLLM
from generation.groq_vision import GroqVLM
from ingestion.loader import load_pdfs
from rag.multimodal import MultimodalRAG
from retrieval.indexes.dense_index import DenseRetriever

chunks = load_pdfs("data/pdfs")
rag = MultimodalRAG(DenseRetriever(chunks), GroqLLM(), GroqVLM(), chunks)
for q in ["In the FLARE paper, what exact-match score does single-time retrieval get on StrategyQA?",
          "What is the main idea of active retrieval augmented generation?"]:
    mode, a = rag.ask(q)
    print(f"\nQ: {q}\nroute: {mode} | cited: {a.cited_chunk_ids}\nA: {a.answer[:300]}")
