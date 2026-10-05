from fastapi.testclient import TestClient

from apps.api.main import create_app
from cache.answer_cache import AnswerCache
from generation.groq_llm import GroqLLM
from ingestion.loader import load_corpus
from retrieval.indexes.dense_index import DenseRetriever

retriever = DenseRetriever(load_corpus("data/corpus_arxiv"))
cache = AnswerCache(embed=lambda t: retriever.model.encode(t, normalize_embeddings=True))
llm = GroqLLM("openai/gpt-oss-120b")
client = TestClient(create_app(retriever, llm, cache=cache))

QUESTIONS = [
    "What is the main idea of active retrieval augmented generation?",
    "What is the main idea of active retrieval augmented generation?",       # identical
    "what is the MAIN idea of active retrieval-augmented generation??",      # trivially different
    "Can you explain the core idea behind active retrieval augmented generation?",  # reworded
    "What is the capital of France?",                                         # unanswerable
    "What is the capital of France?",                                         # unanswerable, repeated
]
total = 0
for q in QUESTIONS:
    r = client.post("/ask", json={"question": q}).json()
    total += r["tokens_used"]
    print(f"cache={r['cache_hit']:8s} tokens={r['tokens_used']:5d} refused_early={r['refused_by_retrieval']!s:5} | {q[:62]}")
print(f"\nTOTAL tokens for {len(QUESTIONS)} requests: {total}   (cache stats: {cache.stats})")
