import re

from evaluation.multimodal_cases import CASES
from generation.groq_llm import GroqLLM
from generation.groq_vision import GroqVLM
from ingestion.loader import load_pdfs
from rag.multimodal import MultimodalRAG
from retrieval.indexes.dense_index import DenseRetriever

chunks = load_pdfs("data/pdfs")
rag = MultimodalRAG(DenseRetriever(chunks), GroqLLM(), GroqVLM(), chunks)
REPS = 3
total = good = 0
for q, pattern, _, kind in CASES:
    if kind != "table":
        continue
    wins = sum(bool(re.search(pattern, rag.ask(q)[1].answer)) for _ in range(REPS))
    total += REPS
    good += wins
    print(f"{wins}/{REPS} | {q[26:92]}")
print(f"\ntable questions with cropping: {good}/{total} correct")
