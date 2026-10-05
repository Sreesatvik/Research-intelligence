import re
import time

from evaluation.multimodal_cases import CASES
from generation.groq_llm import GroqLLM
from generation.groq_vision import GroqVLM
from ingestion.loader import load_pdfs
from rag.multimodal import MultimodalRAG
from retrieval.indexes.dense_index import DenseRetriever

chunks = load_pdfs("data/pdfs")
rag = MultimodalRAG(DenseRetriever(chunks), GroqLLM(), GroqVLM(), chunks)
ok_by_kind = {"table": [0, 0], "figure": [0, 0]}
for q, pattern, _, kind in CASES:
    mode, a = rag.ask(q)
    ok = bool(re.search(pattern, a.answer, re.I | re.S))
    ok_by_kind[kind][0] += ok
    ok_by_kind[kind][1] += 1
    print(f"{kind:6s} route={mode:6s} {'OK ' if ok else 'BAD'} cited={a.cited_chunk_ids} | {q[26:78]}")
    time.sleep(0.3)
print("\n", {k: f"{v[0]}/{v[1]}" for k, v in ok_by_kind.items()})
