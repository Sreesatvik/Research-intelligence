import json
import time

from evaluation.run_eval import load_eval_set
from generation.groq_llm import GroqLLM
from ingestion.loader import load_corpus
from rag.pipeline import RAG
from retrieval.indexes.dense_index import DenseRetriever

rag = RAG(DenseRetriever(load_corpus("data/corpus_arxiv")), GroqLLM(), k=3, min_score=0.25)
answerable = load_eval_set("data/eval/queries_arxiv.json")
unanswerable = json.load(open("data/eval/unanswerable.json", encoding="utf-8"))


def is_refusal(text: str) -> bool:
    return "i don't know" in text.lower().replace("\u2019", "'")


answered = cited_right = 0
for item in answerable:
    a = rag.ask(item["query"])
    ok = not is_refusal(a.answer)
    docs = {c.split("#")[0] for c in a.cited_chunk_ids}
    hit = bool(docs & set(item["relevant"]))
    answered += ok
    cited_right += hit
    if not (ok and hit):
        print(f"  PROBLEM  answered={ok} cited_right={hit} | {item['query']}")
    time.sleep(0.3)

refused_early = refused_llm = 0
for q in unanswerable:
    a = rag.ask(q)
    if a.refused_by_retrieval:
        refused_early += 1
    elif is_refusal(a.answer):
        refused_llm += 1
    else:
        print(f"  HALLUCINATION RISK | {q}\n    -> {a.answer[:200]}")
    time.sleep(0.3)

n, m = len(answerable), len(unanswerable)
print(f"\nANSWERABLE ({n}): answered {answered}/{n}, cited the right paper {cited_right}/{n}")
print(f"UNANSWERABLE ({m}): refused by threshold {refused_early}, "
      f"refused by LLM {refused_llm}, answered anyway {m - refused_early - refused_llm}")
