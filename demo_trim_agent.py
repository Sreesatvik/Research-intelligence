from agent.agent import ResearchAgent
from agent.setup import make_classifier
from agent.tools.research_tools import build_tools
from generation.groq_llm import GroqLLM
from ingestion.loader import load_corpus
from retrieval.indexes.dense_index import DenseRetriever

# (question, [ [acceptable ids for fact 1], [acceptable ids for fact 2], ... ])
CASES = [
    ("Find the paper that localizes injected prompts and the paper that defends against prompt injection "
     "by signing prompts. Give each paper's document_id.", [["2510.12252v2"], ["2401.07612v1"]]),
    ("Which paper speeds up vision transformers by dropping tokens, and which paper proposes linear-time "
     "attention? Give each paper's document_id.", [["2111.15667v3"], ["2412.17810v1"]]),
    ("Find one paper about topic discovery from short texts and one about sentence embeddings with little "
     "training data. Give each paper's document_id.", [["2512.06694v2", "2006.09977v1"], ["2211.04928v2"]]),
    ("Find the paper about running a large language model inside the web browser, give its document_id "
     "and tell me what topic the classifier assigns to it.", [["2412.15803v2"]]),
]
MODEL = "openai/gpt-oss-120b"

chunks = load_corpus("data/corpus_arxiv")
retriever = DenseRetriever(chunks)
registry = build_tools(retriever, chunks, make_classifier())

totals = {"full history": [0, 0, 0, 0], "compacted (keep 1)": [0, 0, 0, 0]}  # tokens, steps, facts ok, facts total
for question, facts in CASES:
    for name, keep in (("full history", None), ("compacted (keep 1)", 1)):
        llm = GroqLLM(MODEL)
        result = ResearchAgent(llm, registry, max_steps=6, keep_last_results=keep).run(question)
        ok = sum(any(i in result.answer for i in group) for group in facts)
        t = totals[name]
        t[0] += llm.tokens_used; t[1] += len(result.calls); t[2] += ok; t[3] += len(facts)
        print(f"{name:17s} tokens={llm.tokens_used:6d} tool_calls={len(result.calls)} "
              f"facts_found={ok}/{len(facts)} stopped={result.stopped_because} | {question[:50]}", flush=True)

print()
for name, (tok, steps, ok, n) in totals.items():
    print(f"TOTAL {name:17s}: {tok:6d} tokens, {steps} tool calls, correct facts {ok}/{n}", flush=True)
