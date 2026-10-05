import json

from generation.groq_llm import GroqLLM
from ingestion.loader import load_corpus
from retrieval.indexes.dense_index import DenseRetriever

retriever = DenseRetriever(load_corpus("data/corpus_arxiv"))
llm = GroqLLM()

# 1. The tool = a real function + a description the model can read
def semantic_search(query: str, k: int = 3) -> str:
    hits = retriever.search(query, k=k)
    return json.dumps([{"chunk_id": h.chunk.chunk_id, "score": round(h.score, 2),
                        "text": h.chunk.text[:200]} for h in hits])

TOOLS = [{"type": "function", "function": {
    "name": "semantic_search",
    "description": "Search the paper collection and return the most relevant text chunks.",
    "parameters": {"type": "object",
                   "properties": {"query": {"type": "string", "description": "what to look for"},
                                  "k": {"type": "integer", "description": "number of results"}},
                   "required": ["query"]}}}]
REGISTRY = {"semantic_search": semantic_search}

for question in ["Which paper helps locate the part of an input that was injected?",
                 "What is 12 times 12?"]:
    print("=" * 70, "\nUSER:", question)
    messages = [{"role": "system", "content": "You are a research assistant. Use tools when you need information from the paper collection."},
                {"role": "user", "content": question}]
    reply = llm.chat(messages, tools=TOOLS)
    if not reply.get("tool_calls"):
        print("MODEL ANSWERED DIRECTLY (no tool call):", reply["content"])
        continue

    call = reply["tool_calls"][0]
    print("MODEL ASKS FOR A TOOL:", call["function"]["name"], call["function"]["arguments"])
    args = json.loads(call["function"]["arguments"])        # structured output -> dict
    result = REGISTRY[call["function"]["name"]](**args)      # WE run the function
    print("WE RAN IT, result starts:", result[:150], "...")

    messages.append({"role": "assistant", "content": reply.get("content"), "tool_calls": reply["tool_calls"]})
    messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
    final = llm.chat(messages, tools=TOOLS)
    print("FINAL ANSWER:", final["content"][:300])
