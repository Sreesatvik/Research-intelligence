"""Shared wiring so demos and the API build the same agent."""
import json
from pathlib import Path

from agent.agent import ResearchAgent
from agent.tools.research_tools import build_tools
from classification.classifier import DocumentClassifier
from generation.groq_llm import GroqLLM
from ingestion.loader import load_corpus
from retrieval.indexes.dense_index import DenseRetriever


def make_classifier() -> DocumentClassifier:
    truth = json.load(open("data/eval/topics.json", encoding="utf-8"))
    paths = sorted(Path("data/corpus_arxiv").glob("*.txt"))
    return DocumentClassifier().fit([p.read_text(encoding="utf-8") for p in paths],
                                    [truth[p.stem] for p in paths])


def make_agent(extra_chunks=None, log_path="data/logs/agent_calls.jsonl", max_steps=6,
               classifier=None, retriever=None, chunks=None, model="openai/gpt-oss-20b"):
    chunks = chunks or load_corpus("data/corpus_arxiv") + (extra_chunks or [])
    registry = build_tools(retriever or DenseRetriever(chunks), chunks, classifier or make_classifier())
    return ResearchAgent(GroqLLM(model), registry, max_steps=max_steps, log_path=log_path)
