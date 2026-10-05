from fastapi.testclient import TestClient

from apps.api.main import create_app
from core.schemas import Chunk, SearchResult


class FakeRetriever:
    def __init__(self, score: float):
        self.score = score

    def search(self, query, k=5):
        chunk = Chunk("docA#0", "docA", "Dogs are mammals.", 0)
        return [SearchResult(chunk, self.score)]


class FakeLLM:
    def generate(self, system, user, temperature=0.0):
        return "Dogs are mammals [1]."


def client(score: float) -> TestClient:
    return TestClient(create_app(FakeRetriever(score), FakeLLM()))


def test_health():
    assert client(0.9).get("/health").json() == {"status": "ok"}


def test_search_returns_provenance():
    body = client(0.9).post("/search", json={"query": "dogs"}).json()
    assert body["results"][0]["chunk_id"] == "docA#0"


def test_ask_cites_sources():
    body = client(0.9).post("/ask", json={"question": "what are dogs"}).json()
    assert body["cited_chunk_ids"] == ["docA#0"]
    assert body["refused_by_retrieval"] is False


def test_ask_refuses_when_evidence_is_weak():
    body = client(0.05).post("/ask", json={"question": "capital of France"}).json()
    assert body["refused_by_retrieval"] is True
    assert body["cited_chunk_ids"] == []


def test_empty_question_is_rejected():
    assert client(0.9).post("/ask", json={"question": ""}).status_code == 422


class CountingLLM:
    """Fake LLM that counts calls and reports token usage like the real adapter."""

    def __init__(self):
        self.calls, self.tokens_used = 0, 0

    def generate(self, system, user, temperature=0.0):
        self.calls += 1
        self.tokens_used += 100
        return "Dogs are mammals [1]."


def test_cache_saves_llm_calls_and_reports_tokens():
    from cache.answer_cache import AnswerCache

    llm = CountingLLM()
    c = TestClient(create_app(FakeRetriever(0.9), llm, cache=AnswerCache()))
    first = c.post("/ask", json={"question": "What are dogs?"}).json()
    second = c.post("/ask", json={"question": "what are dogs"}).json()
    assert (first["cache_hit"], first["tokens_used"]) == ("none", 100)
    assert (second["cache_hit"], second["tokens_used"]) == ("exact", 0)
    assert llm.calls == 1  # the second request never reached the LLM


def test_cache_does_not_mix_different_k():
    from cache.answer_cache import AnswerCache

    llm = CountingLLM()
    c = TestClient(create_app(FakeRetriever(0.9), llm, cache=AnswerCache()))
    c.post("/ask", json={"question": "What are dogs?", "k": 3})
    c.post("/ask", json={"question": "What are dogs?", "k": 5})
    assert llm.calls == 2
