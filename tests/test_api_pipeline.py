from fastapi.testclient import TestClient

from agent.agent import AgentResult, ToolCallRecord
from apps.api.main import create_app
from cache.answer_cache import AnswerCache
from core.schemas import Chunk, SearchResult
from quota.token_quota import TokenQuota


class FakeRetriever:
    def search(self, query, k=5):
        return [SearchResult(Chunk("docA#0", "docA", "Dogs are mammals.", 0), 0.9),
                SearchResult(Chunk("docB#0", "docB", "Cats are mammals.", 0), 0.8)]


class CountingLLM:
    def __init__(self, cost=100):
        self.calls, self.tokens_used, self.cost = 0, 0, cost

    def generate(self, system, user, temperature=0.0):
        self.calls += 1
        self.tokens_used += self.cost
        return "Dogs are mammals [1]."


class FakeAgent:
    def __init__(self, fail=False):
        self.llm = type("L", (), {"tokens_used": 1234})()
        self.fail = fail

    def run(self, question, on_call=None):
        if self.fail:
            raise RuntimeError("model unavailable")
        record = ToolCallRecord(1, "semantic_search", '{"query": "x"}', 500, False, 0.01)
        if on_call:
            on_call(record)
        return AgentResult("Researched answer.", "answered", [record])


def make_client(llm=None, **kwargs):
    llm = llm or CountingLLM()
    return TestClient(create_app(FakeRetriever(), llm, **kwargs)), llm


# ---- routing -------------------------------------------------------------

def test_lookup_question_returns_documents_without_calling_the_llm():
    client, llm = make_client()
    body = client.post("/ask", json={"question": "Which papers are about dogs?"}).json()
    assert body["route"] == "retrieval" and llm.calls == 0 and body["tokens_used"] == 0
    assert "docA" in body["answer"] and "docB" in body["answer"]


def test_multi_step_question_is_pointed_to_research_without_calling_the_llm():
    client, llm = make_client()
    body = client.post("/ask", json={"question": "Compare dogs and cats"}).json()
    assert body["route"] == "agent" and llm.calls == 0 and "/research" in body["answer"]


def test_normal_question_uses_one_rag_call():
    client, llm = make_client()
    body = client.post("/ask", json={"question": "What are dogs?"}).json()
    assert body["route"] == "rag" and llm.calls == 1 and body["tokens_used"] == 100


# ---- quota ---------------------------------------------------------------

def test_quota_blocks_a_user_who_is_over_budget():
    client, llm = make_client(CountingLLM(cost=100), quota=TokenQuota(daily_limit=150))
    headers = {"X-User-Id": "alice"}
    assert client.post("/ask", json={"question": "What are dogs?"}, headers=headers).status_code == 200
    assert client.post("/ask", json={"question": "What are cats?"}, headers=headers).status_code == 200  # 100 < 150 before call
    blocked = client.post("/ask", json={"question": "What are birds?"}, headers=headers)
    assert blocked.status_code == 429 and "daily token limit" in blocked.json()["detail"]
    assert llm.calls == 2  # the blocked request never reached the LLM


def test_quota_is_per_user():
    client, _ = make_client(CountingLLM(cost=100), quota=TokenQuota(daily_limit=50))
    client.post("/ask", json={"question": "What are dogs?"}, headers={"X-User-Id": "alice"})
    assert client.post("/ask", json={"question": "What are cats?"},
                       headers={"X-User-Id": "alice"}).status_code == 429
    assert client.post("/ask", json={"question": "What are cats?"},
                       headers={"X-User-Id": "bob"}).status_code == 200


def test_cache_hits_do_not_consume_quota():
    quota = TokenQuota(daily_limit=1000)
    client, llm = make_client(CountingLLM(cost=100), cache=AnswerCache(), quota=quota)
    for _ in range(3):
        client.post("/ask", json={"question": "What are dogs?"}, headers={"X-User-Id": "alice"})
    assert llm.calls == 1 and quota.remaining("alice") == 900


# ---- research jobs ---------------------------------------------------------

def test_research_job_runs_and_reports_tool_calls_and_tokens():
    client, _ = make_client(agent_factory=lambda: FakeAgent())
    started = client.post("/research", json={"question": "Compare dogs and cats"})
    assert started.status_code == 202
    status = client.get(f"/research/{started.json()['job_id']}").json()
    assert status["status"] == "done" and status["answer"] == "Researched answer."
    assert status["tool_calls"][0]["tool"] == "semantic_search" and status["tokens_used"] == 1234


def test_failed_research_job_is_reported_not_lost():
    client, _ = make_client(agent_factory=lambda: FakeAgent(fail=True))
    job_id = client.post("/research", json={"question": "Compare dogs and cats"}).json()["job_id"]
    status = client.get(f"/research/{job_id}").json()
    assert status["status"] == "failed" and "model unavailable" in status["error"]


def test_unknown_job_id_is_404():
    client, _ = make_client(agent_factory=lambda: FakeAgent())
    assert client.get("/research/nope").status_code == 404


def test_research_without_an_agent_configured_is_503():
    client, _ = make_client()
    assert client.post("/research", json={"question": "Compare dogs and cats"}).status_code == 503


def test_research_job_tokens_count_against_the_quota():
    quota = TokenQuota(daily_limit=2000)
    client, _ = make_client(agent_factory=lambda: FakeAgent(), quota=quota)
    client.post("/research", json={"question": "Compare dogs and cats"}, headers={"X-User-Id": "alice"})
    assert quota.remaining("alice") == 2000 - 1234
