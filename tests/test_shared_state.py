"""Shared state, tested with fakeredis (an in-process Redis). Two "servers" share one client, as they
would share one Redis, so these tests show state really crossing server boundaries."""
import os

import fakeredis
import pytest
from fastapi.testclient import TestClient

from agent.agent import AgentResult, ToolCallRecord
from apps.api.job_store import InMemoryJobStore, RedisJobStore
from apps.api.main import create_app
from apps.api.schemas import AskResponse, ResearchStatus, ToolCallInfo
from cache.answer_cache import AnswerCache
from core.schemas import Chunk, SearchResult
from quota.token_quota import QuotaExceeded, RedisTokenQuota


@pytest.fixture(params=["fake", "real"])
def redis_client(request):
    """Every test runs against fakeredis, and again against a real server when REDIS_TEST_URL is set
    (use a spare database such as redis://localhost:6379/15: it is emptied before and after each test)."""
    if request.param == "fake":
        yield fakeredis.FakeRedis(decode_responses=True)
        return
    url = os.environ.get("REDIS_TEST_URL")
    if not url:
        pytest.skip("set REDIS_TEST_URL to also run against a real Redis server")
    import redis

    client = redis.Redis.from_url(url, decode_responses=True)
    client.flushdb()
    yield client
    client.flushdb()


def make_cache(client):
    return AnswerCache(backend=client, dumps=lambda r: r.model_dump_json(), loads=AskResponse.model_validate_json)


def sample_response(text="answer"):
    return AskResponse(answer=text, cited_chunk_ids=["d#0"], refused_by_retrieval=False, sources=[], tokens_used=7)


# ---- quota ---------------------------------------------------------------

def test_quota_is_one_tally_across_servers(redis_client):
    server_a = RedisTokenQuota(redis_client, daily_limit=1000)
    server_b = RedisTokenQuota(redis_client, daily_limit=1000)
    server_a.record("alice", 600)
    server_b.check("alice")  # 600 < 1000: still allowed
    server_b.record("alice", 500)
    with pytest.raises(QuotaExceeded):
        server_a.check("alice")  # server A sees what server B recorded
    assert server_a.remaining("alice") == 0


def test_quota_counters_expire(redis_client):
    quota = RedisTokenQuota(redis_client)
    quota.record("alice", 10)
    assert 0 < redis_client.ttl(quota._key("alice")) <= 2 * 24 * 3600


def test_quota_is_per_user(redis_client):
    quota = RedisTokenQuota(redis_client, daily_limit=100)
    quota.record("alice", 100)
    quota.check("bob")  # bob is unaffected


# ---- jobs ----------------------------------------------------------------

def test_job_started_on_one_server_is_visible_on_another(redis_client):
    status = ResearchStatus(
        job_id="abc", status="done", answer="x", tokens_used=5,
        tool_calls=[ToolCallInfo(step=1, tool="semantic_search", arguments="{}", is_error=False)])
    RedisJobStore(redis_client).save(status)
    assert RedisJobStore(redis_client).get("abc") == status  # a different store object, same Redis


def test_unknown_job_is_none(redis_client):
    assert RedisJobStore(redis_client).get("nope") is None
    assert InMemoryJobStore().get("nope") is None


def test_jobs_expire(redis_client):
    RedisJobStore(redis_client, ttl_seconds=60).save(ResearchStatus(job_id="abc", status="running"))
    assert 0 < redis_client.ttl("job:abc") <= 60


# ---- cache ---------------------------------------------------------------

def test_exact_cache_hit_crosses_servers(redis_client):
    make_cache(redis_client).put("What is RAG?", sample_response("shared answer"), "k=3")
    hit = make_cache(redis_client).get("what is rag", "k=3")  # a different cache object, same Redis
    assert hit is not None and hit[1] == "exact" and hit[0].answer == "shared answer"


def test_cache_scope_is_respected_remotely(redis_client):
    make_cache(redis_client).put("What is RAG?", sample_response(), "k=3")
    assert make_cache(redis_client).get("What is RAG?", "k=5") is None


def test_cache_entries_expire(redis_client):
    cache = AnswerCache(backend=redis_client, ttl_seconds=120,
                        dumps=lambda r: r.model_dump_json(), loads=AskResponse.model_validate_json)
    cache.put("q", sample_response())
    assert 0 < redis_client.ttl("answer::q") <= 120


def test_clear_also_wipes_the_shared_copies(redis_client):
    cache = make_cache(redis_client)
    cache.put("q", sample_response())
    cache.clear()
    assert redis_client.keys("answer:*") == [] and make_cache(redis_client).get("q") is None


# ---- whole API: two servers, one Redis -----------------------------------------

class FakeRetriever:
    def search(self, query, k=5):
        return [SearchResult(Chunk("docA#0", "docA", "Dogs are mammals.", 0), 0.9)]


class FakeLLM:
    def __init__(self):
        self.calls, self.tokens_used = 0, 0

    def generate(self, system, user, temperature=0.0):
        self.calls += 1
        self.tokens_used += 100
        return "Dogs are mammals [1]."


class FakeAgent:
    def __init__(self):
        self.llm = type("L", (), {"tokens_used": 321})()

    def run(self, question, on_call=None):
        record = ToolCallRecord(1, "semantic_search", "{}", 10, False, 0.0)
        if on_call:
            on_call(record)
        return AgentResult("Researched.", "answered", [record])


def server(client, llm=None):
    return TestClient(create_app(
        FakeRetriever(), llm or FakeLLM(), cache=make_cache(client), agent_factory=FakeAgent,
        quota=RedisTokenQuota(client, daily_limit=150), job_store=RedisJobStore(client)))


def test_research_job_started_on_server_a_is_pollable_on_server_b(redis_client):
    job_id = server(redis_client).post("/research", json={"question": "Compare dogs and cats"}).json()["job_id"]
    status = server(redis_client).get(f"/research/{job_id}").json()
    assert status["status"] == "done" and status["answer"] == "Researched."


def test_answer_cached_by_server_a_costs_server_b_nothing(redis_client):
    llm_a, llm_b = FakeLLM(), FakeLLM()
    server(redis_client, llm_a).post("/ask", json={"question": "What are dogs?"})
    again = server(redis_client, llm_b).post("/ask", json={"question": "what are dogs"}).json()
    assert again["cache_hit"] == "exact" and again["tokens_used"] == 0
    assert llm_a.calls == 1 and llm_b.calls == 0


def test_a_user_cannot_beat_the_limit_by_switching_servers(redis_client):
    headers = {"X-User-Id": "alice"}
    llm = FakeLLM()
    a, b = server(redis_client, llm), server(redis_client, llm)
    assert a.post("/ask", json={"question": "What are dogs?"}, headers=headers).status_code == 200   # 100 used
    assert b.post("/ask", json={"question": "What are cats?"}, headers=headers).status_code == 200   # 200 used
    assert a.post("/ask", json={"question": "What are birds?"}, headers=headers).status_code == 429  # over 150
