import pytest
import requests

from generation import groq_llm
from generation.groq_llm import GroqLLM


class FakeResponse:
    def __init__(self, status, headers=None, body=None, text=""):
        self.status_code, self.headers, self._body, self.text = status, headers or {}, body, text
        self.ok = status < 400

    def json(self):
        return self._body


def make_llm(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    return GroqLLM()


def test_long_rate_limit_wait_fails_fast_with_server_message(monkeypatch):
    slept = []
    monkeypatch.setattr(groq_llm.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(groq_llm.requests, "post", lambda *a, **k: FakeResponse(
        429, {"retry-after": "900"}, text="tokens per day limit"))
    with pytest.raises(requests.HTTPError, match="tokens per day"):
        make_llm(monkeypatch).chat([{"role": "user", "content": "hi"}])
    assert slept == []  # it did NOT sleep for 15 minutes


def test_short_rate_limit_is_retried_and_tokens_are_counted(monkeypatch):
    replies = iter([FakeResponse(429, {"retry-after": "1"}),
                    FakeResponse(200, body={"choices": [{"message": {"content": "ok"}}],
                                            "usage": {"total_tokens": 42}})])
    monkeypatch.setattr(groq_llm.time, "sleep", lambda s: None)
    monkeypatch.setattr(groq_llm.requests, "post", lambda *a, **k: next(replies))
    llm = make_llm(monkeypatch)
    assert llm.chat([{"role": "user", "content": "hi"}])["content"] == "ok"
    assert llm.tokens_used == 42
