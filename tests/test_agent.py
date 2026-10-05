import json

from agent.agent import ResearchAgent
from agent.tools.research_tools import build_tools
from core.schemas import Chunk, SearchResult


class FakeRetriever:
    def search(self, query, k=5):
        return [SearchResult(Chunk("docA#0", "docA", "Dogs are mammals.", 0), 0.9)]


class FakeClassifier:
    def classify(self, text):
        return "animals", 0.8


class ScriptedLLM:
    """Replays a fixed list of replies, recording what it was sent."""

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def chat(self, messages, tools=None, temperature=0.0):
        self.seen.append((len(messages), tools is not None))
        return self.replies.pop(0)


def tool_call(name, args, call_id="c1"):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


REGISTRY = build_tools(FakeRetriever(), [Chunk("docA#0", "docA", "Dogs are mammals.", 0)], FakeClassifier())


def test_agent_calls_tool_then_answers():
    llm = ScriptedLLM([tool_call("semantic_search", {"query": "dogs"}),
                       {"role": "assistant", "content": "Dogs are mammals (docA)."}])
    result = ResearchAgent(llm, REGISTRY).run("what are dogs?")
    assert result.stopped_because == "answered"
    assert [c.tool for c in result.calls] == ["semantic_search"]
    assert llm.seen[1][0] == 4  # system, user, assistant tool request, tool result


def test_step_limit_forces_a_final_answer_without_tools():
    llm = ScriptedLLM([tool_call("semantic_search", {"query": "x"}, f"c{i}") for i in range(3)]
                      + [{"role": "assistant", "content": "best effort"}])
    result = ResearchAgent(llm, REGISTRY, max_steps=3).run("loop forever")
    assert result.stopped_because == "max_steps" and result.answer == "best effort"
    assert len(result.calls) == 3
    assert llm.seen[-1][1] is False  # the final call offered NO tools


def test_unknown_tool_request_is_logged_as_error_and_loop_continues(tmp_path):
    log = tmp_path / "calls.jsonl"
    llm = ScriptedLLM([tool_call("delete_everything", {}), {"role": "assistant", "content": "ok"}])
    result = ResearchAgent(llm, REGISTRY, log_path=str(log)).run("do bad thing")
    assert result.calls[0].is_error and result.stopped_because == "answered"
    assert json.loads(log.read_text().splitlines()[0])["tool"] == "delete_everything"


class FlakyFinalLLM:
    """Always requests a tool until tools are removed, then fails like the real server did."""

    def __init__(self, failures):
        self.failures, self.calls = failures, 0

    def chat(self, messages, tools=None, temperature=0.0):
        import requests
        if tools:
            return tool_call("semantic_search", {"query": "x"}, f"c{self.calls}")
        self.calls += 1
        if self.calls <= self.failures:
            raise requests.HTTPError('400: {"code":"tool_use_failed"}')
        return {"role": "assistant", "content": "plain answer"}


def test_final_call_retries_once_after_tool_use_failed():
    result = ResearchAgent(FlakyFinalLLM(failures=1), REGISTRY, max_steps=1).run("q")
    assert result.stopped_because == "max_steps" and result.answer == "plain answer"


def test_persistent_tool_use_failed_is_reported_not_hidden():
    result = ResearchAgent(FlakyFinalLLM(failures=5), REGISTRY, max_steps=1).run("q")
    assert result.stopped_because == "failed"


def _history(n_results):
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}]
    for i in range(n_results):
        msgs.append({"role": "assistant", "content": None, "tool_calls": [{"id": f"c{i}"}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "X" * 1000})
    return msgs


def test_trim_history_keeps_only_the_newest_results_in_full():
    from agent.agent import trim_history

    out = trim_history(_history(3), keep_last=1)
    tools = [m["content"] for m in out if m["role"] == "tool"]
    assert len(tools[0]) < 400 and len(tools[1]) < 400 and len(tools[2]) == 1000
    assert "[older result, compacted]" in tools[0]


def test_trim_history_does_not_change_the_original_or_drop_tool_requests():
    from agent.agent import trim_history

    original = _history(3)
    out = trim_history(original, keep_last=1)
    assert [len(m["content"] or "") for m in original if m["role"] == "tool"] == [1000, 1000, 1000]
    assert sum(1 for m in out if m["role"] == "assistant") == 3  # every request is still visible


def test_agent_sends_trimmed_history_when_configured():
    sent_sizes = []

    class SpyLLM(ScriptedLLM):
        def chat(self, messages, tools=None, temperature=0.0):
            sent_sizes.append(sum(len(m["content"] or "") for m in messages))
            return super().chat(messages, tools, temperature)

    llm = SpyLLM([tool_call("semantic_search", {"query": "x"}, f"c{i}") for i in range(3)]
                 + [{"role": "assistant", "content": "done"}])
    ResearchAgent(llm, REGISTRY, max_steps=4, keep_last_results=1).run("q")
    assert sent_sizes[-1] <= sent_sizes[-2] + 600  # growth is bounded, not cumulative


def test_compact_result_keeps_document_ids_of_every_hit():
    import json
    from agent.agent import compact_result
    from agent.tools.base import wrap_untrusted

    hits = [{"chunk_id": f"d{i}#0", "document_id": f"doc{i}", "score": 0.5, "text": "T" * 300} for i in range(5)]
    full = wrap_untrusted(json.dumps({"results": hits}))
    small = compact_result(full)
    assert all(f"doc{i}" in small for i in range(5))   # nothing the model must refer back to is lost
    assert len(small) < len(full) / 2
    assert small.startswith('<tool_result untrusted="true">') and small.endswith("</tool_result>")


class MalformedOnceLLM:
    """Raises the server's tool_use_failed error on the first call, then behaves."""

    def __init__(self, failures=1):
        self.failures, self.calls = failures, 0

    def chat(self, messages, tools=None, temperature=0.0):
        import requests
        self.calls += 1
        if self.calls <= self.failures:
            raise requests.HTTPError("400: tool_use_failed: retrieve_document<|channel|>commentary")
        return {"role": "assistant", "content": "recovered"}


def test_malformed_tool_call_is_retried_and_the_run_recovers():
    result = ResearchAgent(MalformedOnceLLM(failures=1), REGISTRY).run("q")
    assert result.answer == "recovered" and result.stopped_because == "answered"


def test_other_http_errors_are_not_swallowed():
    import pytest
    import requests

    class Broken:
        def chat(self, messages, tools=None, temperature=0.0):
            raise requests.HTTPError("401: invalid api key")

    with pytest.raises(requests.HTTPError):
        ResearchAgent(Broken(), REGISTRY).run("q")


def test_gives_up_after_three_malformed_attempts():
    import pytest
    import requests

    with pytest.raises(requests.HTTPError):
        ResearchAgent(MalformedOnceLLM(failures=9), REGISTRY).run("q")


def test_identical_repeated_call_is_not_executed_twice():
    executed = []

    class CountingRetriever(FakeRetriever):
        def search(self, query, k=5):
            executed.append(query)
            return super().search(query, k)

    registry = build_tools(CountingRetriever(), [Chunk("docA#0", "docA", "Dogs.", 0)], FakeClassifier())
    llm = ScriptedLLM([
        tool_call("semantic_search", {"query": "dogs", "k": 3}, "c1"),
        tool_call("semantic_search", {"k": 3, "query": "dogs"}, "c2"),   # same call, keys reordered
        {"role": "assistant", "content": "done"},
    ])
    result = ResearchAgent(llm, registry).run("q")
    assert executed == ["dogs"]                      # the search ran exactly once
    assert [c.is_error for c in result.calls] == [False, True]
    assert result.answer == "done"


def test_on_call_reports_each_tool_call_as_it_happens():
    reported = []
    llm = ScriptedLLM([tool_call("semantic_search", {"query": "a"}, "c1"),
                       tool_call("semantic_search", {"query": "b"}, "c2"),
                       {"role": "assistant", "content": "done"}])
    result = ResearchAgent(llm, REGISTRY).run("q", on_call=lambda rec: reported.append(rec.step))
    assert reported == [1, 2] and len(result.calls) == 2
