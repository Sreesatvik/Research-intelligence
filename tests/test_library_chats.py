import hashlib
import re
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from cache.answer_cache import AnswerCache
from chats.store import ChatStore
from library.store import Library, LibraryError


class FakeModel:
    """Hashes words into a small vector, so similar texts score high without downloading a real model."""

    def encode(self, texts, normalize_embeddings=True, batch_size=32):
        single = isinstance(texts, str)
        out = []
        for text in [texts] if single else texts:
            v = np.zeros(64)
            for word in re.findall(r"[a-z]+", text.lower()):
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
            out.append(v / (np.linalg.norm(v) or 1))
        return out[0] if single else np.array(out)


class FakeLLM:
    def __init__(self):
        self.calls = 0
        self.tokens_used = 0

    def generate(self, system, user, temperature=0.0):
        self.calls += 1
        self.tokens_used += 10
        return "Dogs bark [1]."


DOGS = b"Dogs are loyal animals that bark and wag their tails. Dogs need daily walks and food."
SPACE = b"Rockets burn fuel to reach orbit. The moon orbits the earth and rockets visit the moon."


@pytest.fixture
def lib(tmp_path):
    return Library(str(tmp_path / "users"), str(tmp_path / "app.db"), FakeModel())


# ---- the library -------------------------------------------------------------------

def test_a_users_search_sees_only_their_own_documents(lib):
    lib.add("alice", "dogs.txt", DOGS)
    lib.add("bob", "space.txt", SPACE)
    alice = {r.chunk.document_id for r in lib.retriever("alice").search("dogs", k=5)}
    bob = {r.chunk.document_id for r in lib.retriever("bob").search("dogs", k=5)}
    assert alice == {"dogs"} and bob == {"space"}
    assert lib.retriever("carol") is None  # a new user starts with nothing


def test_documents_are_listed_per_user_and_survive_a_restart(lib, tmp_path):
    lib.add("alice", "dogs.txt", DOGS)
    assert [d["id"] for d in lib.documents("alice")] == ["dogs"] and lib.documents("bob") == []
    reopened = Library(str(tmp_path / "users"), str(tmp_path / "app.db"), FakeModel())
    assert reopened.retriever("alice").search("dogs", k=1)[0].chunk.document_id == "dogs"


def test_adding_after_the_index_is_loaded_updates_it(lib):
    lib.add("alice", "dogs.txt", DOGS)
    lib.retriever("alice")
    lib.add("alice", "space.txt", SPACE)
    assert lib.retriever("alice").search("rockets orbit moon", k=1)[0].chunk.document_id == "space"


def test_delete_removes_the_document_from_search_and_disk(lib):
    lib.add("alice", "dogs.txt", DOGS)
    lib.add("alice", "space.txt", SPACE)
    lib.retriever("alice")
    assert lib.delete("alice", "dogs") and not lib.delete("alice", "dogs")
    assert {r.chunk.document_id for r in lib.retriever("alice").search("dogs", k=5)} == {"space"}
    assert not list(lib._dir("alice").glob("dogs.*"))
    assert not lib.delete("bob", "space")  # cannot delete someone else's document


def test_deleting_the_last_document_empties_the_library(lib):
    lib.add("alice", "dogs.txt", DOGS)
    lib.retriever("alice")
    lib.delete("alice", "dogs")
    assert lib.retriever("alice") is None and lib.corpus_map("alice")["points"] == []


def test_same_filename_gets_a_distinct_id(lib):
    lib.add("alice", "notes.txt", DOGS)
    assert lib.add("alice", "notes.txt", SPACE)["id"] == "notes-2"


@pytest.mark.parametrize("name,data", [("evil.exe", b"x"), ("a.txt", b""), ("a.txt", b"   \n")])
def test_unusable_files_are_rejected(lib, name, data):
    with pytest.raises(LibraryError):
        lib.add("alice", name, data)
    assert lib.documents("alice") == []


def test_a_broken_pdf_is_rejected_and_leaves_nothing_behind(lib):
    with pytest.raises(LibraryError):
        lib.add("alice", "bad.pdf", b"this is not a pdf")
    assert lib.documents("alice") == [] and not list(lib._dir("alice").glob("*"))


def test_size_and_count_limits(tmp_path):
    small = Library(str(tmp_path / "u"), str(tmp_path / "a.db"), FakeModel(), max_documents=1, max_bytes=50)
    with pytest.raises(LibraryError):
        small.add("alice", "big.txt", b"word " * 100)
    small.add("alice", "ok.txt", b"hello world")
    with pytest.raises(LibraryError):
        small.add("alice", "two.txt", b"hello again")


def test_hostile_names_cannot_escape_the_users_folder(lib, tmp_path):
    doc = lib.add("../../etc", "../../evil.txt", DOGS)
    assert doc["id"] == "evil"
    written = [p for p in tmp_path.rglob("*") if p.is_file() and "evil" in p.name]
    assert written and all(str(tmp_path / "users") in str(p) for p in written)


@pytest.mark.parametrize("count", [1, 2, 3, 6])
def test_the_map_works_for_any_library_size(lib, count):
    for i in range(count):
        lib.add("alice", f"doc{i}.txt", (DOGS if i % 2 else SPACE) + f" extra{i}".encode())
    result = lib.corpus_map("alice")
    assert len(result["points"]) == count
    assert sum(c["size"] for c in result["clusters"]) == count
    assert all(set(p) >= {"id", "title", "x", "y", "z", "cluster", "topic"} for p in result["points"])
    assert all(abs(p[axis]) <= 1.0 for p in result["points"] for axis in "xyz")


def test_the_map_separates_two_topics(lib):
    for i in range(3):
        lib.add("alice", f"dog{i}.txt", DOGS + f" pet{i}".encode())
        lib.add("alice", f"rocket{i}.txt", SPACE + f" launch{i}".encode())
    clusters = {p["id"][:3]: p["cluster"] for p in lib.corpus_map("alice")["points"]}
    assert clusters["dog"] != clusters["roc"]


# ---- the API: chats ----------------------------------------------------------------

@pytest.fixture
def app_parts(tmp_path):
    lib = Library(str(tmp_path / "users"), str(tmp_path / "app.db"), FakeModel())
    llm = FakeLLM()
    chats = ChatStore(str(tmp_path / "app.db"))
    cache = AnswerCache(embed=lambda t: FakeModel().encode(t))
    agent_calls = []

    class FakeAgent:
        llm = SimpleNamespace(tokens_used=5)

        def run(self, question, on_call=None):
            record = SimpleNamespace(step=1, tool="semantic_search", arguments='{"query": "dogs"}', is_error=False)
            on_call(record)
            agent_calls.append(question)
            return SimpleNamespace(answer="Dogs bark (dogs).", stopped_because="answered", calls=[record])

    app = create_app(None, llm, cache=cache, library=lib, chats=chats, user_agent_factory=lambda user: FakeAgent())
    return TestClient(app), lib, llm, agent_calls, chats


def ask(client, chat_id, text, user="alice", mode="ask"):
    return client.post(f"/chats/{chat_id}/messages", json={"content": text, "mode": mode}, headers={"X-User-Id": user})


def new_chat(client, user="alice"):
    return client.post("/chats", headers={"X-User-Id": user}).json()["id"]


def test_a_chat_stores_the_conversation_with_sources(app_parts):
    client, lib, llm, _, _ = app_parts
    lib.add("alice", "dogs.txt", DOGS)
    chat = new_chat(client)
    body = ask(client, chat, "What do dogs do?").json()
    assert [m["role"] for m in body["messages"]] == ["user", "assistant"]
    assistant = body["messages"][1]
    assert assistant["content"] == "Dogs bark [1]." and assistant["meta"]["sources"][0]["document_id"] == "dogs"
    assert body["title"] == "What do dogs do?"  # the first message names the chat
    again = client.get(f"/chats/{chat}", headers={"X-User-Id": "alice"}).json()
    assert len(again["messages"]) == 2  # it was really stored


def test_chat_list_is_newest_first_and_private(app_parts):
    client, lib, *_ = app_parts
    lib.add("alice", "dogs.txt", DOGS)
    first, second = new_chat(client), new_chat(client)
    ask(client, first, "dogs?")
    assert [c["id"] for c in client.get("/chats", headers={"X-User-Id": "alice"}).json()] == [first, second]
    assert client.get("/chats", headers={"X-User-Id": "bob"}).json() == []


def test_nobody_can_read_write_or_delete_another_users_chat(app_parts):
    client, lib, *_ = app_parts
    lib.add("alice", "dogs.txt", DOGS)
    chat = new_chat(client)
    ask(client, chat, "dogs?")
    bob = {"X-User-Id": "bob"}
    assert client.get(f"/chats/{chat}", headers=bob).status_code == 404
    assert ask(client, chat, "hi", user="bob").status_code == 404
    assert client.delete(f"/chats/{chat}", headers=bob).status_code == 404
    assert client.patch(f"/chats/{chat}", json={"title": "mine"}, headers=bob).status_code == 404
    assert len(client.get(f"/chats/{chat}", headers={"X-User-Id": "alice"}).json()["messages"]) == 2


def test_delete_and_rename(app_parts):
    client, *_ = app_parts
    chat = new_chat(client)
    alice = {"X-User-Id": "alice"}
    assert client.patch(f"/chats/{chat}", json={"title": "Dogs"}, headers=alice).json()["title"] == "Dogs"
    assert client.delete(f"/chats/{chat}", headers=alice).status_code == 200
    assert client.get(f"/chats/{chat}", headers=alice).status_code == 404


def test_an_empty_library_gets_a_helpful_reply_and_no_llm_call(app_parts):
    client, _, llm, _, _ = app_parts
    chat = new_chat(client)
    reply = ask(client, chat, "What do dogs do?").json()["messages"][1]
    assert "library is empty" in reply["content"].lower() and llm.calls == 0


def test_answers_come_only_from_the_askers_documents(app_parts):
    client, lib, llm, _, _ = app_parts
    lib.add("alice", "dogs.txt", DOGS)
    lib.add("bob", "space.txt", SPACE)
    a = ask(client, new_chat(client), "What do dogs do?").json()["messages"][1]["meta"]["sources"]
    b = ask(client, new_chat(client, "bob"), "What do dogs do?", user="bob").json()["messages"][1]["meta"]["sources"]
    assert {s["document_id"] for s in a} == {"dogs"} and {s["document_id"] for s in b} == {"space"}


def test_the_cache_never_serves_one_users_answer_to_another(app_parts):
    client, lib, llm, _, _ = app_parts
    lib.add("alice", "dogs.txt", DOGS)
    lib.add("bob", "dogs.txt", DOGS)
    ask(client, new_chat(client), "What do dogs do?")
    reply = ask(client, new_chat(client, "bob"), "What do dogs do?", user="bob").json()["messages"][1]
    assert reply["meta"]["cache_hit"] == "none" and llm.calls == 2


def test_the_cache_is_dropped_when_documents_change(app_parts):
    client, lib, llm, _, _ = app_parts
    lib.add("alice", "dogs.txt", DOGS)
    chat = new_chat(client)
    ask(client, chat, "What do dogs do?")
    lib.add("alice", "more.txt", DOGS + b" Puppies play.")
    ask(client, new_chat(client), "What do dogs do?")  # a new chat, so no follow-up rewrite muddies the count
    assert llm.calls == 2


def test_research_mode_runs_the_agent_and_saves_the_steps(app_parts):
    client, lib, _, agent_calls, _ = app_parts
    lib.add("alice", "dogs.txt", DOGS)
    chat = new_chat(client)
    ask(client, chat, "Dogs please", mode="research")
    done = client.get(f"/chats/{chat}", headers={"X-User-Id": "alice"}).json()["messages"][1]
    assert done["status"] == "done" and done["content"] == "Dogs bark (dogs)."
    assert done["meta"]["tool_calls"][0]["tool"] == "semantic_search" and agent_calls == ["Dogs please"]


def test_a_multi_step_question_in_ask_mode_points_to_research(app_parts):
    client, lib, llm, _, _ = app_parts
    lib.add("alice", "dogs.txt", DOGS)
    reply = ask(client, new_chat(client), "Compare dogs and cats").json()["messages"][1]
    assert "Research mode" in reply["content"] and llm.calls == 0


def test_research_interrupted_by_a_restart_is_marked_failed(tmp_path):
    path = str(tmp_path / "app.db")
    store = ChatStore(path)
    chat = store.create("alice")["id"]
    store.add_message("alice", chat, "assistant", "", status="running")
    reopened = ChatStore(path)
    message = reopened.get("alice", chat)["messages"][0]
    assert message["status"] == "failed" and "interrupted" in message["content"]


# ---- the API: library ----------------------------------------------------------------

def put(client, name, data, user="alice"):
    return client.put(f"/library/documents/{name}", content=data, headers={"X-User-Id": user})


def test_upload_list_map_and_delete_over_http(app_parts):
    client, *_ = app_parts
    assert put(client, "dogs.txt", DOGS).status_code == 201
    listing = client.get("/library/documents", headers={"X-User-Id": "alice"}).json()
    assert [d["id"] for d in listing] == ["dogs"]
    assert len(client.get("/library/map", headers={"X-User-Id": "alice"}).json()["points"]) == 1
    assert client.get("/library/documents", headers={"X-User-Id": "bob"}).json() == []
    assert client.delete("/library/documents/dogs", headers={"X-User-Id": "bob"}).status_code == 404
    assert client.delete("/library/documents/dogs", headers={"X-User-Id": "alice"}).status_code == 200


def test_bad_uploads_get_a_clear_400(app_parts):
    client, *_ = app_parts
    response = put(client, "virus.exe", b"x")
    assert response.status_code == 400 and "PDF" in response.json()["detail"]


def test_endpoints_require_a_session_when_accounts_are_on(tmp_path):
    from apps.api.auth import Auth
    from auth.users import UserStore

    lib = Library(str(tmp_path / "users"), str(tmp_path / "app.db"), FakeModel())
    auth = Auth(UserStore(str(tmp_path / "app.db")), "x" * 40)
    client = TestClient(create_app(None, FakeLLM(), auth=auth, library=lib, chats=ChatStore(str(tmp_path / "app.db"))))
    for method, path in [("get", "/chats"), ("post", "/chats"), ("get", "/library/documents"), ("get", "/library/map")]:
        assert getattr(client, method)(path).status_code == 401
    assert client.put("/library/documents/a.txt", content=b"x").status_code == 401


def test_agent_citations_are_shown_as_plain_brackets():
    from rag.pipeline import normalize_agent_citations

    assert normalize_agent_citations("Cats purr.\u3010pets\u3011 More.\u3010dogs\u2020L1-L4\u3011") == "Cats purr.[pets] More.[dogs]"
    assert normalize_agent_citations("Already [1] fine") == "Already [1] fine"


def test_a_windows_style_path_is_reduced_to_the_file_name(lib):
    doc = lib.add("alice", "C:\\Users\\me\\notes\\pets.txt", DOGS)
    assert doc["id"] == "pets" and doc["name"] == "pets.txt"


# ---- duplicates ---------------------------------------------------------------------

def test_the_same_file_cannot_be_added_twice(lib):
    from library.store import DuplicateDocument

    lib.add("alice", "dogs.txt", DOGS)
    with pytest.raises(DuplicateDocument) as caught:
        lib.add("alice", "renamed-copy.txt", DOGS)  # a new name does not make it a new document
    assert caught.value.existing["name"] == "dogs.txt" and len(lib.documents("alice")) == 1


def test_the_same_text_in_another_format_is_a_duplicate(lib):
    from library.store import DuplicateDocument

    lib.add("alice", "dogs.txt", DOGS)
    with pytest.raises(DuplicateDocument):
        lib.add("alice", "dogs.md", DOGS.replace(b" ", b"  ").upper())  # whitespace and case do not matter
    assert [d["id"] for d in lib.documents("alice")] == ["dogs"]


def test_different_content_with_the_same_name_is_kept_as_a_second_document(lib):
    lib.add("alice", "notes.txt", DOGS)
    assert lib.add("alice", "notes.txt", SPACE)["id"] == "notes-2"


def test_two_users_may_each_hold_the_same_file(lib):
    lib.add("alice", "dogs.txt", DOGS)
    assert lib.add("bob", "dogs.txt", DOGS)["id"] == "dogs"


def test_a_deleted_document_can_be_added_again(lib):
    lib.add("alice", "dogs.txt", DOGS)
    lib.delete("alice", "dogs")
    assert lib.add("alice", "dogs.txt", DOGS)["id"] == "dogs"


def test_duplicate_over_http_is_a_409_and_the_check_endpoint_knows_it(app_parts):
    import hashlib as h

    client, *_ = app_parts
    assert put(client, "dogs.txt", DOGS).status_code == 201
    again = put(client, "other.txt", DOGS)
    assert again.status_code == 409 and "dogs.txt" in again.json()["detail"]
    known, unknown = h.sha256(DOGS).hexdigest(), h.sha256(SPACE).hexdigest()
    found = client.post("/library/check", json={"hashes": [known, unknown]}, headers={"X-User-Id": "alice"}).json()
    assert list(found["duplicates"]) == [known] and found["duplicates"][known]["name"] == "dogs.txt"
    assert client.post("/library/check", json={"hashes": [known]}, headers={"X-User-Id": "bob"}).json()["duplicates"] == {}
    assert client.post("/library/check", json={"hashes": ["nope"]}, headers={"X-User-Id": "alice"}).status_code == 422


def test_documents_from_before_duplicate_detection_are_fingerprinted_on_startup(lib, tmp_path):
    import sqlite3

    from library.store import DuplicateDocument

    lib.add("alice", "dogs.txt", DOGS)
    with sqlite3.connect(lib.db_path) as conn:
        conn.execute("UPDATE documents SET file_hash = NULL, text_hash = NULL")
    reopened = Library(str(tmp_path / "users"), lib.db_path, FakeModel())
    with pytest.raises(DuplicateDocument):
        reopened.add("alice", "again.txt", DOGS)


# ---- conversation memory ---------------------------------------------------------------

class ScriptedLLM(FakeLLM):
    """Answers normally, but returns a fixed rewrite when asked to rewrite a question."""

    def __init__(self, rewrite):
        super().__init__()
        self.rewrite, self.seen = rewrite, []

    def generate(self, system, user, temperature=0.0):
        self.seen.append((system, user))
        if system.startswith("Rewrite the user"):
            self.tokens_used += 5
            return self.rewrite
        return super().generate(system, user, temperature)


def test_a_follow_up_is_rewritten_using_the_conversation(tmp_path):
    lib = Library(str(tmp_path / "users"), str(tmp_path / "app.db"), FakeModel())
    lib.add("alice", "dogs.txt", DOGS)
    lib.add("alice", "space.txt", SPACE)
    llm = ScriptedLLM("Do rockets visit the moon?")
    client = TestClient(create_app(None, llm, library=lib, chats=ChatStore(str(tmp_path / "app.db"))))
    chat = new_chat(client)
    first = ask(client, chat, "What do rockets burn?").json()["messages"][-1]
    assert first["meta"]["standalone"] is None and not any(s.startswith("Rewrite") for s, _ in llm.seen)  # nothing to remember yet
    second = ask(client, chat, "Do they visit the moon?").json()["messages"][-1]
    rewrite_prompt = next(u for s, u in llm.seen if s.startswith("Rewrite"))
    assert "What do rockets burn?" in rewrite_prompt and "Do they visit the moon?" in rewrite_prompt
    assert second["meta"]["standalone"] == "Do rockets visit the moon?"
    assert {s["document_id"] for s in second["meta"]["sources"]} >= {"space"}


def test_a_long_question_is_not_rewritten_and_a_failed_rewrite_falls_back(tmp_path):
    lib = Library(str(tmp_path / "users"), str(tmp_path / "app.db"), FakeModel())
    lib.add("alice", "dogs.txt", DOGS)

    class Broken(ScriptedLLM):
        def generate(self, system, user, temperature=0.0):
            if system.startswith("Rewrite the user"):
                raise RuntimeError("model down")
            return super().generate(system, user, temperature)

    llm = Broken("x")
    client = TestClient(create_app(None, llm, library=lib, chats=ChatStore(str(tmp_path / "app.db"))))
    chat = new_chat(client)
    ask(client, chat, "What do dogs do?")
    short = ask(client, chat, "and cats?").json()["messages"][-1]
    assert short["status"] == "done" and short["meta"]["standalone"] is None  # the rewrite failed, the answer still came
    long_question = "Please explain in as much detail as you reasonably can what dogs do during a whole ordinary day"
    ask(client, chat, long_question)
    assert not any(s.startswith("Rewrite") and long_question in u for s, u in llm.seen)


def test_research_gets_the_earlier_conversation_as_context(app_parts):
    client, lib, _, agent_calls, _ = app_parts
    lib.add("alice", "dogs.txt", DOGS)
    chat = new_chat(client)
    ask(client, chat, "What do dogs do?")
    ask(client, chat, "Dig deeper into that", mode="research")
    assert "What do dogs do?" in agent_calls[0] and "Dig deeper into that" in agent_calls[0]


# ---- streaming ---------------------------------------------------------------------------

class StreamingLLM(FakeLLM):
    PIECES = ["Dogs ", "bark ", "loudly ", "[1]."]

    def stream(self, system, user, temperature=0.0):
        self.calls += 1
        for piece in self.PIECES:
            yield piece
        self.tokens_used += 20


def parse_events(text: str) -> list[tuple[str, object]]:
    import json

    events = []
    for block in text.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        events.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return events


def stream(client, chat_id, text, user="alice"):
    response = client.post(f"/chats/{chat_id}/stream", json={"content": text, "mode": "ask"}, headers={"X-User-Id": user})
    return response, (parse_events(response.text) if response.status_code == 200 else [])


@pytest.fixture
def streaming(tmp_path):
    lib = Library(str(tmp_path / "users"), str(tmp_path / "app.db"), FakeModel())
    lib.add("alice", "dogs.txt", DOGS)
    llm = StreamingLLM()
    chats = ChatStore(str(tmp_path / "app.db"))
    return TestClient(create_app(None, llm, library=lib, chats=chats)), llm, chats


def test_an_answer_streams_sources_then_pieces_then_the_saved_chat(streaming):
    client, llm, _ = streaming
    response, events = stream(client, new_chat(client), "What do dogs do?")
    assert response.headers["content-type"].startswith("text/event-stream")
    names = [name for name, _ in events]
    assert names[0] == "sources" and names[-1] == "done" and names.count("delta") == 4
    assert "".join(data["text"] for name, data in events if name == "delta") == "Dogs bark loudly [1]."
    saved = events[-1][1]["messages"][-1]
    assert saved["status"] == "done" and saved["content"] == "Dogs bark loudly [1]." and saved["meta"]["sources"]
    assert saved["meta"]["tokens_used"] == 20


def test_a_streamed_answer_is_cached_for_next_time(tmp_path):
    lib = Library(str(tmp_path / "users"), str(tmp_path / "app.db"), FakeModel())
    lib.add("alice", "dogs.txt", DOGS)
    llm = StreamingLLM()
    client = TestClient(create_app(None, llm, cache=AnswerCache(embed=lambda t: FakeModel().encode(t)), library=lib,
                                   chats=ChatStore(str(tmp_path / "app.db"))))
    stream(client, new_chat(client), "What do dogs do?")
    _, events = stream(client, new_chat(client), "What do dogs do?")
    assert llm.calls == 1 and [name for name, _ in events] == ["done"]  # no model call, no pieces: it was cached
    assert events[-1][1]["messages"][-1]["meta"]["cache_hit"] == "exact"


def test_streaming_works_with_a_model_that_cannot_stream(app_parts):
    client, lib, *_ = app_parts
    lib.add("alice", "dogs.txt", DOGS)
    _, events = stream(client, new_chat(client), "What do dogs do?")
    assert [n for n, _ in events].count("delta") == 1 and events[-1][1]["messages"][-1]["content"] == "Dogs bark [1]."


def test_streaming_an_empty_library_or_a_foreign_chat(streaming, tmp_path):
    client, _, _ = streaming
    response, events = stream(client, new_chat(client, "bob"), "hello?", user="bob")
    assert "library is empty" in events[-1][1]["messages"][-1]["content"].lower()
    assert stream(client, new_chat(client), "x?", user="bob")[0].status_code == 404
    assert client.post("/chats/1/stream", json={"content": "x", "mode": "research"}, headers={"X-User-Id": "alice"}).status_code == 400


def test_a_model_failure_midway_is_reported_in_the_chat(tmp_path):
    lib = Library(str(tmp_path / "users"), str(tmp_path / "app.db"), FakeModel())
    lib.add("alice", "dogs.txt", DOGS)

    class Dying(FakeLLM):
        def stream(self, system, user, temperature=0.0):
            yield "Dogs "
            raise RuntimeError("connection reset")

    client = TestClient(create_app(None, Dying(), library=lib, chats=ChatStore(str(tmp_path / "app.db"))))
    _, events = stream(client, new_chat(client), "What do dogs do?")
    last = events[-1][1]["messages"][-1]
    assert last["status"] == "failed" and "went wrong" in last["content"]


def test_pressing_stop_keeps_the_text_written_so_far(tmp_path):
    lib = Library(str(tmp_path / "users"), str(tmp_path / "app.db"), FakeModel())
    lib.add("alice", "dogs.txt", DOGS)
    chats = ChatStore(str(tmp_path / "app.db"))
    app = create_app(None, StreamingLLM(), library=lib, chats=chats)
    route = next(r for r in app.routes if getattr(r, "path", "") == "/chats/{chat_id}/stream")
    chat = chats.create("alice")["id"]
    from apps.api.schemas import SendMessage

    response = route.endpoint(chat, SendMessage(content="What do dogs do?", mode="ask"), "alice")
    import asyncio
    import gc

    async def read_two_then_leave():
        body = response.body_iterator
        await body.__anext__()  # sources
        await body.__anext__()  # first piece
        await body.aclose()  # the reader went away

    asyncio.run(read_two_then_leave())
    gc.collect()  # the abandoned generator is closed here, which is what saves the partial text
    last = chats.get("alice", chat)["messages"][-1]
    assert last["status"] == "done" and last["content"] == "Dogs " and last["meta"]["stopped"] is True


# ---- library overviews ---------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "breif all pdfs", "give a breif of all pdfs", "Summarize my documents", "what is in my library?",
    "overview of the files", "describe each paper"])
def test_questions_about_the_whole_library_are_recognised(question):
    from rag.cost_router import is_overview

    assert is_overview(question)


@pytest.mark.parametrize("question", [
    "What do dogs do?", "Summarize how rockets reach orbit", "Which paper explains BM25?", "briefly, what is RAG?"])
def test_ordinary_questions_are_not_overviews(question):
    from rag.cost_router import is_overview

    assert not is_overview(question)


def test_an_overview_shows_the_model_every_document_and_not_just_top_hits(tmp_path):
    lib = Library(str(tmp_path / "users"), str(tmp_path / "app.db"), FakeModel())
    lib.add("alice", "dogs.txt", DOGS)
    lib.add("alice", "space.txt", SPACE)

    class Recording(FakeLLM):
        def generate(self, system, user, temperature=0.0):
            self.prompt = (system, user)
            return "- **dogs**: pets\n- **space**: rockets"

    llm = Recording()
    client = TestClient(create_app(None, llm, library=lib, chats=ChatStore(str(tmp_path / "app.db"))))
    reply = ask(client, new_chat(client), "breif all pdfs").json()["messages"][-1]
    system, shown = llm.prompt
    assert "[dogs]" in shown and "[space]" in shown and "summarizing" in system
    assert reply["status"] == "done" and reply["meta"]["route"] == "overview"
    assert {s["document_id"] for s in reply["meta"]["sources"]} == {"dogs", "space"}


def test_an_overview_is_cached_separately_from_ordinary_answers(tmp_path):
    lib = Library(str(tmp_path / "users"), str(tmp_path / "app.db"), FakeModel())
    lib.add("alice", "dogs.txt", DOGS)
    llm = FakeLLM()
    cache = AnswerCache(embed=lambda t: FakeModel().encode(t))
    client = TestClient(create_app(None, llm, cache=cache, library=lib, chats=ChatStore(str(tmp_path / "app.db"))))
    ask(client, new_chat(client), "summarize all documents")
    again = ask(client, new_chat(client), "summarize all documents").json()["messages"][-1]
    assert llm.calls == 1 and again["meta"]["cache_hit"] == "exact"
