import base64
import json
import time

import pytest
from fastapi.testclient import TestClient

from agent.agent import AgentResult, ToolCallRecord
from apps.api.auth import Auth
from apps.api.main import create_app
from auth.limiter import LoginLimiter
from auth.passwords import DUMMY_HASH, hash_password, verify_password
from auth.tokens import issue_token, read_token
from auth.users import EmailTaken, UserStore
from core.schemas import Chunk, SearchResult
from quota.token_quota import TokenQuota

SECRET = "x" * 40
PASSWORD = "correct horse battery"


# ---- password hashing ---------------------------------------------------------

def test_hash_verifies_and_rejects():
    stored = hash_password(PASSWORD)
    assert verify_password(PASSWORD, stored)
    assert not verify_password("wrong password!", stored)


def test_same_password_gets_different_hashes():
    assert hash_password(PASSWORD) != hash_password(PASSWORD)  # unique salt per user


def test_hash_does_not_contain_the_password():
    assert PASSWORD not in hash_password(PASSWORD)


@pytest.mark.parametrize("stored", ["", "garbage", "scrypt$1$2", "md5$a$b$c$d$e", "scrypt$x$y$z$!!$!!"])
def test_malformed_hashes_fail_closed(stored):
    assert verify_password(PASSWORD, stored) is False


def test_dummy_hash_never_matches_a_real_password():
    assert not verify_password(PASSWORD, DUMMY_HASH)


# ---- tokens -----------------------------------------------------------------

def test_token_round_trip():
    claims = read_token(SECRET, issue_token(SECRET, 7, "a@b.co", 60))
    assert claims["sub"] == "7" and claims["email"] == "a@b.co"


def test_expired_token_is_rejected():
    assert read_token(SECRET, issue_token(SECRET, 7, "a@b.co", -10)) is None


def test_tampered_token_is_rejected():
    header, payload, signature = issue_token(SECRET, 7, "a@b.co", 60).split(".")
    forged = base64.urlsafe_b64encode(json.dumps({"sub": "1", "email": "admin@b.co", "exp": int(time.time()) + 999}).encode()).rstrip(b"=").decode()
    assert read_token(SECRET, f"{header}.{forged}.{signature}") is None


def test_token_signed_with_another_secret_is_rejected():
    assert read_token(SECRET, issue_token("y" * 40, 7, "a@b.co", 60)) is None


def test_alg_none_token_is_rejected():
    def b64(data):
        return base64.urlsafe_b64encode(json.dumps(data).encode()).rstrip(b"=").decode()

    unsigned = f"{b64({'alg': 'none', 'typ': 'JWT'})}.{b64({'sub': '1', 'email': 'a@b.co', 'exp': int(time.time()) + 999})}."
    assert read_token(SECRET, unsigned) is None


def test_garbage_token_is_rejected():
    assert read_token(SECRET, "not.a.token") is None


def test_short_secret_is_refused(tmp_path):
    with pytest.raises(ValueError):
        Auth(UserStore(str(tmp_path / "u.db")), secret="short")


# ---- user store ------------------------------------------------------------------

def test_duplicate_email_is_refused_case_insensitively(tmp_path):
    store = UserStore(str(tmp_path / "u.db"))
    store.create("Alice@Example.com", "h")
    with pytest.raises(EmailTaken):
        store.create("alice@example.COM", "h")
    assert store.by_email("ALICE@example.com").email == "alice@example.com"


# ---- the API ---------------------------------------------------------------------

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
        self.llm = type("L", (), {"tokens_used": 50})()

    def run(self, question, on_call=None):
        return AgentResult("Researched.", "answered", [ToolCallRecord(1, "semantic_search", "{}", 1, False, 0.0)])


@pytest.fixture
def parts(tmp_path):
    auth = Auth(UserStore(str(tmp_path / "users.db")), SECRET, limiter=LoginLimiter(max_attempts=5))
    quota = TokenQuota(daily_limit=1000)
    return auth, quota


def client_for(auth, quota):
    return TestClient(create_app(FakeRetriever(), FakeLLM(), agent_factory=FakeAgent, quota=quota, auth=auth))


def register(client, email="alice@example.com", password=PASSWORD):
    return client.post("/auth/register", json={"email": email, "password": password})


def test_register_signs_you_in_with_a_safe_cookie(parts):
    client = client_for(*parts)
    response = register(client)
    assert response.status_code == 201 and response.json() == {"email": "alice@example.com"}
    cookie = response.headers["set-cookie"].lower()
    assert "session=" in cookie and "httponly" in cookie and "samesite=lax" in cookie
    assert client.get("/auth/me").json() == {"email": "alice@example.com"}


def test_logout_ends_the_session(parts):
    client = client_for(*parts)
    register(client)
    client.post("/auth/logout")
    assert client.get("/auth/me").status_code == 401


def test_endpoints_require_a_session(parts):
    client = client_for(*parts)
    assert client.post("/search", json={"query": "dogs"}).status_code == 401
    assert client.post("/ask", json={"question": "What are dogs?"}).status_code == 401
    assert client.post("/research", json={"question": "Compare dogs and cats"}).status_code == 401
    assert client.get("/research/abc").status_code == 401
    assert client.get("/health").status_code == 200  # health stays open


def test_signed_in_user_can_use_the_api(parts):
    client = client_for(*parts)
    register(client)
    assert client.post("/search", json={"query": "dogs"}).status_code == 200
    assert client.post("/ask", json={"question": "What are dogs?"}).status_code == 200


def test_forged_cookie_is_rejected(parts):
    client = client_for(*parts)
    client.cookies.set("session", issue_token("y" * 40, 1, "a@b.co", 60))  # signed with the wrong secret
    assert client.post("/search", json={"query": "dogs"}).status_code == 401


def test_wrong_password_and_unknown_email_look_identical(parts):
    client = client_for(*parts)
    register(client)
    client.post("/auth/logout")
    wrong = client.post("/auth/login", json={"email": "alice@example.com", "password": "not the password"})
    unknown = client.post("/auth/login", json={"email": "nobody@example.com", "password": "not the password"})
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()  # no hint about which emails have accounts


def test_login_works_with_the_right_password_and_any_email_casing(parts):
    client = client_for(*parts)
    register(client)
    client.post("/auth/logout")
    assert client.post("/auth/login", json={"email": "  ALICE@example.com ", "password": PASSWORD}).status_code == 200
    assert client.get("/auth/me").status_code == 200


def test_login_is_locked_after_repeated_failures(parts):
    client = client_for(*parts)
    register(client)
    client.post("/auth/logout")
    for _ in range(5):
        assert client.post("/auth/login", json={"email": "alice@example.com", "password": "guess guess guess"}).status_code == 401
    # Now even the CORRECT password is refused: the lock stops guessing, not just wrong guesses.
    locked = client.post("/auth/login", json={"email": "alice@example.com", "password": PASSWORD})
    assert locked.status_code == 429


def test_weak_or_invalid_registration_is_rejected(parts):
    client = client_for(*parts)
    assert register(client, password="short").status_code == 422
    assert register(client, email="not-an-email").status_code == 422
    assert register(client, password="x" * 500).status_code == 422


def test_duplicate_registration_is_refused(parts):
    client = client_for(*parts)
    assert register(client).status_code == 201
    assert register(client_for(*parts), email="ALICE@example.com").status_code == 409


def test_a_typed_user_id_header_cannot_impersonate_anyone(parts):
    auth, quota = parts
    client = client_for(auth, quota)
    register(client)
    client.post("/ask", json={"question": "What are dogs?"}, headers={"X-User-Id": "somebody-else"})
    assert quota.remaining("somebody-else") == 1000  # the header was ignored
    assert quota.remaining("1") == 900  # the verified account (id 1) was charged


def test_each_account_has_its_own_quota(parts):
    auth, quota = parts
    alice, bob = client_for(auth, quota), client_for(auth, quota)
    register(alice)
    register(bob, email="bob@example.com")
    alice.post("/ask", json={"question": "What are dogs?"})
    assert quota.remaining("1") == 900 and quota.remaining("2") == 1000


def test_research_jobs_are_private_to_their_owner(parts):
    auth, quota = parts
    alice, bob = client_for(auth, quota), client_for(auth, quota)
    register(alice)
    register(bob, email="bob@example.com")
    job_id = alice.post("/research", json={"question": "Compare dogs and cats"}).json()["job_id"]

    mine = alice.get(f"/research/{job_id}")
    assert mine.status_code == 200 and mine.json()["owner"] is None  # the owner field is never exposed
    assert bob.get(f"/research/{job_id}").status_code == 404  # looks exactly like a job that does not exist
    assert bob.get("/research/doesnotexist").json() == bob.get(f"/research/{job_id}").json()
