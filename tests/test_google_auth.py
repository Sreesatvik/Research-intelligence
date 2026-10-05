import base64
import hashlib
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from apps.api.auth import Auth
from apps.api.main import create_app
from auth.google import GoogleOAuth
from auth.tokens import read_payload
from auth.users import UserStore
from core.schemas import Chunk, SearchResult

SECRET = "x" * 40
CALLBACK = "http://localhost:8080/api/auth/google/callback"
PASSWORD = "correct horse battery"


class FakeRetriever:
    def search(self, query, k=5):
        return [SearchResult(Chunk("docA#0", "docA", "Dogs.", 0), 0.9)]


class FakeLLM:
    def generate(self, system, user, temperature=0.0):
        return "ok"


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body or {}

    def json(self):
        return self._body


class SimulatedGoogle:
    """Stands in for Google's token endpoint and ID-token verification."""

    def __init__(self):
        self.token_status = 200
        self.verify_error = None
        self.calls = []
        self.claims = {"iss": "https://accounts.google.com", "sub": "g-123", "email": "carol@example.com",
                       "email_verified": True, "nonce": None}

    def post(self, url, data, timeout):
        self.calls.append((url, data))
        return FakeResponse(self.token_status, {"id_token": "fake-id-token"})

    def verify(self, token, client_id):
        if self.verify_error:
            raise ValueError(self.verify_error)
        assert client_id == "client-id"
        return dict(self.claims)


@pytest.fixture
def world(tmp_path):
    fake = SimulatedGoogle()
    google = GoogleOAuth("client-id", "client-secret", CALLBACK, post=fake.post, verify=fake.verify)
    auth = Auth(UserStore(str(tmp_path / "users.db")), SECRET, google=google)
    client = TestClient(create_app(FakeRetriever(), FakeLLM(), auth=auth))
    return client, fake, auth


def begin(client, next_path="/ask"):
    response = client.get("/auth/google/start", params={"next": next_path}, follow_redirects=False)
    query = {k: v[0] for k, v in parse_qs(urlparse(response.headers["location"]).query).items()}
    return response, query


def finish(client, query, fake, code="auth-code", **overrides):
    """Complete the flow as Google's redirect back would, with the nonce Google echoes into the ID token."""
    fake.claims["nonce"] = overrides.pop("nonce", query["nonce"])
    params = {"code": code, "state": overrides.pop("state", query["state"]), **overrides}
    return client.get("/auth/google/callback", params={k: v for k, v in params.items() if v is not None}, follow_redirects=False)


# ---- starting the sign-in --------------------------------------------------------

def test_start_sends_the_browser_to_google_with_every_protection(world):
    client, _, _ = world
    response, query = begin(client)
    assert response.status_code == 302 and response.headers["location"].startswith("https://accounts.google.com/")
    assert query["client_id"] == "client-id" and query["redirect_uri"] == CALLBACK
    assert query["response_type"] == "code" and "openid" in query["scope"] and "email" in query["scope"]
    assert query["code_challenge_method"] == "S256"
    assert len(query["state"]) >= 30 and len(query["nonce"]) >= 30
    assert "client-secret" not in response.headers["location"]  # the secret never travels through the browser


def test_attempt_cookie_is_httponly_and_pkce_challenge_matches_its_verifier(world):
    client, _, _ = world
    response, query = begin(client)
    cookie = response.headers["set-cookie"]
    assert "oauth_attempt=" in cookie and "HttpOnly" in cookie and "SameSite=lax" in cookie
    attempt = read_payload(SECRET, client.cookies.get("oauth_attempt"))
    expected = base64.urlsafe_b64encode(hashlib.sha256(attempt["verifier"].encode()).digest()).rstrip(b"=").decode()
    assert query["code_challenge"] == expected and attempt["state"] == query["state"]


def test_google_endpoints_do_not_exist_when_google_is_not_configured(tmp_path):
    auth = Auth(UserStore(str(tmp_path / "u.db")), SECRET)
    client = TestClient(create_app(FakeRetriever(), FakeLLM(), auth=auth))
    assert client.get("/auth/providers").json() == {"google": False}
    assert client.get("/auth/google/start", follow_redirects=False).status_code == 404


def test_providers_reports_google_when_configured(world):
    assert world[0].get("/auth/providers").json() == {"google": True}


# ---- a successful sign-in ---------------------------------------------------------

def test_new_google_user_is_created_and_signed_in(world):
    client, fake, _ = world
    _, query = begin(client)
    response = finish(client, query, fake)
    assert response.status_code == 302 and response.headers["location"] == "/ask"
    assert "session=" in response.headers["set-cookie"] and "HttpOnly" in response.headers["set-cookie"]
    assert client.get("/auth/me").json() == {"email": "carol@example.com"}


def test_code_is_redeemed_with_the_pkce_verifier_and_secret_server_side(world):
    client, fake, _ = world
    _, query = begin(client)
    verifier = read_payload(SECRET, client.cookies.get("oauth_attempt"))["verifier"]
    finish(client, query, fake, code="the-code")
    _, data = fake.calls[0]
    assert data["code"] == "the-code" and data["code_verifier"] == verifier and data["client_secret"] == "client-secret"


def test_user_is_sent_to_the_page_they_wanted(world):
    client, fake, _ = world
    _, query = begin(client, "/research")
    assert finish(client, query, fake).headers["location"] == "/research"


@pytest.mark.parametrize("hostile", ["https://evil.example/steal", "//evil.example", "javascript:alert(1)", "/login?x=1"])
def test_next_cannot_be_used_as_an_open_redirect(world, hostile):
    client, fake, _ = world
    _, query = begin(client, hostile)
    assert finish(client, query, fake).headers["location"] == "/ask"


def test_returning_user_is_matched_by_google_id_not_email(world):
    client, fake, auth = world
    _, query = begin(client)
    finish(client, query, fake)
    client.post("/auth/logout")
    fake.claims["email"] = "carol.new@example.com"  # she changed her Google email
    _, query = begin(client)
    finish(client, query, fake)
    assert client.get("/auth/me").json() == {"email": "carol@example.com"}  # same account, no duplicate
    assert auth.users.by_email("carol.new@example.com") is None


# ---- attacks and failures: every one must end at /login?error=google with no session --

def assert_rejected(response, client):
    assert response.status_code == 302 and response.headers["location"] == "/login?error=google"
    assert client.get("/auth/me").status_code == 401


def test_wrong_state_is_rejected(world):
    client, fake, _ = world
    _, query = begin(client)
    assert_rejected(finish(client, query, fake, state="attacker-chosen"), client)


def test_callback_without_the_attempt_cookie_is_rejected(world):
    client, fake, _ = world
    _, query = begin(client)
    client.cookies.clear()  # a different browser, or an attacker injecting their own code
    assert_rejected(finish(client, query, fake), client)


def test_tampered_attempt_cookie_is_rejected(world):
    client, fake, _ = world
    _, query = begin(client)
    client.cookies.set("oauth_attempt", client.cookies.get("oauth_attempt")[:-4] + "AAAA")
    assert_rejected(finish(client, query, fake), client)


def test_nonce_mismatch_is_rejected(world):
    client, fake, _ = world
    _, query = begin(client)
    assert_rejected(finish(client, query, fake, nonce="replayed-from-another-login"), client)


def test_unverified_google_email_is_rejected(world):
    client, fake, _ = world
    _, query = begin(client)
    fake.claims["email_verified"] = False
    assert_rejected(finish(client, query, fake), client)


def test_wrong_issuer_is_rejected(world):
    client, fake, _ = world
    _, query = begin(client)
    fake.claims["iss"] = "https://evil.example"
    assert_rejected(finish(client, query, fake), client)


def test_invalid_id_token_signature_is_rejected(world):
    client, fake, _ = world
    _, query = begin(client)
    fake.verify_error = "Could not verify token signature"
    assert_rejected(finish(client, query, fake), client)


def test_google_refusing_the_code_is_rejected(world):
    client, fake, _ = world
    _, query = begin(client)
    fake.token_status = 400
    assert_rejected(finish(client, query, fake), client)


def test_cancelling_on_googles_screen_is_handled(world):
    client, fake, _ = world
    begin(client)
    cancelled = client.get("/auth/google/callback", params={"error": "access_denied"}, follow_redirects=False)
    assert_rejected(cancelled, client)


def test_an_attempt_cannot_be_replayed(world):
    client, fake, _ = world
    _, query = begin(client)
    assert finish(client, query, fake).headers["location"] == "/ask"
    client.post("/auth/logout")
    assert_rejected(finish(client, query, fake), client)  # the cookie was consumed


# ---- account linking ---------------------------------------------------------------

def test_google_takes_over_a_password_account_and_revokes_the_password(world):
    client, fake, auth = world
    # Someone (maybe an attacker) registered carol's email with a password; email was never verified.
    client.post("/auth/register", json={"email": "carol@example.com", "password": PASSWORD})
    client.post("/auth/logout")
    original = auth.users.by_email("carol@example.com")

    _, query = begin(client)
    finish(client, query, fake)  # the real owner proves the email through Google
    assert client.get("/auth/me").json() == {"email": "carol@example.com"}
    assert auth.users.by_email("carol@example.com").id == original.id  # linked, not duplicated

    client.post("/auth/logout")
    old_password = client.post("/auth/login", json={"email": "carol@example.com", "password": PASSWORD})
    assert old_password.status_code == 401  # the pre-registered password no longer works


def test_google_only_account_cannot_be_logged_into_with_a_password(world):
    client, fake, _ = world
    _, query = begin(client)
    finish(client, query, fake)
    client.post("/auth/logout")
    assert client.post("/auth/login", json={"email": "carol@example.com", "password": "anything at all"}).status_code == 401
    assert client.post("/auth/register", json={"email": "carol@example.com", "password": PASSWORD}).status_code == 409


def test_existing_users_survive_the_schema_migration(tmp_path):
    import sqlite3

    path = str(tmp_path / "old.db")
    old = sqlite3.connect(path)  # a database created before Google sign-in existed
    old.execute("CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT NOT NULL UNIQUE, "
                "password_hash TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
    old.execute("INSERT INTO users (email, password_hash) VALUES ('old@example.com', 'scrypt$x')")
    old.commit()
    old.close()
    store = UserStore(path)  # opening it must add the new column without losing anyone
    user = store.by_email("old@example.com")
    assert user is not None and user.google_sub is None
