import base64
import hashlib
import hmac
import secrets
from dataclasses import dataclass
from urllib.parse import urlencode

import requests

from auth.tokens import read_payload, sign_payload

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
ISSUERS = {"accounts.google.com", "https://accounts.google.com"}
STATE_TTL_SECONDS = 600
DEFAULT_NEXT = "/ask"


class GoogleAuthError(Exception):
    """Anything that makes a Google sign-in untrustworthy or incomplete."""


@dataclass
class GoogleIdentity:
    sub: str  # Google's permanent id for the person; stable even if they change their email
    email: str


def safe_next(value: str | None) -> str:
    """Only same-site paths: a ?next=https://evil.example must never turn sign-in into an open redirect."""
    if value and value.startswith("/") and not value.startswith("//") and not value.startswith("/login"):
        return value
    return DEFAULT_NEXT


def _verify_with_google(id_token_text: str, client_id: str) -> dict:
    """Checks the token's signature against Google's published keys, plus audience, issuer and expiry."""
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token

    return id_token.verify_oauth2_token(id_token_text, google_requests.Request(), client_id)


class GoogleOAuth:
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str,
                 post=requests.post, verify=_verify_with_google) -> None:
        self.client_id, self.client_secret, self.redirect_uri = client_id, client_secret, redirect_uri
        self.post, self.verify = post, verify  # injectable, so tests never talk to Google

    def start(self, secret: str, next_url: str | None) -> tuple[str, str]:
        """Return (the Google URL to send the browser to, a signed cookie value holding the secrets of this attempt)."""
        state = secrets.token_urlsafe(32)  # ties the callback to this browser
        nonce = secrets.token_urlsafe(32)  # ties the ID token to this attempt
        verifier = secrets.token_urlsafe(48)  # PKCE: only we can redeem the code
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        cookie = sign_payload(secret, {"state": state, "nonce": nonce, "verifier": verifier, "next": safe_next(next_url)},
                              STATE_TTL_SECONDS)
        query = urlencode({
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "nonce": nonce,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "prompt": "select_account",
        })
        return f"{AUTH_URL}?{query}", cookie

    def finish(self, secret: str, cookie: str | None, code: str, state: str | None) -> tuple[GoogleIdentity, str]:
        """Complete the sign-in. Returns the verified identity and where to send the user afterwards."""
        attempt = read_payload(secret, cookie or "")
        if attempt is None:
            raise GoogleAuthError("the sign-in attempt expired or was not started here")
        if not hmac.compare_digest(state or "", attempt["state"]):
            raise GoogleAuthError("state mismatch")

        response = self.post(TOKEN_URL, data={
            "code": code,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code",
            "code_verifier": attempt["verifier"],
        }, timeout=10)
        if response.status_code != 200:
            raise GoogleAuthError("Google refused the code")
        try:
            claims = self.verify(response.json().get("id_token", ""), self.client_id)
        except ValueError as e:
            raise GoogleAuthError(f"invalid ID token: {e}") from None

        if claims.get("iss") not in ISSUERS:
            raise GoogleAuthError("wrong issuer")
        if not hmac.compare_digest(str(claims.get("nonce", "")), attempt["nonce"]):
            raise GoogleAuthError("nonce mismatch")
        if claims.get("email_verified") is not True:
            raise GoogleAuthError("Google has not verified this email")
        if not claims.get("sub") or not claims.get("email"):
            raise GoogleAuthError("incomplete identity")
        return GoogleIdentity(sub=str(claims["sub"]), email=str(claims["email"]).lower()), attempt["next"]
