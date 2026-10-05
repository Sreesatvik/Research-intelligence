import time

import jwt

ALGORITHM = "HS256"
MIN_SECRET_LENGTH = 32


def check_secret(secret: str) -> None:
    if len(secret) < MIN_SECRET_LENGTH:
        raise ValueError(f"AUTH_SECRET must be at least {MIN_SECRET_LENGTH} characters")


def issue_token(secret: str, user_id: int, email: str, ttl_seconds: int) -> str:
    now = int(time.time())
    return jwt.encode({"sub": str(user_id), "email": email, "iat": now, "exp": now + ttl_seconds}, secret, algorithm=ALGORITHM)


def read_token(secret: str, token: str) -> dict | None:
    """Return the token's claims, or None if it is forged, tampered with, expired or malformed.
    The algorithm is pinned: accepting whatever the token claims is how "alg: none" attacks work."""
    try:
        return jwt.decode(token, secret, algorithms=[ALGORITHM], options={"require": ["exp", "sub"]})
    except jwt.PyJWTError:
        return None


def sign_payload(secret: str, payload: dict, ttl_seconds: int) -> str:
    """A short-lived signed blob, for things like the OAuth attempt cookie. It has no 'sub', so it can never
    be mistaken for a login session."""
    now = int(time.time())
    return jwt.encode({**payload, "iat": now, "exp": now + ttl_seconds}, secret, algorithm=ALGORITHM)


def read_payload(secret: str, token: str) -> dict | None:
    try:
        return jwt.decode(token, secret, algorithms=[ALGORITHM], options={"require": ["exp"]})
    except jwt.PyJWTError:
        return None
