import base64
import hashlib
import hmac
import os

# scrypt is deliberately slow and memory-hungry, which makes mass password guessing expensive.
# Parameters travel inside every stored hash, so they can be raised later without breaking old accounts.
N, R, P = 2**14, 8, 1
MAX_MEMORY = 64 * 1024 * 1024


def _derive(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p, dklen=32, maxmem=MAX_MEMORY)


def hash_password(password: str) -> str:
    salt = os.urandom(16)  # unique per user: identical passwords produce different hashes
    digest = _derive(password, salt, N, R, P)
    return "$".join(["scrypt", str(N), str(R), str(P), base64.b64encode(salt).decode(), base64.b64encode(digest).decode()])


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, n, r, p, salt, digest = stored.split("$")
        if scheme != "scrypt":
            return False
        expected = base64.b64decode(digest)
        actual = _derive(password, base64.b64decode(salt), int(n), int(r), int(p))
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(actual, expected)  # constant-time comparison


# Verified against when an email is unknown, so "no such user" costs the same time as "wrong password".
DUMMY_HASH = hash_password("not-a-real-password")
