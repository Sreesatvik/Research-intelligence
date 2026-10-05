import contextlib
import sqlite3
from dataclasses import dataclass
from pathlib import Path

# Stored for accounts that sign in with Google only. It is not a valid scrypt hash, so no password ever matches it.
NO_PASSWORD = "google$no-password"


class EmailTaken(Exception):
    pass


@dataclass
class User:
    id: int
    email: str
    password_hash: str
    google_sub: str | None = None


COLUMNS = "id, email, password_hash, google_sub"


class UserStore:
    """Accounts in a SQLite file. Emails are stored lowercase and are unique."""

    def __init__(self, path: str) -> None:
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS users ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "email TEXT NOT NULL UNIQUE, "
                "password_hash TEXT NOT NULL, "
                "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            # Migration: databases created before Google sign-in have no google_sub column.
            if "google_sub" not in [row[1] for row in conn.execute("PRAGMA table_info(users)")]:
                conn.execute("ALTER TABLE users ADD COLUMN google_sub TEXT")
            conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS users_google_sub ON users(google_sub)")

    @contextlib.contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path)  # one short-lived connection per call: safe across request threads
        try:
            with conn:  # commits on success, rolls back on error
                yield conn
        finally:
            conn.close()

    def create(self, email: str, password_hash: str, google_sub: str | None = None) -> User:
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    "INSERT INTO users (email, password_hash, google_sub) VALUES (?, ?, ?)",
                    (email.lower(), password_hash, google_sub))
                return User(cursor.lastrowid, email.lower(), password_hash, google_sub)
        except sqlite3.IntegrityError:
            raise EmailTaken(email) from None

    def _one(self, where: str, value: str) -> User | None:
        with self._connect() as conn:
            row = conn.execute(f"SELECT {COLUMNS} FROM users WHERE {where} = ?", (value,)).fetchone()
        return User(*row) if row else None

    def by_email(self, email: str) -> User | None:
        return self._one("email", email.lower())

    def by_google_sub(self, sub: str) -> User | None:
        return self._one("google_sub", sub)

    def link_google(self, user_id: int, sub: str) -> None:
        """Attach a Google identity to an existing account and revoke its password.

        Google has proved who owns this email; a password set at signup was never verified, so it may
        belong to someone else. Revoking it means only the Google account holder can use this account."""
        with self._connect() as conn:
            conn.execute("UPDATE users SET google_sub = ?, password_hash = ? WHERE id = ?", (sub, NO_PASSWORD, user_id))
