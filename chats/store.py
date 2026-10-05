import contextlib
import json
import sqlite3
from pathlib import Path

TITLE_LENGTH = 60


class ChatStore:
    """Conversations and their messages in SQLite. Every query is scoped by user id, so one user can never
    read, change or even detect another user's chat: a foreign id behaves exactly like a missing one."""

    def __init__(self, path: str) -> None:
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS chats ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, title TEXT NOT NULL, "
                "created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%d %H:%M:%f', 'now')))")
            conn.execute("CREATE INDEX IF NOT EXISTS chats_user ON chats(user_id, updated_at)")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS messages ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE, "
                "role TEXT NOT NULL, content TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'done', "
                "meta TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
            conn.execute("CREATE INDEX IF NOT EXISTS messages_chat ON messages(chat_id, id)")
            # A research job lives in memory; if the server stopped mid-job nobody will ever finish it.
            conn.execute("UPDATE messages SET status = 'failed', content = 'This research was interrupted by a "
                         "server restart. Ask again to retry.' WHERE status = 'running'")

    @contextlib.contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def create(self, user: str, title: str = "New chat") -> dict:
        with self._connect() as conn:
            cursor = conn.execute("INSERT INTO chats (user_id, title) VALUES (?, ?)", (user, title[:TITLE_LENGTH]))
            return self._summary(conn, user, cursor.lastrowid)

    def _summary(self, conn, user: str, chat_id: int) -> dict | None:
        row = conn.execute("SELECT id, title, updated_at FROM chats WHERE id = ? AND user_id = ?",
                           (chat_id, user)).fetchone()
        return {"id": row[0], "title": row[1], "updated_at": row[2]} if row else None

    def list(self, user: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT id, title, updated_at FROM chats WHERE user_id = ? "
                                "ORDER BY updated_at DESC, id DESC", (user,)).fetchall()
        return [{"id": r[0], "title": r[1], "updated_at": r[2]} for r in rows]

    def get(self, user: str, chat_id: int) -> dict | None:
        with self._connect() as conn:
            summary = self._summary(conn, user, chat_id)
            if summary is None:
                return None
            rows = conn.execute("SELECT id, role, content, status, meta, created_at FROM messages "
                                "WHERE chat_id = ? ORDER BY id", (chat_id,)).fetchall()
        return {**summary, "messages": [
            {"id": r[0], "role": r[1], "content": r[2], "status": r[3], "meta": json.loads(r[4]), "created_at": r[5]}
            for r in rows]}

    def rename(self, user: str, chat_id: int, title: str) -> bool:
        with self._connect() as conn:
            return conn.execute("UPDATE chats SET title = ? WHERE id = ? AND user_id = ?",
                                (title.strip()[:TITLE_LENGTH] or "New chat", chat_id, user)).rowcount > 0

    def delete(self, user: str, chat_id: int) -> bool:
        with self._connect() as conn:
            return conn.execute("DELETE FROM chats WHERE id = ? AND user_id = ?", (chat_id, user)).rowcount > 0

    def add_message(self, user: str, chat_id: int, role: str, content: str, status: str = "done",
                    meta: dict | None = None) -> int | None:
        """Append a message; None if the chat is not this user's. Also bumps the chat to the top of the list."""
        with self._connect() as conn:
            if self._summary(conn, user, chat_id) is None:
                return None
            cursor = conn.execute("INSERT INTO messages (chat_id, role, content, status, meta) VALUES (?, ?, ?, ?, ?)",
                                  (chat_id, role, content, status, json.dumps(meta or {})))
            conn.execute("UPDATE chats SET updated_at = strftime('%Y-%m-%d %H:%M:%f', 'now') WHERE id = ?", (chat_id,))
            return cursor.lastrowid

    def update_message(self, message_id: int, content: str | None = None, status: str | None = None,
                       meta: dict | None = None) -> None:
        """Used by background research jobs, which only know the message id they were started for."""
        with self._connect() as conn:
            if content is not None:
                conn.execute("UPDATE messages SET content = ? WHERE id = ?", (content, message_id))
            if status is not None:
                conn.execute("UPDATE messages SET status = ? WHERE id = ?", (status, message_id))
            if meta is not None:
                conn.execute("UPDATE messages SET meta = ? WHERE id = ?", (json.dumps(meta), message_id))

    def message_meta(self, message_id: int) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT meta FROM messages WHERE id = ?", (message_id,)).fetchone()
        return json.loads(row[0]) if row else {}

    def is_untitled(self, user: str, chat_id: int) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT title FROM chats WHERE id = ? AND user_id = ?", (chat_id, user)).fetchone()
        return bool(row) and row[0] == "New chat"
