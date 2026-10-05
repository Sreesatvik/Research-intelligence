import contextlib
import hashlib
import json
import re
import sqlite3
import threading
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from clustering.kmeans import kmeans
from clustering.naming import STOP, cluster_keywords
from clustering.projection import pca_project
from core.schemas import Chunk, SearchResult
from ingestion.chunking import chunk_text
from representations.tfidf.tokenizer import tokenize

ALLOWED = {".pdf", ".txt", ".md"}
MAX_CLUSTERS = 7


class LibraryError(Exception):
    """The upload cannot be accepted; the message is safe to show the user."""


class DuplicateDocument(LibraryError):
    """The same document is already in this user's library."""

    def __init__(self, existing: dict) -> None:
        super().__init__(f'Already in your library as "{existing["name"]}".')
        self.existing = existing


def text_fingerprint(chunks: list[Chunk]) -> str:
    """Identifies a document by what it says, so the same text saved as .txt and .md (or re-exported) still matches."""
    text = re.sub(r"\s+", " ", " ".join(c.text for c in chunks)).strip().lower()
    return hashlib.sha256(text.encode()).hexdigest()


def slug(filename: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(filename).stem).strip("-._")[:80]
    return stem or "document"


class UserIndex:
    """One user's chunks and their embeddings. It is the retriever the RAG pipeline and the agent search."""

    def __init__(self, model, chunks: list[Chunk], vectors: np.ndarray) -> None:
        self.model, self.chunks, self.vectors = model, chunks, vectors

    def search(self, query: str, k: int = 5) -> list[SearchResult]:
        scores = self.vectors @ self.model.encode(query, normalize_embeddings=True)
        return [SearchResult(self.chunks[i], float(scores[i])) for i in np.argsort(-scores)[:k]]


class Library:
    """Every user's own documents: files on disk, a row per document, and an in-memory index built on first use.

    Nothing here is shared between users: each query is scoped by user, and each user's files live in a
    directory named by a hash of their id, so an odd id can never become a path."""

    def __init__(self, root: str, db_path: str, model, max_documents: int = 30,
                 max_bytes: int = 15 * 1024 * 1024) -> None:
        self.root, self.db_path, self.model = Path(root), db_path, model
        self.max_documents, self.max_bytes = max_documents, max_bytes
        self._indexes: dict[str, UserIndex] = {}
        self._maps: dict[str, dict] = {}
        self._lock = threading.RLock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS documents ("
                "user_id TEXT NOT NULL, doc_id TEXT NOT NULL, name TEXT NOT NULL, kind TEXT NOT NULL, "
                "chunks INTEGER NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "PRIMARY KEY (user_id, doc_id))")
            columns = [row[1] for row in conn.execute("PRAGMA table_info(documents)")]
            for column in ("file_hash", "text_hash"):  # migration: databases from before duplicate detection
                if column not in columns:
                    conn.execute(f"ALTER TABLE documents ADD COLUMN {column} TEXT")
        self._backfill_hashes()

    @contextlib.contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _backfill_hashes(self) -> None:
        """Documents added before duplicate detection get their fingerprints computed from the stored files."""
        with self._connect() as conn:
            rows = conn.execute("SELECT user_id, doc_id, kind FROM documents WHERE file_hash IS NULL OR text_hash IS NULL").fetchall()
        for user, doc_id, kind in rows:
            folder = self._dir(user)
            source, chunk_file = folder / f"{doc_id}.{kind}", folder / f"{doc_id}.chunks.json"
            if not (source.exists() and chunk_file.exists()):
                continue
            chunks = [Chunk(**c) for c in json.loads(chunk_file.read_text(encoding="utf-8"))]
            with self._connect() as conn:
                conn.execute("UPDATE documents SET file_hash = ?, text_hash = ? WHERE user_id = ? AND doc_id = ?",
                             (hashlib.sha256(source.read_bytes()).hexdigest(), text_fingerprint(chunks), user, doc_id))

    def _duplicate(self, user: str, column: str, value: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(f"SELECT doc_id, name FROM documents WHERE user_id = ? AND {column} = ?",
                               (user, value)).fetchone()
        return {"id": row[0], "name": row[1]} if row else None

    def known_files(self, user: str, file_hashes: list[str]) -> dict[str, dict]:
        """Which of these file hashes are already in the library (so the browser can say so before uploading)."""
        found = {}
        for value in set(file_hashes):
            existing = self._duplicate(user, "file_hash", value)
            if existing:
                found[value] = existing
        return found

    def _dir(self, user: str) -> Path:
        return self.root / hashlib.sha256(user.encode()).hexdigest()[:24]

    # ---- documents -----------------------------------------------------------------

    def documents(self, user: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT doc_id, name, kind, chunks, created_at FROM documents WHERE user_id = ? "
                                "ORDER BY created_at DESC, doc_id", (user,)).fetchall()
        return [{"id": r[0], "name": r[1], "kind": r[2], "chunks": r[3], "created_at": r[4]} for r in rows]

    def fingerprint(self, user: str) -> str:
        """Changes whenever the user's documents change. Cached answers are keyed by it, so an answer written
        before a document was added or deleted is never served afterwards."""
        with self._connect() as conn:
            rows = conn.execute("SELECT doc_id, created_at FROM documents WHERE user_id = ? ORDER BY doc_id",
                                (user,)).fetchall()
        return hashlib.sha1(json.dumps(rows).encode()).hexdigest()[:10]

    def add(self, user: str, filename: str, data: bytes) -> dict:
        filename = filename.replace("\\", "/").rsplit("/", 1)[-1]  # keep only the file name, never a path
        extension = Path(filename).suffix.lower()
        if extension not in ALLOWED:
            raise LibraryError("Only PDF, TXT and Markdown files are supported.")
        if not data:
            raise LibraryError("The file is empty.")
        if len(data) > self.max_bytes:
            raise LibraryError(f"The file is larger than {self.max_bytes // (1024 * 1024)} MB.")
        with self._lock:
            file_hash = hashlib.sha256(data).hexdigest()
            same = self._duplicate(user, "file_hash", file_hash)
            if same:
                raise DuplicateDocument(same)
            existing = {d["id"] for d in self.documents(user)}
            if len(existing) >= self.max_documents:
                raise LibraryError(f"Your library is full ({self.max_documents} documents). Delete one first.")
            doc_id, n = slug(filename), 1
            while doc_id in existing:
                n += 1
                doc_id = f"{slug(filename)}-{n}"
            folder = self._dir(user)
            folder.mkdir(parents=True, exist_ok=True)
            chunks = self._parse(data, doc_id, extension)  # checked in memory: a bad file never touches the disk
            text_hash = text_fingerprint(chunks)
            same = self._duplicate(user, "text_hash", text_hash)
            if same:
                raise DuplicateDocument(same)
            (folder / f"{doc_id}{extension}").write_bytes(data)
            (folder / f"{doc_id}.chunks.json").write_text(json.dumps([asdict(c) for c in chunks]), encoding="utf-8")
            with self._connect() as conn:
                conn.execute("INSERT INTO documents (user_id, doc_id, name, kind, chunks, file_hash, text_hash) "
                             "VALUES (?, ?, ?, ?, ?, ?, ?)",
                             (user, doc_id, filename[:200], extension[1:], len(chunks), file_hash, text_hash))
            self._maps.pop(user, None)
            index = self._indexes.get(user)
            if index is not None:  # keep a loaded index current without re-embedding everything
                vectors = self.model.encode([c.text for c in chunks], normalize_embeddings=True)
                index.vectors = np.vstack([index.vectors, vectors]) if index.chunks else vectors
                index.chunks = index.chunks + chunks
            return next(d for d in self.documents(user) if d["id"] == doc_id)

    def _parse(self, data: bytes, doc_id: str, extension: str) -> list[Chunk]:
        if extension == ".pdf":
            from ingestion.parsers.pdf import parse_pdf

            try:
                chunks = parse_pdf(data, doc_id)
            except Exception:
                raise LibraryError("Could not read this PDF. It may be damaged or password-protected.") from None
            if not chunks:
                raise LibraryError("This PDF has no selectable text (it may be a scan).")
            return chunks
        chunks = chunk_text(doc_id, data.decode("utf-8", errors="replace"))
        if not chunks:
            raise LibraryError("The file has no text.")
        return chunks

    def delete(self, user: str, doc_id: str) -> bool:
        with self._lock:
            with self._connect() as conn:
                removed = conn.execute("DELETE FROM documents WHERE user_id = ? AND doc_id = ?",
                                       (user, doc_id)).rowcount > 0
            if not removed:
                return False
            for path in self._dir(user).glob(f"{glob_escape(doc_id)}.*"):
                if path.name == f"{doc_id}.chunks.json" or path.stem == doc_id:
                    path.unlink(missing_ok=True)
            self._maps.pop(user, None)
            index = self._indexes.get(user)
            if index is not None:
                keep = np.array([c.document_id != doc_id for c in index.chunks], dtype=bool)
                index.chunks = [c for c, k in zip(index.chunks, keep) if k]
                index.vectors = index.vectors[keep]
            return True

    # ---- search --------------------------------------------------------------------

    def chunks(self, user: str) -> list[Chunk]:
        index = self.retriever(user)
        return index.chunks if index else []

    def retriever(self, user: str) -> UserIndex | None:
        """The user's search index, or None while their library is empty."""
        with self._lock:
            if user not in self._indexes:
                chunks: list[Chunk] = []
                for doc in self.documents(user):
                    path = self._dir(user) / f"{doc['id']}.chunks.json"
                    if path.exists():
                        chunks.extend(Chunk(**c) for c in json.loads(path.read_text(encoding="utf-8")))
                if not chunks:
                    return None
                vectors = self.model.encode([c.text for c in chunks], normalize_embeddings=True, batch_size=64)
                self._indexes[user] = UserIndex(self.model, chunks, vectors)
            index = self._indexes[user]
            return index if index.chunks else None

    # ---- the corpus map ------------------------------------------------------------

    def corpus_map(self, user: str) -> dict:
        """One 3D point per document, grouped by topic. Computed from the user's own documents only."""
        with self._lock:
            if user not in self._maps:
                self._maps[user] = self._build_map(user)
            return self._maps[user]

    def _build_map(self, user: str) -> dict:
        index = self.retriever(user)
        if index is None:
            return {"points": [], "clusters": [], "explained_variance": []}
        names = {d["id"]: d["name"] for d in self.documents(user)}
        ids = list(names)
        rows = {}
        for doc in ids:
            of_doc = [i for i, c in enumerate(index.chunks) if c.document_id == doc]
            rows[doc] = [i for i in of_doc if index.chunks[i].content_type == "text"] or of_doc
        X = np.array([index.vectors[rows[d]].mean(axis=0) for d in ids])
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
        texts = [" ".join(index.chunks[i].text for i in rows[d]) for d in ids]

        n = len(ids)
        k = 1 if n < 3 else min(MAX_CLUSTERS, max(2, round(n ** 0.5)))
        if k > 1:
            labels, _, _ = max((kmeans(X, k, seed=s) for s in range(10)), key=lambda run: run[2])
            keywords = cluster_keywords(texts, labels, k)
        else:
            labels = np.zeros(n, dtype=int)
            words = Counter(w for w in tokenize(" ".join(texts)) if w not in STOP and len(w) > 2)
            keywords = [[w for w, _ in words.most_common(6)]]

        if n >= 2:
            coords, explained = pca_project(X, 3)
            coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))
            coords = coords / (np.abs(coords).max() or 1.0)
            explained = np.nan_to_num(explained)
        else:
            coords, explained = np.zeros((1, 3)), np.zeros(0)

        points = [{
            "id": ids[i], "title": texts[i].split(". ", 1)[0][:110], "topic": names[ids[i]],
            "x": round(float(coords[i, 0]), 4), "y": round(float(coords[i, 1]), 4), "z": round(float(coords[i, 2]), 4),
            "cluster": int(labels[i]),
        } for i in range(n)]
        clusters = [{"id": j, "keywords": keywords[j], "size": int((labels == j).sum())}
                    for j in range(k) if (labels == j).any()]
        return {"points": points, "clusters": clusters,
                "explained_variance": [round(float(v), 4) for v in explained]}


def glob_escape(text: str) -> str:
    return re.sub(r"([\[\]*?])", r"[\1]", text)
