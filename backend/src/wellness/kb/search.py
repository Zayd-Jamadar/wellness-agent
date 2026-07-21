"""Knowledge-base search backed by SQLite + sqlite-vec.

`KBService` owns a single-file SQLite database that stores KB chunks and their
dense embeddings in a `vec0` virtual table, and answers queries via vector KNN.
Embeddings are computed via the OpenAI embeddings API (needs OPENAI_API_KEY).
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import struct
import threading
from dataclasses import dataclass
from pathlib import Path

from wellness import paths
from wellness.config import Settings, get_settings
from wellness.logging import get_logger

log = get_logger(service="kb_search")

_CHUNK_SCHEMA_VERSION = 2


def _serialize_f32(vector: list[float]) -> bytes:
    """Pack a list of floats into the raw f32 blob sqlite-vec expects."""
    return struct.pack(f"{len(vector)}f", *vector)


@dataclass(frozen=True)
class Chunk:
    """A single searchable unit of the knowledge base."""

    id: str
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class SearchResult:
    """A ranked search hit (lower ``distance`` is more similar)."""

    chunk: Chunk
    distance: float


class KBService:
    """Loads, indexes, and searches the wellness knowledge base.

    A single instance owns the SQLite connection (with the sqlite-vec extension
    loaded) and the query interface; embeddings are computed via the OpenAI API.
    Call :meth:`ensure_index` once (or :meth:`build_index` to force a rebuild)
    before :meth:`search`.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.kb_dir: Path = paths.get_kb_dir()
        self.db_path: Path = paths.get_db()
        self._db: sqlite3.Connection | None = None
        self._dim: int | None = None
        self._openai_client = None
        # The service is a process-wide singleton whose connection is warmed on
        # one thread (app lifespan) then reused by tool calls dispatched to
        # executor threads. Allow cross-thread use and serialize access.
        self._db_lock = threading.Lock()

    # ------------------------------------------------------------------ db --
    @property
    def db(self) -> sqlite3.Connection:
        """Return an open connection with sqlite-vec loaded (lazily created)."""
        if self._db is None:
            import sqlite_vec

            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            self._db = conn
        return self._db

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    # ----------------------------------------------------------- embedding --
    @property
    def dim(self) -> int:
        """Embedding dimension, derived from a one-item probe (cached)."""
        if self._dim is None:
            self._dim = len(self._embed(["dimension probe"])[0])
        return self._dim

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via the OpenAI embeddings API (needs OPENAI_API_KEY)."""
        if self._openai_client is None:
            from openai import OpenAI

            # Key comes from Settings (.env), which is not necessarily exported
            # to os.environ; fall back to the env if unset.
            self._openai_client = OpenAI(api_key=self.settings.openai_api_key)
        resp = self._openai_client.embeddings.create(
            model=self.settings.embedding_model,
            input=texts,
        )
        return [item.embedding for item in resp.data]

    # -------------------------------------------------------------- chunks --
    _TOKEN_H1 = re.compile(r"^#\s+(.*)$", re.MULTILINE)

    def _read_title(self, text: str, fallback: str) -> str:
        m = self._TOKEN_H1.search(text)
        return m.group(1).strip() if m else fallback

    def _split_paragraphs(self, text: str) -> list[str]:
        paragraphs: list[str] = []
        for block in text.split("\n\n"):
            block = block.strip()
            if not block or block.startswith("# "):
                continue
            if block.startswith("*") and block.endswith("*") and "\n" not in block:
                continue
            paragraphs.append(" ".join(block.split()))
        return paragraphs

    def load_chunks(self) -> list[Chunk]:
        """Read and chunk every markdown document in the KB directory."""
        md_files = sorted(self.kb_dir.glob("*.md"))
        if not md_files:
            raise FileNotFoundError(f"No markdown documents found in {self.kb_dir}")
        chunks: list[Chunk] = []
        for md in md_files:
            text = md.read_text(encoding="utf-8")
            doc_id = md.stem
            title = self._read_title(text, fallback=doc_id.replace("_", " "))
            for i, para in enumerate(self._split_paragraphs(text)):
                chunks.append(
                    Chunk(id=f"{doc_id}#{i}", doc_id=doc_id, title=title, text=para)
                )
        log.info("chunked", docs=len(md_files), chunks=len(chunks))
        return chunks

    def _signature(self) -> str:
        h = hashlib.sha256()
        h.update(f"v{_CHUNK_SCHEMA_VERSION}:{self.settings.embedding_model}".encode())
        for md in sorted(self.kb_dir.glob("*.md")):
            h.update(md.name.encode())
            h.update(md.read_bytes())
        return h.hexdigest()

    # --------------------------------------------------------------- index --
    def _current_signature(self) -> str | None:
        try:
            with self._db_lock:
                row = self.db.execute(
                    "SELECT value FROM kb_meta WHERE key = 'signature'"
                ).fetchone()
        except sqlite3.OperationalError:
            return None
        return row["value"] if row else None

    def is_indexed(self) -> bool:
        """True when a valid, up-to-date index already exists on disk."""
        return self._current_signature() == self._signature()

    def build_index(self) -> int:
        """(Re)build the SQLite index from scratch. Returns chunk count."""
        chunks = self.load_chunks()
        embeddings = self._embed([c.text for c in chunks])
        dim = self.dim

        with self._db_lock:
            return self._build_index_locked(chunks, embeddings, dim)

    def _build_index_locked(self, chunks, embeddings, dim: int) -> int:
        db = self.db
        db.execute("DROP TABLE IF EXISTS kb_chunks")
        db.execute("DROP TABLE IF EXISTS kb_meta")
        db.execute(
            f"""
            CREATE VIRTUAL TABLE kb_chunks USING vec0(
                embedding float[{dim}],
                +chunk_id TEXT,
                +doc_id TEXT,
                +title TEXT,
                +text TEXT
            )
            """
        )
        db.execute("CREATE TABLE kb_meta (key TEXT PRIMARY KEY, value TEXT)")
        for chunk, emb in zip(chunks, embeddings):
            db.execute(
                """
                INSERT INTO kb_chunks (embedding, chunk_id, doc_id, title, text)
                VALUES (?, ?, ?, ?, ?)
                """,
                (_serialize_f32(emb), chunk.id, chunk.doc_id, chunk.title, chunk.text),
            )
        db.execute(
            "INSERT INTO kb_meta (key, value) VALUES ('signature', ?)",
            (self._signature(),),
        )
        db.commit()
        log.info("index_built", chunks=len(chunks), db=str(self.db_path))
        return len(chunks)

    def ensure_index(self) -> None:
        """Build the index only if it is missing or stale."""
        if not self.is_indexed():
            log.info("index_stale_rebuilding")
            self.build_index()
        else:
            log.info("index_ready")

    # -------------------------------------------------------------- search --
    def search(self, query: str, top_k: int | None = None) -> list[SearchResult]:
        """Return the top-k most similar chunks for a query via vector KNN.

        Args:
            query: Natural-language query.
            top_k: Number of results; defaults to ``settings.kb_top_k``.

        Returns:
            Ranked list of :class:`SearchResult` (nearest first).
        """
        if not query.strip():
            return []
        self.ensure_index()
        k = top_k or self.settings.kb_top_k
        qvec = self._embed([query])[0]
        with self._db_lock:
            rows = self.db.execute(
                """
                SELECT chunk_id, doc_id, title, text, distance
                FROM kb_chunks
                WHERE embedding MATCH ?
                ORDER BY distance
                LIMIT ?
                """,
                (_serialize_f32(qvec), k),
            ).fetchall()
        results = [
            SearchResult(
                chunk=Chunk(
                    id=r["chunk_id"],
                    doc_id=r["doc_id"],
                    title=r["title"],
                    text=r["text"],
                ),
                distance=float(r["distance"]),
            )
            for r in rows
        ]
        log.info(
            "search",
            query=query,
            top_k=k,
            hits=[r.chunk.id for r in results],
        )
        return results


_SERVICE: KBService | None = None


def get_kb_service(settings: Settings | None = None) -> KBService:
    """Return a process-wide singleton :class:`KBService`."""
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = KBService(settings)
    return _SERVICE
