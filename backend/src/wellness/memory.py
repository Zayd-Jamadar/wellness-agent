"""Short-term conversation memory backed by SQLite.

Provides LangGraph checkpointer factories. Conversation state is persisted per
``thread_id`` so multi-turn memory survives across HTTP requests and restarts.

Two variants exist because the sync CLI and the async API have different I/O
models:

- :func:`open_sqlite_saver` - sync ``SqliteSaver`` for ``WellnessAgent.invoke``.
- :func:`open_async_sqlite_saver` - async ``AsyncSqliteSaver`` for the API's
  ``ainvoke`` / ``astream_events`` (async checkpoint I/O off the event loop).

Both are context managers that own their DB connection.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import AsyncIterator, Iterator

# Restrict checkpoint deserialization to known-safe types (package security note).
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

from wellness.config import Settings, get_settings
from wellness.logging import get_logger
from wellness.paths import get_memory_db

log = get_logger(service="memory")


def _db_path(settings: Settings) -> Path:
    """Resolve the memory DB path from settings (or the default state dir)."""
    if settings.memory_db_path:
        return Path(settings.memory_db_path).expanduser()
    return get_memory_db()


@contextmanager
def open_sqlite_saver(settings: Settings | None = None) -> Iterator[object]:
    """Yield a sync ``SqliteSaver`` bound to the memory DB.

    ``check_same_thread=False`` because the connection may be touched from
    worker threads during a run.
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    settings = settings or get_settings()
    path = _db_path(settings)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    try:
        saver = SqliteSaver(conn)
        saver.setup()
        log.info("memory_open", backend="sqlite", path=str(path))
        yield saver
    finally:
        conn.close()


@asynccontextmanager
async def open_async_sqlite_saver(
    settings: Settings | None = None,
) -> AsyncIterator[object]:
    """Yield an async ``AsyncSqliteSaver`` bound to the memory DB."""
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    settings = settings or get_settings()
    path = _db_path(settings)
    conn = await aiosqlite.connect(str(path))
    try:
        saver = AsyncSqliteSaver(conn)
        await saver.setup()
        log.info("memory_open", backend="aiosqlite", path=str(path))
        yield saver
    finally:
        await conn.close()
