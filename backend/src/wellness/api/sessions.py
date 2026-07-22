"""Chat-session helpers built on LangGraph-native checkpointer APIs.

No custom SQL or tables: sessions are derived directly from the LangGraph
checkpointer that already persists conversation state per ``thread_id``.

- List sessions: ``checkpointer.alist(None)`` iterates checkpoints across all
  threads (newest first); we dedupe by ``thread_id`` in Python.
- Resume transcript: ``graph.aget_state(config)`` returns the latest state,
  whose ``messages`` we convert to AI-SDK UI messages.
- Delete: ``checkpointer.adelete_thread(thread_id)`` (native hard delete).
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage

from wellness.api.schemas import (
    SessionDetail,
    SessionSummary,
    UIMessage,
    UIMessagePart,
)
from wellness.logging import get_logger

log = get_logger(service="api_sessions")

_TITLE_MAX = 60
_PREVIEW_MAX = 120


def new_thread_id() -> str:
    """Generate a new thread id (UUID v7, as LangGraph's SDK recommends)."""
    try:
        from uuid_utils import uuid7

        return str(uuid7())
    except Exception:  # pragma: no cover - fallback if uuid-utils absent
        import uuid

        return str(uuid.uuid4())


def _text_of(message: BaseMessage) -> str:
    """Coerce a LangChain message's content to plain text."""
    content = message.content
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "reasoning":
            continue
        elif isinstance(block, dict) and "text" in block:
            parts.append(block["text"])
    return "".join(parts)


def _messages_of(checkpoint: dict[str, Any]) -> list[BaseMessage]:
    """Pull the message list out of a checkpoint's channel values."""
    values = checkpoint.get("channel_values", {}) or {}
    messages = values.get("messages", []) or []
    return [m for m in messages if isinstance(m, BaseMessage)]


def _first_user_text(messages: list[BaseMessage]) -> str:
    for m in messages:
        if m.type == "human":
            text = _text_of(m).strip()
            if text:
                return text
    return ""


def _last_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if m.type in ("human", "ai"):
            text = _text_of(m).strip()
            if text:
                return text
    return ""


def _truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "\u2026"


async def list_sessions(checkpointer: Any) -> list[SessionSummary]:
    """List past chat sessions, newest first (one row per ``thread_id``)."""
    if checkpointer is None:
        return []

    seen: dict[str, SessionSummary] = {}
    async for tup in checkpointer.alist(None):
        cfg = tup.config.get("configurable", {})
        thread_id = cfg.get("thread_id")
        if not thread_id or thread_id in seen:
            continue  # alist is newest-first: first hit is the latest
        messages = _messages_of(tup.checkpoint)
        title = _first_user_text(messages)
        if not title:
            continue  # skip empty/uninitialized threads
        seen[thread_id] = SessionSummary(
            id=thread_id,
            title=_truncate(title, _TITLE_MAX),
            preview=_truncate(_last_text(messages), _PREVIEW_MAX),
            updated_at=tup.checkpoint.get("ts", ""),
        )
    return list(seen.values())


async def get_session(checkpointer: Any, thread_id: str) -> SessionDetail | None:
    """Return a session's transcript as UI messages, or ``None`` if unknown."""
    if checkpointer is None:
        return None

    # Build an agent bound to the shared checkpointer to read graph state.
    from wellness.agent.graph import WellnessAgent

    agent = WellnessAgent(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    state = await agent.graph.aget_state(config)
    raw = (state.values or {}).get("messages", [])
    messages = [m for m in raw if isinstance(m, BaseMessage)]
    if not messages:
        return None

    ui_messages: list[UIMessage] = []
    for m in messages:
        if m.type not in ("human", "ai"):
            continue  # skip system/tool messages in the transcript view
        text = _text_of(m).strip()
        if not text:
            continue
        ui_messages.append(
            UIMessage(
                id=getattr(m, "id", None),
                role="user" if m.type == "human" else "assistant",
                parts=[UIMessagePart(type="text", text=text)],
            )
        )

    if not ui_messages:
        return None

    return SessionDetail(
        id=thread_id,
        title=_truncate(_first_user_text(messages), _TITLE_MAX),
        messages=ui_messages,
    )


async def delete_session(checkpointer: Any, thread_id: str) -> None:
    """Hard-delete all checkpoints/writes for a thread (LangGraph-native)."""
    if checkpointer is None:
        return
    await checkpointer.adelete_thread(thread_id)
