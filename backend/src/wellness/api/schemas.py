"""Request/response models for the chat API.

These mirror the payload the Vercel AI SDK ``useChat`` hook POSTs: a list of
UI messages (each with typed ``parts``) plus a conversation ``id``.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class UIMessagePart(BaseModel):
    """A single part of a UI message (text, tool call, reasoning, ...)."""

    type: str
    text: str | None = None

    model_config = {"extra": "allow"}


class UIMessage(BaseModel):
    """A UI message as sent by the AI SDK frontend."""

    id: str | None = None
    role: str
    parts: list[UIMessagePart] = Field(default_factory=list)


class ChatRequest(BaseModel):
    """Body of ``POST /api/chat`` from ``useChat``."""

    id: str | None = Field(default=None, description="Conversation/thread id.")
    messages: list[UIMessage] = Field(default_factory=list)
    enabled_tools: list[str] | None = Field(
        default=None, description="Optional per-request tool subset."
    )

    def latest_user_text(self) -> str:
        """Return the concatenated text of the most recent user message."""
        for msg in reversed(self.messages):
            if msg.role != "user":
                continue
            text = "".join(
                p.text or "" for p in msg.parts if p.type == "text"
            ).strip()
            if text:
                return text
        return ""

    def thread_id(self) -> str | None:
        """Thread id for short-term memory + trace session grouping."""
        return self.id


class SessionSummary(BaseModel):
    """A past chat session row for the history sidebar."""

    id: str
    title: str
    preview: str = ""
    updated_at: str = ""


class SessionDetail(BaseModel):
    """A resumed session's transcript as AI-SDK UI messages."""

    id: str
    title: str
    messages: list[UIMessage] = Field(default_factory=list)
