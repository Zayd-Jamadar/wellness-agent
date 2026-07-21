"""Translate agent events into the Vercel AI SDK Data Stream Protocol.

The AI SDK ``useChat`` hook consumes an SSE stream of JSON "parts". We map the
agent's normalized :class:`~wellness.agent.graph.AgentEvent`s onto that protocol:

- text:      ``text-start`` -> ``text-delta`` -> ``text-end``
- reasoning: ``reasoning-start`` -> ``reasoning-delta`` -> ``reasoning-end``
- tool:      ``tool-input-start`` -> ``tool-input-available`` -> ``tool-output-available``

The response must carry the ``x-vercel-ai-ui-message-stream: v1`` header.

See: https://ai-sdk.dev/docs/ai-sdk-ui/stream-protocol
"""

from __future__ import annotations

import json
import uuid
from typing import AsyncIterator

from wellness.agent.graph import AgentEvent, WellnessAgent
from wellness.logging import get_logger
from wellness.tracing import flush_langfuse

log = get_logger(service="api_stream")

STREAM_HEADERS = {
    "x-vercel-ai-ui-message-stream": "v1",
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    "connection": "keep-alive",
    "x-accel-buffering": "no",
}


def _sse(part: dict) -> str:
    """Encode one protocol part as an SSE ``data:`` event."""
    return f"data: {json.dumps(part, separators=(',', ':'))}\n\n"


async def ai_stream(
    agent: WellnessAgent, message: str, thread_id: str | None
) -> AsyncIterator[str]:
    """Yield AI SDK data-stream SSE events for one agent turn."""
    text_id: str | None = None
    reasoning_id: str | None = None

    def close_text() -> str | None:
        nonlocal text_id
        if text_id is not None:
            part = _sse({"type": "text-end", "id": text_id})
            text_id = None
            return part
        return None

    def close_reasoning() -> str | None:
        nonlocal reasoning_id
        if reasoning_id is not None:
            part = _sse({"type": "reasoning-end", "id": reasoning_id})
            reasoning_id = None
            return part
        return None

    yield _sse({"type": "start"})
    try:
        async for ev in agent.astream_events(message, thread_id):
            if ev.kind == "text-delta":
                if reasoning_id is not None:
                    yield close_reasoning()  # type: ignore[misc]
                if text_id is None:
                    text_id = uuid.uuid4().hex
                    yield _sse({"type": "text-start", "id": text_id})
                yield _sse(
                    {"type": "text-delta", "id": text_id, "delta": ev.text}
                )

            elif ev.kind == "reasoning-delta":
                if reasoning_id is None:
                    reasoning_id = uuid.uuid4().hex
                    yield _sse({"type": "reasoning-start", "id": reasoning_id})
                yield _sse(
                    {
                        "type": "reasoning-delta",
                        "id": reasoning_id,
                        "delta": ev.text,
                    }
                )

            elif ev.kind == "tool-start":
                closed = close_text()
                if closed:
                    yield closed
                closed = close_reasoning()
                if closed:
                    yield closed
                yield _sse(
                    {
                        "type": "tool-input-start",
                        "toolCallId": ev.tool_call_id,
                        "toolName": ev.tool_name,
                    }
                )
                yield _sse(
                    {
                        "type": "tool-input-available",
                        "toolCallId": ev.tool_call_id,
                        "toolName": ev.tool_name,
                        "input": ev.tool_input,
                    }
                )

            elif ev.kind == "tool-end":
                yield _sse(
                    {
                        "type": "tool-output-available",
                        "toolCallId": ev.tool_call_id,
                        "output": ev.tool_output,
                    }
                )

        closed = close_text()
        if closed:
            yield closed
        closed = close_reasoning()
        if closed:
            yield closed
        yield _sse({"type": "finish"})
    except Exception as exc:  # surface errors to the client + logs
        log.exception("stream_failed", error=str(exc))
        yield _sse({"type": "error", "errorText": str(exc)})
    finally:
        yield "data: [DONE]\n\n"
        flush_langfuse()
