"""Chat endpoint streaming the agent in the AI SDK Data Stream Protocol."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from wellness.agent.graph import WellnessAgent
from wellness.api.ai_stream import STREAM_HEADERS, ai_stream
from wellness.api.schemas import ChatRequest
from wellness.logging import get_logger

log = get_logger(service="api_chat")
router = APIRouter()


@router.post("/api/chat")
async def chat(req: ChatRequest, request: Request) -> StreamingResponse:
    """Stream one assistant turn to the AI SDK ``useChat`` frontend."""
    message = req.latest_user_text()
    thread_id = req.thread_id()
    enabled = set(req.enabled_tools) if req.enabled_tools is not None else None

    log.info("chat_request", thread_id=thread_id, enabled_tools=enabled)
    checkpointer = getattr(request.app.state, "checkpointer", None)
    agent = WellnessAgent(enabled_tools=enabled, checkpointer=checkpointer)

    return StreamingResponse(
        ai_stream(agent, message, thread_id),
        media_type="text/event-stream",
        headers=STREAM_HEADERS,
    )
