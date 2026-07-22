"""Chat-history endpoints: list, resume (detail), and delete sessions."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response

from wellness.api import sessions as sessions_service
from wellness.api.schemas import SessionDetail, SessionSummary
from wellness.logging import get_logger

log = get_logger(service="api_sessions_routes")
router = APIRouter()


def _checkpointer(request: Request):
    return getattr(request.app.state, "checkpointer", None)


@router.post("/api/sessions", status_code=201)
async def create_session() -> dict[str, str]:
    """Mint a new conversation id (server is the source of truth).

    Mirrors LangGraph's ``threads.create`` (a new thread with a
    server-generated UUID). We use UUID v7 like the LangGraph SDK.
    """
    return {"id": sessions_service.new_thread_id()}


@router.get("/api/sessions")
async def list_sessions(request: Request) -> list[SessionSummary]:
    """List past chat sessions, newest first."""
    return await sessions_service.list_sessions(_checkpointer(request))


@router.get("/api/sessions/{thread_id}")
async def get_session(thread_id: str, request: Request) -> SessionDetail:
    """Return a session's transcript for resuming."""
    detail = await sessions_service.get_session(
        _checkpointer(request), thread_id
    )
    if detail is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return detail


@router.delete("/api/sessions/{thread_id}", status_code=204)
async def delete_session(thread_id: str, request: Request) -> Response:
    """Hard-delete a session's checkpoints (LangGraph-native)."""
    await sessions_service.delete_session(_checkpointer(request), thread_id)
    return Response(status_code=204)
