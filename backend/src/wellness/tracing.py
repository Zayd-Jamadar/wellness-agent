"""Langfuse tracing seam.

Provides a LangChain callback handler that traces every LangGraph run (LLM calls,
tool calls, retrieval) to Langfuse. Tracing is fully optional: if Langfuse is not
configured or its SDK fails to initialize, the helpers degrade to a no-op so the
agent always runs.
"""

from __future__ import annotations

import os
from typing import Any

from wellness.config import Settings, get_settings
from wellness.logging import get_logger

log = get_logger(service="tracing")

_warned = False


def _export_env(settings: Settings) -> None:
    """Ensure the Langfuse SDK env vars are present for the singleton client."""
    if settings.langfuse_public_key:
        os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
    if settings.langfuse_secret_key:
        os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
    os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)


def get_langfuse_handler(settings: Settings | None = None) -> Any | None:
    """Return a Langfuse LangChain callback handler, or ``None`` if unavailable.

    Args:
        settings: Optional settings; defaults to :func:`get_settings`.

    Returns:
        A ``langfuse.langchain.CallbackHandler`` instance, or ``None`` when
        tracing is disabled/unconfigured or the SDK cannot be initialized.
    """
    global _warned
    settings = settings or get_settings()
    if not settings.tracing_configured():
        return None

    try:
        _export_env(settings)
        from langfuse.langchain import CallbackHandler

        handler = CallbackHandler()
        log.info("langfuse_enabled", host=settings.langfuse_host)
        return handler
    except Exception as exc:  # noqa: BLE001 - tracing must never break a run
        if not _warned:
            log.warning("langfuse_init_failed", error=str(exc))
            _warned = True
        return None


def flush_langfuse() -> None:
    """Flush pending Langfuse spans (best-effort) before process exit."""
    try:
        from langfuse import get_client

        get_client().flush()
    except Exception as exc:  # noqa: BLE001 - best effort
        log.debug("langfuse_flush_skipped", error=str(exc))
