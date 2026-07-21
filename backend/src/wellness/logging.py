"""Structured logging setup via structlog.

Call :func:`configure` once at process start, then use :func:`get_logger` to
obtain context-bound loggers, e.g. ``get_logger(service="kb_search")``.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_configured = False


def configure(level: str = "INFO", json_logs: bool = False) -> None:
    """Configure structlog + stdlib logging once.

    Args:
        level: Root log level name (e.g. ``"INFO"``, ``"DEBUG"``).
        json_logs: Emit JSON lines instead of the console renderer.
    """
    global _configured
    if _configured:
        return

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(service: str, **initial_values: Any) -> structlog.stdlib.BoundLogger:
    """Return a context-bound logger for a service.

    Args:
        service: Logical service name bound to every log line.
        **initial_values: Extra context to bind.

    Returns:
        A bound structlog logger.
    """
    if not _configured:
        configure()
    return structlog.get_logger(service=service, **initial_values)
