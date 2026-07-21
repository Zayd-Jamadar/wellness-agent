"""FastAPI application factory for the wellness agent service."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from wellness import __version__
from wellness.api.routes.chat import router as chat_router
from wellness.config import get_settings
from wellness.logging import get_logger

log = get_logger(service="api")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Warm the KB index and open the persistent memory checkpointer."""
    settings = get_settings()
    if "lookup_kb" in settings.enabled_tools:
        try:
            from wellness.kb.search import get_kb_service

            # Use the process-wide singleton so the index is reused by the
            # request path (the lookup_kb tool). Builds via the OpenAI
            # embeddings API if the index is missing or stale.
            get_kb_service(settings).ensure_index()
            log.info("kb_warmed")
        except Exception as exc:  # non-fatal: KB just won't be preloaded
            log.warning("kb_warm_failed", error=str(exc))

    app.state.checkpointer = None
    if settings.memory_enabled:
        from wellness.memory import open_async_sqlite_saver

        # Keep one AsyncSqliteSaver open for the whole app lifetime so memory
        # persists across requests. The context manager owns the connection.
        async with open_async_sqlite_saver(settings) as saver:
            app.state.checkpointer = saver
            yield
    else:
        yield


def create_app() -> FastAPI:
    """Build the FastAPI app (CORS, lifespan, routes)."""
    settings = get_settings()
    app = FastAPI(title="Wellness Assistant API", version=__version__, lifespan=_lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    app.include_router(chat_router)
    return app
