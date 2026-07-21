"""Chat model construction (LiteLLM-backed).

A single :func:`build_chat_model` builds a LangChain ``ChatLiteLLM`` from
settings, so the same graph runs on any vendor (OSS or frontier) by changing the
model string / ``api_base``. Gateway/routing concerns now live in the external
LiteLLM proxy, so no in-process gateway abstraction is needed here.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from wellness.config import Settings, get_settings
from wellness.logging import get_logger

log = get_logger(service="llm")


def build_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Return a configured ``ChatLiteLLM`` for the given settings."""
    from langchain_litellm import ChatLiteLLM

    settings = settings or get_settings()
    kwargs: dict[str, object] = {
        "model": settings.model,
        "temperature": settings.temperature,
        # Emit token chunks so LangGraph produces on_chat_model_stream events
        # that the API translates to AI SDK text deltas.
        "streaming": True,
    }
    if settings.max_tokens is not None:
        kwargs["max_tokens"] = settings.max_tokens
    if settings.api_base:
        kwargs["api_base"] = settings.api_base

    log.info("build_llm", model=settings.model, api_base=settings.api_base)
    return ChatLiteLLM(**kwargs)
