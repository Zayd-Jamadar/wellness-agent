"""Chat model construction (provider-routed).

A single :func:`build_chat_model` builds a LangChain chat model from settings,
routing on ``settings.provider`` between hosted OpenAI (``ChatOpenAI``) and
local Ollama (``ChatOllama``). Both integrations implement ``bind_tools``
natively, so the agent graph is provider-agnostic.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from wellness.config import Settings, get_settings
from wellness.logging import get_logger

log = get_logger(service="llm")

_DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


def build_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Return a configured chat model for the selected provider."""
    settings = settings or get_settings()

    log.info(
        "build_llm",
        provider=settings.provider,
        model=settings.model,
        api_base=settings.api_base,
    )

    if settings.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.model,
            base_url=settings.api_base or _DEFAULT_OLLAMA_BASE_URL,
            temperature=settings.temperature,
            # Ollama's max-tokens knob.
            num_predict=settings.max_tokens,
            # Thinking OFF by default: no reasoning trace, reliable tool calls.
            reasoning=settings.reasoning,
        )

    from langchain_openai import ChatOpenAI

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
        kwargs["base_url"] = settings.api_base
    # The key is loaded from .env into Settings by pydantic-settings and is not
    # necessarily present in os.environ (where ChatOpenAI would otherwise look).
    if settings.openai_api_key:
        kwargs["api_key"] = settings.openai_api_key

    return ChatOpenAI(**kwargs)
