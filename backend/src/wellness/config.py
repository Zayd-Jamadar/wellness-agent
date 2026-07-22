"""Application configuration via pydantic-settings.

All runtime knobs live here so the same agent architecture can be pointed at
different LLM vendors (OSS vs frontier) by changing configuration only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Canonical names of tools that can be enabled. Kept here (rather than importing
# from the tools module) to avoid a heavy import at config time.
ALL_TOOLS: frozenset[str] = frozenset({"lookup_kb", "search_web"})

# Package root (wellness/) so the .env is found regardless of the working dir.
_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    """Environment-driven settings.

    Values are read from environment variables (prefixed ``WELLNESS_``) and an
    optional ``.env`` file. Provider API keys (e.g. ``OPENAI_API_KEY``,
    ``ANTHROPIC_API_KEY``, ``TAVILY_API_KEY``) are read by their standard names.
    """

    model_config = SettingsConfigDict(
        env_prefix="WELLNESS_",
        env_file=(".env", str(_ENV_FILE)),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider routing ---
    provider: Literal["openai", "ollama"] = Field(
        default="openai",
        description="Which chat backend to use: hosted OpenAI (ChatOpenAI) or "
        "local Ollama (ChatOllama).",
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="Bare model name for the selected provider, e.g. "
        "'gpt-4o-mini' (openai) or 'qwen2.5' (ollama).",
    )
    api_base: str | None = Field(
        default=None,
        description="Optional base URL. For ollama, defaults to "
        "http://localhost:11434; for openai, leave blank unless using an "
        "OpenAI-compatible endpoint.",
    )
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=1024)
    reasoning: bool = Field(
        default=False,
        description="Ollama thinking/reasoning mode. Off by default so answers "
        "are clean and tool calls fire reliably; set true to re-enable.",
    )

    # --- Tools ---
    enabled_tools: set[str] = Field(
        default_factory=lambda: set(ALL_TOOLS),
        description="Set of tool names the agent is allowed to use.",
    )
    kb_top_k: int = Field(default=4, ge=1, le=20)
    web_max_results: int = Field(default=5, ge=1, le=20)
    tavily_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("TAVILY_API_KEY"),
        description="API key for the Tavily web search tool (search_web).",
    )

    # --- KB / embeddings (OpenAI /v1/embeddings; needs OPENAI_API_KEY) ---
    embedding_model: str = Field(default="text-embedding-3-small")
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY"),
        description="OpenAI key for chat (provider=openai) and KB embeddings.",
    )

    # --- Agent runtime ---
    max_tool_iterations: int = Field(
        default=6, ge=1, description="Safety cap on agent<->tool loops per turn."
    )

    # --- Short-term memory (LangGraph checkpointer) ---
    memory_enabled: bool = Field(
        default=True,
        description="Persist conversation state per thread_id via SQLite. "
        "When false, an in-memory checkpointer is used (evals/tests).",
    )
    memory_db_path: str | None = Field(
        default=None,
        description="Path to the SQLite memory DB. Defaults to the state dir.",
    )

    # --- Observability (Langfuse) ---
    langfuse_enabled: bool = Field(
        default=False,
        description="Master switch for Langfuse tracing.",
    )
    langfuse_public_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGFUSE_PUBLIC_KEY"),
    )
    langfuse_secret_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("LANGFUSE_SECRET_KEY"),
    )
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        validation_alias=AliasChoices("LANGFUSE_HOST", "LANGFUSE_BASE_URL"),
    )

    # --- API server ---
    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8000)
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"],
        description="Allowed CORS origins for the frontend.",
    )

    def tracing_configured(self) -> bool:
        """True when Langfuse tracing is enabled and credentials are present."""
        return bool(
            self.langfuse_enabled
            and self.langfuse_public_key
            and self.langfuse_secret_key
        )

    def resolved_tools(self, enabled: set[str] | None = None) -> set[str]:
        """Validate and return the effective enabled-tool set.

        Args:
            enabled: Optional explicit override. When ``None`` the configured
                ``enabled_tools`` is used.

        Returns:
            The validated set of tool names.

        Raises:
            ValueError: If any requested tool name is unknown.
        """
        requested = set(self.enabled_tools if enabled is None else enabled)
        unknown = requested - set(ALL_TOOLS)
        if unknown:
            raise ValueError(
                f"Unknown tool(s): {sorted(unknown)}. Known: {sorted(ALL_TOOLS)}"
            )
        return requested


def get_settings(**overrides: object) -> Settings:
    """Build a Settings instance, applying any explicit overrides."""
    return Settings(**overrides)
