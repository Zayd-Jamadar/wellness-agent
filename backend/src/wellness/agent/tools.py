"""Tool registry for the wellness agent.

Tools are registered by a stable name so the *set* the agent is given can be
selected at runtime (via CLI flags or eval cases) rather than hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from langchain_core.tools import BaseTool, tool

from wellness.config import Settings, get_settings
from wellness.kb.search import get_kb_service
from wellness.logging import get_logger

log = get_logger(service="tools")


@dataclass(frozen=True)
class ToolSpec:
    """Metadata + factory for a registered tool."""

    name: str
    description: str
    factory: Callable[[Settings], BaseTool]


class KBTool:
    """Builds the ``lookup_kb`` tool bound to a settings instance."""

    name = "lookup_kb"
    description = (
        "Search the curated wellness knowledge base for relevant passages. "
        "Use for questions about diet, exercise, sleep, meditation, habits, "
        "supplements, and nature. Returns passages with their source topic."
    )

    def build(self, settings: Settings) -> BaseTool:
        top_k = settings.kb_top_k

        @tool("lookup_kb")
        def lookup_kb(query: str) -> str:
            """Search the wellness knowledge base and return relevant passages."""
            service = get_kb_service(settings)
            results = service.search(query, top_k=top_k)
            if not results:
                return "No relevant knowledge base entries found."
            blocks = [
                f"[{r.chunk.title}] {r.chunk.text}" for r in results
            ]
            return "\n\n".join(blocks)

        return lookup_kb


class WebSearchTool:
    """Builds the ``search_web`` tool (Tavily) bound to a settings instance."""

    name = "search_web"
    description = (
        "Search the web for current or external information not covered by the "
        "knowledge base. Returns titles, URLs, and snippets."
    )

    def build(self, settings: Settings) -> BaseTool:
        from langchain_tavily import TavilySearch

        # Register the Tavily tool directly under our canonical name so the
        # agent (and the UI) only ever sees a single ``search_web`` tool,
        # rather than a wrapper around a nested ``tavily_search`` tool.
        # Pass the key explicitly: it is loaded from the .env into Settings by
        # pydantic-settings and is not necessarily present in os.environ (which
        # is where TavilySearch would otherwise look for TAVILY_API_KEY). Only
        # forward it when set so TavilySearch can still fall back to os.environ.
        kwargs: dict[str, object] = {
            "name": self.name,
            "description": self.description,
            "max_results": settings.web_max_results,
        }
        if settings.tavily_api_key:
            kwargs["tavily_api_key"] = settings.tavily_api_key
        return TavilySearch(**kwargs)


def _make_spec(builder: KBTool | WebSearchTool) -> ToolSpec:
    return ToolSpec(
        name=builder.name,
        description=builder.description,
        factory=builder.build,
    )


AVAILABLE_TOOLS: dict[str, ToolSpec] = {
    KBTool.name: _make_spec(KBTool()),
    WebSearchTool.name: _make_spec(WebSearchTool()),
}


def build_tools(
    enabled: set[str] | None = None, settings: Settings | None = None
) -> list[BaseTool]:
    """Instantiate the enabled tools.

    Args:
        enabled: Tool names to build; defaults to ``settings.enabled_tools``.
        settings: Settings instance; defaults to :func:`get_settings`.

    Returns:
        The list of instantiated LangChain tools.

    Raises:
        ValueError: If any requested tool name is unknown.
    """
    settings = settings or get_settings()
    names = settings.resolved_tools(enabled)
    unknown = names - set(AVAILABLE_TOOLS)
    if unknown:
        raise ValueError(f"Unknown tool(s): {sorted(unknown)}")
    tools = [AVAILABLE_TOOLS[name].factory(settings) for name in sorted(names)]
    log.info("tools_built", tools=[t.name for t in tools])
    return tools


def describe_tools(enabled: set[str] | None = None, settings: Settings | None = None) -> str:
    """Return a human-readable description block for the enabled tools."""
    settings = settings or get_settings()
    names = sorted(settings.resolved_tools(enabled))
    return "\n".join(f"- {n}: {AVAILABLE_TOOLS[n].description}" for n in names)
