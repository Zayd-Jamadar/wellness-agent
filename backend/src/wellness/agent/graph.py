"""LangGraph runtime for the wellness agent.

`WellnessAgent` wires an LLM (via the LiteLLM gateway) to the enabled tools and
compiles a LangGraph state machine: an ``agent`` node that calls the model and a
``tools`` node that executes tool calls, looping until the model stops. An
in-memory checkpointer provides short-term (in-session) conversation memory.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode

from wellness.agent.state import AgentState
from wellness.agent.tools import build_tools, describe_tools
from wellness.config import Settings, get_settings
from wellness.llm import build_chat_model
from wellness.logging import get_logger
from wellness.prompts import render_prompt
from wellness.tracing import get_langfuse_handler

log = get_logger(service="agent")


@dataclass
class AgentEvent:
    """A normalized, framework-neutral streaming event from an agent turn.

    The API layer translates these into the AI SDK Data Stream Protocol; the
    agent itself stays free of any transport/protocol concerns.
    """

    kind: Literal[
        "text-delta",
        "reasoning-delta",
        "tool-start",
        "tool-end",
        "error",
    ]
    text: str = ""
    tool_call_id: str = ""
    tool_name: str = ""
    tool_input: dict = field(default_factory=dict)
    tool_output: str = ""


class WellnessAgent:
    """Compiles and runs the wellness LangGraph agent."""

    def __init__(
        self,
        settings: Settings | None = None,
        enabled_tools: set[str] | None = None,
        llm: BaseChatModel | None = None,
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.enabled_tools = self.settings.resolved_tools(enabled_tools)
        self.tools = build_tools(self.enabled_tools, self.settings)
        base_llm = llm or build_chat_model(self.settings)
        self.llm = base_llm.bind_tools(self.tools) if self.tools else base_llm
        self.system_prompt = self._render_system_prompt()
        self.checkpointer = checkpointer
        self.graph = self._build_graph()
        handler = get_langfuse_handler(self.settings)
        self._callbacks = [handler] if handler else []

    # ------------------------------------------------------------- prompt --
    def _render_system_prompt(self) -> str:
        return render_prompt(
            "system",
            "system",
            tools=describe_tools(self.enabled_tools, self.settings),
        )

    # -------------------------------------------------------------- graph --
    def _agent_node(self, state: AgentState) -> dict[str, list[BaseMessage]]:
        messages = [SystemMessage(content=self.system_prompt), *state["messages"]]
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    @staticmethod
    def _should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    def _build_graph(self) -> CompiledStateGraph:
        builder = StateGraph(AgentState)
        builder.add_node("agent", self._agent_node)
        builder.set_entry_point("agent")

        if self.tools:
            builder.add_node("tools", ToolNode(self.tools))
            builder.add_conditional_edges(
                "agent", self._should_continue, {"tools": "tools", END: END}
            )
            builder.add_edge("tools", "agent")
        else:
            builder.add_edge("agent", END)

        return builder.compile(checkpointer=self.checkpointer or MemorySaver())

    # --------------------------------------------------------------- run ---
    def _run_config(self, thread_id: str) -> dict[str, object]:
        """Build the LangGraph run config (thread, recursion, tracing)."""
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.settings.max_tool_iterations * 2 + 2,
            "callbacks": self._callbacks,
            "metadata": {
                "langfuse_session_id": thread_id,
                "langfuse_tags": ["wellness", self.settings.model],
                "enabled_tools": sorted(self.enabled_tools),
            },
        }

    def invoke(self, message: str, thread_id: str | None = None) -> str:
        """Run one user turn synchronously and return the assistant's reply.

        Args:
            message: The user message.
            thread_id: Conversation thread id (for short-term memory). A random
                id is generated when omitted.

        Returns:
            The assistant's final text response.
        """
        from langchain_core.messages import HumanMessage

        thread_id = thread_id or uuid.uuid4().hex
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=message)]},
            config=self._run_config(thread_id),
        )
        final = result["messages"][-1]
        return _text_of(final)

    async def ainvoke(self, message: str, thread_id: str | None = None) -> str:
        """Async variant of :meth:`invoke`."""
        from langchain_core.messages import HumanMessage

        thread_id = thread_id or uuid.uuid4().hex
        result = await self.graph.ainvoke(
            {"messages": [HumanMessage(content=message)]},
            config=self._run_config(thread_id),
        )
        return _text_of(result["messages"][-1])

    async def astream_events(
        self, message: str, thread_id: str | None = None
    ) -> AsyncIterator[AgentEvent]:
        """Stream a turn as normalized :class:`AgentEvent`s.

        Consumes LangGraph's event stream and maps the relevant events to
        text/reasoning deltas and tool start/end. This keeps AI-SDK protocol
        formatting out of the agent.

        Args:
            message: The user message.
            thread_id: Conversation thread id.

        Yields:
            :class:`AgentEvent` instances.
        """
        from langchain_core.messages import HumanMessage

        thread_id = thread_id or uuid.uuid4().hex
        async for event in self.graph.astream_events(
            {"messages": [HumanMessage(content=message)]},
            config=self._run_config(thread_id),
            version="v2",
        ):
            etype = event.get("event")
            data = event.get("data", {})

            if etype == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk is None:
                    continue
                # Reasoning content, when the provider supplies it.
                reasoning = _reasoning_of(chunk)
                if reasoning:
                    yield AgentEvent(kind="reasoning-delta", text=reasoning)
                text = _text_of(chunk)
                if text:
                    yield AgentEvent(kind="text-delta", text=text)

            elif etype == "on_tool_start":
                yield AgentEvent(
                    kind="tool-start",
                    tool_call_id=event.get("run_id", uuid.uuid4().hex),
                    tool_name=event.get("name", "tool"),
                    tool_input=data.get("input", {}) or {},
                )

            elif etype == "on_tool_end":
                output = data.get("output")
                yield AgentEvent(
                    kind="tool-end",
                    tool_call_id=event.get("run_id", ""),
                    tool_name=event.get("name", "tool"),
                    tool_output=_tool_output_text(output),
                )


def _reasoning_of(message: BaseMessage) -> str:
    """Extract reasoning text from a chunk, if the provider emits any."""
    extra = getattr(message, "additional_kwargs", {}) or {}
    for key in ("reasoning_content", "reasoning"):
        val = extra.get(key)
        if isinstance(val, str) and val:
            return val
    content = message.content
    if isinstance(content, list):
        parts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "reasoning"
        ]
        return "".join(parts)
    return ""


def _tool_output_text(output: object) -> str:
    """Coerce a tool result (ToolMessage or raw) to a string."""
    content = getattr(output, "content", output)
    if isinstance(content, str):
        return content
    return str(content)


def _text_of(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    # Some providers return content as a list of blocks.
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "reasoning":
            continue
        elif isinstance(block, dict) and "text" in block:
            parts.append(block["text"])
    return "".join(parts)


def build_agent(
    settings: Settings | None = None, enabled_tools: set[str] | None = None
) -> WellnessAgent:
    """Convenience factory for a :class:`WellnessAgent`."""
    return WellnessAgent(settings=settings, enabled_tools=enabled_tools)
