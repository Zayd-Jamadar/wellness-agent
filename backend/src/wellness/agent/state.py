"""Agent graph state."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State threaded through the LangGraph runtime.

    ``messages`` accumulates the conversation (human, AI, tool) using LangGraph's
    reducer so each node can append without clobbering history.
    """

    messages: Annotated[list[AnyMessage], add_messages]
