"""Pluggable evaluation interface.

Defines the data contract (`EvalCase`, `EvalResult`) and a headless `run_agent`
adapter so an external eval framework (LangSmith, promptfoo, deepeval, ...) can
drive the agent without touching agent code. Scoring is intentionally left to the
plugged-in framework; a trivial substring checker is provided for smoke tests.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Protocol

from pydantic import BaseModel, Field

from wellness.agent.graph import WellnessAgent, build_agent
from wellness.config import Settings, get_settings


class EvalCase(BaseModel):
    """A single evaluation case."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    input: str = Field(description="User message to send to the agent.")
    thread_id: str | None = Field(
        default=None, description="Optional thread id for multi-turn cases."
    )
    enabled_tools: set[str] | None = Field(
        default=None,
        description="Pin the tool set for this case (None = config default).",
    )
    expected_substrings: list[str] = Field(
        default_factory=list,
        description="Optional substrings expected in the output (smoke scoring).",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """The outcome of running one :class:`EvalCase`."""

    case_id: str
    input: str
    output: str
    latency_ms: float
    passed: bool | None = Field(
        default=None, description="None when no scorer was applied."
    )
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Scorer(Protocol):
    """A pluggable scoring function."""

    def __call__(self, case: EvalCase, output: str) -> bool: ...


def substring_scorer(case: EvalCase, output: str) -> bool:
    """Default smoke scorer: all expected substrings present (case-insensitive)."""
    lowered = output.lower()
    return all(s.lower() in lowered for s in case.expected_substrings)


def run_agent(
    case: EvalCase,
    settings: Settings | None = None,
    scorer: Scorer | None = None,
    agent: WellnessAgent | None = None,
) -> EvalResult:
    """Run one eval case headlessly and return a structured result.

    Args:
        case: The case to run.
        settings: Optional settings override.
        scorer: Optional scoring function; defaults to substring scoring only
            when the case declares ``expected_substrings``.
        agent: Optional pre-built agent (reused across cases with the same tools).

    Returns:
        An :class:`EvalResult`.
    """
    settings = settings or get_settings()
    if agent is None:
        agent = build_agent(settings=settings, enabled_tools=case.enabled_tools)

    start = time.perf_counter()
    try:
        output = agent.invoke(case.input, thread_id=case.thread_id)
        error = None
    except Exception as exc:  # noqa: BLE001 - capture for the eval report
        output = ""
        error = str(exc)
    latency_ms = (time.perf_counter() - start) * 1000.0

    passed: bool | None = None
    if error is None:
        if scorer is not None:
            passed = scorer(case, output)
        elif case.expected_substrings:
            passed = substring_scorer(case, output)

    return EvalResult(
        case_id=case.id,
        input=case.input,
        output=output,
        latency_ms=latency_ms,
        passed=passed,
        error=error,
        metadata=case.metadata,
    )


def run_suite(
    cases: list[EvalCase],
    settings: Settings | None = None,
    scorer: Scorer | None = None,
) -> list[EvalResult]:
    """Run a list of cases and return their results."""
    settings = settings or get_settings()
    return [run_agent(c, settings=settings, scorer=scorer) for c in cases]


class AgentRun(BaseModel):
    """Agent output plus the retrieved KB context (for grounded scoring)."""

    output: str
    context: str = Field(
        default="", description="Concatenated `lookup_kb` tool outputs for this turn."
    )
    tools_used: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0


def run_agent_with_context(
    message: str,
    settings: Settings | None = None,
    enabled_tools: set[str] | None = None,
    thread_id: str | None = None,
    agent: WellnessAgent | None = None,
) -> AgentRun:
    """Run one turn and return the answer plus retrieved KB context.

    Streams :class:`AgentEvent`s so the ``lookup_kb`` tool outputs can be
    captured as ``context`` for hallucination/faithfulness scoring. This is the
    seam a promptfoo provider uses (Option B).
    """
    settings = settings or get_settings()
    if agent is None:
        agent = build_agent(settings=settings, enabled_tools=enabled_tools)

    async def _collect() -> AgentRun:
        answer_parts: list[str] = []
        kb_context: list[str] = []
        tools_used: list[str] = []
        async for ev in agent.astream_events(message, thread_id=thread_id):
            if ev.kind == "text-delta":
                answer_parts.append(ev.text)
            elif ev.kind == "tool-end":
                tools_used.append(ev.tool_name)
                if ev.tool_name == "lookup_kb" and ev.tool_output:
                    kb_context.append(ev.tool_output)
        return AgentRun(
            output="".join(answer_parts),
            context="\n\n".join(kb_context),
            tools_used=tools_used,
        )

    start = time.perf_counter()
    run = asyncio.run(_collect())
    run.latency_ms = (time.perf_counter() - start) * 1000.0
    return run
