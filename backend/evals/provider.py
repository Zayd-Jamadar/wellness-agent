"""Promptfoo <-> WellnessAgent bridge (Option B: answer + KB context).

Promptfoo calls `call_api(prompt, options, context)` and expects a dict with
an `output` key. We additionally return the retrieved `lookup_kb` passages
under `metadata.context` so `context-faithfulness` / `factuality` assertions
can score the answer against what the agent actually retrieved.

Per-test / per-provider config (from promptfooconfig `config:` block) is read
from `options["config"]`:
  - provider:      chat backend to use ("openai" or "ollama")
  - model:         bare model name (e.g. "gpt-5.4-mini" or "qwen2.5")
  - api_base:      optional base URL (e.g. local Ollama endpoint)
  - enabled_tools: list of tool names to pin for the run

Run standalone to smoke-test:  python provider.py
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

# Fallback so `import wellness` resolves even if a non-venv (3.11+) Python is
# used. Third-party deps still require the backend venv (run via `uv run`).
_SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from wellness.config import get_settings  # noqa: E402
from wellness.evals.interface import run_agent_with_context  # noqa: E402


def call_api(
    prompt: str,
    options: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Any]:
    cfg = (options or {}).get("config", {}) or {}

    overrides: dict[str, Any] = {}
    if cfg.get("provider"):
        overrides["provider"] = cfg["provider"]
    if cfg.get("model"):
        overrides["model"] = cfg["model"]
    if cfg.get("api_base"):
        overrides["api_base"] = cfg["api_base"]
    # Deterministic grading target: memory off so each case is independent.
    overrides["memory_enabled"] = False

    settings = get_settings(**overrides)
    tools = cfg.get("enabled_tools")
    enabled_tools = set(tools) if tools else None

    try:
        run = run_agent_with_context(
            prompt, settings=settings, enabled_tools=enabled_tools
        )
    except Exception as exc:  # noqa: BLE001 - surface errors to the report
        return {"error": str(exc)}

    return {
        "output": run.output,
        "metadata": {
            "context": run.context,
            "tools_used": run.tools_used,
            "latency_ms": round(run.latency_ms, 1),
        },
    }


if __name__ == "__main__":
    result = call_api(
        "What foods should form the base of a healthy diet?",
        {"config": {"provider": "openai", "model": "gpt-5.4-mini"}},
        {},
    )
    print("output:\n", result.get("output", result.get("error")))
    print("\ncontext:\n", result.get("metadata", {}).get("context", "")[:500])
    print("\ntools_used:", result.get("metadata", {}).get("tools_used"))
