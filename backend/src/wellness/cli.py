"""Command-line interface for the wellness assistant."""

from __future__ import annotations

import click

from wellness import __version__
from wellness.config import ALL_TOOLS, get_settings
from wellness.logging import configure, get_logger

log = get_logger(service="cli")


def _resolve_tools(
    kb: bool | None,
    web: bool | None,
    only: str | None,
) -> set[str] | None:
    """Turn CLI flags into an enabled-tool set (or None for config default).

    Args:
        kb: Tri-state for ``lookup_kb`` (--kb/--no-kb; None if unset).
        web: Tri-state for ``search_web`` (--web/--no-web; None if unset).
        only: Comma-separated explicit subset; overrides the on/off flags.
    """
    if only is not None:
        names = {n.strip() for n in only.split(",") if n.strip()}
        unknown = names - set(ALL_TOOLS)
        if unknown:
            raise click.BadParameter(
                f"Unknown tool(s): {sorted(unknown)}. Known: {sorted(ALL_TOOLS)}"
            )
        return names

    if kb is None and web is None:
        return None  # use config default

    enabled = set(get_settings().enabled_tools)
    if kb is not None:
        enabled = (enabled | {"lookup_kb"}) if kb else (enabled - {"lookup_kb"})
    if web is not None:
        enabled = (enabled | {"search_web"}) if web else (enabled - {"search_web"})
    return enabled


def _tool_options(func):
    """Shared per-tool enable/disable options for commands."""
    func = click.option(
        "--only",
        default=None,
        metavar="NAMES",
        help="Comma-separated explicit tool subset (e.g. 'lookup_kb').",
    )(func)
    func = click.option(
        "--web/--no-web", "web", default=None, help="Enable/disable search_web."
    )(func)
    func = click.option(
        "--kb/--no-kb", "kb", default=None, help="Enable/disable lookup_kb."
    )(func)
    return func


@click.group(invoke_without_command=True)
@click.option("-v", "--version", is_flag=True, help="Show version and exit.")
@click.option("--log-level", default="INFO", help="Log level (DEBUG, INFO, ...).")
@click.option("--json-logs", is_flag=True, help="Emit JSON log lines.")
@click.pass_context
def main(ctx: click.Context, version: bool, log_level: str, json_logs: bool) -> None:
    """Wellness Assistant — an evals-ready LangGraph agent."""
    configure(level=log_level, json_logs=json_logs)
    if version:
        click.echo(__version__)
        ctx.exit()
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command("serve")
@click.option("--host", default=None, help="Bind host (default from config).")
@click.option("--port", default=None, type=int, help="Bind port (default from config).")
@click.option("--reload", is_flag=True, help="Enable auto-reload (dev).")
def serve(host: str | None, port: int | None, reload: bool) -> None:
    """Run the FastAPI server exposing the agent to the frontend."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "wellness.api.app:create_app",
        factory=True,
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=reload,
    )


@main.command("ask")
@click.argument("question", nargs=-1, required=True)
@click.option(
    "-t",
    "--thread",
    default=None,
    metavar="ID",
    help="Conversation thread id to continue (enables persistent memory).",
)
@_tool_options
def ask(
    question: tuple[str, ...],
    thread: str | None,
    kb: bool | None,
    web: bool | None,
    only: str | None,
) -> None:
    """Ask a single question headlessly (no TUI) and print the answer.

    Pass ``--thread ID`` to continue a prior conversation; the agent recalls
    earlier turns stored in the SQLite memory DB.
    """
    from contextlib import nullcontext

    from wellness.agent.graph import WellnessAgent
    from wellness.config import get_settings
    from wellness.memory import open_sqlite_saver
    from wellness.tracing import flush_langfuse

    enabled = _resolve_tools(kb, web, only)
    settings = get_settings()
    saver_cm = (
        open_sqlite_saver(settings) if settings.memory_enabled else nullcontext(None)
    )
    try:
        with saver_cm as checkpointer:
            agent = WellnessAgent(enabled_tools=enabled, checkpointer=checkpointer)
            answer = agent.invoke(" ".join(question), thread_id=thread)
            click.echo(answer)
    finally:
        flush_langfuse()


@main.command("index")
@click.option("--force", is_flag=True, help="Rebuild even if the index is current.")
def index(force: bool) -> None:
    """Build (or rebuild) the knowledge-base search index."""
    from wellness.kb.index import build_index

    count = build_index(force=force)
    click.echo(f"Indexed {count} chunks.")


@main.command("eval")
@_tool_options
def eval_cmd(kb: bool | None, web: bool | None, only: str | None) -> None:
    """Run a tiny built-in smoke eval suite (stub for a real framework)."""
    from wellness.evals.interface import EvalCase, run_suite
    from wellness.tracing import flush_langfuse

    enabled = _resolve_tools(kb, web, only)
    cases = [
        EvalCase(
            input="How many minutes of exercise are recommended per week?",
            enabled_tools=enabled,
            expected_substrings=["150"],
        ),
        EvalCase(
            input="What are the benefits of meditation?",
            enabled_tools=enabled,
        ),
    ]
    try:
        results = run_suite(cases)
        for r in results:
            status = "?" if r.passed is None else ("PASS" if r.passed else "FAIL")
            click.echo(f"[{status}] ({r.latency_ms:.0f} ms) {r.input}")
            if r.error:
                click.echo(f"    error: {r.error}")
            else:
                click.echo(f"    {r.output[:200]}")
    finally:
        flush_langfuse()


if __name__ == "__main__":
    main()
