# Wellness Assistant

An evals-ready, multi-turn **Wellness Assistant** built on **LangGraph**. It answers
health and lifestyle questions grounded in a curated knowledge base, with two tools:

- `lookup_kb` — hybrid search (BM25 keyword + local sentence-transformers vectors,
  combined via Reciprocal Rank Fusion) over the `kb/` markdown corpus.
- `search_web` — Tavily web search for anything outside the KB.

The LLM vendor is swappable (open-source vs frontier) via a simple provider flag:
set `WELLNESS_PROVIDER` to `openai` or `ollama`, keep the same architecture.

## Install

Uses [uv](https://docs.astral.sh/uv/).

```bash
cd wellness
uv sync
```

## Configure

Settings are read from the environment (prefix `WELLNESS_`) and an optional `.env`.
Provider keys use their standard names.

```bash
# Frontier example (hosted OpenAI)
export WELLNESS_PROVIDER="openai"
export WELLNESS_MODEL="gpt-4o-mini"
export OPENAI_API_KEY="sk-..."

# OSS example (local Ollama: `ollama serve` + `ollama pull qwen2.5`)
export WELLNESS_PROVIDER="ollama"
export WELLNESS_MODEL="qwen2.5"
export WELLNESS_API_BASE="http://localhost:11434"   # optional; this is the default

# Web search
export TAVILY_API_KEY="tvly-..."
```

Note: `OPENAI_API_KEY` is always required for KB embeddings, even when
`WELLNESS_PROVIDER=ollama`. Ollama "thinking" mode is off by default; set
`WELLNESS_REASONING=true` to enable it.

Key settings: `WELLNESS_PROVIDER`, `WELLNESS_MODEL`, `WELLNESS_API_BASE`,
`WELLNESS_REASONING`, `WELLNESS_TEMPERATURE`, `WELLNESS_EMBEDDING_MODEL`,
`WELLNESS_KB_TOP_K`, `WELLNESS_WEB_MAX_RESULTS`.

## Usage

```bash
# Build (or rebuild) the KB search index
uv run wellness index

# Serve the API (for the frontend)
uv run wellness serve            # http://127.0.0.1:8000

# One-shot headless ask
uv run wellness ask "How much exercise per week is recommended?"

# Multi-turn: reuse a thread id to continue a conversation (persistent memory)
uv run wellness ask -t sam "My name is Sam."
uv run wellness ask -t sam "What is my name?"   # -> recalls "Sam"

# Disable a tool for a session
uv run wellness ask --no-web "..."
uv run wellness ask --only lookup_kb "..."

# Evals stub
uv run wellness eval
```

## Docker Compose (frontend + backend)

Run the whole stack (Next.js UI + FastAPI agent) with one command. Uses OpenAI
via an API key; no self-hosted model stack required.

```bash
cp .env.example .env        # then set OPENAI_API_KEY (and optional TAVILY_API_KEY)
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend:  http://localhost:8000 (health: http://localhost:8000/health)

The SQLite DB (KB index + conversation memory) persists in the `wellness-data`
volume. `OPENAI_API_KEY` is required — it powers both chat completions and KB
embeddings, which build on first boot.

## Architecture

```
src/wellness/
  cli.py        Click CLI (serve / ask / index / eval)
  config.py     pydantic-settings (provider flag, model, tools, memory, KB knobs)
  llm.py        build_chat_model() -> ChatOpenAI/ChatOllama by WELLNESS_PROVIDER
  memory.py     SQLite checkpointer factories (sync + async)
  prompts.py    YAML prompt loader (no prompts in Python)
  logging.py    structlog context loggers
  agent/
    graph.py    LangGraph runtime (agent node + ToolNode)
    state.py    AgentState
    tools.py    tool registry + build_tools(enabled)
  kb/
    index.py    build/load vector index over kb/*.md
    search.py   sqlite-vec KNN search (KBService)
  api/          FastAPI app streaming the AI SDK Data Stream Protocol
  evals/        pluggable eval interface (EvalCase / EvalResult / run_agent)
prompts/
  system.yml
```

## Short-term memory

Conversation state is persisted per `thread_id` via a LangGraph SQLite
checkpointer (`langgraph-checkpoint-sqlite`), so multi-turn memory survives
across HTTP requests and process restarts:

- The API uses an async `AsyncSqliteSaver` opened once in the FastAPI lifespan;
  the CLI uses a sync `SqliteSaver` per invocation.
- The memory key is the conversation id — the frontend's `useChat` session `id`
  over the API, or the `--thread/-t` flag from the CLI.
- DB location: `backend/data/wellness.db` — a single SQLite file shared by the
  KB vector index and conversation memory (override the memory path with
  `WELLNESS_MEMORY_DB_PATH`). Disable with `WELLNESS_MEMORY_ENABLED=false` to
  fall back to an in-memory checkpointer (useful for evals/tests).
