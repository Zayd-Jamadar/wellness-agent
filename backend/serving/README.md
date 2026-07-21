# serving/ — OSS inference deployment

Clean, portable inference path for the wellness agent, powered by Ollama:

```
app (ChatLiteLLM) --OpenAI /v1--> LiteLLM proxy :4000 --ollama api--> Ollama :11434
```

Everything is OpenAI-compatible at the proxy seam, so moving from a local Ollama
to a remote Ollama host later is a one-env-var change (`OLLAMA_API_BASE`) — no app
code changes.

```
serving/
  README.md            this file
  benchmark.py         one-time cost + latency test (OpenAI client -> proxy)
  litellm/config.yaml  proxy model_list -> Ollama (env-driven)
```

## Prerequisites

- [Ollama](https://ollama.com) installed and running (`ollama serve`). Runs on
  macOS, Linux, and Windows — CPU or GPU.
- Proxy: `pip install 'litellm[proxy]'` (or `uvx --from 'litellm[proxy]' litellm ...`).
- Benchmark: just the `openai` client (already in the backend venv).

## Run order (local)

1. **Ollama** (pull a model once, then serve):

   ```bash
   ollama serve                     # usually already running as a service
   ollama pull qwen2.5              # verify: curl http://localhost:11434/api/tags
   ```

2. **LiteLLM proxy** (reads OLLAMA_API_BASE / OLLAMA_MODEL from the env):

   ```bash
   OLLAMA_API_BASE=http://localhost:11434 OLLAMA_MODEL=qwen2.5 \
     litellm --config serving/litellm/config.yaml --port 4000
   # verify: curl http://localhost:4000/v1/models
   ```

3. **App via proxy** — set in `backend/.env`:

   ```
   WELLNESS_MODEL=openai/wellness-local
   WELLNESS_API_BASE=http://localhost:4000
   OPENAI_API_KEY=sk-anything
   ```

   ```bash
   cd backend && uv run wellness ask "What makes a healthy diet?"
   ```

> Prefer no proxy? Point the app straight at Ollama:
> `WELLNESS_MODEL=ollama/qwen2.5` and `WELLNESS_API_BASE=http://localhost:11434`.

## Cost + latency benchmark (one-time)

Measures TTFT, end-to-end latency (p50/p95), throughput (tok/s), and prints a
table comparing self-hosted cost against an equivalent hosted API price
(sequential vs concurrent). Runs against the LiteLLM proxy or Ollama directly:

```bash
cd backend
uv run python ../serving/benchmark.py \
  --base-url http://localhost:4000 --model wellness-local \
  --n 20 --concurrency 4 \
  --gpu-hourly 2.00 \                 # your host $/hr (0 for a machine you own)
  --api-in-price 0.15 --api-out-price 0.60   # hosted comparison ($/1M tok)
```

Cost model:
- OSS $/1M output tokens = `gpu_hourly / (tok_per_sec * 3600) * 1e6`
- Hosted API $/request   = `prompt/1e6 * in_price + completion/1e6 * out_price`

Add `--json out.json` to dump raw per-request results.

## Deploy elsewhere later

- **Remote Ollama / any OpenAI-compatible host**: run the model on another box
  and set that URL as `OLLAMA_API_BASE` (or point `api_base` in
  `litellm/config.yaml` at any OpenAI-compatible endpoint). Nothing else changes.
- **Vercel** hosts only the app/frontend — run the model host separately.
