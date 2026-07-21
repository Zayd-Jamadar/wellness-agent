# serving/ — OSS inference deployment

Clean, portable inference path for the wellness agent:

```
app (ChatLiteLLM) --OpenAI /v1--> LiteLLM proxy :4000 --OpenAI /v1--> vLLM :8001
```

Everything is OpenAI-compatible, so moving from local to Modal/RunPod later is a
one-URL change in `litellm/config.yaml` — no app code changes.

```
serving/
  README.md            this file
  benchmark.py         one-time cost + latency test (OpenAI client -> proxy)
  vllm/run-local.sh    launch vLLM (prefix caching, tool calling, :8001)
  litellm/config.yaml  proxy model_list -> vLLM
```

> vLLM CUDA serving requires an NVIDIA GPU host and does **not** run on Apple
> Silicon. On a Mac you can still run the proxy + benchmark against a remote
> vLLM by setting `api_base` to the remote URL.

## Prerequisites

- GPU host: `pip install vllm` (pulls CUDA PyTorch). vLLM is intentionally not a
  backend dependency, so the Mac `uv` env is unaffected.
- Proxy: `pip install 'litellm[proxy]'` (or `uvx --from 'litellm[proxy]' litellm ...`).
- Benchmark: just the `openai` client (already in the backend venv).

## Run order (local)

1. **vLLM** (GPU host):

   ```bash
   bash serving/vllm/run-local.sh
   # verify: curl http://localhost:8001/v1/models
   ```

2. **LiteLLM proxy**:

   ```bash
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
   cd backend && uv run wellness ask -q "What makes a healthy diet?"
   ```

## Cost + latency benchmark (one-time)

Measures TTFT, end-to-end latency (p50/p95), throughput (tok/s), and prints a
table comparing self-hosted GPU cost against an equivalent hosted API price
(sequential vs concurrent, so continuous-batching throughput shows up):

```bash
cd backend
uv run python ../serving/benchmark.py \
  --base-url http://localhost:4000 --model wellness-local \
  --n 20 --concurrency 4 \
  --gpu-hourly 2.00 \                 # your GPU $/hr (RunPod/Modal/etc.)
  --api-in-price 0.15 --api-out-price 0.60   # hosted comparison ($/1M tok)
```

Cost model:
- OSS $/1M output tokens = `gpu_hourly / (tok_per_sec * 3600) * 1e6`
- Hosted API $/request   = `prompt/1e6 * in_price + completion/1e6 * out_price`

Add `--json out.json` to dump raw per-request results.

## Deploy elsewhere later

- **Modal / RunPod**: host vLLM there (both give an OpenAI `/v1` URL with
  PagedAttention + continuous batching), then set that URL as `api_base` in
  `litellm/config.yaml`. Nothing else changes.
- Point `HF_HOME` at a mounted persistent volume so weights are downloaded once
  and survive cold starts (the biggest knob for avoiding repeat multi-GB pulls).
- **Vercel** hosts only the app/frontend — not vLLM (no persistent GPU).
