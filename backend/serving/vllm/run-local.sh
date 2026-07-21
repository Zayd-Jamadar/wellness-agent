#!/usr/bin/env bash
# Launch vLLM as an OpenAI-compatible server for the wellness OSS deployment.
#
# vLLM does PagedAttention + continuous batching + KV caching by default; we
# additionally enable prefix caching and tool calling (the agent uses tools).
#
# Requires an NVIDIA GPU host with the CUDA vLLM build installed:
#   pip install vllm
# (Does NOT run on Apple Silicon. On the Mac, run only the proxy + benchmark
# against a remote vLLM by setting its api_base.)
set -euo pipefail

# Persistent HF cache so the model is downloaded ONCE and reused on every run.
# Override HF_HOME to point at a persistent disk (esp. on ephemeral cloud GPUs
# like RunPod/Modal, where it should be a mounted volume).
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Optional: faster downloads on first pull.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
# To go fully offline after the first pull, export HF_HUB_OFFLINE=1 before running.

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-8001}"

vllm serve "$MODEL" \
  --port "$PORT" \
  --served-model-name qwen \
  --download-dir "$HF_HOME/hub" \
  --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser hermes
