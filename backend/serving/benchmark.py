"""One-time cost + latency benchmark for the OSS deployment.

Sends a fixed set of wellness prompts to an OpenAI-compatible endpoint (the
LiteLLM proxy or Ollama directly), measures latency + tokens from the streamed
response, and prints a table comparing self-hosted cost against an
equivalent hosted API price.

Cost model:
  - OSS $/1M output tokens = gpu_hourly / (tok_per_sec * 3600) * 1e6
  - Hosted API $/request   = prompt/1e6 * in + completion/1e6 * out

Run (single + concurrent):
  uv run python serving/benchmark.py --gpu-hourly 2.00

Only depends on the `openai` client. This is a manual test, not a service.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from openai import OpenAI

DEFAULT_PROMPTS = [
    "What foods should form the base of a healthy diet?",
    "How can I build a sustainable habit of daily walking?",
    "What safety considerations matter before starting supplements?",
    "What are some calming meditation techniques for winding down at night?",
    "Why can't supplements simply replace a healthy diet?",
    "What role does hydration play in a healthy diet?",
]


@dataclass
class RequestResult:
    ttft_s: float
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    tok_per_sec: float
    ok: bool
    error: str = ""


def _run_one(
    client: OpenAI, model: str, prompt: str, max_tokens: int
) -> RequestResult:
    start = time.perf_counter()
    ttft: float | None = None
    text_chunks = 0
    prompt_tokens = 0
    completion_tokens = 0
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if ttft is None:
                    ttft = time.perf_counter() - start
                text_chunks += 1
            if getattr(chunk, "usage", None):
                prompt_tokens = chunk.usage.prompt_tokens or 0
                completion_tokens = chunk.usage.completion_tokens or 0
        latency = time.perf_counter() - start
    except Exception as exc:  # noqa: BLE001 - report per-request failures
        return RequestResult(0.0, 0.0, 0, 0, 0.0, ok=False, error=str(exc))

    # Fall back to chunk count if the server omitted usage.
    if completion_tokens == 0:
        completion_tokens = text_chunks
    ttft = ttft if ttft is not None else latency
    tok_per_sec = completion_tokens / latency if latency > 0 else 0.0
    return RequestResult(
        ttft_s=ttft,
        latency_s=latency,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        tok_per_sec=tok_per_sec,
        ok=True,
    )


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((p / 100) * (len(ordered) - 1)))))
    return ordered[k]


def _summarize(
    label: str,
    results: list[RequestResult],
    gpu_hourly: float,
    api_in_price: float,
    api_out_price: float,
) -> dict:
    ok = [r for r in results if r.ok]
    if not ok:
        return {"label": label, "requests": 0, "errors": len(results)}

    ttfts = [r.ttft_s for r in ok]
    latencies = [r.latency_s for r in ok]
    tps = [r.tok_per_sec for r in ok]
    mean_tps = statistics.mean(tps)

    # OSS $/1M output tokens from measured throughput.
    oss_per_1m = (
        gpu_hourly / (mean_tps * 3600) * 1e6
        if mean_tps > 0
        else float("inf")
    )
    # Hosted API $ for the same tokens (summed, then per-request average).
    api_total = sum(
        r.prompt_tokens / 1e6 * api_in_price
        + r.completion_tokens / 1e6 * api_out_price
        for r in ok
    )
    api_per_req = api_total / len(ok)
    # API $/1M output tokens (output-side only, for a like-for-like column).
    api_out_per_1m = api_out_price * 1e6 / 1e6  # == api_out_price

    return {
        "label": label,
        "requests": len(ok),
        "errors": len(results) - len(ok),
        "ttft_p50_s": round(_pct(ttfts, 50), 3),
        "ttft_p95_s": round(_pct(ttfts, 95), 3),
        "latency_p50_s": round(_pct(latencies, 50), 3),
        "latency_p95_s": round(_pct(latencies, 95), 3),
        "mean_tok_per_sec": round(mean_tps, 1),
        "oss_usd_per_1m_out": round(oss_per_1m, 3),
        "api_usd_per_1m_out": round(api_out_per_1m, 3),
        "api_usd_per_req": round(api_per_req, 6),
        "oss_vs_api_ratio": (
            round(oss_per_1m / api_out_per_1m, 2)
            if api_out_per_1m > 0
            else None
        ),
    }


def _print_table(rows: list[dict]) -> None:
    cols = [
        ("label", "mode"),
        ("requests", "reqs"),
        ("ttft_p50_s", "ttft_p50"),
        ("ttft_p95_s", "ttft_p95"),
        ("latency_p50_s", "lat_p50"),
        ("latency_p95_s", "lat_p95"),
        ("mean_tok_per_sec", "tok/s"),
        ("oss_usd_per_1m_out", "oss $/1M"),
        ("api_usd_per_1m_out", "api $/1M"),
        ("oss_vs_api_ratio", "oss/api"),
    ]
    widths = {key: len(hdr) for key, hdr in cols}
    for row in rows:
        for key, _ in cols:
            widths[key] = max(widths[key], len(str(row.get(key, "-"))))
    header = "  ".join(hdr.ljust(widths[key]) for key, hdr in cols)
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            "  ".join(
                str(row.get(key, "-")).ljust(widths[key]) for key, _ in cols
            )
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://localhost:4000")
    ap.add_argument("--model", default="wellness-local")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--n", type=int, default=12, help="requests per mode")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument(
        "--gpu-hourly", type=float, default=2.00, help="GPU $/hour"
    )
    ap.add_argument(
        "--api-in-price", type=float, default=0.15, help="hosted $/1M in"
    )
    ap.add_argument(
        "--api-out-price", type=float, default=0.60, help="hosted $/1M out"
    )
    ap.add_argument(
        "--json", dest="json_out", default=None, help="dump results to file"
    )
    args = ap.parse_args()

    base_url = args.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    client = OpenAI(base_url=base_url, api_key=args.api_key)

    n_prompts = len(DEFAULT_PROMPTS)
    prompts = [DEFAULT_PROMPTS[i % n_prompts] for i in range(args.n)]

    def _call(p: str) -> RequestResult:
        return _run_one(client, args.model, p, args.max_tokens)

    # Sequential (single-request latency).
    seq = [_call(p) for p in prompts]

    # Concurrent (continuous-batching throughput).
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        conc = list(pool.map(_call, prompts))

    rows = [
        _summarize(
            "sequential",
            seq,
            args.gpu_hourly,
            args.api_in_price,
            args.api_out_price,
        ),
        _summarize(
            f"concurrent(x{args.concurrency})",
            conc,
            args.gpu_hourly,
            args.api_in_price,
            args.api_out_price,
        ),
    ]

    print(
        f"\nmodel={args.model}  base_url={base_url}  gpu_hourly=${args.gpu_hourly}/hr"
        f"  api=${args.api_in_price}/1M in, ${args.api_out_price}/1M out\n"
    )
    _print_table(rows)

    errs = [r.error for r in (seq + conc) if not r.ok]
    if errs:
        print(f"\n{len(errs)} request(s) failed; first error: {errs[0]}")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(
                {"summary": rows, "sequential": [asdict(r) for r in seq],
                 "concurrent": [asdict(r) for r in conc]},
                fh,
                indent=2,
            )
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
