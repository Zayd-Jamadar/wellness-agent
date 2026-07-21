# Wellness Assistant — Promptfoo Evals

Evaluates and compares the wellness agent on three axes:

1. **Hallucination** — answers grounded in retrieved KB context (`context-faithfulness`), no fabricated studies/doses.
2. **Bias & harmful outputs** — stereotypes, discrimination, unsafe advice (`bias:*`, `harmful:*`).
3. **Content safety** — jailbreak resistance, refusal handling, robustness (`jailbreak`, `prompt-injection`, `crescendo`, `donotanswer`, `xstest`).

The same agent is driven through [`provider.py`](./provider.py); only the model
(and optional `api_base`) change between providers, so results are a fair
side-by-side grid — including a hosted model vs a local OSS model on Ollama.

## How it works

Promptfoo calls `provider.py:call_api(prompt, options, context)`, which runs the
`WellnessAgent` and returns:

```json
{ "output": "<answer>", "metadata": { "context": "<retrieved KB passages>", "tools_used": [...] } }
```

The retrieved KB passages are surfaced under `metadata.context` so
`context-faithfulness` / `factuality` can score answers against what the agent
actually retrieved (Option B). `metadata.tools_used` shows whether the model
actually called `lookup_kb`, which helps explain hallucination scores.

`provider.py` forwards the provider `config.model` and `config.api_base` into the
agent's chat model, so no code changes are needed to add a model — just add
a provider/target entry.

---

## 1. Setup (once)

Requires Node.js 18+ and the backend installed (its `.venv` has the `wellness`
package + all deps). promptfoo is pinned to `0.117.7` because newer releases
require Node `>=22.22.0` (this machine has 22.18); bump it once Node is updated.

This project lives inside the backend (`wellness/backend/evals/`), so running it
under `uv run` from `backend/` gives promptfoo the backend venv's Python — the
one that has the `wellness` package. No `PROMPTFOO_PYTHON` export is needed.

```bash
cd wellness/backend/evals
npm install                      # installs the pinned promptfoo locally
cp .env.example .env             # add OPENAI_API_KEY (LLM under test + grader)
                                 # and TAVILY_API_KEY (search_web tool)
```

Index the KB once (creates `backend/data/wellness.db`):

```bash
cd wellness/backend && uv run wellness index
```

All commands below run from `backend/` under `uv run` so promptfoo spawns the
venv Python.

### Smoke-test the bridge (no promptfoo)

```bash
cd wellness/backend
uv run python evals/provider.py
```

---

## 2. Functional suite (hallucination + refusal)

```bash
cd wellness/backend
uv run npx promptfoo eval -c evals/promptfooconfig.yaml
uv run npx promptfoo view
```

Providers under test live in [`promptfooconfig.yaml`](./promptfooconfig.yaml):
`gpt-5.4-mini` and `wellness-local (Ollama)`. The grader
(`defaultTest.options.provider`) is a fixed hosted model so both providers are
judged the same way.

---

## 3. Red team (bias / harmful / content safety)

Red teaming is split into two phases so the generated attack dataset is created
once and reused for every run:

- [`redteam.config.yaml`](./redteam.config.yaml) — **generation source**
  (plugins, strategies, targets; no cases).
- [`redteam.yaml`](./redteam.yaml) — **generated dataset** (the ~590 attack
  cases with their assertions; this is what you evaluate).

### 3a. Generate the dataset (run once, or when you change plugins/strategies)

```bash
cd wellness/backend
uv run npx promptfoo redteam generate -c evals/redteam.config.yaml -o evals/redteam.yaml
```

This (re)writes `redteam.yaml`. Use `--force` to regenerate when the config is
unchanged.

### 3b. Evaluate the dataset (repeatable — does NOT regenerate)

```bash
cd wellness/backend
uv run npx promptfoo eval -c evals/redteam.yaml
uv run npx promptfoo redteam report      # or: uv run npx promptfoo view
```

> Important: do **not** use `promptfoo redteam run` for routine scoring — it is a
> generate-then-evaluate command and will overwrite `redteam.yaml` with a fresh
> dataset. Use `redteam generate` (3a) only when you intentionally want a new
> attack set, and `promptfoo eval` (3b) for everything else.

Both `gpt-5.4-mini` and `wellness-local (Ollama)` are listed as targets in
`redteam.yaml`, so a single `eval` scores the same cases against both models.

---

## 4. Compare a local OSS model (Ollama)

The `wellness-local (Ollama)` provider/target reaches an OSS model directly on a
local Ollama server:

- provider: `ollama`
- model: `qwen2.5`
- api_base: `http://localhost:11434`

Bring up Ollama (pull a model once, then serve):

```bash
ollama serve                            # usually already running as a service
ollama pull qwen2.5
```

No eval config changes are needed — `provider.py` forwards `provider` + `model` +
`api_base` into the agent. If Ollama runs on a remote host, set that URL as
`api_base` in the target config.

Until Ollama is up, the `wellness-local` column errors (connection refused)
while `gpt-5.4-mini` still runs — a harmless way to verify the config wiring
first.

If the OSS model runs on a different machine than the hosted model (and that
machine can't reach OpenAI), see section 5 for the split run + consolidate flow.

Note: a small OSS model's tool-calling is weaker; `metadata.tools_used` shows
whether it actually retrieved from the KB, which helps explain hallucination
scores.

---

## 5. Running on two machines (OpenAI on Mac, Ollama on another host)

If the hosted and OSS models run on different machines (e.g. `gpt-5.4-mini` on a
Mac and `wellness-local` on a host that cannot reach OpenAI), evaluate each
model separately and consolidate afterwards. promptfoo does not merge two
independent runs automatically, so we export each run to JSON, import both on one
machine, and compare in the viewer.

Exports are committed under
[`../data/eval-results/`](../data/eval-results/) so both machines share results
via git.

### 5a. Pick the model on each machine

Comment out the provider/target you are NOT running:
- In [`promptfooconfig.yaml`](./promptfooconfig.yaml) (`providers:`) for the functional suite.
- In [`redteam.yaml`](./redteam.yaml) (`targets:`) for the red-team suite.

Both machines must evaluate the SAME `redteam.yaml` (identical cases), so
generate it once (step 3a) and share it via git — never regenerate per machine.

### 5b. Functional suite, per machine

```bash
cd wellness/backend

# Mac (gpt-5.4-mini; comment out wellness-local first):
uv run npx promptfoo eval -c evals/promptfooconfig.yaml \
  --description "functional / gpt-5.4-mini (mac)"
uv run npx promptfoo export latest -o data/eval-results/functional-gpt-5.4-mini.json

# Other host (wellness-local; Ollama up; comment out gpt-5.4-mini first):
uv run npx promptfoo eval -c evals/promptfooconfig.yaml \
  --description "functional / wellness-local (pc)"
uv run npx promptfoo export latest -o data/eval-results/functional-wellness-local.json
```

### 5c. Red-team suite, per machine

```bash
cd wellness/backend

# Each machine: comment out the other target, then evaluate the SHARED dataset:
uv run npx promptfoo eval -c evals/redteam.yaml \
  --description "redteam / gpt-5.4-mini (mac)"           # or wellness-local (pc)
uv run npx promptfoo export latest -o data/eval-results/redteam-gpt-5.4-mini.json
# (on the PC: -o data/eval-results/redteam-wellness-local.json)
```

### 5d. Consolidate on one machine

```bash
cd wellness/backend
# Pull the committed JSONs (git), then import each into the local promptfoo DB:
uv run npx promptfoo import data/eval-results/functional-gpt-5.4-mini.json
uv run npx promptfoo import data/eval-results/functional-wellness-local.json
uv run npx promptfoo import data/eval-results/redteam-gpt-5.4-mini.json
uv run npx promptfoo import data/eval-results/redteam-wellness-local.json

uv run npx promptfoo view    # compare the runs side-by-side (use the --description labels)
```

The `--description` on each run makes the two models easy to tell apart in the
viewer. `data/eval-results/*.json` is committed as the versioned record of each
run.

---

## Notes

- Grader: `gpt-5.4-mini` (see `defaultTest.options.provider` in
  `promptfooconfig.yaml` and `redteam.provider` in `redteam.config.yaml`).
  For high-stakes bias/safety grading a stronger judge is more reliable.
- Attack Success Rate isn't comparable across tools — treat it as a relative
  signal between your own models.
- `memory_enabled` is forced off in the provider so each eval case is independent.
