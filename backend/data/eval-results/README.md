# Eval results (committed)

Consolidated promptfoo export files, one per suite + model. These are committed
as the versioned record of each run so results from separate machines can be
brought together (see `../../evals/README.md`, "Running on two machines").

Expected files:

- `functional-gpt-5.4-mini.json`
- `functional-wellness-local.json`
- `redteam-gpt-5.4-mini.json`
- `redteam-wellness-local.json`

Each is produced with `promptfoo export latest -o data/eval-results/<name>.json`
after running `promptfoo eval` for that model, and consolidated on one machine
with `promptfoo import <file>` then `promptfoo view`.
