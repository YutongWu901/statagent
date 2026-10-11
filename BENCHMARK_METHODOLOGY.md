# Benchmark Methodology

## Research question

> **Can rule-based statistical verification improve the reliability of
> LLM-powered data analysis?**

Studied on a DeepSeek-driven statistical analysis agent (StatAgent).

## Experimental systems

Three systems share the same task set, the same datasets and (where
applicable) the same strict execution engine:

| System | Planning | Computation | Verification |
|---|---|---|---|
| **A — LLM only** | LLM reasons in one JSON call | *None* — numbers come from the model's head | None |
| **B — LLM + tools** | LLM planner | Registered Python tools | None |
| **C — LLM + tools + validation** | LLM planner | Registered Python tools | 26-rule validation engine + clarification loop |

Notes on fairness:
- B receives **no** validation and **no** clarification loop: when design
  facts are unknown it proceeds on naive assumptions (independent samples,
  observational, non-causal). C asks the user and proceeds only after
  answers; the benchmark simulates a user who answers with the true facts.
- A has no tools by definition; its numeric answers are scored directly
  against the same ground truth.
- All three use the same model, temperature and task prompts (A uses a
  different prompt because it must answer inline).

## Task suite

60 tasks in 7 categories, each a seeded synthetic dataset + natural-language
question + ground truth:

| Category | n | What it tests |
|---|---|---|
| `method_selection` | 12 | The correct method is unambiguous. |
| `wrong_method` | 10 | The obvious method is wrong (t-test on skewed/ordinal data, chi-square on continuous data, correlation of two binaries, inference on n=8…). |
| `numerical_accuracy` | 10 | The computation must match reference values. |
| `insufficient_design` | 10 | Critical design facts are missing → the system must ask (C) instead of guessing. |
| `assumption_violation` | 8 | Naive methods break assumptions (skew, outliers, heteroscedasticity, sparse cells, ordinal scales, tiny samples). |
| `causal_misuse` | 5 | Causal claims on observational data must be refused/flagged. |
| `interpretation_error` | 5 | Correct numbers, but the interpretation must avoid classic traps (p>0.05 ≠ "no effect", significance ≠ importance, MW ≠ median test…). |

**Ground truth** per task: expected tool (+acceptable alternatives),
parameter-level trap matchers (e.g. `correlation_test` with
`method=pearson` where Spearman is correct), reference answers computed by
*independent* code (direct scipy/numpy formulas — not the analysis engine),
interpretation rubrics (required points + forbidden patterns, including a
negation-aware causal-claim detector), and `must_clarify` flags.

## Metrics

- **Task completion rate** — fraction of tasks ending in a completed report.
- **Method selection accuracy** — primary tool ∈ allowed set (B/C only; A
  computes no methods).
- **Numerical accuracy** — per task: fraction of reference answers matched
  within tolerance (2% relative / 1e-3 absolute); averaged over tasks.
- **Interpretation accuracy** — fraction of rubric items satisfied
  (required caveats present, forbidden misinterpretations absent).
- **Appropriate clarification rate** — on insufficient-design tasks only:
  the system asked about the missing design fact (A has no mechanism).
- **Invalid analysis rate** — on trap tasks: the system executed a
  ground-truth-invalid analysis **and did not flag/block it**.
- **Execution time, token usage, cost** — from usage trackers (USD computed
  from configurable per-token prices); **cost per correct task** = total
  cost / fully-correct tasks.

## Statistical comparison

- Proportions reported with 95% **Wilson** confidence intervals.
- B vs C compared on shared binary outcomes with **McNemar's test**
  (continuity-corrected) on discordant pairs — a paired comparison that
  respects the matched task design.

## Procedure

```
python scripts/run_benchmark.py [--limit N] [--smoke] [--label NAME] [--out DIR]
```

For each task the runner builds a fresh seeded dataset and executes every
selected system; **every raw outcome** (status, clarifications asked, method
+ params, full step summaries, interpretation text, validation findings,
errors, timing, usage) is appended to `results.jsonl`. Scoring is then
computed deterministically from those raw records, and the ground truth is
snapshotted to `ground_truth.json` — the whole run is auditable after the
fact, and re-running the scorer on the same jsonl reproduces the metrics.

## Current status (honest accounting)

- **Pipeline validation run** (60 tasks, `llm_mode=offline`, heuristic
  planner, scripted mock for System A): executed and archived under
  `runs/benchmark/smoke60/` — C achieves 100% numerical accuracy on
  completed tasks and 0% invalid-analysis rate; B, without validation, leaks
  3% invalid analyses; the mock A scores 0% numeric accuracy. This validates
  the framework, not the LLM.
- **Live-LLM run**: **not yet executed** — requires an API key. The protocol
  is identical; results will be appended to `RESEARCH_REPORT.md` when run.

## Pilot-study framing

If the live run is performed on a subset (≥ 20 tasks) or only offline, the
results are reported as a *pilot study*; general claims about LLM behaviour
require the full live run.
