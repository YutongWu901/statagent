# Architecture

StatAgent is a deliberately **monolithic, layered** Python application: one
Streamlit process, no database, no microservices. Layers are separated by
package boundaries and Pydantic contracts so each can be tested in isolation.

## Layer overview

```
┌────────────────────────────── UI Layer (streamlit_app) ──────────────────────────┐
│ Dashboard · Data Explorer · AI Analysis · Report Center · Benchmark Lab           │
└───────────────────────────────────────┬───────────────────────────────────────────┘
                                        │ session state (DatasetStore, records)
┌───────────────────────────────────────▼───────────────────────────────────────────┐
│ Agent Layer (statagent.agent)                                                      │
│  DeepSeekClient ── JSON mode / function calling, retries, cost tracking            │
│  PlannerAgent   ── question+profile → AnalysisPlan (Pydantic, self-repair)         │
│  HeuristicPlanner ── deterministic offline fallback (no LLM)                       │
│  InterpreterAgent ── results → honest natural language                             │
│  AnalysisPipeline ── orchestration: plan → clarification → gate → execute →        │
│                       interpret → post-hoc validation → record                     │
└───────────────┬───────────────────────────────┬───────────────────────────────────┘
                │ plans (tool names + params)   │ results (numbers only)
┌───────────────▼───────────────┐   ┌───────────▼────────────────────────────────────┐
│ Validation Engine              │   │ Statistical Engine (statagent.stats)          │
│ 26 rules, 4 categories:        │   │ ToolRegistry: 11 registered tools, each with  │
│  DATA / DESIGN / METHOD /      │   │ a Pydantic parameter schema and a hard-coded  │
│  INTERP (PASS/WARNING/FAIL/    │   │ scipy/numpy implementation.                   │
│  UNKNOWN + evidence + fix)     │   │ execute_tool() validates params strictly,     │
│ Gate: blocks FAIL/DESIGN-      │   │ wraps errors, serializes figures, and records │
│ UNKNOWN before execution       │   │ a reproducible code snippet per call.         │
└───────────────┬───────────────┘   └───────────┬────────────────────────────────────┘
                │                              │
┌───────────────▼──────────────────────────────▼────────────────────────────────────┐
│ Data Layer (statagent.data) — CSV/XLSX loading (encoding fallback), automatic      │
│ profiling (variable types, missingness, outliers, duplicates, sha256),             │
│ in-memory DatasetStore, seeded synthetic-data generators                            │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Key design decisions

### 1. The LLM never touches numbers
The planner returns tool names and parameters, never results. The executor
(`statagent.stats.engine.execute_tool`) validates parameters against each
tool's Pydantic schema (unknown keys rejected), runs the hard-coded function,
and returns a `ToolResult`. There is no code-execution path for LLM output of
any kind (no `eval`, no SQL, no shell).

### 2. Validation is rules, not a second LLM
Every validation check is an explicit, testable Python rule with an id
(`DATA-01` … `INTERP-06`), a category, evidence and a suggested action. A rule
returns PASS / WARNING / FAIL / UNKNOWN — UNKNOWN meaning "insufficient
information", which for research-design facts triggers a **clarification
question** instead of a guess. The engine catches rule exceptions so a broken
rule degrades to UNKNOWN rather than breaking analysis.

The interpretation check includes a **number-fidelity rule** (INTERP-05):
every number appearing in the natural-language interpretation must match a
number in the computed results — the cheapest possible defense against LLM
hallucinated statistics.

### 3. Pydantic contracts end-to-end
`DatasetProfile`, `AnalysisPlan` (with `StudyDesign` facts), `StepResult`,
`ValidationReport`, `Interpretation` and `AnalysisRecord` are the interfaces
between layers. The planner's output is validated against
`AnalysisPlan.model_json_schema()` and re-prompted on failure (up to two
repair rounds), then sanitized: steps referencing unknown tools or invalid
parameters are dropped with a recorded note.

### 4. Offline fallback, identical computation
Without an API key, `HeuristicPlanner` (keyword/method-name/type-based rules)
replaces the LLM planner and a deterministic template replaces the LLM
interpreter. Everything else — execution, validation, reporting — is
unchanged. This makes the app demoable without credentials and gives the
benchmark a deterministic pipeline-validation mode; runs are labelled
`llm_mode=offline` so they can never be confused with live-LLM experiments.

### 5. Reproducibility by construction
- Every synthetic dataset is seeded.
- Bootstrap takes an explicit seed.
- `AnalysisRecord` stores the dataset sha256, library versions, the seed, the
  LLM configuration (never the key) and the exact `code` string for every
  executed step.
- Benchmark runs persist raw `results.jsonl`, `ground_truth.json` and the
  full config in `summary.json`.

## Data flow for one analysis

1. `Data Explorer` upload → `load_dataframe` → `DatasetStore.add` →
   `profile_frame` (types, missingness, outliers, duplicates).
2. `AI Analysis` submits a question → `AnalysisPipeline.analyze`.
3. `PlannerAgent.plan(question, profile, design_answers)` → `AnalysisPlan`
   (LLM JSON mode with schema + self-repair, or heuristic fallback).
4. If `plan.needs_clarification()` → return questions to the UI (or the
   benchmark simulates user answers); on the next call the answers are merged
   into `StudyDesign` via `design.fill`.
5. `ValidationEngine.validate_before_execution` (DATA/DESIGN/METHOD) — if
   blocked, the UI shows findings and offers an explicit, recorded override.
6. `execute_step` per plan step → `StepResult`s (numbers + serialized figures).
7. `InterpreterAgent.interpret` → `Interpretation` text.
8. `ValidationEngine.validate_interpretation` (INTERP rules, including number
   fidelity) → post-hoc report.
9. `AnalysisRecord` assembled; saved under `runs/reports/`; Markdown and HTML
   reports generated on demand.

## Technology choices

| Choice | Rationale |
|---|---|
| Streamlit | Fastest path to a professional interactive app; st.navigation gives a real multi-page app without a JS build. |
| scipy / statsmodels-grade numpy formulas | Statistical correctness; OLS implemented with explicit numpy formulas so it works on any supported Python and is cross-checked against statsmodels in tests. |
| Pydantic v2 | Strict contracts for plan/params/results; unknown parameters rejected at the executor. |
| DeepSeek via the `openai` SDK | OpenAI-compatible endpoint; JSON output mode + function calling; env-configurable to any compatible provider. |
| Plotly | Interactive charts that serialize to JSON, store in records, and re-render in UI and HTML reports. |
| No database | Session-scoped `DatasetStore` plus files under `runs/` — auditable and dead simple. |
