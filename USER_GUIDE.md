# User Guide

## Getting started

1. Start the app: `streamlit run app.py` and open http://localhost:8501.
2. Go to **📊 Data Explorer** and either upload a CSV/XLSX file or pick one
   of the demo datasets. StatAgent automatically detects variable types,
   missing values, duplicates and outliers — the Overview tab shows the
   profile, the Variable explorer shows per-column statistics and charts,
   Relationships shows correlation heatmaps/scatters, Quality checks lists
   data problems.
3. Go to **🤖 AI Analysis**, choose the dataset, and type your question in
   plain language. Examples:

   - *Is study time associated with exam score?*
   - *Does the new teaching method produce higher posttest scores than the
     traditional method?*
   - *Is smoking associated with hypertension?*
   - *Predict monthly spending from income and education years.*
   - *Do the scores change from before to after the training?*

4. Click **Run analysis**.

## What happens when you run an analysis

- If the question needs research-design information (e.g. are two groups
  independent or paired?), StatAgent **asks first** — answer the questions
  and it continues. It never silently assumes facts that change the method.
- If the validation engine finds a blocking problem (e.g. a chi-square test
  requested on a continuous variable), the analysis stops and shows the
  rules that failed, the evidence and a suggestion. You can force-run with
  an explicit checkbox — the override is recorded in the report.
- Otherwise you get: the analysis plan (goal, design facts, tools with
  rationale), the computed results with charts, validation findings
  (PASS/WARNING/FAIL/UNKNOWN), and a plain-language interpretation.

## Understanding validation findings

- ✅ **PASS** — check passed.
- ⚠️ **WARNING** — something to know (e.g. outliers present, normality
  violated, small sample). The analysis proceeds; read the suggestion.
- ⛔ **FAIL** — the analysis as planned is statistically unsound; execution
  is blocked unless you explicitly override.
- ❓ **UNKNOWN** — not enough information; usually triggers a clarification
  question.

Every finding has a rule id (e.g. `METHOD-03`), evidence and a suggestion.

## Interpretation rules StatAgent enforces

- Association is **not** causation (observational data).
- Statistical significance is **not** practical importance (effect sizes are
  reported and required).
- p > 0.05 does **not** prove "no effect" — it means insufficient evidence.
- Confidence intervals are **not** probability statements about the
  parameter.
- Every number in the interpretation must match the computed results —
  invented numbers are flagged.

## Exporting reports

In **🤖 AI Analysis** or **📄 Report Center** download:
- **Markdown** — full report with tables, tool calls and findings.
- **HTML** — self-contained report with interactive Plotly charts (needs
  internet for the Plotly CDN when opened locally).
- **JSON** — the raw record for archival/re-analysis.

The report includes the dataset hash, engine version, seed, LLM mode and
the exact code of every executed statistical call — reproducible by design.

## Offline vs live mode

The sidebar shows the LLM mode:
- 🟡 **Offline** — no API key set: a rule-based planner handles common
  question patterns. All statistics are identical and real; only the
  planning is heuristic.
- 🟢 **Live** — DeepSeek plans the analysis and writes the interpretation.

To enable live mode, copy `.env.example` to `.env`, set
`STATAGENT_DEEPSEEK_API_KEY`, and restart the app.

## Benchmark Lab

See **🧪 Benchmark Lab** to inspect past evaluation runs (three-system
comparison with metrics, per-category breakdown, raw artifacts) or start a
new run. Runs without an API key are labelled `offline` and validate the
pipeline only.

## Tips

- Column names are matched exactly; rename cryptic columns
  (`VAR001`) to meaningful ones for better AI planning.
- Very small samples (< 10 rows) cannot support inference — StatAgent will
  say so instead of pretending.
- For causal questions, expect StatAgent to push back: observational data
  supports associations, not causal claims.
