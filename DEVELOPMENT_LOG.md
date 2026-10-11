# Development Log

Chronological record of significant decisions, bugs found and fixes made
during the build. Written as the work happened, not retrofitted.

## 2026-10-11 — Initial build (single session, ~4.5 hours)

### Environment

- Windows 10, Python 3.14.3 (D:\Python), pip 25.3, Git 2.53.
- No `DEEPSEEK_API_KEY` available → the system was designed from the start
  with an offline fallback (rule-based planner + template interpreter) so
  that every component can be built, tested and demoed without credentials;
  the live path uses the identical pipeline.

### Decisions

1. **Monolithic layered app** (Streamlit + 7 Python packages), no database,
   no multi-agent framework — per the project brief; state lives in
   `DatasetStore`/session and audit artifacts in `runs/`.
2. **Tool registry instead of code generation.** The LLM can only pick
   registered tools with Pydantic-validated parameters; `extra="forbid"`
   rejects unknown keys. No eval/exec path exists for LLM output.
3. **Regression implemented in numpy/scipy formulas** rather than depending
   on statsmodels at runtime, then cross-checked against statsmodels in
   tests. Chosen for robustness across Python versions; statsmodels 0.15.0
   also installed and used as a test oracle.
4. **Validation = deterministic rules, not a second LLM.** 26 rules with
   ids/evidence/suggestions; INTERP-05 checks that every number in the
   interpretation appears in the computed results.
5. **System B gets no clarification loop and no ground-truth injection**
   (naive assumptions only) so the B/C contrast is honest.
6. **`needs_clarification()` is context-aware**: only facts that change the
   analysis (sample structure for group comparisons, design/confounding for
   causal questions) block execution — descriptive questions never prompt.

### Bugs found and fixed

| Bug | Symptom | Fix |
|---|---|---|
| `AnalysisPlan` unresolvable in `AnalysisRecord` | PydanticUserError "not fully defined" | imported `AnalysisPlan` in `results.py` |
| `parameters.model_dump` on a dict | AttributeError on every executed step | use the validated `params` object |
| `_synthesize_questions` pasted mid-method | orphaned code, analyze() returned None | moved the staticmethod after `analyze` |
| `needs_clarification()` too eager | even association questions demanded design facts | context-aware version (see Decisions) |
| Heuristic matched columns by exact name only | "study time" never matched `study_hours` | token + prefix matching (`satisfied`~`satisfaction`) |
| Single-letter category levels ("C") substring-matched everything | outliers tasks picked the wrong group column | word-boundary match for 1-char levels |
| `group_col.top_values` is count-ordered, not question-ordered | "R&D vs Sales" picked Sales/Support | `group_pair()` prefers levels mentioned in the question |
| `posttest` contains substring `ttest` | explicit-method detection fired for "posttest scores" | word boundary in the t-test regex |
| `sorted(a, b)` instead of `sorted([a, b])` | TypeError on 13 benchmark tasks | fixed call sites |
| VIF `inf` → JSON `null` (`to_jsonable` maps non-finite to None) | TypeError comparing None > 10 | cap VIF at 1e6 in the engine |
| VIF `inf` filtered by `np.isfinite` | collinearity never flagged | drop the isfinite filter |
| Number regex `(?![\w.])` rejects "0.98." | INTERP-05 saw no numbers at all | `(?!\w|\.\d)` lookahead |
| "does not imply causation" flagged as a causal claim | good caveats failed INTERP-01 / benchmark rubric | negation-aware `has_causal_claim()` |
| pandas 3.0 `get_dummies(drop_first)` drops alphabetically-first level | test expected first-observed dropped | correct expectations; documented encoding order |
| `st.plotly_chart` fed raw `Figure` objects | "Object of type Figure is not JSON serializable" in Data Explorer | route UI chart calls through `execute_tool` (serialized figures) |
| `_fmt` used before definition in benchmark_lab.py | NameError on the Benchmark Lab page | moved above first use |
| jsonl written with indented JSON | multi-line records broke the jsonl parser | compact single-line records |
| Regression extract paths didn't match nested summary shape | `coef_income` missing in benchmark extraction | nested list paths with index support |
| MW ground truth direction (U depends on group order) | wrong "u" scores | alphabetical group ordering agreed between engine heuristic and truth |
| Streamlit page navigation via URL reloads the session | dataset lost between pages during browser test | sidebar links (client-side routing) keep session state |

### Verification performed

- `pytest`: **152 passed** in ~5 s (stats cross-checked against scipy and
  statsmodels; every validation rule triggered by a crafted error case;
  planner self-repair tested with a mocked LLM — zero API cost).
- 60-task benchmark offline run archived at `runs/benchmark/smoke60/`:
  C = 100% numerical accuracy on completed tasks, 0% invalid analyses,
  100% clarification rate; B = 3% invalid unflagged; A (mock) = 0%
  numeric. Relabelled honestly as a pipeline validation, not LLM evidence.
- End-to-end UI verified in a real browser (Streamlit on :8501): dashboard,
  demo-data loading and profiling, the clarification loop (group-comparison
  question paused and asked for sample design), Welch t-test results with
  correct non-significant phrasing, Report Center downloads, Benchmark Lab
  run history. Screenshots in `docs/screenshots/`.

### Outstanding / next steps

- **Live DeepSeek run of the benchmark** — blocked on an API key; the
  command is `python scripts/run_benchmark.py`. Results go into
  `RESEARCH_REPORT.md` §5.2 verbatim.
- Optional next features: ANOVA/Kruskal-Wallis, BCa bootstrap, multiple-
  comparison corrections, PDF export, CI workflow.
