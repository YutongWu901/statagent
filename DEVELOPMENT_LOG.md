# Development Log

Chronological record of significant decisions, bugs found and fixes made
during the build. Written as the work happened, not retrofitted.

## 2026-10-11 — Bug-fix session (user-reported issues)

### Bug 1 — association question planned describe_variable instead of Pearson

Reproduced with the user's data (`hours` 1..10, `score` 10..100):
`"Is study time associated with exam score?"` planned
`describe_variable(score)`. Three root causes, all fixed:

1. The ordinal name-hint regex contained `score|grade`, so a column literally
   named `score` (n_unique ≤ 10) was typed `categorical_ordinal`.
   → removed `score`/`grade` from the hints; outcomes named "score" are now
   continuous.
2. `_DISCRETE_MAX_UNIQUE = 10` typed `hours = 1..10` as discrete, pushing
   correlation toward Spearman. → lowered to 5; rounded continuous data
   stays continuous (Likert scales are still caught by the name hints).
3. The planner could not associate "study time" with the `hours` column and
   gave up after matching only `score`. → added a small question→column
   synonym table (time→hours, grade/result→score, income↔salary) plus a
   fallback: an association question with ≥1 matched column on a dataset
   with exactly two numeric columns now correlates both columns instead of
   describing one. The synonym table is directional (question words map to
   column tokens); the first draft had it backwards, which briefly broke
   MS-09 in the benchmark and was caught by the full-suite rerun.

Also added: when the question shows clear analytic intent but no column can
be matched at all, the pipeline now returns an explicit error
("Could not find any columns matching the variables in the question.
Available columns: …") instead of silently summarizing an unrelated column.

Verification: end-to-end run on the user's data now plans
`correlation_test(method=pearson, x=hours, y=score)` and computes
**r = 1.000** (p < 1e-6). Regression tests in `test_planner.py`,
`test_profiler.py`, `test_pipeline.py`.

### Bug 2 — INTERP-03 false positive on a correct disclaimer

`"…insufficient evidence to detect an effect, not that no effect exists."`
was flagged as "proving the null" by the regex
`\bno (effect|…) exists\b`. → INTERP-03 now uses negation-aware matching
(`null_claim_matches`): a claim is only counted when the preceding context
contains no negation ("not", "cannot", "rather than", "不代表", "并非"…).
The benchmark rubric for IE-01 uses the same helper via the `NULL_CLAIM`
forbidden-marker, so correct disclaimers no longer lose interpretation
points. Positive/negative/boundary tests added in `test_validation.py`,
`test_interpreter.py`, `test_benchmark.py`.

### Bug 3 — .env existed but the UI stayed offline

`load_dotenv()` was called with no path, so it searched the *process working
directory* — launching the app from anywhere other than the project root
silently skipped the `.env`. → the loader now reads
`<project root>/.env` explicitly (override via `STATAGENT_ENV_FILE`), and
`load_config()` reloads it so restarts pick up edits. Both
`DEEPSEEK_API_KEY` and `STATAGENT_DEEPSEEK_API_KEY` work (the prefixed one
wins). API failures now surface in the UI: `PlannerAgent` records
`last_error`, the pipeline propagates it as `outcome.llm_warning`, and the
AI Analysis page shows a warning banner ("LLM API issue — offline fallback
used: …") instead of failing silently into the heuristic. Session config is
refreshed on every rerun so a restarted app immediately reflects `.env`.

Verified live: the user's `.env` was loaded without reading its contents;
the sidebar now shows 🟢 live / deepseek-chat; one **minimal real API call**
succeeded (9+1 tokens, "OK", $0.000004) — the key is never logged and
`.env` stays gitignored. New tests in `test_synthetic.py` cover the env-file
override and key precedence.

### Bug 4 — example buttons did not fill the question box

The text area had no explicit key (auto-keyed by label) while the buttons
wrote to `st.session_state.ai_question` — the box kept its own empty state,
Clear couldn't clear the box, and Run preferred the stale session value
over edits. → the box now has key `question_input`; the four example
buttons and Clear use `on_click` callbacks that set the key *before* the
widget is instantiated (the supported Streamlit pattern — writing to a
widget key after instantiation raises StreamlitAPIException). Run uses the
widget's current value. Verified by 8 new `AppTest` tests
(`tests/test_ui_ai_analysis.py`: all four buttons fill, consecutive
switching, manual edits, Clear, Run-uses-current-question, variable
mismatch error, rerun persistence). Browser note: the buttons were verified
in the earlier session and the page renders correctly; in this session the
IAB automation could not open the demo-dataset selectbox popover (all
interaction channels timed out) so the literal in-browser click pass for
Bug 4 relies on the AppTest coverage plus the earlier browser verification
of the same page.

### Regression checks

- `pytest`: **180 passed** (~17 s).
- 60-task offline benchmark (new `--offline` CLI flag): C numeric accuracy
  100%, invalid-analysis rate 0%, clarification rate 100%; B 93%/3%;
  results archived in `runs/benchmark/smoke60/`.

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
