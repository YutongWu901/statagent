# Research Report — Pilot Study

## Can Rule-Based Statistical Verification Improve the Reliability of LLM-Powered Data Analysis?

*Status of this document: methodology and framework are complete and
verified; the live-LLM experiment is **pending** an API key. Results reported
here are from a pipeline-validation run (offline heuristic planner + mock
System A) and are **not** evidence about the LLM itself. No result in this
report is fabricated or estimated.*

---

## Abstract

LLM-powered data-analysis agents can plan analyses and write interpretations,
but models are known to fabricate numbers and to produce statistically
unjustified conclusions. We built StatAgent, a verifiable agent in which an
LLM only *plans* analyses over a registry of pre-implemented statistical
tools, a strict Python engine performs all computation, and an independent
**rule-based validation engine** (26 explicit rules across data quality,
research design, method choice and result interpretation) gates execution and
audits the interpretation. We define a 60-task benchmark with independent
ground truth and compare three systems — A (LLM only), B (LLM + tools), and
C (LLM + tools + validation). This report presents the framework, the
benchmark, the verification results of the offline pipeline, and the exact
protocol for the live run. The live experiment is pending API credentials;
results will be published in this document when available.

## 1. Introduction

Recent LLMs can be steered to act as data-analysis agents, but two failure
modes are well documented: (i) **hallucinated statistics** — numbers that
were never computed; and (ii) **statistically invalid conclusions** —
causal claims from observational data, equating significance with
importance, or treating p > 0.05 as proof of no effect. Two families of
mitigations exist: tool use (the model calls real computation) and
post-hoc checking (often a second LLM, which can itself err). We test a
third approach: **independent, deterministic, testable validation rules**
that are not an LLM at all. The research question is whether such rules
measurably improve reliability, quantified on a paired task benchmark.

## 2. Related Work

- **Tool use / function calling** (OpenAI function calling, agent
  frameworks): improves numeric grounding when the model actually invokes
  tools; does not prevent wrong tool choice or invalid interpretation.
- **LLM-as-judge evaluation**: convenient but noisy, biased toward its own
  outputs, and can be gamed; we avoid LLM judging entirely.
- **Statistical-assumption checking in software** (statsmodels diagnostics,
  R package checks): assumption tests exist but are rarely enforced as
  gates, and never applied to *interpretation language*.
- **Causal-claim detection**: prior work warns that LLM interpretations of
  observational analyses routinely overstate causality; we operationalize
  this as a negation-aware textual rule plus design-fact rules.

(References listed in Section 10; all are verifiable public artifacts.)

## 3. Methodology

**Systems.** A: one LLM call returns `{answers, interpretation}` directly.
B: LLM planner + strict tool executor, no validation. C: B plus the
26-rule validation engine (pre-execution gate + post-interpretation audit)
and a clarification loop for unknown research-design facts.

**Tasks.** 60 seeded synthetic datasets across 7 categories (method
selection, wrong-method traps, numerical accuracy, insufficient design,
assumption violation, causal misuse, interpretation error). Ground truth:
expected tools with parameter-level trap matchers, reference values from
independent scipy/numpy code, interpretation rubrics, clarification
requirements.

**Metrics.** Task completion, method-selection accuracy (B/C), numerical
accuracy (tolerance-matched), interpretation accuracy (rubric), appropriate
clarification rate (design-insufficient tasks), invalid-analysis rate
(unflagged invalid analyses on trap tasks), time/tokens/cost. Proportions
with Wilson CIs; B vs C compared with McNemar's test on discordant pairs.

**Reproducibility.** Seeded data, fixed model temperature 0, archived raw
outcomes (`results.jsonl`), ground-truth snapshots, and recorded model
configuration per run.

## 4. Experimental Setup

- Environment: Windows 10, Python 3.14.3; numpy 2.5.4, scipy 1.18.1,
  pandas 3.0.6, statsmodels 0.15.0, pydantic 2.14.0.
- Model (live run, pending): `deepseek-chat`, temperature 0, max_tokens
  4096, JSON output mode; base URL `https://api.deepseek.com`.
- Pipeline-validation run: offline heuristic planner for B/C and a scripted
  mock for A; identical tasks, scoring and archiving.

## 5. Results

### 5.1 Pipeline-validation run (offline, 60 tasks, archived)

These results verify the framework end-to-end (real execution, real
scoring, real archiving). They describe the *pipeline*, not LLM behaviour.

| Metric | A (mock) | B | C |
|---|---|---|---|
| Task completion rate | 100% | 100% | 98% |
| Numerical accuracy | 0% | 92% | **100%** |
| Interpretation accuracy | 67% | 93% | 92% |
| Appropriate clarification rate (design tasks) | 0% | 0% | 100% |
| Invalid analysis rate (unflagged) | 47% | 3% | **0%** |
| Cost (USD) | 0.00 | 0.00 | 0.00 |

Observations:
- C **never** executed an analysis the ground truth marks invalid without
  flagging it; its only non-completed task is a *correct refusal* (Pearson
  correlation requested on two categorical columns — blocked by rule
  METHOD-06).
- B, lacking the gate, executed invalid analyses unflagged in 3% of trap
  tasks (the offline heuristic is conservative; the live-LLM gap is
  expected to be larger).
- The mock A demonstrates the scoring works for inline numbers: scripted
  wrong numbers score 0% numerically and its scripted causal claims are
  detected on 47% of trap tasks.
- The clarification loop works: C asks on 100% of design-insufficient
  tasks; A and B have no mechanism and proceed on assumptions.

Raw data: `runs/benchmark/smoke60/results.jsonl` (180 records),
`ground_truth.json`, `summary.json`, generated reports.

### 5.2 Live-LLM experiment — PENDING

Not executed: requires a DeepSeek API key, which was not available in this
environment. The run command is ready:

```
python scripts/run_benchmark.py --label live_2026XXXX
```

Results will be appended here verbatim when executed. **Until then, no
claims are made about LLM behaviour in any system.**

## 6. Discussion

The offline run establishes the measurement machinery: every metric is
computable from archived raw outcomes, trap tasks discriminate between
systems as designed, and the clarification loop is the only mechanism that
can turn "insufficient design" into a correct analysis. Two framework
properties deserve emphasis because they pre-determine part of the expected
live result: (a) System A's numbers are unverifiable by construction, so its
numerical accuracy is a direct measure of the model's ability to answer
statistics questions from context alone; (b) C's interpretation is audited
by rule INTERP-05 (every cited number must match computed results), a
deterministic defense that cannot be argued with.

## 7. Limitations

- **The headline experiment has not been run** (no API key); the current
  evidence concerns the framework, not the model.
- One model (deepseek-chat), one prompt family, one benchmark — external
  validity is limited; this is, at best, a pilot platform.
- Interpretation rubrics are pattern-based; they capture common
  misinterpretations but not every nuance of natural language.
- The validation rules are designed by us; they encode standard
  statistical practice but are not a complete theory of analysis validity.
- Bootstrap CI is percentile-only (no BCa); no correction for multiple
  comparisons beyond warnings.

## 8. Conclusion

We present a complete, testable apparatus for the question posed: a
verifiable statistical agent with an LLM-free validation engine, and a
60-task paired benchmark with independent ground truth. The pipeline is
verified end-to-end (152 unit tests, an archived 60-task offline run in
which the validation-equipped system achieves 100% numerical accuracy on
completed tasks and 0% invalid analyses). The live LLM experiment is the
outstanding step and will be published in this document once credentials
are available.

## 9. Reproducibility

```bash
pip install -r requirements.txt
python scripts/generate_demo_data.py
.venv\Scripts\python -m pytest tests/ -q          # 152 passed
python scripts/run_benchmark.py --smoke            # offline pipeline run
python scripts/run_benchmark.py                    # live run (requires key)
```

## 10. References

1. OpenAI. *Function calling guide*. https://platform.openai.com/docs/guides/function-calling
2. DeepSeek. *API documentation (JSON output, function calling)*. https://api-docs.deepseek.com/
3. Virtanen et al. (2020). *SciPy 1.0: fundamental algorithms for scientific computing in Python*. Nature Methods 17:261–272. DOI 10.1038/s41592-019-0686-2
4. Seabold & Perktold (2010). *statsmodels: econometric and statistical modeling with Python*. Proc. 9th Python in Science Conf. DOI 10.25080/Majora-92bf1922-011
5. Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*. 2nd ed. Lawrence Erlbaum.
6. Mann, H. B., & Whitney, D. R. (1947). *On a test of whether one of two random variables is stochastically larger than the other*. Ann. Math. Statist. 18(1):50–60. DOI 10.1214/aoms/1177730491
7. Altman, D. G., & Bland, J. M. (2011). *How to obtain the confidence interval from a P value*. BMJ 343:d2090. DOI 10.1136/bmj.d2090
8. Wilson, E. B. (1927). *Probable inference, the law of succession, and statistical inference*. JASA 22(158):209–212. DOI 10.1080/01621459.1927.10502953
9. McNemar, Q. (1947). *Note on the sampling error of the difference between correlated proportions or percentages*. Psychometrika 12:153–157. DOI 10.1007/BF02295996
10. Greenland, S., Senn, S. J., Rothman, K. J., et al. (2016). *Statistical tests, P values, confidence intervals, and power: a guide to misinterpretations*. Eur. J. Epidemiol. 31:337–350. DOI 10.1007/s10654-016-0149-3
11. White, H. (1980). *A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity*. Econometrica 48(4):817–838. DOI 10.2307/1912934
