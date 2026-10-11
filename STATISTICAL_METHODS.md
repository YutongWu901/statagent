# Statistical Methods Reference

Every method implemented in StatAgent is listed here with what it estimates,
its assumptions, how the engine implements it, and its limitations. The
engine's implementations are unit-tested against scipy / statsmodels.

## 1. Descriptive statistics (`describe_variable`)

**What it does.** Computes n, missing count, mean, median, standard
deviation, variance, min/max, quartiles, IQR, skewness, kurtosis, the
standard error of the mean, and a 95% CI for the mean.

**Implementation notes.**
- Missing values are excluded per column and the dropped count is reported.
- The CI for the mean is `mean ± t(0.975, n−1)·s/√n` — a t-based interval,
  valid regardless of sample size for normal data and approximately valid
  otherwise via the CLT.
- Skewness/kurtosis use scipy's bias-corrected estimators.

**Limitations.** The mean is not robust to outliers — the engine reports
outliers separately (DATA-05) and suggests robust alternatives.

## 2. Normality check (`check_normality`)

Shapiro-Wilk test (valid for 3 ≤ n ≤ 5000; beyond that only skewness/kurtosis
are reported with a CLT note) plus a normal Q-Q plot.

**Interpretation caveats.**
- A small p-value rejects normality — it does not quantify the deviation.
- At large n the test has high power and flags trivially small deviations;
  at very small n it has little power (that is why METHOD-01 only *warns*
  for t-tests, suggesting robust alternatives rather than forbidding).

## 3. Correlation (`correlation_test`, `correlation_matrix`)

**Pearson's r** measures *linear* association between two continuous
variables; it assumes interval data, roughly linear relationship, and no
gross outliers (it is sensitive to them). p-value tests H₀: ρ = 0 via the
t-transformation; the 95% CI for r uses Fisher's z-transform
(`tanh(atanh(r) ± 1.96/√(n−3))`).

**Spearman's ρ** is Pearson's r applied to ranks: a monotonic-association
measure, appropriate for ordinal data, skewed distributions and outlier-heavy
data. The same Fisher CI is used as an approximation.

**Limitations.** Correlation measures association, never causation. Absence
of linear correlation does not imply independence. Complete-case pairs are
used and n is reported.

## 4. t-tests (`ttest_one_sample`, `ttest_two_sample`, `ttest_paired`)

**One-sample** — tests whether the mean of one sample equals μ₀.
Assumes approximate normality (robust for n ≳ 30 via CLT); reports Cohen's d
against μ₀.

**Two-sample** — StatAgent uses **Welch's t-test by default** (does *not*
assume equal variances); Levene's test is reported as evidence, Student's
reference result is included, and the mean-difference CI uses the
Welch-Satterthwaite degrees of freedom. Cohen's d uses the pooled standard
deviation. `paired=true` (equal group sizes, matched by row order) runs a
paired t-test instead.

**Paired** — compares two matched columns (before/after). Reports the mean
of the differences with its t-based CI and Cohen's d_z.

**Assumptions / violations.** Normality of the data (or paired
differences) — checked via Shapiro (METHOD-01 warns when violated); equal
variance — irrelevant for Welch; independence of observations — a *design*
property the system asks about when unknown (DESIGN-01).

## 5. Chi-square test of independence (`chi_square_test`)

Tests whether two categorical variables are independent, via
`χ² = Σ (O−E)²/E` with `E = row·col / n`. The engine reports expected
counts, Cramér's V as the effect size, and automatically also runs Fisher's
exact test for 2×2 tables.

**Validity condition** (enforced as METHOD-03): the approximation requires
expected counts ≳ 5; below 1 the validation gate **blocks** the analysis
(and suggests collapsing categories or an exact test). The test says
*nothing* about the strength or direction of association beyond Cramér's V.

## 6. Mann-Whitney U (`mann_whitney_u`)

Nonparametric two-sample test of **stochastic dominance** — whether values
from one group tend to be larger than those of the other. Uses scipy's
exact/asymptotic `mannwhitneyu`; reports the rank-biserial correlation
`r = 1 − 2U/(n₁n₂)` as the effect size.

**What it is NOT.** A test of medians in general — it can reject even with
equal medians if distributions differ in shape. The engine's output states
this explicitly, and the INTERP rubric (IE-03) penalizes interpretations
that call it a median test.

**When to use.** Ordinal outcomes, skewed data, outliers — the situations
where t-tests are fragile.

## 7. OLS linear regression (`linear_regression`)

Fits `y = β₀ + Σ βⱼxⱼ + ε` by least squares. Reports per-coefficient:
estimate, standard error, t, p-value, 95% CI; plus R², adjusted R², F-test,
AIC/BIC, and n/k. Categorical predictors are one-hot encoded (first level is
the reference; the mapping is reported).

**Diagnostics computed and reported.**
- Durbin-Watson — residual autocorrelation.
- Jarque-Bera — residual non-normality.
- Breusch-Pagan — heteroscedasticity (a violation the engine flags in notes).
- VIF — predictor collinearity (warned at VIF > 10).
- Cook's distance (max) — influential points.
- Residual-vs-fitted and Q-Q plots.

**Implementation.** Explicit numpy formulas (normal equations with
`lstsq`, t/F/χ² reference distributions from scipy); verified against
statsmodels in the test suite.

**Assumptions.** Linear in parameters, exogeneity, homoscedasticity,
approximately normal errors (for inference at small n), no perfect
collinearity. Regression coefficients are *associations* unless the design
is experimental and confounders are controlled — DESIGN-05/06 and
INTERP-01/06 enforce this in language and claims.

## 8. Bootstrap confidence intervals (`bootstrap_ci`)

Nonparametric percentile bootstrap for the mean, median, standard deviation,
or a Pearson/Spearman correlation. Resamples with replacement (default
2000, user-settable **seed** for reproducibility); reports the observed
estimate, bootstrap bias, bootstrap SE and the percentile CI.

**Caveats.** The percentile CI does not correct for bias (no BCa — stated in
the docs); bootstrap is unreliable for very small samples (METHOD-10 warns
below n = 20); results depend on the seed, which is recorded.

## How assumptions are enforced

Assumption checking is split between the *engine* (which always computes
diagnostics and states them in notes) and the *validation engine* (which
turns them into gateable decisions):

| Rule | Check | Effect |
|---|---|---|
| METHOD-01 | Shapiro p < 0.05 at n < 100 for t/Pearson | WARNING → suggest MW/Spearman |
| METHOD-02 | Levene p < 0.05 | informational — Welch already used |
| METHOD-03 | min expected count < 5 / < 1 for χ² | WARNING / **block** |
| METHOD-05 | VIF > 10 | WARNING |
| METHOD-06 | method vs variable type mismatch | WARNING or **block** |
| METHOD-07 | group size < 3 / < 30 | **block** / WARNING |
| METHOD-10 | bootstrap with n < 20 | WARNING |

The full rule catalogue lives in `statagent/validation/` and is unit-tested
one rule at a time.
