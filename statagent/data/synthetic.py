"""Seeded synthetic data generators.

Used both for demo data and for the benchmark task datasets. Every generator
is deterministic: same seed -> same data. Ground-truth statistics for the
benchmark are computed independently of the analysis engine (see
``statagent/benchmark/tasks.py``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

_rng = np.random.default_rng(20261011)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def gen_study_hours(n: int = 200, seed: int = 11) -> pd.DataFrame:
    """Study hours vs exam score: linear association with noise."""
    rng = _rng(seed)
    hours = rng.uniform(0.5, 40, n)
    score = 45.0 + 1.6 * hours + rng.normal(0, 8.0, n)
    attendance = rng.uniform(50, 100, n)
    prior_grade = np.clip(score - 20 + rng.normal(0, 12, n), 0, 100)
    df = pd.DataFrame(
        {
            "student_id": [f"S{i:03d}" for i in range(1, n + 1)],
            "study_hours": np.round(hours, 1),
            "exam_score": np.round(score, 1),
            "attendance_pct": np.round(attendance, 1),
            "prior_grade": np.round(prior_grade, 1),
            "gender": rng.choice(["Female", "Male"], n, p=[0.52, 0.48]),
        }
    )
    df.loc[rng.random(n) < 0.02, "exam_score"] = np.nan  # ~2% missing
    return df


def gen_teaching_method(n: int = 120, seed: int = 12) -> pd.DataFrame:
    """Two-group comparison with a true mean difference (~5 points)."""
    rng = _rng(seed)
    group = np.where(rng.random(n) < 0.5, "Traditional", "New")
    ability = rng.normal(0, 8, n)
    score = np.where(group == "New", 74 + ability, 69 + ability)
    pretest = np.clip(score - 8 + rng.normal(0, 10, n), 0, 100)
    return pd.DataFrame(
        {
            "method": group,
            "posttest_score": np.round(score, 1),
            "pretest_score": np.round(pretest, 1),
        }
    )


def gen_prepost(n: int = 60, seed: int = 13) -> pd.DataFrame:
    """Paired before/after measurements with a true ~4 point gain."""
    rng = _rng(seed)
    before = rng.normal(60, 10, n)
    after = before + 4.0 + rng.normal(0, 3.5, n)
    return pd.DataFrame(
        {"before_score": np.round(before, 1), "after_score": np.round(after, 1)}
    )


def gen_health_survey(n: int = 300, seed: int = 14) -> pd.DataFrame:
    """Smoking / hypertension contingency data (observational, confounded)."""
    rng = _rng(seed)
    age = rng.uniform(20, 70, n)
    smoking = rng.random(n) < (0.18 + 0.004 * (age - 20))
    # Smoking raises hypertension risk; age confounds both.
    p_ht = np.clip(0.10 + 0.12 * smoking.astype(float) + 0.006 * (age - 20), 0.05, 0.9)
    hypertension = rng.random(n) < p_ht
    bmi = np.clip(rng.normal(25, 4, n), 16, 45)
    return pd.DataFrame(
        {
            "age": np.round(age, 0),
            "smoking": np.where(smoking, "Yes", "No"),
            "hypertension": np.where(hypertension, "Yes", "No"),
            "bmi": np.round(bmi, 1),
        }
    )


def gen_salaries(n: int = 150, seed: int = 15) -> pd.DataFrame:
    """Right-skewed salary data: normality violations for t-based methods."""
    rng = _rng(seed)
    salary = rng.exponential(40000, n) + 20000
    dept = rng.choice(["R&D", "Sales", "Support"], n)
    return pd.DataFrame(
        {"salary": np.round(salary, 0), "department": dept, "years_experience": np.round(rng.uniform(0, 25, n), 1)}
    )


def gen_small_sample(n: int = 8, seed: int = 16) -> pd.DataFrame:
    """Tiny sample: underpowered, normality checks unreliable."""
    rng = _rng(seed)
    return pd.DataFrame(
        {
            "group": ["A"] * (n // 2) + ["B"] * (n - n // 2),
            "value": np.round(rng.normal(10, 3, n), 2),
        }
    )


def gen_outliers(n: int = 60, seed: int = 17) -> pd.DataFrame:
    """Data with extreme outliers that distort means and t-tests."""
    rng = _rng(seed)
    value = rng.normal(50, 5, n)
    value[:4] = [150, 160, 8, 5]  # four extreme observations
    return pd.DataFrame(
        {"value": np.round(value, 1), "group": rng.choice(["C", "D"], n)}
    )


def gen_confounded(n: int = 300, seed: int = 18) -> pd.DataFrame:
    """Education -> spending association driven mostly by income (confounder).

    education and spending correlate strongly in the raw data, but the
    correlation is largely spurious: income drives both.
    """
    rng = _rng(seed)
    income = rng.normal(50000, 15000, n)
    education = np.clip(8 + income / 8000 + rng.normal(0, 1.5, n), 0, 22)
    spending = 5000 + 0.30 * income + 40 * education + rng.normal(0, 6000, n)
    return pd.DataFrame(
        {
            "income": np.round(income, 0),
            "education_years": np.round(education, 1),
            "monthly_spending": np.round(spending, 0),
        }
    )


def gen_ordinal_survey(n: int = 200, seed: int = 19) -> pd.DataFrame:
    """Ordinal Likert satisfaction + numeric spend: rank methods apply."""
    rng = _rng(seed)
    base = rng.normal(0, 1, n)
    satisfaction = np.clip(np.round(3 + base + rng.normal(0, 0.6, n)), 1, 5).astype(int)
    spend = 200 + 30 * satisfaction + rng.normal(0, 60, n)
    return pd.DataFrame(
        {
            "satisfaction": satisfaction,
            "monthly_spend": np.round(np.clip(spend, 0, None), 1),
            "region": rng.choice(["North", "South", "East", "West"], n),
        }
    )


def gen_heteroscedastic(n: int = 200, seed: int = 20) -> pd.DataFrame:
    """Variance grows with x: residual assumptions of OLS are violated."""
    rng = _rng(seed)
    x = rng.uniform(1, 30, n)
    y = 10 + 1.2 * x + rng.normal(0, 0.5 * x, n)
    return pd.DataFrame({"x": np.round(x, 2), "y": np.round(y, 2)})


def gen_likert_groups(n: int = 100, seed: int = 21) -> pd.DataFrame:
    """Two ordinal-scale groups with a location shift (Mann-Whitney case)."""
    rng = _rng(seed)
    g1 = rng.choice([1, 2, 3, 4, 5], n // 2, p=[0.2, 0.3, 0.3, 0.15, 0.05])
    g2 = rng.choice([1, 2, 3, 4, 5], n // 2, p=[0.05, 0.15, 0.3, 0.3, 0.2])
    return pd.DataFrame(
        {
            "group": ["Control"] * (n // 2) + ["Treatment"] * (n // 2),
            "rating": np.concatenate([g1, g2]),
        }
    )


def gen_noise(n: int = 200, seed: int = 22) -> pd.DataFrame:
    """Two weakly associated variables: r ≈ 0.08, typically p > 0.05."""
    rng = _rng(seed)
    x = rng.normal(50, 10, n)
    y = 50 + 0.08 * x + rng.normal(0, 9.97, n)
    return pd.DataFrame({"x": np.round(x, 2), "y": np.round(y, 2)})


def gen_sparse_cat(n: int = 60, seed: int = 23) -> pd.DataFrame:
    """Contingency data with one sparse category (expected counts 1-5)."""
    rng = _rng(seed)
    region = rng.choice(["North", "South", "East"], n, p=[0.40, 0.40, 0.20])
    p_satisfied = np.where(region == "East", 0.30, 0.55)
    satisfied = rng.random(n) < p_satisfied
    return pd.DataFrame(
        {
            "region": region,
            "satisfied": np.where(satisfied, "Yes", "No"),
        }
    )


GENERATORS = {
    "study_hours": gen_study_hours,
    "teaching_method": gen_teaching_method,
    "prepost": gen_prepost,
    "health_survey": gen_health_survey,
    "salaries": gen_salaries,
    "small_sample": gen_small_sample,
    "outliers": gen_outliers,
    "confounded": gen_confounded,
    "ordinal_survey": gen_ordinal_survey,
    "heteroscedastic": gen_heteroscedastic,
    "likert_groups": gen_likert_groups,
    "noise": gen_noise,
    "sparse_cat": gen_sparse_cat,
}
