"""Tests for hypothesis tests against scipy reference results."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from statagent.data.store import DatasetStore
from statagent.stats.engine import execute_tool


def run(store, tool, params):
    return execute_tool(store, tool, params)


@pytest.fixture()
def two_group_store():
    s = DatasetStore()
    rng = np.random.default_rng(11)
    a = rng.normal(50, 8, 40)
    b = rng.normal(55, 8, 40)
    df = pd.DataFrame(
        {"value": np.round(np.concatenate([a, b]), 2), "group": ["G1"] * 40 + ["G2"] * 40}
    )
    s.add(df, "groups.csv")
    return s


def test_one_sample_ttest_matches_scipy(store):
    r = run(store, "ttest_one_sample", {"dataset_id": "ds1", "column": "x", "mu0": 10.0})
    df = store.get("ds1")
    ref = stats.ttest_1samp(df["x"], 10.0)
    assert r.summary["t"] == pytest.approx(ref.statistic, abs=1e-6)
    assert r.summary["p_value"] == pytest.approx(ref.pvalue, abs=1e-6)
    assert r.summary["mean"] == pytest.approx(df["x"].mean(), abs=1e-6)


def test_welch_ttest_matches_scipy(two_group_store):
    r = run(two_group_store, "ttest_two_sample",
            {"dataset_id": "ds1", "column": "value", "group_column": "group", "group_a": "G1", "group_b": "G2"})
    df = two_group_store.get("ds1")
    a = df[df.group == "G1"].value
    b = df[df.group == "G2"].value
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert r.summary["t"] == pytest.approx(ref.statistic, abs=1e-6)
    assert r.summary["p_value"] == pytest.approx(ref.pvalue, abs=1e-6)
    assert r.summary["mean_diff"] == pytest.approx(a.mean() - b.mean(), abs=1e-6)
    # CI must bracket the point estimate
    lo, hi = r.summary["ci95_mean_diff"]
    assert lo <= r.summary["mean_diff"] <= hi
    # Levene reported
    assert "levene_p" in r.details


def test_welch_ci_matches_manual(two_group_store):
    r = run(two_group_store, "ttest_two_sample",
            {"dataset_id": "ds1", "column": "value", "group_column": "group", "group_a": "G1", "group_b": "G2"})
    df = two_group_store.get("ds1")
    a = df[df.group == "G1"].value.to_numpy()
    b = df[df.group == "G2"].value.to_numpy()
    n1, n2 = len(a), len(b)
    se = np.sqrt(a.var(ddof=1) / n1 + b.var(ddof=1) / n2)
    df_w = se**4 / ((a.var(ddof=1) / n1) ** 2 / (n1 - 1) + (b.var(ddof=1) / n2) ** 2 / (n2 - 1))
    crit = stats.t.ppf(0.975, df_w)
    diff = a.mean() - b.mean()
    assert r.summary["ci95_mean_diff"] == pytest.approx([diff - crit * se, diff + crit * se], abs=1e-6)


def test_paired_ttest_matches_scipy():
    s = DatasetStore()
    rng = np.random.default_rng(12)
    before = rng.normal(60, 10, 30)
    after = before + 4 + rng.normal(0, 3, 30)
    s.add(pd.DataFrame({"before": before, "after": after}), "p.csv")
    r = run(s, "ttest_paired", {"dataset_id": "ds1", "column_a": "before", "column_b": "after"})
    ref = stats.ttest_rel(before, after)
    assert r.summary["t"] == pytest.approx(ref.statistic, abs=1e-6)
    assert r.summary["p_value"] == pytest.approx(ref.pvalue, abs=1e-6)


def test_ttest_unknown_group_level(two_group_store):
    r = run(two_group_store, "ttest_two_sample",
            {"dataset_id": "ds1", "column": "value", "group_column": "group", "group_a": "G1", "group_b": "NOPE"})
    assert not r.ok


def test_paired_flag_requires_equal_sizes(store):
    r = run(store, "ttest_two_sample",
            {"dataset_id": "ds1", "column": "x", "group_column": "group", "group_a": "A", "group_b": "B", "paired": True})
    assert r.ok  # 25 vs 25 — equal sizes


def test_chi_square_matches_scipy():
    s = DatasetStore()
    rng = np.random.default_rng(13)
    n = 200
    smoke = rng.random(n) < 0.3
    disease = rng.random(n) < (0.15 + 0.2 * smoke)
    df = pd.DataFrame({"smoke": np.where(smoke, "Y", "N"), "disease": np.where(disease, "Y", "N")})
    s.add(df, "h.csv")
    r = run(s, "chi_square_test", {"dataset_id": "ds1", "column_a": "smoke", "column_b": "disease"})
    table = pd.crosstab(df.smoke, df.disease)
    ref = stats.chi2_contingency(table.values, correction=False)
    assert r.summary["chi2"] == pytest.approx(ref[0], abs=1e-6)
    assert r.summary["p_value"] == pytest.approx(ref[1], abs=1e-6)
    assert 0 <= r.summary["cramers_v"] <= 1


def test_chi_square_2x2_reports_fisher():
    s = DatasetStore()
    df = pd.DataFrame({"a": ["x"] * 8 + ["y"] * 7, "b": ["u"] * 4 + ["v"] * 4 + ["u"] * 2 + ["v"] * 5})
    s.add(df, "f.csv")
    r = run(s, "chi_square_test", {"dataset_id": "ds1", "column_a": "a", "column_b": "b"})
    assert "fisher_p" in r.details


def test_mann_whitney_matches_scipy(two_group_store):
    r = run(two_group_store, "mann_whitney_u",
            {"dataset_id": "ds1", "column": "value", "group_column": "group", "group_a": "G1", "group_b": "G2"})
    df = two_group_store.get("ds1")
    a = df[df.group == "G1"].value
    b = df[df.group == "G2"].value
    ref = stats.mannwhitneyu(a, b, alternative="two-sided")
    assert r.summary["u"] == pytest.approx(ref.statistic, abs=1e-6)
    assert r.summary["p_value"] == pytest.approx(ref.pvalue, abs=1e-6)
    assert -1 <= r.summary["rank_biserial_r"] <= 1


def test_cohens_d_sign_and_magnitude(two_group_store):
    r = run(two_group_store, "ttest_two_sample",
            {"dataset_id": "ds1", "column": "value", "group_column": "group", "group_a": "G1", "group_b": "G2"})
    assert r.summary["cohens_d"] < 0  # G1 mean lower than G2


def test_tiny_groups_rejected():
    s = DatasetStore()
    df = pd.DataFrame({"v": [1.0, 2.0, 3.0, 4.0], "g": ["a", "a", "b", "b"]})
    s.add(df, "tiny.csv")
    r = run(s, "mann_whitney_u", {"dataset_id": "ds1", "column": "v", "group_column": "g", "group_a": "a", "group_b": "b"})
    assert not r.ok
