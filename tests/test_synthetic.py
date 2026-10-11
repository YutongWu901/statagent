"""Tests for synthetic data generators and LLM config/client helpers."""
from __future__ import annotations

import numpy as np

from statagent.config import load_config
from statagent.agent.llm import MockLLM, parse_json
from statagent.data.synthetic import GENERATORS


def test_generators_deterministic():
    for name, gen in GENERATORS.items():
        a, b = gen(), gen()
        assert a.equals(b), f"{name} is not deterministic"
        assert len(a) > 0


def test_study_hours_has_expected_structure():
    from statagent.data.synthetic import gen_study_hours

    df = gen_study_hours()
    assert {"student_id", "study_hours", "exam_score"} <= set(df.columns)
    assert df["study_hours"].between(0, 40).all()
    assert df["exam_score"].isna().sum() > 0  # ~2% missing values


def test_teaching_method_true_effect():
    from statagent.data.synthetic import gen_teaching_method

    df = gen_teaching_method()
    new_mean = df[df.method == "New"].posttest_score.mean()
    old_mean = df[df.method == "Traditional"].posttest_score.mean()
    assert new_mean > old_mean + 3  # true ~5 point difference


def test_prepost_true_gain():
    from statagent.data.synthetic import gen_prepost

    df = gen_prepost()
    assert (df.after_score - df.before_score).mean() > 2


def test_config_from_env():
    cfg = load_config({})
    assert not cfg.llm.is_configured
    assert cfg.llm_mode == "offline"
    cfg2 = load_config({"STATAGENT_DEEPSEEK_API_KEY": "sk-test"})
    assert cfg2.llm.is_configured
    assert cfg2.llm_mode == "live"
    assert cfg2.llm.base_url == "https://api.deepseek.com"


def test_config_overrides():
    cfg = load_config({"STATAGENT_MODEL": "deepseek-reasoner",
                       "STATAGENT_BASE_URL": "https://example.com/v1",
                       "STATAGENT_TEMPERATURE": "0.5"})
    assert cfg.llm.model == "deepseek-reasoner"
    assert cfg.llm.base_url == "https://example.com/v1"
    assert cfg.llm.temperature == 0.5


def test_parse_json_tolerates_fences():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('here is {"a": 1} ok') == {"a": 1}
    try:
        parse_json("no json here")
        raise AssertionError("should have raised")
    except Exception as exc:
        assert "JSON" in str(exc) or "json" in str(exc)


def test_mock_llm_records_calls_and_zero_cost():
    mock = MockLLM()
    mock.on(lambda m: "plan" in str(m), lambda m: type("R", (), {"content": "{}", "tool_calls": []})())
    resp = mock.complete([{"role": "user", "content": "make a plan"}], json_mode=True)
    assert resp.content == "{}"
    assert len(mock.calls) == 1
    assert mock.usage.cost_usd == 0.0


def test_numpy_seed_independence():
    rng = np.random.default_rng(42)
    a = rng.normal(size=10)
    rng2 = np.random.default_rng(42)
    b = rng2.normal(size=10)
    assert np.array_equal(a, b)
