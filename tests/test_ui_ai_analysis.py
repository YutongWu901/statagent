"""UI tests for the AI Analysis page's example buttons (Bug 4).

Uses Streamlit's AppTest so the real view script runs: buttons, callbacks,
the keyed question box, and the Run flow are exercised end-to-end without a
browser. These tests must stay offline: the config seeded into the session
is built from an empty environment, so no API key is ever used.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from statagent.config import load_config
from statagent.data.store import DatasetStore
from statagent.data.synthetic import gen_study_hours

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

VIEW = (
    Path(__file__).resolve().parent.parent / "streamlit_app" / "views" / "ai_analysis.py"
)

EXAMPLE_QUESTIONS = [
    "Is study time associated with exam score?",
    "Does the new teaching method produce higher posttest scores than the traditional method?",
    "Is smoking associated with hypertension?",
    "Predict monthly spending from income and education years.",
]


@pytest.fixture(autouse=True)
def _force_offline_env(tmp_path, monkeypatch):
    """Neutralise any real API key (e.g. from a project .env) so these UI
    tests always run the offline pipeline and never call the API."""
    monkeypatch.setenv("STATAGENT_ENV_FILE", str(tmp_path / "empty.env"))
    monkeypatch.delenv("STATAGENT_DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)


def _app():
    at = streamlit_testing.AppTest.from_file(str(VIEW), default_timeout=60)
    store = DatasetStore()
    store.add(gen_study_hours(), "s.csv")
    at.session_state["store"] = store
    at.session_state["cfg"] = load_config({})  # force offline, never the API
    at.run()
    return at


def _question_value(at) -> str:
    return at.text_area(key="question_input").value


def _click_example(at, index: int) -> None:
    button = next(b for b in at.button if b.key == f"example_{index}")
    button.click().run()


def test_all_four_example_buttons_fill_question():
    at = _app()
    for i, expected in enumerate(EXAMPLE_QUESTIONS):
        _click_example(at, i)
        assert _question_value(at) == expected, f"button {i} filled wrong text"


def test_consecutive_example_clicks_update_question():
    at = _app()
    _click_example(at, 0)
    _click_example(at, 3)
    assert _question_value(at) == EXAMPLE_QUESTIONS[3]
    _click_example(at, 1)
    assert _question_value(at) == EXAMPLE_QUESTIONS[1]


def test_clear_button_empties_question():
    at = _app()
    _click_example(at, 2)
    assert _question_value(at) == EXAMPLE_QUESTIONS[2]
    clear = next(b for b in at.button if b.label == "Clear")
    clear.click().run()
    assert _question_value(at) == ""


def test_user_edits_question_after_example():
    at = _app()
    _click_example(at, 0)
    at.text_area(key="question_input").set_value("My custom question here").run()
    assert _question_value(at) == "My custom question here"


def test_run_uses_current_question_and_completes():
    at = _app()
    _click_example(at, 0)
    # Edit the question to something else and run: the analysis must use the
    # edited text, not the example's session value.
    at.text_area(key="question_input").set_value(
        "Is study time associated with exam score? edited"
    ).run()
    run_button = next(b for b in at.button if b.label == "Run analysis")
    run_button.click().run()
    assert not at.exception
    records = at.session_state["records"]
    assert len(records) == 1
    record = next(iter(records.values()))
    assert record.question == "Is study time associated with exam score? edited"
    assert record.step_results[0].tool == "correlation_test"


def test_run_with_mismatched_variables_shows_error():
    at = _app()
    # study_hours_scores has no smoking/hypertension columns.
    at.text_area(key="question_input").set_value(
        "Is smoking associated with hypertension?"
    ).run()
    run_button = next(b for b in at.button if b.label == "Run analysis")
    run_button.click().run()
    assert not at.exception
    assert at.session_state["error"] is not None
    assert "Could not find any columns" in at.session_state["error"]


def test_rerun_does_not_clear_edited_question():
    at = _app()
    at.text_area(key="question_input").set_value("keep me").run()
    at.run()  # simulate an unrelated rerun
    assert _question_value(at) == "keep me"
