"""Tests for report generation: Markdown/HTML structure and completeness."""
from __future__ import annotations

from pathlib import Path

from statagent.agent.pipeline import AnalysisPipeline
from statagent.config import load_config
from statagent.data.store import DatasetStore
from statagent.data.synthetic import gen_study_hours
from statagent.report.generator import build_html, build_markdown, save_report

REQUIRED_SECTIONS = [
    "Research question",
    "Analysis plan",
    "Results",
    "Interpretation",
    "Reproducibility",
]


def make_record():
    store = DatasetStore()
    ds = store.add(gen_study_hours(), "s.csv")
    out = AnalysisPipeline(load_config({})).analyze(
        "Is study time associated with exam score?", store, ds
    )
    assert out.status == "completed"
    return out.record


def test_markdown_contains_required_sections():
    md = build_markdown(make_record())
    for section in REQUIRED_SECTIONS:
        assert section in md
    assert "correlation_test" in md
    assert "sha256" in md.lower()


def test_markdown_includes_validation_findings():
    md = build_markdown(make_record())
    assert "Validation before execution" in md


def test_html_is_full_document():
    html = build_html(make_record())
    assert html.startswith("<!DOCTYPE html>")
    assert "<style>" in html
    assert "correlation_test" in html


def test_html_embeds_plotly():
    html = build_html(make_record())
    # The record includes a scatter figure rendered by Plotly.
    assert "plotly" in html.lower()


def test_save_report_writes_three_files(tmp_path: Path):
    record = make_record()
    paths = save_report(record, tmp_path)
    assert paths["json"].exists()
    assert paths["md"].exists()
    assert paths["html"].exists()
    assert "Reproducibility" in paths["md"].read_text(encoding="utf-8")


def test_report_reflects_actual_numbers():
    record = make_record()
    r_value = record.step_results[0].result.summary["r"]
    md = build_markdown(record)
    assert f"{r_value:.4f}" in md.replace("<", " ")  # key results table
