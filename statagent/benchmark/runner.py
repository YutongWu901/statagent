"""Benchmark runner: executes the three systems over the task suite and
persists every raw outcome for audit."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from statagent.benchmark.metrics import category_breakdown, score_task, summarize
from statagent.benchmark.tasks import TaskSpec
from statagent.config import AppConfig
from statagent.data.store import DatasetStore
from statagent.utils import json_dumps, to_jsonable


@dataclass
class BenchmarkRun:
    label: str
    out_dir: Path
    systems: list[str] = field(default_factory=list)
    tasks_run: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    breakdown: dict[str, Any] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""
    llm_mode: str = "offline"

    @property
    def results_path(self) -> Path:
        return self.out_dir / "results.jsonl"

    @property
    def summary_path(self) -> Path:
        return self.out_dir / "summary.json"


class BenchmarkRunner:
    def __init__(self, config: AppConfig, systems: dict[str, Any], out_dir: Path) -> None:
        self.config = config
        self.systems = systems
        self.out_dir = Path(out_dir)

    def run(
        self,
        tasks: list[TaskSpec],
        label: str | None = None,
        progress: Any = None,
    ) -> BenchmarkRun:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        run = BenchmarkRun(
            label=label or f"run_{datetime.now():%Y%m%d_%H%M%S}",
            out_dir=self.out_dir,
            systems=list(self.systems),
            tasks_run=[t.id for t in tasks],
            started_at=datetime.now().isoformat(),
            llm_mode=self.config.llm_mode,
        )
        results_path = run.results_path
        with results_path.open("w", encoding="utf-8") as fh:
            for i, task in enumerate(tasks):
                df = task.build_dataframe()
                store = DatasetStore()
                dataset_id = store.add(df, f"{task.dataset}.csv")
                profile = store.profile(dataset_id)
                for system_name, system in self.systems.items():
                    try:
                        outcome = system.run(task, store, profile, dataset_id)
                    except Exception as exc:  # noqa: BLE001 - record, don't crash the run
                        outcome = {
                            "task_id": task.id,
                            "system": system_name,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "timing_s": 0.0,
                            "clarifications_asked": [],
                            "answers": {},
                            "interpretation": "",
                        }
                    fh.write(
                        json.dumps(to_jsonable(outcome), ensure_ascii=False) + "\n"
                    )
                    fh.flush()
                if progress is not None:
                    progress(i + 1, len(tasks))

        run.summary, run.breakdown = self._score(tasks)
        run.finished_at = datetime.now().isoformat()
        self._save_artifacts(run, tasks)
        return run

    def _load_outcomes(self) -> dict[str, list[dict[str, Any]]]:
        outcomes: dict[str, list[dict[str, Any]]] = {s: [] for s in self.systems}
        with self.out_dir.joinpath("results.jsonl").open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj["system"] in outcomes:
                    outcomes[obj["system"]].append(obj)
        return outcomes

    def _score(self, tasks: list[TaskSpec]) -> tuple[dict[str, Any], dict[str, Any]]:
        outcomes = self._load_outcomes()
        # Keep per-system rows aligned with the task order.
        scores: dict[str, list[dict[str, Any]]] = {}
        for system, outs in outcomes.items():
            by_id = {o["task_id"]: o for o in outs}
            rows = []
            for task in tasks:
                o = by_id.get(task.id)
                if o is None:
                    rows.append({
                        "task_id": task.id, "category": task.category,
                        "completed": False, "status": "missing",
                        "method_correct": None, "numeric_accuracy": 0.0,
                        "interpretation_accuracy": 0.0, "invalid_analysis": False,
                        "invalid_unflagged": False, "clarification_asked": False,
                        "timing_s": None, "usage": {},
                    })
                else:
                    rows.append(score_task(task, o))
            scores[system] = rows
        return summarize(scores, tasks), category_breakdown(scores, tasks)

    def _save_artifacts(self, run: BenchmarkRun, tasks: list[TaskSpec]) -> None:
        from statagent.utils import json_dumps as jd

        # Ground truth snapshot for auditability.
        truth = {
            t.id: {
                "category": t.category,
                "question": t.question,
                "dataset": t.dataset,
                "method": t.ground_truth.method,
                "allowed": t.ground_truth.allowed_methods,
                "answers": t.ground_truth.answers,
                "must_clarify": t.ground_truth.must_clarify,
            }
            for t in tasks
        }
        (self.out_dir / "ground_truth.json").write_text(jd(truth), encoding="utf-8")
        run_payload = {
            "label": run.label,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "llm_mode": run.llm_mode,
            "model": self.config.llm.model,
            "temperature": self.config.llm.temperature,
            "systems": run.systems,
            "n_tasks": len(run.tasks_run),
            "summary": run.summary,
            "category_breakdown": run.breakdown,
            "config": {
                "seed": self.config.seed,
                "base_url": self.config.llm.base_url,
                "pricing": {
                    "input_per_mtok": self.config.pricing.input_per_mtok,
                    "output_per_mtok": self.config.pricing.output_per_mtok,
                },
            },
        }
        run.summary_path.write_text(jd(run_payload), encoding="utf-8")
        # Markdown + HTML report.
        from statagent.benchmark.report import build_report_md, build_report_html

        run.summary_path.with_suffix("").with_name(f"{run.label}_report.md").write_text(
            build_report_md(run, run_payload), encoding="utf-8"
        )
        run.out_dir.joinpath(f"{run.label}_report.html").write_text(
            build_report_html(run, run_payload), encoding="utf-8"
        )
