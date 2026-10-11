"""Benchmark & evaluation layer."""
from statagent.benchmark.metrics import (
    category_breakdown,
    score_task,
    summarize,
    wilson_ci,
)
from statagent.benchmark.runner import BenchmarkRun, BenchmarkRunner
from statagent.benchmark.systems import SystemA, SystemB, SystemC, build_systems
from statagent.benchmark.tasks import TASKS, GroundTruth, TaskSpec

__all__ = [
    "BenchmarkRun",
    "BenchmarkRunner",
    "GroundTruth",
    "SystemA",
    "SystemB",
    "SystemC",
    "TASKS",
    "TaskSpec",
    "build_systems",
    "category_breakdown",
    "score_task",
    "summarize",
    "wilson_ci",
]
