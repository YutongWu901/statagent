"""Run the StatAgent benchmark from the command line.

Usage:
    python scripts/run_benchmark.py [--tasks N] [--out DIR] [--label NAME]
                                    [--smoke] [--systems A,B,C] [--limit N]

With an API key configured (DEEPSEEK_API_KEY) the run uses the live model;
without one it validates the pipeline with the offline heuristic planner
(recorded honestly as llm_mode=offline).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from statagent.agent.llm import MockLLM, create_client  # noqa: E402
from statagent.benchmark.runner import BenchmarkRunner  # noqa: E402
from statagent.benchmark.systems import SystemA, SystemB, SystemC  # noqa: E402
from statagent.benchmark.tasks import TASKS  # noqa: E402
from statagent.config import get_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the StatAgent benchmark.")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N tasks (0 = all).")
    parser.add_argument("--out", type=str, default="", help="Output directory (default: runs/benchmark/<label>).")
    parser.add_argument("--label", type=str, default="", help="Run label (default: timestamped).")
    parser.add_argument("--smoke", action="store_true", help="Pipeline smoke run with a scripted mock LLM for System A.")
    parser.add_argument("--offline", action="store_true", help="Force the offline heuristic planner (never calls the API).")
    args = parser.parse_args()

    config = get_config()
    if args.offline:
        from statagent.config import load_config

        config = load_config({})
    client = create_client(config.llm, config.pricing)
    print(f"LLM mode: {config.llm_mode} (model: {config.llm.model})")

    systems = {}
    if client is not None:
        systems["A_llm_only"] = SystemA(client)
    elif args.smoke:
        mock = MockLLM()
        mock.on(
            lambda m: any("DIRECTLY" in str(x.get("content", "")) for x in m),
            lambda m: type(
                "R", (), {"content": '{"answers": {"r": 0.30, "p": 0.05}, "interpretation": "These variables are strongly causally related (r = 0.30, p = 0.05)."}', "tool_calls": []},
            )(),
        )
        systems["A_llm_only"] = SystemA(mock)
        print("System A: using scripted MockLLM (pipeline smoke test).")
    systems["B_llm_tools"] = SystemB(config, client)
    systems["C_llm_tools_validation"] = SystemC(config, client)

    tasks = TASKS[: args.limit] if args.limit else TASKS
    out_dir = Path(args.out) if args.out else config.benchmark_dir / (args.label or "run")
    runner = BenchmarkRunner(config, systems, out_dir)
    print(f"Running {len(tasks)} tasks over systems: {list(systems)}")

    def progress(done: int, total: int) -> None:
        if done % 5 == 0 or done == total:
            print(f"  ... {done}/{total} tasks done")

    run = runner.run(tasks, label=args.label, progress=progress)
    print(f"\nRun '{run.label}' finished in {run.out_dir}")
    print("Headline metrics:")
    for system, m in run.summary.items():
        if not isinstance(m, dict) or "task_completion_rate" not in m:
            continue
        print(
            f"  {system}: completion={m['task_completion_rate']:.0%}, "
            f"numeric={m['numerical_accuracy']:.0%}, "
            f"interp={m['interpretation_accuracy']:.0%}, "
            f"invalid_unflagged={m['invalid_analysis_rate']:.0%}, "
            f"cost=${m['cost_usd']:.4f}"
        )
    print("Artifacts: results.jsonl, summary.json, ground_truth.json, report.md/html")


if __name__ == "__main__":
    main()
