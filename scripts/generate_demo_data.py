"""Generate demo datasets into demo_data/.

Usage: python scripts/generate_demo_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from statagent.data.synthetic import (  # noqa: E402
    gen_health_survey,
    gen_ordinal_survey,
    gen_prepost,
    gen_study_hours,
    gen_teaching_method,
)


def main() -> None:
    out = Path(__file__).resolve().parent.parent / "demo_data"
    out.mkdir(exist_ok=True)

    gen_study_hours(200, seed=11).to_csv(out / "study_hours_scores.csv", index=False)
    gen_teaching_method(120, seed=12).to_csv(out / "teaching_method.csv", index=False)
    gen_prepost(60, seed=13).to_csv(out / "paired_prepost.csv", index=False)
    gen_health_survey(300, seed=14).to_csv(out / "health_survey.csv", index=False)
    survey = gen_ordinal_survey(200, seed=19).rename(
        columns={"satisfaction": "satisfaction_1_5", "monthly_spend": "monthly_spend"}
    )
    survey.to_excel(out / "customer_survey.xlsx", index=False, sheet_name="survey")

    print("Demo data written to demo_data/:")
    for p in sorted(out.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
