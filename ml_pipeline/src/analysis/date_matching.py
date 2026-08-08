from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.io.read_lightcurves import discover_available_dates


RESULTS_DIR = Path("results")


def write_date_matching_report(out_path: Path = RESULTS_DIR / "date_matching_report.csv") -> pd.DataFrame:
    _, availability = discover_available_dates()
    report = availability.rename(columns={"status": "matched", "reason": "skip_reason"}).copy()
    report["matched"] = report["matched"].eq("matched")
    report = report[["date", "has_solexs", "has_hel1os", "matched", "skip_reason"]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out_path, index=False)
    return report


def main() -> None:
    report = write_date_matching_report()
    print("Date matching report:")
    print(report.to_string(index=False))
    print(f"\nSaved: {RESULTS_DIR / 'date_matching_report.csv'}")
