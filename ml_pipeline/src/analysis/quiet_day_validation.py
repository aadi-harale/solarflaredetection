from __future__ import annotations

from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")
QUALITY_PATH = RESULTS_DIR / "data_quality_report.csv"
SKIPPED_PATH = RESULTS_DIR / "skipped_event_candidates.csv"
OUT_PATH = RESULTS_DIR / "quiet_day_validation.csv"


def normalize_date(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def cleaned_event_counts() -> dict[str, int]:
    counts = {}
    for path in sorted(RESULTS_DIR.glob("*_nowcast_catalogue_clean.csv")):
        if path.name.startswith("combined_") or path.name.startswith("june03_"):
            continue
        date = normalize_date(path.name.split("_", 1)[0])
        counts[date] = len(pd.read_csv(path))
    return counts


def skipped_candidate_counts() -> dict[str, int]:
    if not SKIPPED_PATH.exists():
        return {}
    skipped = pd.read_csv(SKIPPED_PATH)
    if skipped.empty or "date" not in skipped.columns:
        return {}
    skipped["date"] = skipped["date"].map(normalize_date)
    return skipped.groupby("date").size().to_dict()


def quiet_status(row: pd.Series) -> str:
    label = str(row["quality_label"]).upper()
    cleaned_count = int(row["cleaned_event_count"])
    skipped_count = int(row["skipped_candidate_count"])

    if label == "BROKEN":
        return "REJECT_CONTROL"
    if cleaned_count > 0:
        return "ACTIVE_OR_FLARE_DAY"
    if label == "QUESTIONABLE":
        return "QUESTIONABLE_CONTROL"
    if label != "GOOD":
        return "REJECT_CONTROL"
    if skipped_count > 0:
        return "LOW_ACTIVITY_CONTROL"

    low_counts = (
        row["max_soft_count"] < 500
        and row["max_cdte_count"] < 500
        and row["max_czt_count"] < 500
    )
    if low_counts:
        return "CONFIRMED_QUIET"
    return "LOW_ACTIVITY_CONTROL"


def build_quiet_day_validation() -> pd.DataFrame:
    if not QUALITY_PATH.exists():
        raise FileNotFoundError(f"Missing data quality report: {QUALITY_PATH.resolve()}")

    quality = pd.read_csv(QUALITY_PATH)
    required = {
        "date",
        "overlap_quality_label",
        "solexs_finite_percent",
        "cdte_finite_percent",
        "czt_finite_percent",
        "soft_max_count",
        "cdte_5_20_max_count",
        "czt_20_40_max_count",
    }
    missing = required - set(quality.columns)
    if missing:
        raise ValueError(f"Data quality report is missing required columns: {sorted(missing)}")

    event_counts = cleaned_event_counts()
    skipped_counts = skipped_candidate_counts()

    out = pd.DataFrame(
        {
            "date": quality["date"].map(normalize_date),
            "quality_label": quality["overlap_quality_label"].astype(str).str.upper(),
            "solexs_finite_percent": quality["solexs_finite_percent"],
            "cdte_finite_percent": quality["cdte_finite_percent"],
            "czt_finite_percent": quality["czt_finite_percent"],
            "max_soft_count": quality["soft_max_count"],
            "max_cdte_count": quality["cdte_5_20_max_count"],
            "max_czt_count": quality["czt_20_40_max_count"],
        }
    )
    out["cleaned_event_count"] = out["date"].map(event_counts).fillna(0).astype(int)
    out["skipped_candidate_count"] = out["date"].map(skipped_counts).fillna(0).astype(int)
    out["quiet_status"] = out.apply(quiet_status, axis=1)

    out = out[
        [
            "date",
            "quality_label",
            "cleaned_event_count",
            "skipped_candidate_count",
            "solexs_finite_percent",
            "cdte_finite_percent",
            "czt_finite_percent",
            "max_soft_count",
            "max_cdte_count",
            "max_czt_count",
            "quiet_status",
        ]
    ]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    return out


def main() -> None:
    report = build_quiet_day_validation()
    print("Quiet/control validation:")
    print(report.to_string(index=False))
    print(f"\nSaved: {OUT_PATH}")
