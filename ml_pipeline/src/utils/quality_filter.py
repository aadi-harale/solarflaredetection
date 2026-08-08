from __future__ import annotations

from pathlib import Path

import pandas as pd


QUALITY_REPORT_PATH = Path("results/data_quality_report.csv")
EVALUATION_SUMMARY_PATH = Path("results/evaluation_dataset_summary.md")


def normalize_date(date: object) -> str:
    text = str(date).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def load_quality_report(path: Path = QUALITY_REPORT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing data quality report: {path.resolve()}")

    report = pd.read_csv(path)
    required = {"date", "overlap_quality_label"}
    missing = required - set(report.columns)
    if missing:
        raise ValueError(f"Data quality report is missing required columns: {sorted(missing)}")

    report = report.copy()
    report["date"] = report["date"].map(normalize_date)
    report["overlap_quality_label"] = report["overlap_quality_label"].astype(str).str.upper()
    return report


def get_quality_label(date: object) -> str:
    report = load_quality_report()
    date_key = normalize_date(date)
    match = report[report["date"] == date_key]
    if match.empty:
        raise KeyError(f"No quality label found for date {date_key}")
    return str(match.iloc[0]["overlap_quality_label"])


def is_broken_date(date: object) -> bool:
    return get_quality_label(date) == "BROKEN"


def is_usable_for_forecast(date: object) -> bool:
    return get_quality_label(date) in {"GOOD", "QUESTIONABLE"}


def quality_lookup() -> dict[str, str]:
    report = load_quality_report()
    return dict(zip(report["date"], report["overlap_quality_label"]))


def apply_quality_gate_to_forecast_df(df: pd.DataFrame) -> pd.DataFrame:
    if "source_date" not in df.columns:
        return df

    lookup = quality_lookup()
    out = df.copy()
    out["source_date"] = out["source_date"].map(normalize_date)
    out["quality_label"] = out["source_date"].map(lookup)
    return out[out["quality_label"].isin(["GOOD", "QUESTIONABLE"])].copy()


def apply_quality_gate_to_catalogue(cat: pd.DataFrame) -> pd.DataFrame:
    if "source_date" not in cat.columns:
        return cat

    lookup = quality_lookup()
    out = cat.copy()
    out["source_date"] = out["source_date"].map(normalize_date)
    out["quality_label"] = out["source_date"].map(lookup)
    return out[out["quality_label"].isin(["GOOD", "QUESTIONABLE"])].copy()


def write_evaluation_dataset_summary(
    included_dates: list[str],
    excluded_dates: dict[str, str],
    cleaned_events_by_date: dict[str, int],
    quiet_dates: list[str],
    questionable_dates: list[str],
    path: Path = EVALUATION_SUMMARY_PATH,
) -> Path:
    total_events = sum(cleaned_events_by_date.get(date, 0) for date in included_dates)

    lines = [
        "# Evaluation Dataset Summary",
        "",
        "Quality gate source: `results/data_quality_report.csv`.",
        "",
        "BROKEN dates are excluded from supervised forecasting training and validation. GOOD and QUESTIONABLE dates are included. Quiet GOOD dates remain usable as negative examples.",
        "",
        "## Included dates",
    ]

    for date in included_dates:
        label = get_quality_label(date)
        events = cleaned_events_by_date.get(date, 0)
        quiet = "yes" if date in quiet_dates else "no"
        lines.append(f"- `{date}`: quality={label}, cleaned_events={events}, quiet_day={quiet}")

    lines.extend(["", "## Excluded dates"])
    if excluded_dates:
        for date, reason in excluded_dates.items():
            lines.append(f"- `{date}`: {reason}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## Counts",
            f"- Total cleaned events used: {total_events}",
            f"- Quiet days used: {len(quiet_dates)}",
            f"- QUESTIONABLE dates used: {len(questionable_dates)}",
            "",
            "Warning: QUESTIONABLE dates are usable, but missing chunks or reduced finite coverage may affect metrics.",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
