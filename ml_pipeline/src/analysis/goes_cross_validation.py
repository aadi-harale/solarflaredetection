from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path(".")
RESULTS_DIR = ROOT / "results"
GOES_PATH = ROOT / "data" / "external" / "goes_flare_events.csv"
CATALOGUE_PATH = RESULTS_DIR / "combined_nowcast_catalogue_clean.csv"
ALERT_EPISODES_PATH = RESULTS_DIR / "combined_alert_episodes.csv"
FALSE_EARLY_PATH = RESULTS_DIR / "false_early_followup_analysis.csv"
VALID_ALERTS_PATH = RESULTS_DIR / "valid_alerts_by_quality_group.csv"
QUALITY_PATH = RESULTS_DIR / "data_quality_report.csv"
DIAGNOSTIC_REPORT_PATH = RESULTS_DIR / "hackathon_diagnostic_report.md"

REQUIRED_GOES_COLUMNS = [
    "event_id",
    "source",
    "start_time_utc",
    "peak_time_utc",
    "end_time_utc",
    "goes_class",
    "active_region",
    "notes",
]


def parse_timestamp_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    parsed = df.copy()
    for col in columns:
        parsed[col] = pd.to_datetime(parsed[col], utc=True, format="mixed", errors="coerce")
    return parsed


def normalize_date(value: object) -> str:
    text = str(value)
    return text.replace("-", "")[:8]


def class_group(goes_class: object) -> str:
    text = "" if pd.isna(goes_class) else str(goes_class).strip().upper()
    return text[:1] if re.fullmatch(r"[ABCMX]\d+(\.\d+)?", text) else "UNKNOWN"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    text = df.copy()
    for col in text.columns:
        text[col] = text[col].map(lambda value: "" if pd.isna(value) else str(value))
    headers = list(text.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in text.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in headers) + " |")
    return "\n".join(lines)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [
        path
        for path in [
            GOES_PATH,
            CATALOGUE_PATH,
            ALERT_EPISODES_PATH,
            FALSE_EARLY_PATH,
            VALID_ALERTS_PATH,
            QUALITY_PATH,
            DIAGNOSTIC_REPORT_PATH,
        ]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(str(path) for path in missing))

    goes = pd.read_csv(GOES_PATH)
    catalogue = pd.read_csv(CATALOGUE_PATH)
    episodes = pd.read_csv(ALERT_EPISODES_PATH)
    false_early = pd.read_csv(FALSE_EARLY_PATH)
    valid_alerts = pd.read_csv(VALID_ALERTS_PATH)
    quality = pd.read_csv(QUALITY_PATH)

    goes = parse_timestamp_columns(goes, ["start_time_utc", "peak_time_utc", "end_time_utc"])
    catalogue = parse_timestamp_columns(catalogue, ["event_start", "event_end", "soft_peak_time"])
    episodes = parse_timestamp_columns(episodes, ["alert_start", "alert_end", "soft_peak_time"])
    false_early = parse_timestamp_columns(false_early, ["alert_start", "alert_end"])
    valid_alerts = parse_timestamp_columns(valid_alerts, ["event_start", "soft_peak_time", "first_valid_alert_time"])

    goes["date"] = goes["start_time_utc"].dt.strftime("%Y%m%d")
    catalogue["date"] = catalogue["source_date"].map(normalize_date)
    episodes["date"] = episodes["date"].map(normalize_date)
    false_early["date"] = false_early["date"].map(normalize_date)
    valid_alerts["date"] = valid_alerts["date"].map(normalize_date)
    quality["date"] = quality["date"].map(normalize_date)

    return goes, catalogue, episodes, false_early, valid_alerts, quality


def validate_goes(goes: pd.DataFrame) -> dict[str, object]:
    missing_columns = [col for col in REQUIRED_GOES_COLUMNS if col not in goes.columns]
    timestamp_nulls = {
        col: int(goes[col].isna().sum()) if col in goes.columns else None
        for col in ["start_time_utc", "peak_time_utc", "end_time_utc"]
    }
    ordering_bad = int(
        (~((goes["start_time_utc"] <= goes["peak_time_utc"]) & (goes["peak_time_utc"] <= goes["end_time_utc"]))).sum()
    )
    class_bad_mask = ~goes["goes_class"].astype(str).str.upper().str.match(r"^[CMX]\d+(\.\d+)?$", na=False)
    duplicate_event_ids = goes[goes["event_id"].duplicated(keep=False)]["event_id"].dropna().unique().tolist()
    coverage = sorted(goes["date"].dropna().unique().tolist())

    report = f"""# GOES Label Validation Report

## Input

- File: `{GOES_PATH}`
- Rows: {len(goes)}

## Checks

- Required columns missing: {", ".join(missing_columns) if missing_columns else "none"}
- Invalid or missing start timestamps: {timestamp_nulls["start_time_utc"]}
- Invalid or missing peak timestamps: {timestamp_nulls["peak_time_utc"]}
- Invalid or missing end timestamps: {timestamp_nulls["end_time_utc"]}
- Rows failing start <= peak <= end: {ordering_bad}
- Rows failing GOES class format: {int(class_bad_mask.sum())}
- Duplicate event_id values: {len(duplicate_event_ids)}
- Date coverage: {", ".join(coverage)}

## Notes

The GOES `source` column was not filtered by exact text; all rows in the file were validated.
"""
    (RESULTS_DIR / "goes_label_validation_report.md").write_text(report, encoding="utf-8")
    return {
        "missing_columns": missing_columns,
        "timestamp_nulls": timestamp_nulls,
        "ordering_bad": ordering_bad,
        "class_bad": int(class_bad_mask.sum()),
        "duplicate_event_ids": duplicate_event_ids,
        "coverage": coverage,
    }


def match_one_surya_event(event: pd.Series, goes_for_date: pd.DataFrame) -> dict[str, object]:
    if goes_for_date.empty:
        return {
            "goes_match_status": "NO_MATCH",
            "goes_event_id": "",
            "goes_start_time_utc": pd.NaT,
            "goes_peak_time_utc": pd.NaT,
            "goes_end_time_utc": pd.NaT,
            "goes_class": "",
            "goes_class_group": "",
            "goes_active_region": "",
            "peak_delta_min": np.nan,
            "window_overlap": False,
            "notes": "No GOES events on same date.",
        }

    peak_delta = (goes_for_date["peak_time_utc"] - event["soft_peak_time"]).abs().dt.total_seconds() / 60
    exact = goes_for_date[peak_delta <= 10].copy()
    if not exact.empty:
        exact["peak_delta_min"] = peak_delta.loc[exact.index]
        match = exact.sort_values("peak_delta_min").iloc[0]
        status = "EXACT_PEAK_MATCH"
        notes = "SuryaAlert soft peak is within +/-10 minutes of GOES peak."
    else:
        overlap = goes_for_date[
            (event["event_start"] <= goes_for_date["end_time_utc"]) & (event["event_end"] >= goes_for_date["start_time_utc"])
        ].copy()
        if not overlap.empty:
            overlap["peak_delta_min"] = peak_delta.loc[overlap.index]
            match = overlap.sort_values("peak_delta_min").iloc[0]
            status = "WINDOW_OVERLAP_MATCH"
            notes = "SuryaAlert event window overlaps GOES event window."
        else:
            nearest = goes_for_date.copy()
            nearest["peak_delta_min"] = peak_delta
            match = nearest.sort_values("peak_delta_min").iloc[0]
            status = "NEAREST_ONLY"
            notes = "Nearest GOES event exists on same date, but no peak/window match."

    return {
        "goes_match_status": status,
        "goes_event_id": match["event_id"],
        "goes_start_time_utc": match["start_time_utc"],
        "goes_peak_time_utc": match["peak_time_utc"],
        "goes_end_time_utc": match["end_time_utc"],
        "goes_class": match["goes_class"],
        "goes_class_group": class_group(match["goes_class"]),
        "goes_active_region": match["active_region"],
        "peak_delta_min": float(abs((match["peak_time_utc"] - event["soft_peak_time"]).total_seconds()) / 60),
        "window_overlap": bool((event["event_start"] <= match["end_time_utc"]) and (event["event_end"] >= match["start_time_utc"])),
        "notes": notes,
    }


def match_surya_to_goes(goes: pd.DataFrame, catalogue: pd.DataFrame, valid_alerts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    valid_keys = set(zip(valid_alerts["date"].astype(str), valid_alerts["event_id"].astype(str)))
    for _, event in catalogue.iterrows():
        date = str(event["date"])
        goes_for_date = goes[goes["date"] == date]
        match = match_one_surya_event(event, goes_for_date)
        surya_id = str(event.get("global_event_id", event.get("event_id", "")))
        event_key = (date, str(event.get("event_id", "")))
        rows.append(
            {
                "surya_event_id": surya_id,
                "date": date,
                "quality_label": event.get("quality_label", ""),
                "surya_event_start": event["event_start"],
                "surya_event_end": event["event_end"],
                "surya_soft_peak_time": event["soft_peak_time"],
                "surya_lead_time_min": event.get("lead_time_min", np.nan),
                "surya_alert_type": event.get("alert_type", ""),
                "valid_alerted": event_key in valid_keys,
                **match,
            }
        )
    report = pd.DataFrame(rows)
    report.to_csv(RESULTS_DIR / "goes_matching_report.csv", index=False)
    return report


def write_detection_summary(goes: pd.DataFrame, catalogue: pd.DataFrame, matching: pd.DataFrame) -> pd.DataFrame:
    matched = matching[matching["goes_match_status"].isin(["EXACT_PEAK_MATCH", "WINDOW_OVERLAP_MATCH"])]
    class_counts = matched["goes_class_group"].value_counts().to_dict()
    summary = pd.DataFrame(
        [
            {
                "total_surya_cleaned_events": len(catalogue),
                "total_goes_events": len(goes),
                "exact_peak_match_count": int((matching["goes_match_status"] == "EXACT_PEAK_MATCH").sum()),
                "window_overlap_match_count": int((matching["goes_match_status"] == "WINDOW_OVERLAP_MATCH").sum()),
                "nearest_only_count": int((matching["goes_match_status"] == "NEAREST_ONLY").sum()),
                "no_match_count": int((matching["goes_match_status"] == "NO_MATCH").sum()),
                "matched_c_count": int(class_counts.get("C", 0)),
                "matched_m_count": int(class_counts.get("M", 0)),
                "matched_x_count": int(class_counts.get("X", 0)),
                "valid_alerted_events_goes_matched": int(matched["valid_alerted"].sum()),
                "goes_dates": ", ".join(sorted(goes["date"].dropna().unique())),
                "surya_dates": ", ".join(sorted(catalogue["date"].dropna().unique())),
            }
        ]
    )
    summary.to_csv(RESULTS_DIR / "goes_event_detection_summary.csv", index=False)
    return summary


def false_early_goes_followup(goes: pd.DataFrame, false_early: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, alert in false_early.iterrows():
        date = str(alert["date"])
        candidates = goes[(goes["date"] == date) & (goes["start_time_utc"] >= alert["alert_start"])].copy()
        if candidates.empty:
            next_goes = None
            minutes_to_start = np.nan
            minutes_to_peak = np.nan
        else:
            candidates["minutes_to_start"] = (candidates["start_time_utc"] - alert["alert_start"]).dt.total_seconds() / 60
            next_goes = candidates.sort_values("minutes_to_start").iloc[0]
            minutes_to_start = float(next_goes["minutes_to_start"])
            minutes_to_peak = float((next_goes["peak_time_utc"] - alert["alert_start"]).total_seconds() / 60)

        within_30 = pd.notna(minutes_to_start) and minutes_to_start <= 30
        within_45 = pd.notna(minutes_to_start) and minutes_to_start <= 45
        within_60 = pd.notna(minutes_to_start) and minutes_to_start <= 60
        within_90 = pd.notna(minutes_to_start) and minutes_to_start <= 90
        quality_label = str(alert.get("quality_label", "MISSING"))

        if quality_label == "QUESTIONABLE":
            reclassified = "AMBIGUOUS_DEGRADED_DATA"
            notes = "QUESTIONABLE data quality prevents a clean GOES-based false-alarm conclusion."
        elif within_60:
            reclassified = "GOES_SUPPORTED_EARLY_PRECURSOR"
            notes = "GOES event follows alert start within 60 minutes."
        elif within_90:
            reclassified = "LONG_GOES_PRECURSOR"
            notes = "GOES event follows alert start within 90 minutes but not within 60 minutes."
        else:
            reclassified = "TRUE_ISOLATED_FALSE_ALARM"
            notes = "No GOES event follows alert start within 90 minutes on the same date."

        rows.append(
            {
                "date": date,
                "quality_label": quality_label,
                "alert_start": alert["alert_start"],
                "alert_end": alert["alert_end"],
                "alert_duration_sec": alert.get("alert_duration_sec", np.nan),
                "alert_score_or_probability": alert.get("alert_score_or_probability", np.nan),
                "original_episode_label": alert.get("original_episode_label", "FALSE_EARLY_ALERT"),
                "next_goes_event_id": "" if next_goes is None else next_goes["event_id"],
                "next_goes_start_time": pd.NaT if next_goes is None else next_goes["start_time_utc"],
                "next_goes_peak_time": pd.NaT if next_goes is None else next_goes["peak_time_utc"],
                "next_goes_end_time": pd.NaT if next_goes is None else next_goes["end_time_utc"],
                "next_goes_class": "" if next_goes is None else next_goes["goes_class"],
                "next_goes_class_group": "" if next_goes is None else class_group(next_goes["goes_class"]),
                "minutes_to_next_goes_start": minutes_to_start,
                "minutes_to_next_goes_peak": minutes_to_peak,
                "followed_by_goes_within_30min": within_30,
                "followed_by_goes_within_45min": within_45,
                "followed_by_goes_within_60min": within_60,
                "followed_by_goes_within_90min": within_90,
                "reclassified_as": reclassified,
                "notes": notes,
            }
        )

    analysis = pd.DataFrame(rows)
    analysis.to_csv(RESULTS_DIR / "false_early_goes_followup_analysis.csv", index=False)
    return analysis


def write_class_aware_summary(matching: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "surya_event_id": matching["surya_event_id"],
            "date": matching["date"],
            "quality_label": matching["quality_label"],
            "soft_peak_time": matching["surya_soft_peak_time"],
            "lead_time_min": matching["surya_lead_time_min"],
            "goes_match_status": matching["goes_match_status"],
            "goes_class": matching["goes_class"],
            "goes_class_group": matching["goes_class_group"],
            "class_source": np.where(
                matching["goes_match_status"].isin(["EXACT_PEAK_MATCH", "WINDOW_OVERLAP_MATCH"]),
                "NOAA/SWPC GOES XRA",
                "",
            ),
            "valid_alerted": matching["valid_alerted"],
            "notes": matching["notes"],
        }
    )
    summary.to_csv(RESULTS_DIR / "class_aware_event_summary.csv", index=False)
    return summary


def write_cross_validation_summary(
    goes: pd.DataFrame,
    catalogue: pd.DataFrame,
    matching: pd.DataFrame,
    detection_summary: pd.DataFrame,
    false_goes: pd.DataFrame,
) -> str:
    row = detection_summary.iloc[0]
    matched = matching[matching["goes_match_status"].isin(["EXACT_PEAK_MATCH", "WINDOW_OVERLAP_MATCH"])]
    false_30 = int(false_goes["followed_by_goes_within_30min"].sum()) if not false_goes.empty else 0
    false_45 = int(false_goes["followed_by_goes_within_45min"].sum()) if not false_goes.empty else 0
    false_60 = int(false_goes["followed_by_goes_within_60min"].sum()) if not false_goes.empty else 0
    false_90 = int(false_goes["followed_by_goes_within_90min"].sum()) if not false_goes.empty else 0
    true_isolated = int((false_goes["reclassified_as"] == "TRUE_ISOLATED_FALSE_ALARM").sum()) if not false_goes.empty else 0

    status_counts = matching["goes_match_status"].value_counts().rename_axis("status").reset_index(name="count")
    matched_class_counts = matched["goes_class_group"].value_counts().rename_axis("goes_class_group").reset_index(name="count")

    md = f"""# GOES/SWPC Cross-Validation Summary

## Dataset

- Total SuryaAlert cleaned events: {len(catalogue)}
- Total GOES C/M/X events in external file: {len(goes)}
- GOES-covered dates: {", ".join(sorted(goes["date"].dropna().unique()))}

## SuryaAlert Cleaned Event Matching

{dataframe_to_markdown(status_counts)}

- EXACT_PEAK_MATCH count: {int(row["exact_peak_match_count"])}
- WINDOW_OVERLAP_MATCH count: {int(row["window_overlap_match_count"])}
- NEAREST_ONLY count: {int(row["nearest_only_count"])}
- NO_MATCH count: {int(row["no_match_count"])}

## Matched GOES Classes

{dataframe_to_markdown(matched_class_counts)}

- Matched C-class count: {int(row["matched_c_count"])}
- Matched M-class count: {int(row["matched_m_count"])}
- Matched X-class count: {int(row["matched_x_count"])}
- Valid alerted events that are GOES matched: {int(row["valid_alerted_events_goes_matched"])}

## False-Early GOES Follow-Up

- False-early alerts followed by GOES event within 30 minutes: {false_30}
- False-early alerts followed by GOES event within 45 minutes: {false_45}
- False-early alerts followed by GOES event within 60 minutes: {false_60}
- False-early alerts followed by GOES event within 90 minutes: {false_90}
- True isolated false alarms after GOES check: {true_isolated}

## Meaning For Space Agency Criteria

- Class coverage: matched events now carry NOAA/SWPC GOES C/M/X classes where an external match exists.
- TPR/FAR: GOES matching supports external true-positive accounting for matched cleaned events and reclassifies some false-early alerts, but this remains limited by the small date set and QUESTIONABLE days.
- Lead time: SuryaAlert lead time can now be interpreted by GOES class for matched events; lead-time claims should still be framed as prototype evidence, not final scientific reliability.
"""
    (RESULTS_DIR / "goes_cross_validation_summary.md").write_text(md, encoding="utf-8")
    return md


def update_diagnostic_report(summary_md: str) -> None:
    existing = DIAGNOSTIC_REPORT_PATH.read_text(encoding="utf-8")
    section_title = "## GOES/SWPC External Validation"
    body = summary_md.split("\n", 1)[1].strip()
    body = re.sub(r"^## ", "### ", body, flags=re.MULTILINE)
    section = section_title + "\n\n" + body + "\n"

    old_status = "\n## GOES Cross-Check Status"
    if old_status in existing:
        start = existing.index(old_status)
        rest = existing[start + 1 :]
        next_section = re.search(r"\n## ", rest)
        end = start + 1 + next_section.start() if next_section else len(existing)
        existing = existing[:start].rstrip() + "\n\n" + existing[end:].lstrip()

    marker = "\n## Space Agency Criteria Mapping"
    if section_title in existing and marker in existing:
        start = existing.index(section_title)
        end = existing.index(marker)
        updated = existing[:start].rstrip() + "\n\n" + section + existing[end:]
    elif section_title in existing:
        before = existing.split(section_title, 1)[0].rstrip()
        updated = before + "\n\n" + section
    elif marker in existing:
        updated = existing.replace(marker, "\n" + section + marker, 1)
    else:
        updated = existing.rstrip() + "\n\n" + section
    DIAGNOSTIC_REPORT_PATH.write_text(updated, encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    goes, catalogue, episodes, false_early, valid_alerts, quality = load_inputs()
    validation = validate_goes(goes)
    matching = match_surya_to_goes(goes, catalogue, valid_alerts)
    detection_summary = write_detection_summary(goes, catalogue, matching)
    false_goes = false_early_goes_followup(goes, false_early)
    class_summary = write_class_aware_summary(matching)
    summary_md = write_cross_validation_summary(goes, catalogue, matching, detection_summary, false_goes)
    update_diagnostic_report(summary_md)

    row = detection_summary.iloc[0]
    print("GOES/SWPC cross-validation complete")
    print(f"GOES label rows: {len(goes)}")
    print(f"SuryaAlert cleaned events: {len(catalogue)}")
    print(
        "Matches: "
        f"EXACT={int(row['exact_peak_match_count'])}, "
        f"WINDOW={int(row['window_overlap_match_count'])}, "
        f"NEAREST_ONLY={int(row['nearest_only_count'])}, "
        f"NO_MATCH={int(row['no_match_count'])}"
    )
    print(
        "Matched classes: "
        f"C={int(row['matched_c_count'])}, "
        f"M={int(row['matched_m_count'])}, "
        f"X={int(row['matched_x_count'])}"
    )
    print(f"Valid alerted GOES-matched events: {int(row['valid_alerted_events_goes_matched'])}")
    print(
        "False-early followed by GOES within 30/45/60/90 min: "
        f"{int(false_goes['followed_by_goes_within_30min'].sum())}/"
        f"{int(false_goes['followed_by_goes_within_45min'].sum())}/"
        f"{int(false_goes['followed_by_goes_within_60min'].sum())}/"
        f"{int(false_goes['followed_by_goes_within_90min'].sum())}"
    )
    print(f"True isolated false alarms after GOES check: {int((false_goes['reclassified_as'] == 'TRUE_ISOLATED_FALSE_ALARM').sum())}")
    print("Reports written under results/")


if __name__ == "__main__":
    main()
