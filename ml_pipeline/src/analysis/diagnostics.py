from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.alert_policy import apply_policy, f1_score_from_precision_recall


RESULTS_DIR = Path("results")
EXTERNAL_DIR = Path("data/external")

GOOD_ONLY_DATES = ["20260222", "20260224", "20260310", "20260311", "20260603", "20260605"]
GOOD_PLUS_QUESTIONABLE_DATES = [
    "20260222",
    "20260224",
    "20260310",
    "20260311",
    "20260603",
    "20260605",
    "20260201",
    "20260202",
    "20260204",
    "20260223",
    "20260312",
    "20260313",
]


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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    episodes = pd.read_csv(RESULTS_DIR / "combined_alert_episodes.csv")
    catalogue = pd.read_csv(RESULTS_DIR / "combined_nowcast_catalogue_clean.csv")
    quality = pd.read_csv(RESULTS_DIR / "quiet_day_validation.csv")
    policy = pd.read_csv(RESULTS_DIR / "operational_metrics_summary.csv").iloc[0]

    episodes["date"] = episodes["date"].astype(str)
    episodes["alert_start"] = pd.to_datetime(episodes["alert_start"], utc=True, format="mixed", errors="coerce")
    episodes["alert_end"] = pd.to_datetime(episodes["alert_end"], utc=True, format="mixed", errors="coerce")
    episodes["soft_peak_time"] = pd.to_datetime(episodes["soft_peak_time"], utc=True, format="mixed", errors="coerce")
    episodes["duration_sec"] = pd.to_numeric(episodes["duration_sec"], errors="coerce")
    episodes["max_probability_or_score"] = pd.to_numeric(episodes["max_probability_or_score"], errors="coerce")

    catalogue["source_date"] = catalogue["source_date"].astype(str)
    for col in ["event_start", "event_end", "soft_peak_time"]:
        catalogue[col] = pd.to_datetime(catalogue[col], utc=True, format="mixed", errors="coerce")

    quality["date"] = quality["date"].astype(str)
    return episodes, catalogue, quality, policy


def policy_filtered_episodes(episodes: pd.DataFrame, policy: pd.Series) -> pd.DataFrame:
    return apply_policy(
        episodes=episodes,
        horizon_min=int(policy["horizon_min"]),
        threshold=float(policy["threshold"]),
        min_duration_sec=int(policy["min_duration_sec"]),
        cooldown_min=int(policy["cooldown_min"]),
        min_score_or_probability=float(policy["min_score_or_probability"]),
    )


def group_metrics(
    group_name: str,
    dates: list[str],
    episodes: pd.DataFrame,
    catalogue: pd.DataFrame,
    quality: pd.DataFrame,
    policy: pd.Series,
) -> dict:
    quality_dates = quality[quality["date"].isin(dates)]
    group_catalogue = catalogue[catalogue["source_date"].isin(dates)]
    group_episodes = episodes[episodes["date"].isin(dates)]
    filtered = policy_filtered_episodes(group_episodes, policy)

    useful = filtered[filtered["episode_type"].isin(["TRUE_VALID_ALERT", "OVERLAP_ALERT"])].copy()
    first_useful = useful.sort_values("alert_start").drop_duplicates("heldout_event_id")

    total_alerts = len(filtered)
    useful_count = len(useful)
    false_early = int((filtered["episode_type"] == "FALSE_EARLY_ALERT").sum())
    valid_events = int(useful["heldout_event_id"].nunique())
    total_events = int(len(group_catalogue))

    precision = useful_count / total_alerts if total_alerts else 0.0
    recall = valid_events / total_events if total_events else 0.0

    notes = []
    if quality_dates["quality_label"].eq("QUESTIONABLE").any():
        notes.append("Includes QUESTIONABLE dates; metrics may reflect data-quality limitations.")
    if total_events == 0:
        notes.append("No cleaned events in this group; recall is undefined operationally and set to 0.")

    return {
        "group_name": group_name,
        "dates_used": ", ".join(dates),
        "total_dates": len(dates),
        "good_dates_count": int((quality_dates["quality_label"] == "GOOD").sum()),
        "questionable_dates_count": int((quality_dates["quality_label"] == "QUESTIONABLE").sum()),
        "total_events": total_events,
        "total_alert_episodes_after_policy": total_alerts,
        "useful_alert_episodes_after_policy": useful_count,
        "false_early_alert_episodes_after_policy": false_early,
        "valid_alerted_events": valid_events,
        "event_level_precision": precision,
        "event_level_recall": recall,
        "event_level_f1": f1_score_from_precision_recall(precision, recall),
        "false_alerts_per_day": false_early / len(dates) if dates else np.nan,
        "mean_lead_time_min": float(first_useful["lead_time_min"].mean()) if not first_useful.empty else np.nan,
        "median_lead_time_min": float(first_useful["lead_time_min"].median()) if not first_useful.empty else np.nan,
        "notes": " ".join(notes) if notes else "GOOD-only quality group.",
    }


def write_quality_group_comparison(
    episodes: pd.DataFrame, catalogue: pd.DataFrame, quality: pd.DataFrame, policy: pd.Series
) -> pd.DataFrame:
    comparison = pd.DataFrame(
        [
            group_metrics("GOOD-only", GOOD_ONLY_DATES, episodes, catalogue, quality, policy),
            group_metrics(
                "GOOD + QUESTIONABLE",
                GOOD_PLUS_QUESTIONABLE_DATES,
                episodes,
                catalogue,
                quality,
                policy,
            ),
        ]
    )
    out_csv = RESULTS_DIR / "quality_group_metric_comparison.csv"
    comparison.to_csv(out_csv, index=False)

    good = comparison[comparison["group_name"] == "GOOD-only"].iloc[0]
    allq = comparison[comparison["group_name"] == "GOOD + QUESTIONABLE"].iloc[0]
    precision_delta = allq["event_level_precision"] - good["event_level_precision"]
    recall_delta = allq["event_level_recall"] - good["event_level_recall"]
    f1_delta = allq["event_level_f1"] - good["event_level_f1"]

    if f1_delta > 0:
        better = "GOOD + QUESTIONABLE has the higher F1 because it adds more alerted events, despite weaker precision."
    elif f1_delta < 0:
        better = "GOOD-only has the higher F1."
    else:
        better = "Both groups have the same F1."

    md = f"""# Quality Group Metric Comparison

## Scorecard

{dataframe_to_markdown(comparison)}

## Interpretation

- GOOD-only event count: {int(good["total_events"])} cleaned events across {int(good["total_dates"])} dates.
- GOOD + QUESTIONABLE event count: {int(allq["total_events"])} cleaned events across {int(allq["total_dates"])} dates.
- Precision change after adding QUESTIONABLE dates: {precision_delta:.3f}.
- Recall change after adding QUESTIONABLE dates: {recall_delta:.3f}.
- F1 change after adding QUESTIONABLE dates: {f1_delta:.3f}.

{better}

QUESTIONABLE dates do not simply degrade every metric: they add many labelled events, which raises recall from {good["event_level_recall"]:.3f} to {allq["event_level_recall"]:.3f}, but precision remains weak. This looks like a mix of data-quality sensitivity and deeper prototype weakness: the system can catch more events when more event-bearing QUESTIONABLE days are included, but it still produces many non-useful alert episodes.
"""
    out_md = RESULTS_DIR / "quality_group_metric_comparison.md"
    out_md.write_text(md, encoding="utf-8")
    return comparison


def next_event_after(alert_start: pd.Timestamp, events: pd.DataFrame) -> pd.Series | None:
    future = events[events["event_start"] >= alert_start].sort_values("event_start")
    if future.empty:
        return None
    return future.iloc[0]


def write_false_early_followup(
    filtered: pd.DataFrame, catalogue: pd.DataFrame, quality: pd.DataFrame
) -> pd.DataFrame:
    quality_lookup = dict(zip(quality["date"], quality["quality_label"]))
    rows = []

    false_early = filtered[filtered["episode_type"] == "FALSE_EARLY_ALERT"].copy()
    for _, episode in false_early.iterrows():
        date = str(episode["date"])
        label = quality_lookup.get(date, "MISSING")
        date_events = catalogue[catalogue["source_date"] == date]
        next_event = next_event_after(episode["alert_start"], date_events)

        if next_event is None:
            next_event_id = np.nan
            next_start = pd.NaT
            next_peak = pd.NaT
            minutes_to_start = np.nan
            minutes_to_peak = np.nan
        else:
            next_event_id = next_event.get("event_id", next_event.get("global_event_id", np.nan))
            next_start = next_event["event_start"]
            next_peak = next_event["soft_peak_time"]
            minutes_to_start = (next_start - episode["alert_start"]).total_seconds() / 60
            minutes_to_peak = (next_peak - episode["alert_start"]).total_seconds() / 60

        within_30 = pd.notna(minutes_to_start) and minutes_to_start <= 30
        within_45 = pd.notna(minutes_to_start) and minutes_to_start <= 45
        within_60 = pd.notna(minutes_to_start) and minutes_to_start <= 60
        within_90 = pd.notna(minutes_to_start) and minutes_to_start <= 90

        if label != "GOOD":
            reclassified = "AMBIGUOUS"
            notes = "QUESTIONABLE or missing data quality; do not reinterpret as clean false alarm."
        elif within_60:
            reclassified = "POSSIBLE_VALID_EARLY_PRECURSOR"
            notes = "A cleaned event follows within 60 minutes."
        elif within_90:
            reclassified = "LONG_EARLY_PRECURSOR"
            notes = "A cleaned event follows within 90 minutes but outside 60 minutes."
        else:
            reclassified = "ISOLATED_FALSE_ALARM"
            notes = "No cleaned event follows within 90 minutes."

        rows.append(
            {
                "date": date,
                "quality_label": label,
                "alert_start": episode["alert_start"],
                "alert_end": episode["alert_end"],
                "alert_duration_sec": episode["duration_sec"],
                "alert_score_or_probability": episode["max_probability_or_score"],
                "original_episode_label": episode["episode_type"],
                "next_cleaned_event_id": next_event_id,
                "next_event_start": next_start,
                "next_soft_peak_time": next_peak,
                "minutes_to_next_event_start": minutes_to_start,
                "minutes_to_next_soft_peak": minutes_to_peak,
                "followed_by_event_within_30min": within_30,
                "followed_by_event_within_45min": within_45,
                "followed_by_event_within_60min": within_60,
                "followed_by_event_within_90min": within_90,
                "reclassified_as": reclassified,
                "notes": notes,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "false_early_followup_analysis.csv", index=False)

    total = len(out)
    within_counts = {
        window: int(out[f"followed_by_event_within_{window}min"].sum()) if total else 0
        for window in [30, 45, 60, 90]
    }
    isolated = int((out["reclassified_as"] == "ISOLATED_FALSE_ALARM").sum()) if total else 0
    split = out.groupby("quality_label").size().to_dict() if total else {}

    md = f"""# False-Early Follow-Up Analysis

- Total false-early episodes after policy: {total}
- Followed by a cleaned event within 30 minutes: {within_counts[30]}
- Followed by a cleaned event within 45 minutes: {within_counts[45]}
- Followed by a cleaned event within 60 minutes: {within_counts[60]}
- Followed by a cleaned event within 90 minutes: {within_counts[90]}
- Isolated false alarms: {isolated}
- GOOD split: {split.get("GOOD", 0)}
- QUESTIONABLE split: {split.get("QUESTIONABLE", 0)}

The current 5/10/30-minute labels appear strict for some alerts: several false-early episodes are followed by cleaned events outside the evaluated horizon. However, many episodes remain isolated or ambiguous, so this is not enough to claim they are scientifically valid precursors without external labels and more data.
"""
    (RESULTS_DIR / "false_early_followup_summary.md").write_text(md, encoding="utf-8")
    return out


def write_valid_alerts_by_quality(filtered: pd.DataFrame, catalogue: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    quality_lookup = dict(zip(quality["date"], quality["quality_label"]))
    useful = filtered[filtered["episode_type"].isin(["TRUE_VALID_ALERT", "OVERLAP_ALERT"])].copy()
    first = useful.sort_values("alert_start").drop_duplicates("heldout_event_id")

    rows = []
    for _, alert in first.iterrows():
        event = catalogue[catalogue["global_event_id"] == int(alert["heldout_event_id"])]
        if event.empty:
            continue
        event = event.iloc[0]
        date = str(event["source_date"])
        rows.append(
            {
                "date": date,
                "quality_label": quality_lookup.get(date, str(event.get("quality_label", ""))),
                "event_id": int(event["event_id"]),
                "event_start": event["event_start"],
                "soft_peak_time": event["soft_peak_time"],
                "first_valid_alert_time": alert["alert_start"],
                "lead_time_min": alert["lead_time_min"],
                "event_class": event.get("alert_type", ""),
                "class_source": "nowcast_catalogue_alert_type" if "alert_type" in event.index else "",
                "belongs_to_good_only_group": date in GOOD_ONLY_DATES,
                "belongs_to_good_plus_questionable_group": date in GOOD_PLUS_QUESTIONABLE_DATES,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "valid_alerts_by_quality_group.csv", index=False)

    good_count = int((out["quality_label"] == "GOOD").sum()) if not out.empty else 0
    questionable_count = int((out["quality_label"] == "QUESTIONABLE").sum()) if not out.empty else 0
    concentration = (
        "Successes are concentrated on QUESTIONABLE dates."
        if questionable_count > good_count
        else "Successes are concentrated on GOOD dates."
        if good_count > questionable_count
        else "Successes are evenly split between GOOD and QUESTIONABLE dates."
    )

    md = f"""# Valid Alerts By Quality Group

- Valid alerted events: {len(out)}
- From GOOD dates: {good_count}
- From QUESTIONABLE dates: {questionable_count}

{concentration}

This split suggests that useful alerts are not limited to clean data. Because QUESTIONABLE days contain many of the labelled events, the current successes reflect both real event-bearing days and degraded coverage conditions.
"""
    (RESULTS_DIR / "valid_alerts_by_quality_group.md").write_text(md, encoding="utf-8")
    return out


def write_goes_outputs(catalogue: pd.DataFrame) -> str:
    goes_path = EXTERNAL_DIR / "goes_flare_events.csv"
    if not goes_path.exists():
        EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
        template = pd.DataFrame(
            columns=[
                "event_id",
                "source",
                "start_time_utc",
                "peak_time_utc",
                "end_time_utc",
                "goes_class",
                "active_region",
                "notes",
            ]
        )
        template.to_csv(EXTERNAL_DIR / "goes_flare_events_template.csv", index=False)
        status = """# GOES/SWPC Cross-Check Status

GOES external label file not found.

Created template: `data/external/goes_flare_events_template.csv`

Required columns:

- event_id
- source
- start_time_utc
- peak_time_utc
- end_time_utc
- goes_class
- active_region
- notes

GOES matching skipped until labels are added.
"""
        (RESULTS_DIR / "goes_crosscheck_status.md").write_text(status, encoding="utf-8")
        return "GOES external label file not found; template created and matching skipped."

    goes = pd.read_csv(goes_path)
    for col in ["start_time_utc", "peak_time_utc", "end_time_utc"]:
        goes[col] = pd.to_datetime(goes[col], utc=True, format="mixed", errors="coerce")

    rows = []
    for _, event in catalogue.iterrows():
        peak = event["soft_peak_time"]
        event_start = event["event_start"]
        event_end = event["event_end"]
        matched = goes[
            ((goes["peak_time_utc"] - peak).abs() <= pd.Timedelta(minutes=10))
            | ((goes["start_time_utc"] <= event_end) & (goes["end_time_utc"] >= event_start))
        ]
        for _, goes_event in matched.iterrows():
            rows.append(
                {
                    "surya_global_event_id": event["global_event_id"],
                    "surya_date": event["source_date"],
                    "surya_soft_peak_time": peak,
                    "goes_event_id": goes_event["event_id"],
                    "goes_peak_time_utc": goes_event["peak_time_utc"],
                    "goes_class": goes_event.get("goes_class", ""),
                    "match_reason": "peak within +/-10 min or event-window overlap",
                }
            )

    report = pd.DataFrame(rows)
    report.to_csv(RESULTS_DIR / "goes_matching_report.csv", index=False)
    summary = pd.DataFrame(
        [
            {
                "surya_events": len(catalogue),
                "goes_events": len(goes),
                "matched_surya_events": report["surya_global_event_id"].nunique() if not report.empty else 0,
                "matches": len(report),
            }
        ]
    )
    summary.to_csv(RESULTS_DIR / "goes_event_detection_summary.csv", index=False)
    return "GOES label file found; matching report created."


def write_final_report(
    comparison: pd.DataFrame,
    false_early: pd.DataFrame,
    valid_alerts: pd.DataFrame,
    goes_status: str,
    quality: pd.DataFrame,
    catalogue: pd.DataFrame,
) -> None:
    good = comparison[comparison["group_name"] == "GOOD-only"].iloc[0]
    allq = comparison[comparison["group_name"] == "GOOD + QUESTIONABLE"].iloc[0]
    confirmed_quiet = int((quality["quiet_status"] == "CONFIRMED_QUIET").sum())
    broken = ", ".join(quality[quality["quality_label"] == "BROKEN"]["date"].tolist())

    md = f"""# SuryaAlert Hackathon Diagnostic Report

## Dataset Summary

- Matched quality-gated evaluation dates: {int(allq["total_dates"])}
- Cleaned events used: {len(catalogue)}
- Confirmed quiet/control days: {confirmed_quiet}
- BROKEN dates excluded: {broken}

## GOOD-Only vs GOOD + QUESTIONABLE Scorecard

{dataframe_to_markdown(comparison)}

GOOD + QUESTIONABLE improves recall because it adds event-bearing dates, but precision remains weak. This points to both data-quality sensitivity and prototype model/label limitations.

## False-Early Alert Interpretation

- False-early episodes after policy: {len(false_early)}
- Possible valid early precursors within 60 minutes: {int((false_early["reclassified_as"] == "POSSIBLE_VALID_EARLY_PRECURSOR").sum()) if not false_early.empty else 0}
- Long early precursors within 90 minutes: {int((false_early["reclassified_as"] == "LONG_EARLY_PRECURSOR").sum()) if not false_early.empty else 0}
- Isolated false alarms: {int((false_early["reclassified_as"] == "ISOLATED_FALSE_ALARM").sum()) if not false_early.empty else 0}
- Ambiguous episodes: {int((false_early["reclassified_as"] == "AMBIGUOUS").sum()) if not false_early.empty else 0}

Some false-early alerts may be outside the current 5/10/30-minute labelling windows, but many remain isolated or ambiguous.

## Valid Alerts By Quality Group

- Valid alerted events: {len(valid_alerts)}
- GOOD-date valid alerts: {int((valid_alerts["quality_label"] == "GOOD").sum()) if not valid_alerts.empty else 0}
- QUESTIONABLE-date valid alerts: {int((valid_alerts["quality_label"] == "QUESTIONABLE").sum()) if not valid_alerts.empty else 0}

## GOES Cross-Check Status

{goes_status}

## Space Agency Criteria Mapping

- Low/high flare detection: current cleaned catalogue records candidate event detections and soft/hard X-ray alert type, but external GOES classes are needed to map low/high flare classes.
- True positive rate / false alarm rate: event-level recall and false-early alert episodes are reported; external labels are needed for a final scientific false-alarm rate.
- Lead time: mean and median valid lead time are reported from first useful alert episode per event.

## Recommended Next Steps

- Add GOES/SWPC event labels using the generated template.
- Recompute diagnostics with external flare classes.
- Separate results by GOOD-only and QUESTIONABLE dates in the dashboard.
- Investigate isolated false alarms on GOOD quiet days before tuning any thresholds.
"""
    (RESULTS_DIR / "hackathon_diagnostic_report.md").write_text(md, encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    episodes, catalogue, quality, policy = load_inputs()
    filtered = policy_filtered_episodes(episodes, policy)

    comparison = write_quality_group_comparison(episodes, catalogue, quality, policy)
    false_early = write_false_early_followup(filtered, catalogue, quality)
    valid_alerts = write_valid_alerts_by_quality(filtered, catalogue, quality)
    goes_status = write_goes_outputs(catalogue)
    write_final_report(comparison, false_early, valid_alerts, goes_status, quality, catalogue)

    created = [
        RESULTS_DIR / "quality_group_metric_comparison.csv",
        RESULTS_DIR / "quality_group_metric_comparison.md",
        RESULTS_DIR / "false_early_followup_analysis.csv",
        RESULTS_DIR / "false_early_followup_summary.md",
        RESULTS_DIR / "valid_alerts_by_quality_group.csv",
        RESULTS_DIR / "valid_alerts_by_quality_group.md",
        RESULTS_DIR / "goes_crosscheck_status.md",
        EXTERNAL_DIR / "goes_flare_events_template.csv",
        RESULTS_DIR / "hackathon_diagnostic_report.md",
    ]
    print("Created files:")
    for path in created:
        if path.exists():
            print(path)

    good = comparison[comparison["group_name"] == "GOOD-only"].iloc[0]
    allq = comparison[comparison["group_name"] == "GOOD + QUESTIONABLE"].iloc[0]
    print("\nKey findings:")
    print(f"GOOD-only: events={int(good['total_events'])}, precision={good['event_level_precision']:.3f}, recall={good['event_level_recall']:.3f}, f1={good['event_level_f1']:.3f}")
    print(f"GOOD+QUESTIONABLE: events={int(allq['total_events'])}, precision={allq['event_level_precision']:.3f}, recall={allq['event_level_recall']:.3f}, f1={allq['event_level_f1']:.3f}")
    print(f"False-early episodes audited: {len(false_early)}")
    print(f"Valid alerts from GOOD dates: {int((valid_alerts['quality_label'] == 'GOOD').sum()) if not valid_alerts.empty else 0}")
    print(f"Valid alerts from QUESTIONABLE dates: {int((valid_alerts['quality_label'] == 'QUESTIONABLE').sum()) if not valid_alerts.empty else 0}")
    print(goes_status)
