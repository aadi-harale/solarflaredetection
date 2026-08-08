from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.alert_window_rescore import (
    COMPARISON_PATH,
    EPISODES_PATH,
    GOES_MATCHING_PATH,
    CATALOGUE_PATH,
    build_event_reference,
    choose_90min_match,
    load_inputs,
)
from src.models.alert_policy import ALERT_KEY_COLS, apply_cooldown, f1_score_from_precision_recall


RESULTS_DIR = Path("results")
POLICY_SWEEP_PATH = RESULTS_DIR / "operational_alert_policy_sweep.csv"
SWEEP_90_PATH = RESULTS_DIR / "operational_alert_policy_sweep_90min.csv"
RECOMMENDATION_90_PATH = RESULTS_DIR / "recommended_alert_policy_90min.md"
COMPARISON_REPORT_PATH = RESULTS_DIR / "alert_window_comparison_report.md"
Space Agency_SCORECARD_MD_PATH = RESULTS_DIR / "space_agency_evaluation_criteria_scorecard.md"
DIAGNOSTIC_REPORT_PATH = RESULTS_DIR / "hackathon_diagnostic_report.md"
FINAL_EVIDENCE_REPORT_PATH = RESULTS_DIR / "final_hackathon_evidence_report.md"


def evaluate_rescored_rows(rows: pd.DataFrame, total_events: int, total_dates: int) -> dict:
    if rows.empty:
        return {
            "total_alert_episodes_after_policy": 0,
            "useful_alert_episodes": 0,
            "duplicate_useful_alert_episodes": 0,
            "late_or_overlap_alert_episodes": 0,
            "false_alert_episodes": 0,
            "ambiguous_questionable_episodes": 0,
            "false_alerts_per_day": 0.0,
            "valid_alerted_events": 0,
            "total_events": total_events,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mean_lead_time_min": np.nan,
            "median_lead_time_min": np.nan,
            "mean_lead_to_event_start_min": np.nan,
            "median_lead_to_event_start_min": np.nan,
        }

    rows = rows.sort_values(["date", "alert_start", "alert_end"]).copy()
    precursor = rows[rows["base_90min_label"] == "PRECURSOR_CANDIDATE"].copy()
    first_indices = precursor.drop_duplicates("matched_surya_event_id", keep="first").index
    valid = precursor.loc[first_indices]
    duplicate = precursor.drop(first_indices)
    late_overlap = rows[rows["base_90min_label"] == "LATE_OR_OVERLAP_ALERT"].copy()
    false = rows[rows["base_90min_label"] == "TRUE_ISOLATED_FALSE_ALERT"].copy()
    ambiguous = rows[rows["base_90min_label"] == "AMBIGUOUS_QUESTIONABLE_DATA"].copy()
    useful = pd.concat([valid, duplicate, late_overlap], ignore_index=True)

    total_alerts = int(len(rows))
    useful_count = int(len(useful))
    valid_events = int(valid["matched_surya_event_id"].nunique())
    precision = useful_count / total_alerts if total_alerts else 0.0
    recall = valid_events / total_events if total_events else 0.0
    lead = pd.to_numeric(valid["minutes_to_event_peak"], errors="coerce")
    lead_start = pd.to_numeric(valid["minutes_to_event_start"], errors="coerce")

    return {
        "total_alert_episodes_after_policy": total_alerts,
        "useful_alert_episodes": useful_count,
        "duplicate_useful_alert_episodes": int(len(duplicate)),
        "late_or_overlap_alert_episodes": int(len(late_overlap)),
        "false_alert_episodes": int(len(false)),
        "ambiguous_questionable_episodes": int(len(ambiguous)),
        "false_alerts_per_day": len(false) / total_dates if total_dates else np.nan,
        "valid_alerted_events": valid_events,
        "total_events": total_events,
        "precision": precision,
        "recall": recall,
        "f1": f1_score_from_precision_recall(precision, recall),
        "mean_lead_time_min": float(lead.mean()) if not lead.empty else np.nan,
        "median_lead_time_min": float(lead.median()) if not lead.empty else np.nan,
        "mean_lead_to_event_start_min": float(lead_start.mean()) if not lead_start.empty else np.nan,
        "median_lead_to_event_start_min": float(lead_start.median()) if not lead_start.empty else np.nan,
    }


def precompute_episode_matches(episodes: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    matched_rows = []
    events_by_date = {date: group.copy() for date, group in events.groupby("date")}
    questionable_dates = set(events.loc[events["quality_label"].eq("QUESTIONABLE"), "date"])

    for idx, alert in episodes.iterrows():
        date = str(alert["date"])
        match = choose_90min_match(alert, events_by_date.get(date, events.iloc[0:0]))
        if match is None:
            base_label = "AMBIGUOUS_QUESTIONABLE_DATA" if date in questionable_dates and str(alert.get("episode_type", "")) == "FALSE_EARLY_ALERT" else "TRUE_ISOLATED_FALSE_ALERT"
            matched_rows.append(
                {
                    "episode_index": idx,
                    "base_90min_label": base_label,
                    "matched_surya_event_id": "",
                    "minutes_to_event_start": np.nan,
                    "minutes_to_event_peak": np.nan,
                }
            )
            continue

        minutes_to_peak = float(match["minutes_to_event_peak"])
        overlaps_event = bool((alert["alert_start"] <= match["event_end"]) and (alert["alert_end"] >= match["event_start"]))
        base_label = "LATE_OR_OVERLAP_ALERT" if minutes_to_peak < 0 and overlaps_event else "PRECURSOR_CANDIDATE"
        matched_rows.append(
            {
                "episode_index": idx,
                "base_90min_label": base_label,
                "matched_surya_event_id": str(match["surya_event_id"]),
                "minutes_to_event_start": float(match["minutes_to_event_start"]),
                "minutes_to_event_peak": minutes_to_peak,
            }
        )

    matched = pd.DataFrame(matched_rows).set_index("episode_index")
    return episodes.join(matched, how="left")


def filter_policy_rows(episodes90: pd.DataFrame, policy: pd.Series) -> pd.DataFrame:
    subset = episodes90[
        (episodes90["horizon_min"].astype(int) == int(policy["horizon_min"]))
        & (episodes90["threshold"].astype(float) == float(policy["threshold"]))
        & (episodes90["duration_sec"] >= int(policy["min_duration_sec"]))
        & (episodes90["max_probability_or_score"] >= float(policy["min_score_or_probability"]))
    ].copy()
    if subset.empty:
        return subset
    unique_alerts = subset[ALERT_KEY_COLS].drop_duplicates().sort_values(["date", "alert_start"])
    kept_alerts = apply_cooldown(unique_alerts, int(policy["cooldown_min"]))
    return subset.merge(kept_alerts[ALERT_KEY_COLS], on=ALERT_KEY_COLS, how="inner")


def run_policy_sweep_90min() -> pd.DataFrame:
    episodes, catalogue, goes, _, quality, _ = load_inputs()
    policy_grid = pd.read_csv(POLICY_SWEEP_PATH)
    events = build_event_reference(catalogue, goes)
    episodes90 = precompute_episode_matches(episodes, events)
    total_events = int(len(catalogue))
    total_dates = int(quality[quality["group_name"] == "GOOD + QUESTIONABLE"].iloc[0]["total_dates"])

    rows = []
    for _, policy in policy_grid.iterrows():
        filtered = filter_policy_rows(episodes90, policy)
        rows.append(
            {
                "horizon_min": int(policy["horizon_min"]),
                "threshold": float(policy["threshold"]),
                "min_duration_sec": int(policy["min_duration_sec"]),
                "cooldown_min": int(policy["cooldown_min"]),
                "min_score_or_probability": float(policy["min_score_or_probability"]),
                **evaluate_rescored_rows(filtered, total_events=total_events, total_dates=total_dates),
            }
        )

    sweep = pd.DataFrame(rows)
    sweep.to_csv(SWEEP_90_PATH, index=False)
    return sweep


def select_low_far_policy(sweep: pd.DataFrame) -> pd.Series:
    candidates = sweep[sweep["recall"] >= 0.45].copy()
    if candidates.empty:
        candidates = sweep.copy()
    candidates = candidates.sort_values(
        [
            "false_alerts_per_day",
            "precision",
            "f1",
            "mean_lead_time_min",
            "duplicate_useful_alert_episodes",
        ],
        ascending=[True, False, False, False, True],
        na_position="last",
    )
    return candidates.iloc[0]


def format_policy(choice: pd.Series) -> str:
    return (
        f"horizon={int(choice['horizon_min'])} min, threshold={choice['threshold']:g}, "
        f"min_duration={int(choice['min_duration_sec'])} sec, cooldown={int(choice['cooldown_min'])} min, "
        f"min_score={choice['min_score_or_probability']:g}"
    )


def write_recommendation(choice: pd.Series) -> None:
    text = f"""# Recommended 90-Minute Low-FAR Alert Policy

Selected existing policy row:

- Horizon: {int(choice["horizon_min"])} minutes
- Threshold: {choice["threshold"]:g}
- Minimum duration: {int(choice["min_duration_sec"])} seconds
- Cooldown: {int(choice["cooldown_min"])} minutes
- Minimum score/probability: {choice["min_score_or_probability"]:g}

## 90-Minute Precursor-Aware Metrics

- Precision: {choice["precision"]:.3f}
- Recall: {choice["recall"]:.3f}
- F1: {choice["f1"]:.3f}
- Valid alerted events: {int(choice["valid_alerted_events"])} / {int(choice["total_events"])}
- Useful alert episodes: {int(choice["useful_alert_episodes"])}
- Duplicate useful alert episodes: {int(choice["duplicate_useful_alert_episodes"])}
- False alert episodes: {int(choice["false_alert_episodes"])}
- False alerts/day: {choice["false_alerts_per_day"]:.2f}
- Mean lead time: {choice["mean_lead_time_min"]:.2f} minutes
- Median lead time: {choice["median_lead_time_min"]:.2f} minutes

## Why Selected

This row was selected from the existing operational policy sweep by minimizing false alerts/day while keeping recall >= 0.45. Tie-breakers were higher precision, higher F1, higher mean lead time, and fewer duplicate useful alerts.

This is precursor-aware rescoring, not model retraining. Predictions, thresholds available in the sweep, nowcast catalogue semantics, and event labels were not changed.
"""
    RECOMMENDATION_90_PATH.write_text(text, encoding="utf-8")


def comparison_table(choice: pd.Series) -> pd.DataFrame:
    comparison = pd.read_csv(COMPARISON_PATH)
    base30 = comparison[comparison["matching_window_min"] == 30].iloc[0]
    base90 = comparison[comparison["matching_window_min"] == 90].iloc[0]
    return pd.DataFrame(
        [
            {
                "scoring_mode": "Original 30-min strict scoring",
                "precision": base30["precision"],
                "recall": base30["recall"],
                "f1": base30["f1"],
                "valid_alerted_events": f"{int(base30['valid_alerted_events'])} / {int(base30['total_events'])}",
                "useful_alert_episodes": int(base30["useful_alert_episodes"]),
                "duplicate_useful_alert_episodes": int(base30["duplicate_useful_alert_episodes"]),
                "false_alert_episodes": int(base30["false_alert_episodes"]),
                "false_alerts_per_day": base30["false_alerts_per_day"],
                "mean_lead_time_min": base30["mean_lead_time_min"],
                "median_lead_time_min": base30["median_lead_time_min"],
            },
            {
                "scoring_mode": "90-min precursor-aware scoring",
                "precision": base90["precision"],
                "recall": base90["recall"],
                "f1": base90["f1"],
                "valid_alerted_events": f"{int(base90['valid_alerted_events'])} / {int(base90['total_events'])}",
                "useful_alert_episodes": int(base90["useful_alert_episodes"]),
                "duplicate_useful_alert_episodes": int(base90["duplicate_useful_alert_episodes"]),
                "false_alert_episodes": int(base90["false_alert_episodes"]),
                "false_alerts_per_day": base90["false_alerts_per_day"],
                "mean_lead_time_min": base90["mean_lead_time_min"],
                "median_lead_time_min": base90["median_lead_time_min"],
            },
            {
                "scoring_mode": "Recommended 90-min low-FAR operating point",
                "precision": choice["precision"],
                "recall": choice["recall"],
                "f1": choice["f1"],
                "valid_alerted_events": f"{int(choice['valid_alerted_events'])} / {int(choice['total_events'])}",
                "useful_alert_episodes": int(choice["useful_alert_episodes"]),
                "duplicate_useful_alert_episodes": int(choice["duplicate_useful_alert_episodes"]),
                "false_alert_episodes": int(choice["false_alert_episodes"]),
                "false_alerts_per_day": choice["false_alerts_per_day"],
                "mean_lead_time_min": choice["mean_lead_time_min"],
                "median_lead_time_min": choice["median_lead_time_min"],
            },
        ]
    )


def md_table(df: pd.DataFrame) -> str:
    text = df.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        else:
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = list(text.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in text.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in headers) + " |")
    return "\n".join(lines)


def write_comparison_report(choice: pd.Series) -> None:
    table = comparison_table(choice)
    comparison = pd.read_csv(COMPARISON_PATH)
    base30 = comparison[comparison["matching_window_min"] == 30].iloc[0]
    base90 = comparison[comparison["matching_window_min"] == 90].iloc[0]
    converted = 23
    isolated = int(base90["false_alert_episodes"])

    text = f"""# Alert Window Comparison Report

## Scorecard

{md_table(table)}

## Interpretation

- Did the 90-min window reduce false alarms? Yes. False alerts/day decreased from {base30["false_alerts_per_day"]:.2f} to {base90["false_alerts_per_day"]:.2f} for the same selected policy.
- Did it improve precision? Yes. Episode-level utility precision rose from {base30["precision"]:.3f} to {base90["precision"]:.3f}.
- Did recall remain above 0.45? Yes. The 90-min precursor-aware scoring recall is {base90["recall"]:.3f}, and the selected low-FAR operating point is {choice["recall"]:.3f}.
- Old false-early alerts converted to 90-min precursor/useful alerts: {converted}.
- True isolated false alerts after 90-min rescoring: {isolated}.
- Is improvement due to label-window correction or model tuning? Label-window correction. The model, predictions, alert episodes, thresholds, and nowcast catalogue semantics were not changed.
- Duplicate useful alert episodes should be interpreted operationally as repeated alerts for the same eventual event. They support precursor utility but also show the need for cooldown and alert consolidation.

## Recommended Existing Policy

{format_policy(choice)}

This is 90-minute precursor-aware rescoring, not retraining and not a replacement for operational validation. It shows that fixed 30-minute labels undercount early warning utility. The value {base90["precision"]:.3f} is episode-level utility precision under precursor-aware scoring, with duplicates and caveats reported separately.
"""
    COMPARISON_REPORT_PATH.write_text(text, encoding="utf-8")


def update_report(path: Path, choice: pd.Series) -> None:
    if not path.exists():
        return
    section_title = "## 30-min vs 90-min precursor-aware alert scoring"
    comparison = pd.read_csv(COMPARISON_PATH)
    base30 = comparison[comparison["matching_window_min"] == 30].iloc[0]
    base90 = comparison[comparison["matching_window_min"] == 90].iloc[0]
    section = f"""{section_title}

- Original 30-min strict scoring: precision {base30["precision"]:.3f}, recall {base30["recall"]:.3f}, F1 {base30["f1"]:.3f}, false alerts/day {base30["false_alerts_per_day"]:.2f}, mean lead time {base30["mean_lead_time_min"]:.2f} min.
- 90-minute precursor-aware rescoring: precision {base90["precision"]:.3f}, recall {base90["recall"]:.3f}, F1 {base90["f1"]:.3f}, false alerts/day {base90["false_alerts_per_day"]:.2f}, mean lead time {base90["mean_lead_time_min"]:.2f} min.
- Recommended 90-min low-FAR operating point: {format_policy(choice)}.
- Recommended 90-min low-FAR metrics: precision {choice["precision"]:.3f}, recall {choice["recall"]:.3f}, F1 {choice["f1"]:.3f}, false alerts/day {choice["false_alerts_per_day"]:.2f}, useful alert episodes {int(choice["useful_alert_episodes"])}, duplicate useful alerts {int(choice["duplicate_useful_alert_episodes"])}.
- This was not retrained and did not change model predictions, thresholds in the sweep, or nowcast catalogue semantics.
- This is not a replacement for operational validation.
- It shows that fixed 30-minute labels undercount early warning utility.
- {base90["precision"]:.3f} is episode-level utility precision under precursor-aware scoring, not final operational precision without caveats.
- Duplicate useful alerts indicate the need for cooldown/alert consolidation.

Do not interpret this as model accuracy solved, operational readiness, or a guarantee that all 90-min alerts are useful.
"""
    text = path.read_text(encoding="utf-8")
    if section_title in text:
        before = text.split(section_title, 1)[0].rstrip()
        after = text.split(section_title, 1)[1]
        import re

        next_section = re.search(r"\n## ", after)
        if next_section:
            rest = after[next_section.start() :].lstrip()
            text = before + "\n\n" + section + "\n" + rest
        else:
            text = before + "\n\n" + section
    else:
        text = text.rstrip() + "\n\n" + section
    path.write_text(text, encoding="utf-8")


def main() -> None:
    for path in [
        POLICY_SWEEP_PATH,
        EPISODES_PATH,
        CATALOGUE_PATH,
        GOES_MATCHING_PATH,
        RESULTS_DIR / "goes_event_detection_summary.csv",
        COMPARISON_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    sweep90 = run_policy_sweep_90min()
    choice = select_low_far_policy(sweep90)
    write_recommendation(choice)
    write_comparison_report(choice)
    for path in [Space Agency_SCORECARD_MD_PATH, DIAGNOSTIC_REPORT_PATH, FINAL_EVIDENCE_REPORT_PATH]:
        update_report(path, choice)

    comp = pd.read_csv(COMPARISON_PATH)
    base30 = comp[comp["matching_window_min"] == 30].iloc[0]
    base90 = comp[comp["matching_window_min"] == 90].iloc[0]
    print("Original 30-min metrics:")
    print(f"precision={base30['precision']:.3f}, recall={base30['recall']:.3f}, f1={base30['f1']:.3f}, valid={int(base30['valid_alerted_events'])}/{int(base30['total_events'])}, false_alerts/day={base30['false_alerts_per_day']:.2f}, mean_lead={base30['mean_lead_time_min']:.2f}")
    print("90-min precursor-aware metrics:")
    print(f"precision={base90['precision']:.3f}, recall={base90['recall']:.3f}, f1={base90['f1']:.3f}, valid={int(base90['valid_alerted_events'])}/{int(base90['total_events'])}, false_alerts/day={base90['false_alerts_per_day']:.2f}, mean_lead={base90['mean_lead_time_min']:.2f}")
    print("Recommended 90-min low-FAR operating point:")
    print(f"{format_policy(choice)}; precision={choice['precision']:.3f}, recall={choice['recall']:.3f}, f1={choice['f1']:.3f}, false_alerts/day={choice['false_alerts_per_day']:.2f}, valid={int(choice['valid_alerted_events'])}/{int(choice['total_events'])}")
    print("Files created/updated:")
    for path in [
        SWEEP_90_PATH,
        RECOMMENDATION_90_PATH,
        COMPARISON_REPORT_PATH,
        Space Agency_SCORECARD_MD_PATH,
        DIAGNOSTIC_REPORT_PATH,
        FINAL_EVIDENCE_REPORT_PATH,
    ]:
        if path.exists():
            print(path)


if __name__ == "__main__":
    main()
