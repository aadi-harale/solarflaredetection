from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.models.alert_policy import apply_policy, f1_score_from_precision_recall, load_alert_episodes


RESULTS_DIR = Path("results")
EPISODES_PATH = RESULTS_DIR / "combined_alert_episodes.csv"
CATALOGUE_PATH = RESULTS_DIR / "combined_nowcast_catalogue_clean.csv"
GOES_MATCHING_PATH = RESULTS_DIR / "goes_matching_report.csv"
FALSE_EARLY_GOES_PATH = RESULTS_DIR / "false_early_goes_followup_analysis.csv"
QUALITY_COMPARISON_PATH = RESULTS_DIR / "quality_group_metric_comparison.csv"
OPERATIONAL_METRICS_PATH = RESULTS_DIR / "operational_metrics_summary.csv"
RESCORED_PATH = RESULTS_DIR / "alert_90min_rescored_episodes.csv"
COMPARISON_PATH = RESULTS_DIR / "alert_window_comparison.csv"


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    for path in [
        EPISODES_PATH,
        CATALOGUE_PATH,
        GOES_MATCHING_PATH,
        FALSE_EARLY_GOES_PATH,
        QUALITY_COMPARISON_PATH,
        OPERATIONAL_METRICS_PATH,
    ]:
        require_file(path)

    episodes = load_alert_episodes(EPISODES_PATH)
    catalogue = pd.read_csv(CATALOGUE_PATH)
    goes = pd.read_csv(GOES_MATCHING_PATH)
    false_goes = pd.read_csv(FALSE_EARLY_GOES_PATH)
    quality = pd.read_csv(QUALITY_COMPARISON_PATH)
    policy = pd.read_csv(OPERATIONAL_METRICS_PATH).iloc[0]

    for col in ["event_start", "event_end", "soft_peak_time"]:
        catalogue[col] = pd.to_datetime(catalogue[col], utc=True, format="mixed", errors="coerce")
    catalogue["source_date"] = catalogue["source_date"].astype(str)
    catalogue["event_id"] = catalogue["event_id"].astype(str)
    catalogue["global_event_id"] = catalogue["global_event_id"].astype(str)

    for col in ["surya_event_start", "surya_event_end", "surya_soft_peak_time", "goes_start_time_utc", "goes_peak_time_utc", "goes_end_time_utc"]:
        goes[col] = pd.to_datetime(goes[col], utc=True, format="mixed", errors="coerce")
    goes["date"] = goes["date"].astype(str)
    goes["surya_event_id"] = goes["surya_event_id"].astype(str)

    for col in ["alert_start", "alert_end"]:
        false_goes[col] = pd.to_datetime(false_goes[col], utc=True, format="mixed", errors="coerce")

    return episodes, catalogue, goes, false_goes, quality, policy


def current_policy_filtered(episodes: pd.DataFrame, policy: pd.Series) -> pd.DataFrame:
    return apply_policy(
        episodes=episodes,
        horizon_min=int(policy["horizon_min"]),
        threshold=float(policy["threshold"]),
        min_duration_sec=int(policy["min_duration_sec"]),
        cooldown_min=int(policy["cooldown_min"]),
        min_score_or_probability=float(policy["min_score_or_probability"]),
    ).copy()


def build_event_reference(catalogue: pd.DataFrame, goes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    goes_by_surya = goes.set_index("surya_event_id", drop=False)
    for _, event in catalogue.iterrows():
        gid = str(event["global_event_id"])
        g = goes_by_surya.loc[gid] if gid in goes_by_surya.index else None
        use_goes = g is not None and str(g.get("goes_match_status", "")) in ["EXACT_PEAK_MATCH", "WINDOW_OVERLAP_MATCH"]

        rows.append(
            {
                "date": str(event["source_date"]),
                "quality_label": event.get("quality_label", ""),
                "surya_event_id": gid,
                "surya_local_event_id": str(event["event_id"]),
                "event_start": g["goes_start_time_utc"] if use_goes else event["event_start"],
                "event_peak": g["goes_peak_time_utc"] if use_goes else event["soft_peak_time"],
                "event_end": g["goes_end_time_utc"] if use_goes else event["event_end"],
                "goes_event_id": g.get("goes_event_id", "") if g is not None else "",
                "goes_class": g.get("goes_class", "") if g is not None else "",
                "goes_class_group": g.get("goes_class_group", "") if g is not None else "",
                "goes_match_status": g.get("goes_match_status", "") if g is not None else "",
                "event_source": "GOES" if use_goes else "SuryaAlert",
            }
        )
    return pd.DataFrame(rows)


def choose_90min_match(alert: pd.Series, events_for_date: pd.DataFrame) -> pd.Series | None:
    if events_for_date.empty or pd.isna(alert["alert_start"]):
        return None

    candidates = events_for_date.copy()
    candidates["minutes_to_event_peak"] = (candidates["event_peak"] - alert["alert_start"]).dt.total_seconds() / 60
    candidates["minutes_to_event_start"] = (candidates["event_start"] - alert["alert_start"]).dt.total_seconds() / 60
    overlaps = (alert["alert_start"] <= candidates["event_end"]) & (alert["alert_end"] >= candidates["event_start"])
    precedes_peak = candidates["minutes_to_event_peak"].between(0, 90, inclusive="both")
    precedes_start = candidates["minutes_to_event_start"].between(0, 90, inclusive="both")
    candidates = candidates[overlaps | precedes_peak | precedes_start].copy()
    if candidates.empty:
        return None

    candidates["is_goes_preferred"] = candidates["event_source"].eq("GOES")
    candidates["is_future_precursor"] = (candidates["minutes_to_event_peak"] >= 0) | (candidates["minutes_to_event_start"] >= 0)
    candidates["abs_peak_minutes"] = candidates["minutes_to_event_peak"].abs()
    candidates["overlaps"] = overlaps.loc[candidates.index]
    candidates = candidates.sort_values(
        ["is_goes_preferred", "is_future_precursor", "minutes_to_event_start", "overlaps", "abs_peak_minutes"],
        ascending=[False, False, True, False, True],
    )
    return candidates.iloc[0]


def rescore_90min(filtered: pd.DataFrame, events: pd.DataFrame, output_path: Path | None = RESCORED_PATH) -> pd.DataFrame:
    columns = [
        "date",
        "quality_label",
        "alert_start",
        "alert_end",
        "original_episode_label",
        "rescored_90min_label",
        "matched_surya_event_id",
        "matched_goes_event_id",
        "matched_goes_class",
        "matched_goes_class_group",
        "minutes_to_event_start",
        "minutes_to_event_peak",
        "is_first_alert_for_event",
        "is_duplicate_alert_for_event",
        "notes",
    ]
    rows = []
    first_alert_for_event: set[str] = set()
    events_by_date = {date: group.copy() for date, group in events.groupby("date")}

    for _, alert in filtered.sort_values(["date", "alert_start", "alert_end"]).iterrows():
        date = str(alert["date"])
        match = choose_90min_match(alert, events_by_date.get(date, events.iloc[0:0]))
        original_label = str(alert.get("episode_type", ""))

        if match is None:
            label = "AMBIGUOUS_QUESTIONABLE_DATA" if original_label == "FALSE_EARLY_ALERT" and date in set(events.loc[events["quality_label"].eq("QUESTIONABLE"), "date"]) else "TRUE_ISOLATED_FALSE_ALERT"
            rows.append(
                {
                    "date": date,
                    "quality_label": "",
                    "alert_start": alert["alert_start"],
                    "alert_end": alert["alert_end"],
                    "original_episode_label": original_label,
                    "rescored_90min_label": label,
                    "matched_surya_event_id": "",
                    "matched_goes_event_id": "",
                    "matched_goes_class": "",
                    "matched_goes_class_group": "",
                    "minutes_to_event_start": np.nan,
                    "minutes_to_event_peak": np.nan,
                    "is_first_alert_for_event": False,
                    "is_duplicate_alert_for_event": False,
                    "notes": "No GOES/SuryaAlert event found within 90 minutes after alert start or overlapping the alert.",
                }
            )
            continue

        event_id = str(match["surya_event_id"])
        minutes_to_start = float(match["minutes_to_event_start"])
        minutes_to_peak = float(match["minutes_to_event_peak"])
        overlaps_event = bool((alert["alert_start"] <= match["event_end"]) and (alert["alert_end"] >= match["event_start"]))
        is_late = minutes_to_peak < 0 and overlaps_event
        is_first = event_id not in first_alert_for_event
        is_duplicate = not is_first

        if is_late:
            label = "LATE_OR_OVERLAP_ALERT"
            note = "Alert overlaps a matched event after the event peak; counted separately from precursor alerts."
        elif is_duplicate:
            label = "DUPLICATE_ALERT_FOR_SAME_EVENT"
            note = "Useful under 90-minute window but not an additional event-level true positive."
        else:
            label = "VALID_90MIN_PRECURSOR_ALERT"
            note = f"First alert for matched event using {match['event_source']}-preferred timing."
            first_alert_for_event.add(event_id)

        rows.append(
            {
                "date": date,
                "quality_label": match["quality_label"],
                "alert_start": alert["alert_start"],
                "alert_end": alert["alert_end"],
                "original_episode_label": original_label,
                "rescored_90min_label": label,
                "matched_surya_event_id": event_id,
                "matched_goes_event_id": match.get("goes_event_id", ""),
                "matched_goes_class": match.get("goes_class", ""),
                "matched_goes_class_group": match.get("goes_class_group", ""),
                "minutes_to_event_start": minutes_to_start,
                "minutes_to_event_peak": minutes_to_peak,
                "is_first_alert_for_event": is_first and label == "VALID_90MIN_PRECURSOR_ALERT",
                "is_duplicate_alert_for_event": is_duplicate and label == "DUPLICATE_ALERT_FOR_SAME_EVENT",
                "notes": note,
            }
        )

    rescored = pd.DataFrame(rows, columns=columns)
    if output_path is not None:
        rescored.to_csv(output_path, index=False)
    return rescored


def original_30min_row(quality: pd.DataFrame) -> dict:
    row = quality[quality["group_name"] == "GOOD + QUESTIONABLE"].iloc[0]
    precision = float(row["event_level_precision"])
    recall = float(row["event_level_recall"])
    return {
        "matching_window_min": 30,
        "precision": precision,
        "recall": recall,
        "f1": float(row["event_level_f1"]),
        "valid_alerted_events": int(row["valid_alerted_events"]),
        "total_events": int(row["total_events"]),
        "total_alert_episodes_after_policy": int(row["total_alert_episodes_after_policy"]),
        "useful_alert_episodes": int(row["useful_alert_episodes_after_policy"]),
        "duplicate_useful_alert_episodes": 0,
        "false_alert_episodes": int(row["false_early_alert_episodes_after_policy"]),
        "false_alerts_per_day": float(row["false_alerts_per_day"]),
        "mean_lead_time_min": float(row["mean_lead_time_min"]),
        "median_lead_time_min": float(row["median_lead_time_min"]),
        "notes": "Original event-level operational policy metrics using 30-minute scoring window.",
    }


def rescored_90min_row(rescored: pd.DataFrame, total_events: int, total_dates: int) -> dict:
    valid = rescored[rescored["rescored_90min_label"] == "VALID_90MIN_PRECURSOR_ALERT"].copy()
    duplicate = rescored[rescored["rescored_90min_label"] == "DUPLICATE_ALERT_FOR_SAME_EVENT"].copy()
    late_overlap = rescored[rescored["rescored_90min_label"] == "LATE_OR_OVERLAP_ALERT"].copy()
    useful = pd.concat([valid, duplicate, late_overlap], ignore_index=True)
    false = rescored[rescored["rescored_90min_label"] == "TRUE_ISOLATED_FALSE_ALERT"]

    valid_events = int(valid["matched_surya_event_id"].nunique())
    useful_count = int(len(useful))
    total_alerts = int(len(rescored))
    precision = useful_count / total_alerts if total_alerts else 0.0
    recall = valid_events / total_events if total_events else 0.0

    lead = pd.to_numeric(valid["minutes_to_event_peak"], errors="coerce")
    return {
        "matching_window_min": 90,
        "precision": precision,
        "recall": recall,
        "f1": f1_score_from_precision_recall(precision, recall),
        "valid_alerted_events": valid_events,
        "total_events": total_events,
        "total_alert_episodes_after_policy": total_alerts,
        "useful_alert_episodes": useful_count,
        "duplicate_useful_alert_episodes": int(len(duplicate)),
        "false_alert_episodes": int(len(false)),
        "false_alerts_per_day": len(false) / total_dates if total_dates else np.nan,
        "mean_lead_time_min": float(lead.mean()) if not lead.empty else np.nan,
        "median_lead_time_min": float(lead.median()) if not lead.empty else np.nan,
        "notes": "Same alerts rescored with 90-minute precursor-aware event matching; no predictions or thresholds changed.",
    }


def main() -> None:
    episodes, catalogue, goes, false_goes, quality, policy = load_inputs()
    filtered = current_policy_filtered(episodes, policy)
    events = build_event_reference(catalogue, goes)
    rescored = rescore_90min(filtered, events)

    baseline = original_30min_row(quality)
    total_dates = int(quality[quality["group_name"] == "GOOD + QUESTIONABLE"].iloc[0]["total_dates"])
    ninety = rescored_90min_row(rescored, total_events=int(len(catalogue)), total_dates=total_dates)
    comparison = pd.DataFrame([baseline, ninety])
    comparison.to_csv(COMPARISON_PATH, index=False)

    converted = int(
        (
            rescored["original_episode_label"].eq("FALSE_EARLY_ALERT")
            & rescored["rescored_90min_label"].isin(["VALID_90MIN_PRECURSOR_ALERT", "DUPLICATE_ALERT_FOR_SAME_EVENT"])
        ).sum()
    )
    true_false = int((rescored["rescored_90min_label"] == "TRUE_ISOLATED_FALSE_ALERT").sum())

    print("Original 30-min metrics:")
    print(
        f"precision={baseline['precision']:.3f}, recall={baseline['recall']:.3f}, "
        f"f1={baseline['f1']:.3f}, valid_events={baseline['valid_alerted_events']}/{baseline['total_events']}, "
        f"alerts={baseline['total_alert_episodes_after_policy']}, useful={baseline['useful_alert_episodes']}, "
        f"false_early={baseline['false_alert_episodes']}, false_alerts/day={baseline['false_alerts_per_day']:.2f}, "
        f"mean_lead={baseline['mean_lead_time_min']:.2f} min, median_lead={baseline['median_lead_time_min']:.2f} min"
    )
    print("New 90-min metrics:")
    print(
        f"precision={ninety['precision']:.3f}, recall={ninety['recall']:.3f}, "
        f"f1={ninety['f1']:.3f}, valid_events={ninety['valid_alerted_events']}/{ninety['total_events']}, "
        f"alerts={ninety['total_alert_episodes_after_policy']}, useful={ninety['useful_alert_episodes']}, "
        f"duplicate_useful={ninety['duplicate_useful_alert_episodes']}, true_isolated_false={ninety['false_alert_episodes']}, "
        f"false_alerts/day={ninety['false_alerts_per_day']:.2f}, mean_lead={ninety['mean_lead_time_min']:.2f} min, "
        f"median_lead={ninety['median_lead_time_min']:.2f} min"
    )
    print(f"Old false-early alerts converted to 90-min precursor/useful alerts: {converted}")
    print(f"True isolated false alerts: {true_false}")
    print("Files created:")
    print(RESCORED_PATH)
    print(COMPARISON_PATH)


if __name__ == "__main__":
    main()
