from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


EPISODES_PATH = Path("results/combined_alert_episodes.csv")
SWEEP_PATH = Path("results/operational_alert_policy_sweep.csv")
RECOMMENDATION_PATH = Path("results/recommended_alert_policy.md")
METRICS_SUMMARY_PATH = Path("results/operational_metrics_summary.csv")
COMBINED_FORECAST_DATASET_PATH = Path("data/processed/combined_forecast_dataset.csv")

MIN_DURATION_SEC = [5, 10, 20, 30, 60]
COOLDOWN_MIN = [0, 5, 10, 15, 30]
PROBABILITY_THRESHOLDS = [0.3, 0.5, 0.7, 0.9]
HARD_SCORE_THRESHOLDS = [4, 6, 8, 10, 12]

ALERT_KEY_COLS = [
    "date",
    "horizon_min",
    "threshold",
    "alert_start",
    "alert_end",
    "duration_sec",
    "max_probability_or_score",
]


def load_alert_episodes(path: Path = EPISODES_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing alert episodes file: {path.resolve()}. Run scripts/08_threshold_sweep.py first."
        )

    episodes = pd.read_csv(path)
    required = {
        "date",
        "heldout_event_id",
        "horizon_min",
        "threshold",
        "alert_start",
        "alert_end",
        "duration_sec",
        "max_probability_or_score",
        "episode_type",
        "lead_time_min",
    }
    missing = required - set(episodes.columns)
    if missing:
        raise ValueError(f"Alert episode file is missing required columns: {sorted(missing)}")

    episodes = episodes.copy()
    episodes["date"] = episodes["date"].astype(str)
    for col in ["alert_start", "alert_end", "valid_window_start", "valid_window_end", "soft_peak_time"]:
        if col in episodes.columns:
            episodes[col] = pd.to_datetime(episodes[col], utc=True, format="mixed", errors="coerce")

    episodes["duration_sec"] = pd.to_numeric(episodes["duration_sec"], errors="coerce")
    episodes["max_probability_or_score"] = pd.to_numeric(episodes["max_probability_or_score"], errors="coerce")
    episodes["horizon_min"] = pd.to_numeric(episodes["horizon_min"], errors="coerce").astype("Int64")
    episodes["threshold"] = pd.to_numeric(episodes["threshold"], errors="coerce")
    episodes["heldout_event_id"] = pd.to_numeric(episodes["heldout_event_id"], errors="coerce").astype("Int64")
    episodes = episodes.dropna(subset=["alert_start", "alert_end", "duration_sec", "max_probability_or_score"])
    return episodes


def score_thresholds_for(episodes: pd.DataFrame) -> list[float]:
    scores = episodes["max_probability_or_score"].dropna()
    if not scores.empty and scores.between(0, 1).all():
        return PROBABILITY_THRESHOLDS
    return HARD_SCORE_THRESHOLDS


def apply_cooldown(unique_alerts: pd.DataFrame, cooldown_min: int) -> pd.DataFrame:
    if unique_alerts.empty:
        return unique_alerts.copy()

    kept_groups = []
    cooldown = pd.Timedelta(minutes=cooldown_min)
    for _, group in unique_alerts.sort_values("alert_start").groupby("date", sort=False):
        kept_indices = []
        previous_kept_start = None
        for idx, row in group.iterrows():
            alert_start = row["alert_start"]
            if previous_kept_start is None or alert_start > previous_kept_start + cooldown:
                kept_indices.append(idx)
                previous_kept_start = alert_start
        kept_groups.append(group.loc[kept_indices])

    if not kept_groups:
        return unique_alerts.iloc[0:0].copy()
    return pd.concat(kept_groups, ignore_index=True)


def apply_policy(
    episodes: pd.DataFrame,
    horizon_min: int,
    threshold: float,
    min_duration_sec: int,
    cooldown_min: int,
    min_score_or_probability: float,
) -> pd.DataFrame:
    subset = episodes[
        (episodes["horizon_min"].astype(int) == horizon_min)
        & (episodes["threshold"].astype(float) == float(threshold))
        & (episodes["duration_sec"] >= min_duration_sec)
        & (episodes["max_probability_or_score"] >= min_score_or_probability)
    ].copy()
    if subset.empty:
        return subset

    unique_alerts = subset[ALERT_KEY_COLS].drop_duplicates().sort_values(["date", "alert_start"])
    kept_alerts = apply_cooldown(unique_alerts, cooldown_min)
    return subset.merge(kept_alerts[ALERT_KEY_COLS], on=ALERT_KEY_COLS, how="inner")


def merge_timeseries_alerts(alerts: pd.DataFrame, score_col: str = "hard_score") -> pd.DataFrame:
    columns = ["date", "alert_start", "alert_end", "duration_sec", "max_probability_or_score"]
    if alerts.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for date, group in alerts.sort_values("time_utc").groupby("source_date", sort=False):
        start = group.iloc[0]["time_utc"]
        end = start
        max_score = float(group.iloc[0][score_col])

        for _, row in group.iloc[1:].iterrows():
            timestamp = row["time_utc"]
            gap_sec = (timestamp - end).total_seconds()
            if gap_sec <= 60:
                end = timestamp
                max_score = max(max_score, float(row[score_col]))
                continue

            rows.append(
                {
                    "date": date,
                    "alert_start": start,
                    "alert_end": end,
                    "duration_sec": (end - start).total_seconds(),
                    "max_probability_or_score": max_score,
                }
            )
            start = timestamp
            end = timestamp
            max_score = float(row[score_col])

        rows.append(
            {
                "date": date,
                "alert_start": start,
                "alert_end": end,
                "duration_sec": (end - start).total_seconds(),
                "max_probability_or_score": max_score,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def f1_score_from_precision_recall(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def evaluate_policy(filtered: pd.DataFrame, total_heldout_events: int) -> dict:
    total_alerts = len(filtered)
    useful_mask = filtered["episode_type"].isin(["TRUE_VALID_ALERT", "OVERLAP_ALERT"])
    useful = int(useful_mask.sum())
    false_early = int((filtered["episode_type"] == "FALSE_EARLY_ALERT").sum())
    late = int((filtered["episode_type"] == "LATE_ALERT").sum())
    overlap = int((filtered["episode_type"] == "OVERLAP_ALERT").sum())

    useful_rows = filtered[useful_mask].sort_values("alert_start")
    valid_alerted_events = int(useful_rows["heldout_event_id"].nunique())
    precision = useful / total_alerts if total_alerts else 0.0
    recall = valid_alerted_events / total_heldout_events if total_heldout_events else 0.0

    first_useful = useful_rows.drop_duplicates("heldout_event_id")
    mean_lead = float(first_useful["lead_time_min"].mean()) if not first_useful.empty else np.nan

    return {
        "total_alert_episodes_after_policy": total_alerts,
        "useful_alert_episodes_after_policy": useful,
        "false_early_alert_episodes_after_policy": false_early,
        "late_alert_episodes_after_policy": late,
        "overlap_alert_episodes_after_policy": overlap,
        "valid_alerted_events": valid_alerted_events,
        "total_heldout_events": total_heldout_events,
        "event_level_precision": precision,
        "event_level_recall": recall,
        "event_level_f1": f1_score_from_precision_recall(precision, recall),
        "mean_valid_lead_time_min": mean_lead,
    }


def quiet_day_false_alerts(choice: pd.Series) -> int | float:
    if not COMBINED_FORECAST_DATASET_PATH.exists():
        return np.nan

    df = pd.read_csv(COMBINED_FORECAST_DATASET_PATH)
    required = {"source_date", "time_utc", "hard_score", "is_quiet_day"}
    if required - set(df.columns):
        return np.nan

    df = df.copy()
    df["source_date"] = df["source_date"].astype(str)
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, format="mixed", errors="coerce")
    df["hard_score"] = pd.to_numeric(df["hard_score"], errors="coerce").fillna(0)
    quiet_mask = df["is_quiet_day"].astype(str).str.lower().isin(["true", "1", "yes"])
    quiet = df[quiet_mask & df["time_utc"].notna()]
    if quiet.empty:
        return 0

    positives = quiet[quiet["hard_score"] >= float(choice["threshold"])]
    episodes = merge_timeseries_alerts(positives)
    if episodes.empty:
        return 0

    episodes = episodes[
        (episodes["duration_sec"] >= int(choice["min_duration_sec"]))
        & (episodes["max_probability_or_score"] >= float(choice["min_score_or_probability"]))
    ]
    kept = apply_cooldown(episodes, int(choice["cooldown_min"]))
    return int(len(kept))


def write_operational_metrics_summary(choice: pd.Series, path: Path = METRICS_SUMMARY_PATH) -> Path:
    total_eval_dates = np.nan
    good_dates = np.nan
    questionable_dates = np.nan
    quiet_days = np.nan

    if COMBINED_FORECAST_DATASET_PATH.exists():
        df = pd.read_csv(COMBINED_FORECAST_DATASET_PATH, usecols=lambda col: col in {"source_date", "quality_label", "is_quiet_day"})
        if not df.empty and "source_date" in df.columns:
            dates = df.drop_duplicates("source_date")
            total_eval_dates = int(len(dates))
            if "quality_label" in dates.columns:
                good_dates = int((dates["quality_label"].astype(str) == "GOOD").sum())
                questionable_dates = int((dates["quality_label"].astype(str) == "QUESTIONABLE").sum())
            if "is_quiet_day" in dates.columns:
                quiet_days = int(dates["is_quiet_day"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())

    false_early = int(choice["false_early_alert_episodes_after_policy"])
    false_alerts_per_day = false_early / total_eval_dates if pd.notna(total_eval_dates) and total_eval_dates else np.nan
    quiet_false_alerts = quiet_day_false_alerts(choice)

    summary = pd.DataFrame(
        [
            {
                "horizon_min": int(choice["horizon_min"]),
                "threshold": float(choice["threshold"]),
                "min_duration_sec": int(choice["min_duration_sec"]),
                "cooldown_min": int(choice["cooldown_min"]),
                "min_score_or_probability": float(choice["min_score_or_probability"]),
                "total_eval_dates": total_eval_dates,
                "good_dates": good_dates,
                "questionable_dates": questionable_dates,
                "quiet_days": quiet_days,
                "event_level_precision": float(choice["event_level_precision"]),
                "event_level_recall": float(choice["event_level_recall"]),
                "event_level_f1": float(choice["event_level_f1"]),
                "valid_alerted_events": int(choice["valid_alerted_events"]),
                "total_heldout_events": int(choice["total_heldout_events"]),
                "total_alert_episodes_after_policy": int(choice["total_alert_episodes_after_policy"]),
                "useful_alert_episodes_after_policy": int(choice["useful_alert_episodes_after_policy"]),
                "false_early_alert_episodes_after_policy": false_early,
                "false_alerts_per_day": false_alerts_per_day,
                "quiet_day_false_alert_episodes": quiet_false_alerts,
                "mean_valid_lead_time_min": float(choice["mean_valid_lead_time_min"]),
            }
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(path, index=False)
    return path


def run_policy_sweep(episodes: pd.DataFrame) -> pd.DataFrame:
    total_heldout_events = int(episodes["heldout_event_id"].nunique())
    min_score_values = score_thresholds_for(episodes)
    rows = []

    base_pairs = episodes[["horizon_min", "threshold"]].drop_duplicates().sort_values(["horizon_min", "threshold"])
    for _, pair in base_pairs.iterrows():
        horizon_min = int(pair["horizon_min"])
        threshold = float(pair["threshold"])
        for min_duration_sec, cooldown_min, min_score in product(
            MIN_DURATION_SEC, COOLDOWN_MIN, min_score_values
        ):
            filtered = apply_policy(
                episodes=episodes,
                horizon_min=horizon_min,
                threshold=threshold,
                min_duration_sec=min_duration_sec,
                cooldown_min=cooldown_min,
                min_score_or_probability=min_score,
            )
            rows.append(
                {
                    "horizon_min": horizon_min,
                    "threshold": threshold,
                    "min_duration_sec": min_duration_sec,
                    "cooldown_min": cooldown_min,
                    "min_score_or_probability": min_score,
                    **evaluate_policy(filtered, total_heldout_events),
                }
            )

    return pd.DataFrame(rows)


def select_recommended_policy(summary: pd.DataFrame) -> pd.Series:
    candidates = summary.copy()
    candidates = candidates.sort_values(
        by=[
            "event_level_f1",
            "event_level_precision",
            "false_early_alert_episodes_after_policy",
            "valid_alerted_events",
            "mean_valid_lead_time_min",
        ],
        ascending=[False, False, True, False, False],
        na_position="last",
    )
    return candidates.iloc[0]


def write_recommendation(choice: pd.Series, path: Path = RECOMMENDATION_PATH) -> Path:
    text = f"""# Recommended Operational Alert Policy

Recommended post-processing policy:

- Horizon: {int(choice["horizon_min"])} minutes
- Hard-score threshold: {choice["threshold"]:g}
- Minimum episode duration: {int(choice["min_duration_sec"])} seconds
- Cooldown: {int(choice["cooldown_min"])} minutes
- Minimum score/probability: {choice["min_score_or_probability"]:g}
- Event-level precision: {choice["event_level_precision"]:.3f}
- Event-level recall: {choice["event_level_recall"]:.3f}
- Event-level F1: {choice["event_level_f1"]:.3f}
- Total alert episodes after policy: {int(choice["total_alert_episodes_after_policy"])}
- Useful alert episodes after policy: {int(choice["useful_alert_episodes_after_policy"])}
- False early alert episodes after policy: {int(choice["false_early_alert_episodes_after_policy"])}
- Valid alerted events: {int(choice["valid_alerted_events"])} / {int(choice["total_heldout_events"])}
- Mean valid lead time: {choice["mean_valid_lead_time_min"]:.2f} minutes

This is an operational post-processing policy layered on top of the existing alert episodes. It reduces noisy repeated alerts using duration, score, and cooldown rules; it does not change the model, quality gate, nowcast catalogue, or event labels.

The recommendation is selected by event-level F1, then event-level precision, fewer false early alert episodes, valid alerted events, and useful mean lead time. Weak metrics are intentionally reported as-is.

This does not prove scientific forecasting reliability yet. The dataset is still small, QUESTIONABLE dates are included but marked in the evaluation summary, and BROKEN dates are excluded.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    episodes = load_alert_episodes()
    summary = run_policy_sweep(episodes)
    SWEEP_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SWEEP_PATH, index=False)
    choice = select_recommended_policy(summary)
    recommendation_path = write_recommendation(choice)
    metrics_path = write_operational_metrics_summary(choice)

    print("Operational alert policy sweep complete.")
    print(f"Saved: {SWEEP_PATH}")
    print(f"Saved: {recommendation_path}")
    print(f"Saved: {metrics_path}")
    print("\nRecommended policy:")
    print(choice.to_string())
