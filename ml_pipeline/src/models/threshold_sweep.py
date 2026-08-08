from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from src.utils.quality_filter import apply_quality_gate_to_catalogue, apply_quality_gate_to_forecast_df


COMBINED_PREDICTIONS_PATH = Path("results/combined_baseline_predictions.csv")
PREDICTIONS_PATH = Path("results/june03_baseline_predictions.csv")
COMBINED_FORECAST_DATASET_PATH = Path("data/processed/combined_forecast_dataset.csv")
FORECAST_DATASET_PATH = Path("data/processed/june03_forecast_dataset.csv")
COMBINED_CLEAN_CATALOGUE_PATH = Path("results/combined_nowcast_catalogue_clean.csv")
CLEAN_CATALOGUE_PATH = Path("results/june03_nowcast_catalogue_clean.csv")
SUMMARY_PATH = Path("results/threshold_sweep_summary.csv")
RECOMMENDATION_PATH = Path("results/recommended_operating_point.md")
ALERT_EPISODES_PATH = Path("results/combined_alert_episodes.csv")

HORIZONS_MIN = [5, 10, 30]
HARD_SCORE_THRESHOLDS = [4, 6, 8, 10, 12, 15, 20, 30, 40, 50, 75, 100, 200, 500, 1000]


def resolve_inputs() -> tuple[Path, Path, Path, str]:
    if (
        COMBINED_PREDICTIONS_PATH.exists()
        and COMBINED_FORECAST_DATASET_PATH.exists()
        and COMBINED_CLEAN_CATALOGUE_PATH.exists()
    ):
        return (
            COMBINED_PREDICTIONS_PATH,
            COMBINED_FORECAST_DATASET_PATH,
            COMBINED_CLEAN_CATALOGUE_PATH,
            "combined",
        )
    return PREDICTIONS_PATH, FORECAST_DATASET_PATH, CLEAN_CATALOGUE_PATH, "june03"


def write_probability_sweep(predictions_path: Path, output_prefix: str) -> pd.DataFrame:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing baseline predictions: {predictions_path.resolve()}")

    pred_df = pd.read_csv(predictions_path)
    required = {"y_true", "flare_probability"}
    missing = required - set(pred_df.columns)
    if missing:
        raise ValueError(f"Missing required prediction columns: {sorted(missing)}")

    y_true = pred_df["y_true"].astype(int)
    prob = pred_df["flare_probability"].astype(float)

    rows = []
    for threshold in [i / 100 for i in range(5, 100, 5)]:
        y_pred = (prob >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "alerts": int(y_pred.sum()),
            }
        )

    results = pd.DataFrame(rows)
    out_path = Path("results") / f"{output_prefix}_threshold_sweep.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)

    print("Threshold sweep:")
    print(results.to_string(index=False))
    print(f"\nSaved: {out_path}")
    return results


def _load_forecast_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing forecast dataset: {path.resolve()}")

    df = pd.read_csv(path)
    if "time_utc" not in df.columns:
        df = df.rename(columns={df.columns[0]: "time_utc"})

    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, format="mixed", errors="coerce")
    df = df.dropna(subset=["time_utc"]).sort_values("time_utc")
    df = apply_quality_gate_to_forecast_df(df)

    required = {"hard_score", "inside_detected_event"} | {f"flare_next_{h}min" for h in HORIZONS_MIN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required forecast dataset columns: {sorted(missing)}")

    return df


def _load_catalogue(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing clean nowcast catalogue: {path.resolve()}")

    catalogue = pd.read_csv(path)
    if "soft_peak_time" not in catalogue.columns:
        raise ValueError("Missing required clean catalogue column: soft_peak_time")

    catalogue = apply_quality_gate_to_catalogue(catalogue)
    catalogue = catalogue.copy()
    catalogue["soft_peak_time"] = pd.to_datetime(
        catalogue["soft_peak_time"], utc=True, format="mixed", errors="coerce"
    )
    catalogue = catalogue.dropna(subset=["soft_peak_time"]).sort_values("soft_peak_time")
    if "source_date" not in catalogue.columns:
        catalogue["source_date"] = catalogue["soft_peak_time"].dt.strftime("%Y%m%d")
    return catalogue


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _merge_alert_episodes(alerts: pd.DataFrame, score_col: str = "hard_score") -> list[dict]:
    if alerts.empty:
        return []

    alerts = alerts.sort_values("time_utc")
    episodes = []
    start = alerts.iloc[0]["time_utc"]
    end = start
    max_score = float(alerts.iloc[0][score_col])

    for _, row in alerts.iloc[1:].iterrows():
        timestamp = row["time_utc"]
        gap_sec = (timestamp - end).total_seconds()
        if gap_sec <= 60:
            end = timestamp
            max_score = max(max_score, float(row[score_col]))
            continue

        episodes.append({"alert_start": start, "alert_end": end, "max_probability_or_score": max_score})
        start = timestamp
        end = timestamp
        max_score = float(row[score_col])

    episodes.append({"alert_start": start, "alert_end": end, "max_probability_or_score": max_score})
    return episodes


def _classify_episode(alert_start: pd.Timestamp, alert_end: pd.Timestamp, valid_start: pd.Timestamp, valid_end: pd.Timestamp) -> str:
    if valid_start <= alert_start <= valid_end:
        return "TRUE_VALID_ALERT"
    if alert_start < valid_start and alert_end >= valid_start:
        return "OVERLAP_ALERT"
    if alert_start < valid_start:
        return "FALSE_EARLY_ALERT"
    return "LATE_ALERT"


def _episode_rows_for_event(
    event_row: pd.Series,
    event_df: pd.DataFrame,
    horizon_min: int,
    threshold: int,
) -> list[dict]:
    peak_time = event_row["soft_peak_time"]
    valid_start = peak_time - pd.Timedelta(minutes=horizon_min)
    valid_end = peak_time
    event_id = event_row.get("global_event_id", event_row.get("event_id"))
    date = str(event_row.get("source_date", peak_time.strftime("%Y%m%d")))

    positives = event_df[event_df["hard_score"] >= threshold]
    rows = []
    for episode in _merge_alert_episodes(positives):
        alert_start = episode["alert_start"]
        alert_end = episode["alert_end"]
        episode_type = _classify_episode(alert_start, alert_end, valid_start, valid_end)
        if episode_type in {"TRUE_VALID_ALERT", "OVERLAP_ALERT"}:
            lead_time_min = (peak_time - alert_start).total_seconds() / 60
        else:
            lead_time_min = np.nan

        rows.append(
            {
                "date": date,
                "heldout_event_id": int(event_id),
                "horizon_min": horizon_min,
                "threshold": threshold,
                "alert_start": alert_start,
                "alert_end": alert_end,
                "duration_sec": (alert_end - alert_start).total_seconds(),
                "max_probability_or_score": episode["max_probability_or_score"],
                "valid_window_start": valid_start,
                "valid_window_end": valid_end,
                "soft_peak_time": peak_time,
                "episode_type": episode_type,
                "lead_time_min": lead_time_min,
            }
        )
    return rows


def _event_level_metrics(episodes: pd.DataFrame, catalogue: pd.DataFrame, horizon_min: int, threshold: int) -> dict:
    subset = episodes[(episodes["horizon_min"] == horizon_min) & (episodes["threshold"] == threshold)]
    total_alert_episodes = len(subset)
    true_valid = int((subset["episode_type"] == "TRUE_VALID_ALERT").sum())
    overlap = int((subset["episode_type"] == "OVERLAP_ALERT").sum())
    false_early = int((subset["episode_type"] == "FALSE_EARLY_ALERT").sum())
    late = int((subset["episode_type"] == "LATE_ALERT").sum())
    useful = true_valid + overlap
    useful_subset = subset[subset["episode_type"].isin(["TRUE_VALID_ALERT", "OVERLAP_ALERT"])]
    valid_alerted_events = int(useful_subset["heldout_event_id"].nunique())
    total_heldout_events = len(catalogue)

    event_precision = useful / total_alert_episodes if total_alert_episodes else 0.0
    event_recall = valid_alerted_events / total_heldout_events if total_heldout_events else 0.0

    first_useful = useful_subset.sort_values("alert_start").drop_duplicates("heldout_event_id")
    mean_lead = float(first_useful["lead_time_min"].mean()) if not first_useful.empty else np.nan

    return {
        "event_level_precision": event_precision,
        "event_level_recall": event_recall,
        "event_level_f1": _f1(event_precision, event_recall),
        "total_alert_episodes": total_alert_episodes,
        "useful_alert_episodes": useful,
        "true_valid_alert_episodes": true_valid,
        "false_early_alert_episodes": false_early,
        "overlap_alert_episodes": overlap,
        "late_alert_episodes": late,
        "valid_alerted_events": valid_alerted_events,
        "total_heldout_events": total_heldout_events,
        "mean_valid_lead_time_min": mean_lead,
    }


def write_summary_sweep(forecast_dataset_path: Path, clean_catalogue_path: Path, output_prefix: str) -> pd.DataFrame:
    df = _load_forecast_data(forecast_dataset_path)
    catalogue = _load_catalogue(clean_catalogue_path)

    eval_df = df[df["inside_detected_event"] == 0].copy()
    eval_df["hard_score"] = pd.to_numeric(eval_df["hard_score"], errors="coerce").fillna(0)

    episode_rows = []
    rows = []
    for horizon_min in HORIZONS_MIN:
        label_col = f"flare_next_{horizon_min}min"
        y_true = eval_df[label_col].astype(int)

        for threshold in HARD_SCORE_THRESHOLDS:
            y_pred = (eval_df["hard_score"] >= threshold).astype(int)
            threshold_alerts = eval_df[y_pred == 1]

            for _, event_row in catalogue.iterrows():
                if "source_date" in threshold_alerts.columns:
                    event_alerts = threshold_alerts[
                        threshold_alerts["source_date"].astype(str) == str(event_row["source_date"])
                    ]
                else:
                    event_alerts = threshold_alerts
                episode_rows.extend(_episode_rows_for_event(event_row, event_alerts, horizon_min, threshold))

            episodes_so_far = pd.DataFrame(episode_rows)
            if episodes_so_far.empty:
                episode_metrics = _event_level_metrics(
                    pd.DataFrame(columns=["horizon_min", "threshold", "episode_type", "heldout_event_id", "lead_time_min"]),
                    catalogue,
                    horizon_min,
                    threshold,
                )
            else:
                episode_metrics = _event_level_metrics(episodes_so_far, catalogue, horizon_min, threshold)

            rows.append(
                {
                    "horizon_min": horizon_min,
                    "threshold": threshold,
                    "row_level_precision": precision_score(y_true, y_pred, zero_division=0),
                    "row_level_recall": recall_score(y_true, y_pred, zero_division=0),
                    "row_level_f1": f1_score(y_true, y_pred, zero_division=0),
                    **episode_metrics,
                }
            )

    episode_columns = [
        "date",
        "heldout_event_id",
        "horizon_min",
        "threshold",
        "alert_start",
        "alert_end",
        "duration_sec",
        "max_probability_or_score",
        "valid_window_start",
        "valid_window_end",
        "soft_peak_time",
        "episode_type",
        "lead_time_min",
    ]
    episodes = pd.DataFrame(episode_rows, columns=episode_columns)
    episodes.to_csv(ALERT_EPISODES_PATH, index=False)

    summary = pd.DataFrame(rows)
    summary["mean_precision"] = summary["event_level_precision"]
    summary["mean_recall"] = summary["event_level_recall"]
    summary["mean_f1"] = summary["event_level_f1"]
    summary["alerts"] = summary["total_alert_episodes"]
    summary["false_alerts_before_valid_window"] = summary["false_early_alert_episodes"]
    summary_path = Path("results") / f"{output_prefix}_threshold_sweep_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\nThreshold sweep summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved: {ALERT_EPISODES_PATH}")
    print(f"\nSaved: {summary_path}")
    print(f"Saved: {SUMMARY_PATH}")
    return summary


def select_operating_point(summary: pd.DataFrame) -> pd.Series:
    candidates = summary[
        (summary["event_level_recall"] > 0)
        & (summary["event_level_precision"] > 0)
        & (summary["valid_alerted_events"] > 0)
        & (summary["mean_valid_lead_time_min"].notna())
    ].copy()

    if candidates.empty:
        candidates = summary[summary["valid_alerted_events"] > 0].copy()
    if candidates.empty:
        candidates = summary.copy()

    candidates = candidates.sort_values(
        by=[
            "event_level_f1",
            "false_early_alert_episodes",
            "mean_valid_lead_time_min",
            "valid_alerted_events",
        ],
        ascending=[False, True, False, False],
    )
    return candidates.iloc[0]


def write_recommendation(summary: pd.DataFrame) -> pd.Series:
    choice = select_operating_point(summary)

    text = f"""# Recommended Operating Point

Recommended prototype setting:

- Horizon: {int(choice["horizon_min"])} minutes
- Hard-score threshold: {choice["threshold"]:g}
- Event-level precision: {choice["event_level_precision"]:.3f}
- Event-level recall: {choice["event_level_recall"]:.3f}
- Event-level F1: {choice["event_level_f1"]:.3f}
- Row-level precision: {choice["row_level_precision"]:.3f}
- Row-level recall: {choice["row_level_recall"]:.3f}
- Row-level F1: {choice["row_level_f1"]:.3f}
- Total alert episodes: {int(choice["total_alert_episodes"])}
- Useful alert episodes: {int(choice["useful_alert_episodes"])}
- False early alert episodes: {int(choice["false_early_alert_episodes"])}
- Valid alerted events: {int(choice["valid_alerted_events"])}
- Total held-out events: {int(choice["total_heldout_events"])}
- Mean valid lead time: {choice["mean_valid_lead_time_min"]:.2f} minutes

Rationale: the recommendation is selected using event-level alert episode metrics, prioritizing event-level F1, fewer false early alert episodes, useful lead time, and valid alerted events. Row-level metrics are diagnostic only because they count per-second predictions rather than operational alert episodes.

Quality gate: QUESTIONABLE dates are included but marked in the evaluation summary. BROKEN dates are excluded from supervised forecasting evaluation.

Important limitation: this project currently uses a small local matched-date dataset. Scientific reliability needs evaluation over more flare days and quiet days.
"""

    RECOMMENDATION_PATH.write_text(text, encoding="utf-8")
    print(f"\nSaved: {RECOMMENDATION_PATH}")
    return choice


def main() -> None:
    predictions_path, forecast_dataset_path, clean_catalogue_path, output_prefix = resolve_inputs()
    write_probability_sweep(predictions_path, output_prefix)
    summary = write_summary_sweep(forecast_dataset_path, clean_catalogue_path, output_prefix)
    write_recommendation(summary)
