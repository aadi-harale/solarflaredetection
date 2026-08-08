from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline

from src.utils.config import random_forest_estimators
from src.utils.quality_filter import apply_quality_gate_to_forecast_df, get_quality_label, normalize_date, quality_lookup


RESULTS_DIR = Path("results")
PROCESSED_DIR = Path("data") / "processed"

COMBINED_FORECAST_PATH = PROCESSED_DIR / "combined_forecast_dataset.csv"
MASTER_CLASSIFIED_PATH = RESULTS_DIR / "master_flare_catalogue_classified_v2.csv"
MASTER_V1_PATH = RESULTS_DIR / "master_flare_catalogue.csv"
ALERT_WINDOW_COMPARISON_PATH = RESULTS_DIR / "alert_window_comparison.csv"

DATASET_OUT = RESULTS_DIR / "forecast_dataset_precursor_v2.csv"
METRICS_OUT = RESULTS_DIR / "precursor_forecast_v2_metrics.csv"
PREDICTIONS_OUT = RESULTS_DIR / "precursor_forecast_v2_predictions.csv"
REPORT_OUT = RESULTS_DIR / "precursor_forecast_v2_report.md"
COMPARISON_OUT = RESULTS_DIR / "precursor_forecast_v2_comparison.csv"
PLOT_COMPARISON_OUT = RESULTS_DIR / "plot_precursor_v2_comparison.csv"
PLOT_LEAD_TIME_OUT = RESULTS_DIR / "plot_lead_time_distribution_v2.csv"

TARGET_HORIZONS = [60, 90]
RULE_SCORE_THRESHOLD = 4.0
RF_PROBABILITY_THRESHOLD = 0.5
MAX_NEGATIVE_RATIO = 20
EPISODE_GAP_SECONDS = 60
EVALUATION_CADENCE_SECONDS = 10


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path.resolve()}")


def load_forecast_base() -> pd.DataFrame:
    require_file(COMBINED_FORECAST_PATH)
    df = pd.read_csv(COMBINED_FORECAST_PATH)
    if "time_utc" not in df.columns:
        df = df.rename(columns={df.columns[0]: "time_utc"})
    if "source_date" not in df.columns:
        raise ValueError("Combined forecast dataset is missing required column: source_date")

    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, format="mixed", errors="coerce")
    df = df.dropna(subset=["time_utc"]).copy()
    df["source_date"] = df["source_date"].map(normalize_date)
    df["date"] = df["source_date"]
    df = apply_quality_gate_to_forecast_df(df)
    df["date"] = df["source_date"].map(normalize_date)
    return df.sort_values(["date", "time_utc"]).reset_index(drop=True)


def load_supported_events() -> pd.DataFrame:
    path = MASTER_CLASSIFIED_PATH if MASTER_CLASSIFIED_PATH.exists() else MASTER_V1_PATH
    require_file(path)
    events = pd.read_csv(path)
    required = {"event_id", "date", "soft_peak", "combined_start", "combined_end", "goes_match_status", "goes_class_group"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    events = events.copy()
    events["date"] = events["date"].map(normalize_date)
    for col in ["soft_peak", "combined_start", "combined_end"]:
        events[col] = pd.to_datetime(events[col], utc=True, format="mixed", errors="coerce")

    events = events[
        events["goes_match_status"].isin(["EXACT_PEAK_MATCH", "WINDOW_OVERLAP_MATCH"])
        & events["soft_peak"].notna()
    ].copy()
    events["quality_label"] = events["date"].map(get_quality_label)
    events = events[events["quality_label"].isin(["GOOD", "QUESTIONABLE"])].copy()
    events["goes_class_group"] = events["goes_class_group"].astype(str).str.upper()
    events["goes_low_high_group"] = np.where(events["goes_class_group"].isin(["M", "X"]), "HIGH", "LOW_OR_MODERATE")
    return events.sort_values(["date", "soft_peak"]).reset_index(drop=True)


def add_precursor_labels(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    lookup = quality_lookup()
    out["quality_label"] = out["date"].map(lookup)
    out["current_nowcast_state"] = np.select(
        [
            pd.to_numeric(out.get("inside_detected_event", 0), errors="coerce").fillna(0).astype(int).eq(1),
            pd.to_numeric(out.get("hard_score", 0), errors="coerce").fillna(0).ge(RULE_SCORE_THRESHOLD),
            pd.to_numeric(out.get("soft_score", 0), errors="coerce").fillna(0).ge(RULE_SCORE_THRESHOLD),
        ],
        ["INSIDE_DETECTED_EVENT", "HARD_SCORE_ACTIVE", "SOFT_SCORE_ACTIVE"],
        default="QUIET_OR_BACKGROUND",
    )
    out["hard_nowcast_active"] = pd.to_numeric(out.get("hard_score", 0), errors="coerce").fillna(0).ge(RULE_SCORE_THRESHOLD).astype(int)
    out["soft_nowcast_active"] = pd.to_numeric(out.get("soft_score", 0), errors="coerce").fillna(0).ge(RULE_SCORE_THRESHOLD).astype(int)

    for horizon in TARGET_HORIZONS:
        out[f"flare_next_{horizon}min"] = 0
        out[f"time_to_peak_within_{horizon}min"] = np.nan
    out["high_class_flare_next_90min"] = 0
    out["any_flare_next_90min"] = 0

    for date, group in out.groupby("date", sort=False):
        event_rows = events[events["date"] == date].copy()
        if event_rows.empty:
            continue
        times_ns = datetime_ns(group["time_utc"])
        for horizon in TARGET_HORIZONS:
            peaks_ns = np.sort(datetime_ns(event_rows["soft_peak"]))
            positions = np.searchsorted(peaks_ns, times_ns, side="right")
            has_next = positions < len(peaks_ns)
            next_peak = np.full(len(times_ns), np.nan)
            next_peak[has_next] = peaks_ns[positions[has_next]]
            lead_min = (next_peak - times_ns) / 1e9 / 60.0
            label = has_next & (lead_min > 0) & (lead_min <= horizon)
            out.loc[group.index, f"flare_next_{horizon}min"] = label.astype(int)
            out.loc[group.index, f"time_to_peak_within_{horizon}min"] = np.where(label, lead_min, np.nan)

        high_events = event_rows[event_rows["goes_low_high_group"] == "HIGH"]
        if not high_events.empty:
            high_peaks_ns = np.sort(datetime_ns(high_events["soft_peak"]))
            high_positions = np.searchsorted(high_peaks_ns, times_ns, side="right")
            has_next_high = high_positions < len(high_peaks_ns)
            next_high_peak = np.full(len(times_ns), np.nan)
            next_high_peak[has_next_high] = high_peaks_ns[high_positions[has_next_high]]
            high_lead_min = (next_high_peak - times_ns) / 1e9 / 60.0
            high_label = has_next_high & (high_lead_min > 0) & (high_lead_min <= 90)
            out.loc[group.index, "high_class_flare_next_90min"] = high_label.astype(int)
        out.loc[group.index, "any_flare_next_90min"] = out.loc[group.index, "flare_next_90min"]

    return out


def datetime_ns(series: pd.Series) -> np.ndarray:
    return pd.to_datetime(series, utc=True, format="mixed", errors="coerce").dt.tz_convert(None).to_numpy(
        dtype="datetime64[ns]"
    ).astype("int64")


def feature_columns(df: pd.DataFrame) -> list[str]:
    forbidden_exact = {
        "time_utc",
        "timestamp",
        "date",
        "source_date",
        "quality_label",
        "is_quiet_day",
        "current_nowcast_state",
        "flare_next_5min",
        "flare_next_10min",
        "flare_next_30min",
        "flare_next_60min",
        "flare_next_90min",
        "high_class_flare_next_90min",
        "any_flare_next_90min",
    }
    cols = []
    for col in df.columns:
        if col in forbidden_exact:
            continue
        if col.startswith("time_to_peak_within_"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def downsample_training(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    positives = y[y == 1].index
    negatives = y[y == 0].index
    if len(positives) == 0 or len(negatives) <= len(positives) * MAX_NEGATIVE_RATIO:
        return X, y
    sampled_negatives = pd.Index(negatives).to_series().sample(
        n=len(positives) * MAX_NEGATIVE_RATIO,
        random_state=42,
    ).index
    keep = positives.union(sampled_negatives).sort_values()
    return X.loc[keep], y.loc[keep]


def merge_alert_episodes(pred: pd.DataFrame, score_col: str) -> pd.DataFrame:
    positives = pred[pred["predicted_positive"] == 1].copy()
    if positives.empty:
        return pd.DataFrame(
            columns=["date", "alert_start", "alert_end", "duration_sec", "max_score", "row_count"]
        )
    rows = []
    for date, group in positives.sort_values(["date", "time_utc"]).groupby("date"):
        start = None
        end = None
        max_score = -np.inf
        row_count = 0
        previous = None
        for _, row in group.iterrows():
            ts = row["time_utc"]
            score = row[score_col]
            if start is None or (ts - previous).total_seconds() > EPISODE_GAP_SECONDS:
                if start is not None:
                    rows.append(
                        {
                            "date": date,
                            "alert_start": start,
                            "alert_end": end,
                            "duration_sec": max(0.0, (end - start).total_seconds()),
                            "max_score": max_score,
                            "row_count": row_count,
                        }
                    )
                start = ts
                max_score = score
                row_count = 1
            else:
                max_score = max(max_score, score)
                row_count += 1
            end = ts
            previous = ts
        if start is not None:
            rows.append(
                {
                    "date": date,
                    "alert_start": start,
                    "alert_end": end,
                    "duration_sec": max(0.0, (end - start).total_seconds()),
                    "max_score": max_score,
                    "row_count": row_count,
                }
            )
    return pd.DataFrame(rows)


def evaluate_episodes(
    pred: pd.DataFrame,
    events: pd.DataFrame,
    horizon_min: int,
    method: str,
    score_col: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    episodes = merge_alert_episodes(pred, score_col)
    if episodes.empty:
        total_events = len(events)
        return (
            {
                "method": method,
                "target": f"flare_next_{horizon_min}min",
                "horizon_min": horizon_min,
                "validation_method": "leave-one-date-out blocked validation" if method != "rule_score_precursor_policy" else "fixed rule score applied to all quality-gated dates",
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "false_alerts_per_day": 0.0,
                "valid_alerted_events": 0,
                "total_events": total_events,
                "total_alert_episodes": 0,
                "useful_alert_episodes": 0,
                "false_alert_episodes": 0,
                "mean_lead_time_min": np.nan,
                "median_lead_time_min": np.nan,
                "q1_lead_time_min": np.nan,
                "q3_lead_time_min": np.nan,
                "iqr_lead_time_min": np.nan,
                "positive_lead_time_percent": np.nan,
                "notes": "No alert episodes produced.",
            },
            episodes,
            pd.DataFrame(),
        )

    event_hits = {}
    episode_rows = []
    for _, episode in episodes.iterrows():
        same_date_events = events[events["date"] == episode["date"]].copy()
        future_events = same_date_events[
            (same_date_events["soft_peak"] >= episode["alert_start"])
            & (same_date_events["soft_peak"] <= episode["alert_start"] + pd.Timedelta(minutes=horizon_min))
        ].copy()
        if future_events.empty:
            episode_type = "FALSE_ALERT_EPISODE"
            matched = None
            lead_time = np.nan
        else:
            future_events["lead_time_min"] = (future_events["soft_peak"] - episode["alert_start"]).dt.total_seconds() / 60.0
            matched = future_events.sort_values("lead_time_min").iloc[0]
            event_key = str(matched["event_id"])
            episode_type = "USEFUL_PRECURSOR_ALERT"
            lead_time = float(matched["lead_time_min"])
            if event_key not in event_hits or episode["alert_start"] < event_hits[event_key]["first_alert_time"]:
                event_hits[event_key] = {
                    "event_id": event_key,
                    "date": episode["date"],
                    "quality_label": matched["quality_label"],
                    "goes_class_group": matched["goes_class_group"],
                    "first_alert_time": episode["alert_start"],
                    "soft_peak": matched["soft_peak"],
                    "lead_time_min": lead_time,
                }

        episode_rows.append(
            {
                **episode.to_dict(),
                "method": method,
                "horizon_min": horizon_min,
                "episode_type": episode_type,
                "matched_event_id": "" if matched is None else matched["event_id"],
                "matched_goes_class_group": "" if matched is None else matched["goes_class_group"],
                "lead_time_min": lead_time,
            }
        )

    episode_eval = pd.DataFrame(episode_rows)
    hits = pd.DataFrame(event_hits.values())
    total_events = len(events)
    useful = int((episode_eval["episode_type"] == "USEFUL_PRECURSOR_ALERT").sum())
    false_alerts = int((episode_eval["episode_type"] == "FALSE_ALERT_EPISODE").sum())
    total_alerts = len(episode_eval)
    valid_events = len(hits)
    precision = useful / total_alerts if total_alerts else 0.0
    recall = valid_events / total_events if total_events else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    dates_used = max(1, pred["date"].nunique())
    lead = pd.to_numeric(hits.get("lead_time_min", pd.Series(dtype=float)), errors="coerce").dropna()

    metrics = {
        "method": method,
        "target": f"flare_next_{horizon_min}min",
        "horizon_min": horizon_min,
        "validation_method": "leave-one-date-out blocked validation" if method != "rule_score_precursor_policy" else "fixed rule score applied to all quality-gated dates",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alerts_per_day": false_alerts / dates_used,
        "valid_alerted_events": valid_events,
        "total_events": total_events,
        "total_alert_episodes": total_alerts,
        "useful_alert_episodes": useful,
        "false_alert_episodes": false_alerts,
        "mean_lead_time_min": float(lead.mean()) if not lead.empty else np.nan,
        "median_lead_time_min": float(lead.median()) if not lead.empty else np.nan,
        "q1_lead_time_min": float(lead.quantile(0.25)) if not lead.empty else np.nan,
        "q3_lead_time_min": float(lead.quantile(0.75)) if not lead.empty else np.nan,
        "iqr_lead_time_min": float(lead.quantile(0.75) - lead.quantile(0.25)) if not lead.empty else np.nan,
        "positive_lead_time_percent": float((lead > 0).mean() * 100.0) if not lead.empty else np.nan,
        "notes": "Episode-level precision counts useful precursor alert episodes; recall counts unique GOES-supported SuryaAlert events alerted.",
    }
    return metrics, episode_eval, hits


def make_rule_predictions(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    pred = df[["time_utc", "date", "quality_label", f"flare_next_{horizon}min", f"time_to_peak_within_{horizon}min"]].copy()
    pred["model"] = "rule_score_precursor_policy"
    pred["horizon_min"] = horizon
    pred["score"] = pd.to_numeric(df["hard_score"], errors="coerce").fillna(0)
    pred["predicted_positive"] = pred["score"].ge(RULE_SCORE_THRESHOLD).astype(int)
    pred["y_true"] = pred[f"flare_next_{horizon}min"].astype(int)
    return pred


def make_rf_predictions(df: pd.DataFrame, horizon: int, features: list[str]) -> tuple[pd.DataFrame, str]:
    rows = []
    target = f"flare_next_{horizon}min"
    for date in sorted(df["date"].unique()):
        train = df[df["date"] != date].copy()
        test = df[df["date"] == date].copy()
        if train[target].nunique() < 2 or test.empty:
            continue
        X_train = train[features].replace([np.inf, -np.inf], np.nan)
        y_train = train[target].astype(int)
        X_train, y_train = downsample_training(X_train, y_train)
        X_test = test[features].replace([np.inf, -np.inf], np.nan)

        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=random_forest_estimators(),
                max_depth=8,
                min_samples_leaf=20,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        )
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1]
        temp = test[["time_utc", "date", "quality_label", target, f"time_to_peak_within_{horizon}min"]].copy()
        temp["model"] = "random_forest_precursor_v2"
        temp["horizon_min"] = horizon
        temp["score"] = prob
        temp["predicted_positive"] = (prob >= RF_PROBABILITY_THRESHOLD).astype(int)
        temp["y_true"] = temp[target].astype(int)
        rows.append(temp)
    if not rows:
        return pd.DataFrame(), "NOT_AVAILABLE_FAIRLY"
    return pd.concat(rows, ignore_index=True), "leave-one-date-out blocked validation"


def row_metrics(pred: pd.DataFrame, method: str, horizon: int) -> dict:
    if pred.empty:
        return {}
    return {
        "method": method,
        "target": f"flare_next_{horizon}min",
        "horizon_min": horizon,
        "row_level_precision_diagnostic": precision_score(pred["y_true"], pred["predicted_positive"], zero_division=0),
        "row_level_recall_diagnostic": recall_score(pred["y_true"], pred["predicted_positive"], zero_division=0),
        "row_level_f1_diagnostic": f1_score(pred["y_true"], pred["predicted_positive"], zero_division=0),
    }


def class_group_metrics(hits: pd.DataFrame, events: pd.DataFrame, method: str, horizon: int) -> pd.DataFrame:
    rows = []
    hit_ids = set(hits["event_id"].astype(str)) if not hits.empty and "event_id" in hits.columns else set()
    for group in ["LOW_OR_MODERATE", "HIGH"]:
        subset = events[events["goes_low_high_group"] == group]
        total = len(subset)
        alerted = int(subset["event_id"].astype(str).isin(hit_ids).sum()) if total else 0
        rows.append(
            {
                "method": method,
                "target": f"flare_next_{horizon}min",
                "class_group": group,
                "total_events": total,
                "valid_alerted_events": alerted,
                "class_group_recall": alerted / total if total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if ALERT_WINDOW_COMPARISON_PATH.exists():
        prior = pd.read_csv(ALERT_WINDOW_COMPARISON_PATH)
        strict = prior[prior["matching_window_min"] == 30]
        rescored = prior[prior["matching_window_min"] == 90]
        if not strict.empty:
            r = strict.iloc[0]
            rows.append(
                {
                    "system": "original 5/10/30-min target model",
                    "evaluation": "30-min strict scoring",
                    "precision": r.get("precision"),
                    "recall": r.get("recall"),
                    "f1": r.get("f1"),
                    "false_alerts_per_day": r.get("false_alerts_per_day"),
                    "mean_lead_time_min": r.get("mean_lead_time_min"),
                    "notes": "v1 target labels and operational policy output.",
                }
            )
        if not rescored.empty:
            r = rescored.iloc[0]
            rows.append(
                {
                    "system": "v1 predictions with 90-min precursor-aware rescoring",
                    "evaluation": "90-min precursor-aware rescoring",
                    "precision": r.get("precision"),
                    "recall": r.get("recall"),
                    "f1": r.get("f1"),
                    "false_alerts_per_day": r.get("false_alerts_per_day"),
                    "mean_lead_time_min": r.get("mean_lead_time_min"),
                    "notes": "No retraining; fixed-window labels reinterpreted with wider precursor window.",
                }
            )
    for _, row in metrics[metrics["method"].eq("random_forest_precursor_v2")].iterrows():
        rows.append(
            {
                "system": f"new v2 trained target ({row['horizon_min']}-min)",
                "evaluation": "leave-one-date-out blocked validation",
                "precision": row.get("precision"),
                "recall": row.get("recall"),
                "f1": row.get("f1"),
                "false_alerts_per_day": row.get("false_alerts_per_day"),
                "mean_lead_time_min": row.get("mean_lead_time_min"),
                "notes": "Precursor-aware labels trained directly; diagnostic due small sample.",
            }
        )
    return pd.DataFrame(rows)


def update_report_sections(metrics: pd.DataFrame, comparison: pd.DataFrame) -> None:
    best_90 = metrics[(metrics["method"] == "random_forest_precursor_v2") & (metrics["horizon_min"] == 90)]
    if best_90.empty:
        interpretation = "90-min rescoring remains useful, but training a robust precursor model requires more events."
        metric_line = "- Random Forest precursor v2 was not available fairly for the 90-minute target."
    else:
        r = best_90.iloc[0]
        prior_f1 = np.nan
        if ALERT_WINDOW_COMPARISON_PATH.exists():
            prior = pd.read_csv(ALERT_WINDOW_COMPARISON_PATH)
            prior_90 = prior[prior["matching_window_min"] == 90]
            if not prior_90.empty:
                prior_f1 = prior_90.iloc[0].get("f1")
        improved = pd.notna(prior_f1) and r["f1"] > prior_f1
        interpretation = (
            "Precursor-aware labels better match the observed hard-X-ray lead-time behavior."
            if improved
            else "90-min rescoring remains useful, but training a robust precursor model requires more events."
        )
        metric_line = (
            f"- Random Forest v2 90-min target: precision={r['precision']:.3f}, recall={r['recall']:.3f}, "
            f"F1={r['f1']:.3f}, false alerts/day={r['false_alerts_per_day']:.2f}, "
            f"mean lead time={r['mean_lead_time_min']:.2f} min."
        )

    section = f"""## Precursor-aware forecasting v2

SuryaAlert now includes a v2 precursor-aware forecasting diagnostic. This path creates 60-minute and 90-minute future-flare targets directly from GOES-supported SuryaAlert events, then evaluates predictions with blocked date-wise validation.

The v2 labels are generated only from future event timing/class information. Model features use only current and past SoLEXS/HEL1OS features already present at prediction time.

{metric_line}

Interpretation: {interpretation}

Caveats:

- This is a v2 diagnostic forecasting experiment, not a replacement for the original 5/10/30-minute labels.
- It does not change v1 catalogues, thresholds, or alert semantics.
- The dataset remains small, so calibration and class-group performance are diagnostic only.
"""
    for name in ["final_hackathon_evidence_report.md", "space_agency_evaluation_criteria_scorecard.md", "hackathon_diagnostic_report.md"]:
        path = RESULTS_DIR / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        title = "## Precursor-aware forecasting v2"
        if title in text:
            before = text.split(title, 1)[0].rstrip()
            after = text.split(title, 1)[1]
            import re

            match = re.search(r"\n## ", after)
            text = before + "\n\n" + section + ("\n" + after[match.start() :].lstrip() if match else "")
        else:
            text = text.rstrip() + "\n\n" + section
        path.write_text(text, encoding="utf-8")


def write_report(
    dataset: pd.DataFrame,
    metrics: pd.DataFrame,
    comparison: pd.DataFrame,
    class_perf: pd.DataFrame,
    model_status: dict[str, str],
) -> None:
    target_counts = []
    for col in ["flare_next_60min", "flare_next_90min", "high_class_flare_next_90min", "any_flare_next_90min"]:
        counts = dataset[col].value_counts().sort_index().to_dict()
        target_counts.append(f"- `{col}`: {counts}")

    rf90 = metrics[(metrics["method"] == "random_forest_precursor_v2") & (metrics["horizon_min"] == 90)]
    if rf90.empty:
        interpretation = "90-min rescoring remains useful, but training a robust precursor model requires more events."
    else:
        prior = comparison[comparison["system"].eq("v1 predictions with 90-min precursor-aware rescoring")]
        prior_f1 = prior.iloc[0]["f1"] if not prior.empty else np.nan
        interpretation = (
            "Precursor-aware labels better match the observed hard-X-ray lead-time behavior."
            if pd.notna(prior_f1) and rf90.iloc[0]["f1"] > prior_f1
            else "90-min rescoring remains useful, but training a robust precursor model requires more events."
        )

    report = f"""# Precursor-Aware Forecasting v2

## Purpose

This v2 diagnostic makes the forecasting target itself precursor-aware. It predicts whether a GOES-supported SuryaAlert flare event is expected within the next 60 or 90 minutes using only current and past SoLEXS + HEL1OS features.

No v1 result files, catalogues, or original 5/10/30-minute labels are overwritten.

## Target Distributions

{chr(10).join(target_counts)}

## Validation

Headline validation uses blocked leave-one-date-out splits. Random row splits are not used as headline metrics. Prediction/evaluation rows use a fixed {EVALUATION_CADENCE_SECONDS}-second cadence plus active nowcast rows to keep the blocked diagnostic reproducible and tractable on per-second telemetry.

## Models Evaluated

- Rule-score precursor policy: fixed `hard_score >= 4`, no training.
- Random Forest precursor v2: existing project baseline style, evaluated with blocked date-wise splits.
- Logistic Regression: {model_status.get("logistic_regression", "NOT_AVAILABLE_FAIRLY")}

## Metrics

See `results/precursor_forecast_v2_metrics.csv`.

## Comparison

See `results/precursor_forecast_v2_comparison.csv`.

Interpretation: **{interpretation}**

## Caveats

- This is a diagnostic v2 target experiment, not operational validation.
- Scores are not fully calibrated operational probabilities.
- The dataset remains small, with limited quiet/control days and limited true-negative windows.
- Class-group performance is reported only where GOES-supported class labels are available.
"""
    REPORT_OUT.write_text(report, encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    base = load_forecast_base()
    base = base[
        base.groupby("date").cumcount().mod(EVALUATION_CADENCE_SECONDS).eq(0)
        | pd.to_numeric(base.get("hard_score", 0), errors="coerce").fillna(0).ge(RULE_SCORE_THRESHOLD)
        | pd.to_numeric(base.get("soft_score", 0), errors="coerce").fillna(0).ge(RULE_SCORE_THRESHOLD)
    ].copy()
    events = load_supported_events()
    dataset = add_precursor_labels(base, events)
    dataset = dataset.rename(columns={"time_utc": "timestamp"})
    dataset.to_csv(DATASET_OUT, index=False)

    eval_df = dataset.rename(columns={"timestamp": "time_utc"}).copy()
    # Keep evaluation outside already-detected event interiors so the model is a precursor forecast, not a post-detection echo.
    if "inside_detected_event" in eval_df.columns:
        eval_df = eval_df[pd.to_numeric(eval_df["inside_detected_event"], errors="coerce").fillna(0).astype(int) == 0].copy()

    features = feature_columns(eval_df)
    all_predictions = []
    all_metrics = []
    all_episode_rows = []
    all_hits = []
    class_perf_rows = []
    model_status = {"logistic_regression": "NOT_AVAILABLE_FAIRLY: small event count and imbalanced blocked folds."}

    for horizon in TARGET_HORIZONS:
        target = f"flare_next_{horizon}min"
        rule_pred = make_rule_predictions(eval_df, horizon)
        all_predictions.append(rule_pred)
        metrics, episodes, hits = evaluate_episodes(rule_pred, events, horizon, "rule_score_precursor_policy", "score")
        metrics.update(row_metrics(rule_pred, "rule_score_precursor_policy", horizon))
        all_metrics.append(metrics)
        all_episode_rows.append(episodes)
        all_hits.append(hits.assign(method="rule_score_precursor_policy", horizon_min=horizon) if not hits.empty else hits)
        class_perf_rows.append(class_group_metrics(hits, events, "rule_score_precursor_policy", horizon))

        rf_pred, rf_status = make_rf_predictions(eval_df, horizon, features)
        if not rf_pred.empty:
            all_predictions.append(rf_pred)
            metrics, episodes, hits = evaluate_episodes(rf_pred, events, horizon, "random_forest_precursor_v2", "score")
            metrics.update(row_metrics(rf_pred, "random_forest_precursor_v2", horizon))
            all_metrics.append(metrics)
            all_episode_rows.append(episodes)
            all_hits.append(hits.assign(method="random_forest_precursor_v2", horizon_min=horizon) if not hits.empty else hits)
            class_perf_rows.append(class_group_metrics(hits, events, "random_forest_precursor_v2", horizon))
        else:
            all_metrics.append(
                {
                    "method": "random_forest_precursor_v2",
                    "target": target,
                    "horizon_min": horizon,
                    "validation_method": rf_status,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1": np.nan,
                    "false_alerts_per_day": np.nan,
                    "valid_alerted_events": np.nan,
                    "total_events": len(events),
                    "notes": "NOT_AVAILABLE_FAIRLY",
                }
            )

    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    predictions.to_csv(PREDICTIONS_OUT, index=False)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(METRICS_OUT, index=False)

    comparison = build_comparison(metrics_df)
    comparison.to_csv(COMPARISON_OUT, index=False)
    comparison.to_csv(PLOT_COMPARISON_OUT, index=False)

    lead_frames = [hits for hits in all_hits if not hits.empty]
    lead_dist = pd.concat(lead_frames, ignore_index=True) if lead_frames else pd.DataFrame()
    lead_dist.to_csv(PLOT_LEAD_TIME_OUT, index=False)

    class_perf = pd.concat(class_perf_rows, ignore_index=True) if class_perf_rows else pd.DataFrame()
    if not class_perf.empty:
        class_perf.to_csv(RESULTS_DIR / "precursor_forecast_v2_class_group_performance.csv", index=False)

    write_report(dataset, metrics_df, comparison, class_perf, model_status)
    update_report_sections(metrics_df, comparison)

    print("Precursor-aware forecasting v2 complete.")
    print("Target distributions:")
    for col in ["flare_next_60min", "flare_next_90min", "high_class_flare_next_90min", "any_flare_next_90min"]:
        print(col)
        print(dataset[col].value_counts().sort_index().to_string())
    print("\nModels evaluated:")
    print("- rule_score_precursor_policy")
    print("- random_forest_precursor_v2")
    print("- Logistic Regression: NOT_AVAILABLE_FAIRLY")
    print(
        "\nValidation method: leave-one-date-out blocked validation for Random Forest; "
        "fixed rule applied to all quality-gated dates. Evaluation cadence: "
        f"{EVALUATION_CADENCE_SECONDS} seconds plus active nowcast rows."
    )
    print("\n60-min metrics:")
    print(metrics_df[metrics_df["horizon_min"] == 60][["method", "precision", "recall", "f1", "false_alerts_per_day", "valid_alerted_events", "mean_lead_time_min"]].to_string(index=False))
    print("\n90-min metrics:")
    print(metrics_df[metrics_df["horizon_min"] == 90][["method", "precision", "recall", "f1", "false_alerts_per_day", "valid_alerted_events", "mean_lead_time_min"]].to_string(index=False))
    print("\nSaved:")
    for path in [DATASET_OUT, METRICS_OUT, PREDICTIONS_OUT, REPORT_OUT, COMPARISON_OUT, PLOT_COMPARISON_OUT, PLOT_LEAD_TIME_OUT]:
        print(path)
    print("\nCaveat: v2 targets are diagnostic and precursor-aware; this is not operational readiness or probability calibration.")


if __name__ == "__main__":
    main()
