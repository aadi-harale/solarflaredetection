from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OUT_DIR = Path("results") / "forecasting_v3"
DATASET_PATH = OUT_DIR / "forecasting_v3_dataset.csv"
MASTER_CLASSIFIED_PATH = Path("results") / "master_flare_catalogue_classified_v2.csv"
MASTER_PATH = Path("results") / "master_flare_catalogue.csv"
MODEL_COMPARISON_PATH = OUT_DIR / "forecasting_v3_model_comparison.csv"
PREDICTIONS_PATH = OUT_DIR / "forecasting_v3_predictions.csv"
IMPORTANCE_PATH = OUT_DIR / "forecasting_v3_feature_importance.csv"
REPORT_PATH = OUT_DIR / "forecasting_v3_model_report.md"

TARGETS = [
    ("flare_onset_next_30min", 30, "onset", "primary"),
    ("flare_onset_next_60min", 60, "onset", "secondary"),
    ("flare_peak_next_15min", 15, "peak", "diagnostic"),
]
LABEL_COLUMNS = {
    "flare_onset_next_30min",
    "flare_onset_next_60min",
    "flare_peak_next_15min",
    "high_class_onset_next_60min",
}
PREDICTION_CADENCE_SECONDS = 30
EPISODE_GAP_SECONDS = 60
V1_BASELINE = {
    "precision": 0.7241379310344828,
    "recall": 0.5294117647058824,
    "f1": 0.6116504854368933,
    "false_alerts_per_day": 1.6666666666666667,
    "valid_alerted_events": "9 / 17",
    "mean_lead_time_min": 40.50740740740741,
    "median_lead_time_min": 31.933333333333334,
}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path.resolve()}")


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed", errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df["date"] = df["date"].astype(str).str.replace(r"\.0$", "", regex=True)
    return df.sort_values(["date", "timestamp"]).reset_index(drop=True)


def load_events() -> pd.DataFrame:
    path = MASTER_CLASSIFIED_PATH if MASTER_CLASSIFIED_PATH.exists() else MASTER_PATH
    require_file(path)
    events = pd.read_csv(path)
    events["date"] = events["date"].astype(str).str.replace(r"\.0$", "", regex=True)
    onset_col = "combined_start" if "combined_start" in events.columns else "soft_start"
    events["event_onset_time"] = pd.to_datetime(events[onset_col], utc=True, format="mixed", errors="coerce")
    events["event_peak_time"] = pd.to_datetime(events["soft_peak"], utc=True, format="mixed", errors="coerce")
    events = events.dropna(subset=["event_onset_time", "event_peak_time"]).copy()
    if "surya_estimated_class_group" not in events.columns:
        events["surya_estimated_class_group"] = ""
    if "goes_class_group" not in events.columns:
        events["goes_class_group"] = ""
    events["event_id"] = events["event_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    return events


def modelling_frame(df: pd.DataFrame) -> pd.DataFrame:
    sampled = df.groupby("date").cumcount().mod(PREDICTION_CADENCE_SECONDS).eq(0)
    # Keep active rule-score rows as well, so short hard-X-ray bursts are not removed by cadence sampling.
    active = pd.to_numeric(df.get("hard_score", 0), errors="coerce").fillna(0).ge(4)
    out = df[sampled | active].copy()
    return out.reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = {"timestamp", "date", "quality_label"} | LABEL_COLUMNS
    cols = []
    for col in df.columns:
        if col in blocked:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any():
            cols.append(col)
    return cols


def merge_alert_episodes(pred: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    positives = pred[pred["predicted_positive"] == 1].copy()
    if positives.empty:
        return pd.DataFrame(columns=["date", "alert_start", "alert_end", "max_score", "row_count"])
    rows = []
    for date, group in positives.sort_values(["date", "timestamp"]).groupby("date"):
        start = None
        end = None
        previous = None
        max_score = -np.inf
        count = 0
        for _, row in group.iterrows():
            ts = row["timestamp"]
            if start is None or (ts - previous).total_seconds() > EPISODE_GAP_SECONDS:
                if start is not None:
                    rows.append({"date": date, "alert_start": start, "alert_end": end, "max_score": max_score, "row_count": count})
                start = ts
                max_score = float(row[score_col])
                count = 1
            else:
                max_score = max(max_score, float(row[score_col]))
                count += 1
            end = ts
            previous = ts
        if start is not None:
            rows.append({"date": date, "alert_start": start, "alert_end": end, "max_score": max_score, "row_count": count})
    return pd.DataFrame(rows)


def evaluate_episode_metrics(pred: pd.DataFrame, events: pd.DataFrame, target: str, horizon_min: int, anchor: str, model_name: str) -> tuple[dict, pd.DataFrame]:
    episodes = merge_alert_episodes(pred)
    event_hits: dict[str, dict] = {}
    episode_rows = []
    anchor_col = "event_onset_time" if anchor == "onset" else "event_peak_time"
    for _, episode in episodes.iterrows():
        same_date = events[events["date"] == str(episode["date"])].copy()
        candidates = same_date[
            (same_date[anchor_col] > episode["alert_start"])
            & (same_date[anchor_col] <= episode["alert_start"] + pd.Timedelta(minutes=horizon_min))
        ].copy()
        if candidates.empty:
            episode_type = "ISOLATED_FALSE_ALERT"
            matched_event_id = ""
            lead_time = np.nan
        else:
            candidates["lead_time_min"] = (candidates[anchor_col] - episode["alert_start"]).dt.total_seconds() / 60.0
            match = candidates.sort_values("lead_time_min").iloc[0]
            episode_type = "USEFUL_ALERT_EPISODE"
            matched_event_id = str(match["event_id"])
            lead_time = float(match["lead_time_min"])
            if matched_event_id not in event_hits or episode["alert_start"] < event_hits[matched_event_id]["first_alert_time"]:
                event_hits[matched_event_id] = {
                    "event_id": matched_event_id,
                    "date": episode["date"],
                    "first_alert_time": episode["alert_start"],
                    "event_time": match[anchor_col],
                    "lead_time_min": lead_time,
                    "goes_class_group": match.get("goes_class_group", ""),
                    "surya_estimated_class_group": match.get("surya_estimated_class_group", ""),
                }
        episode_rows.append(
            {
                **episode.to_dict(),
                "model": model_name,
                "target": target,
                "horizon_min": horizon_min,
                "episode_type": episode_type,
                "matched_event_id": matched_event_id,
                "lead_time_min": lead_time,
            }
        )

    episode_eval = pd.DataFrame(episode_rows)
    hit_df = pd.DataFrame(event_hits.values())
    useful = int((episode_eval.get("episode_type", pd.Series(dtype=str)) == "USEFUL_ALERT_EPISODE").sum()) if not episode_eval.empty else 0
    false_alerts = int((episode_eval.get("episode_type", pd.Series(dtype=str)) == "ISOLATED_FALSE_ALERT").sum()) if not episode_eval.empty else 0
    total_episodes = len(episode_eval)
    total_events = len(events)
    valid_events = len(hit_df)
    missed_events = max(0, total_events - valid_events)
    precision = useful / total_episodes if total_episodes else 0.0
    recall = valid_events / total_events if total_events else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    csi = valid_events / (valid_events + false_alerts + missed_events) if (valid_events + false_alerts + missed_events) else np.nan
    lead = pd.to_numeric(hit_df.get("lead_time_min", pd.Series(dtype=float)), errors="coerce").dropna()
    dates = max(1, pred["date"].nunique())
    metrics = {
        "model": model_name,
        "target": target,
        "target_role": "diagnostic" if "peak" in target else ("primary" if "30min" in target else "secondary"),
        "validation_method": "leave-one-date-out blocked validation" if model_name != "rule_score_baseline" else "fixed rule score evaluated on blocked-cadence v3 frame",
        "prediction_cadence_sec": PREDICTION_CADENCE_SECONDS,
        "precision": precision,
        "recall_pod": recall,
        "f1": f1,
        "false_alerts_per_day": false_alerts / dates,
        "valid_alerted_events": valid_events,
        "total_events": total_events,
        "useful_alert_episodes": useful,
        "isolated_false_alerts": false_alerts,
        "total_alert_episodes": total_episodes,
        "mean_lead_time_min": float(lead.mean()) if not lead.empty else np.nan,
        "median_lead_time_min": float(lead.median()) if not lead.empty else np.nan,
        "q1_lead_time_min": float(lead.quantile(0.25)) if not lead.empty else np.nan,
        "q3_lead_time_min": float(lead.quantile(0.75)) if not lead.empty else np.nan,
        "iqr_lead_time_min": float(lead.quantile(0.75) - lead.quantile(0.25)) if not lead.empty else np.nan,
        "positive_lead_time_percent": float((lead > 0).mean() * 100.0) if not lead.empty else np.nan,
        "csi": csi,
        "tss": "UNDEFINED_SMALL_SAMPLE",
        "hss": "UNDEFINED_SMALL_SAMPLE",
        "notes": "TSS/HSS not used as headline because episode evaluation lacks robust true-negative units.",
    }
    return metrics, episode_eval


def make_models() -> dict[str, object]:
    return {
        "logistic_regression_l2": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                penalty="l2",
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                solver="liblinear",
                random_state=42,
            ),
        ),
        "extra_trees_challenger": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=80,
                max_depth=6,
                min_samples_leaf=30,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        "shallow_random_forest_diagnostic": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=80,
                max_depth=6,
                min_samples_leaf=30,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
    }


def extract_importance(model_name: str, estimator: object, feature_cols: list[str], target: str) -> pd.DataFrame:
    final = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
    if hasattr(final, "feature_importances_"):
        importance = final.feature_importances_
        kind = "tree_feature_importance"
    elif hasattr(final, "coef_"):
        importance = np.abs(final.coef_[0])
        kind = "absolute_standardized_coefficient"
    else:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "model": model_name,
            "target": target,
            "feature": feature_cols,
            "importance": importance,
            "importance_type": kind,
        }
    ).sort_values(["target", "model", "importance"], ascending=[True, True, False])


def run_rule_baseline(df: pd.DataFrame, target: str) -> pd.DataFrame:
    score = pd.to_numeric(df.get("hard_score", 0), errors="coerce").fillna(0)
    pred = df[["timestamp", "date", target]].copy()
    pred["model"] = "rule_score_baseline"
    pred["target"] = target
    pred["y_true"] = pred[target].astype(int)
    pred["score"] = score
    pred["predicted_positive"] = score.ge(4).astype(int)
    return pred


def run_model_blocked(df: pd.DataFrame, target: str, feature_cols: list[str], model_name: str, model: object) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    rows = []
    importances = []
    valid_folds = 0
    for date in sorted(df["date"].unique()):
        train = df[df["date"] != date].copy()
        test = df[df["date"] == date].copy()
        if test.empty or train[target].nunique() < 2:
            continue
        valid_folds += 1
        X_train = train[feature_cols].replace([np.inf, -np.inf], np.nan)
        y_train = train[target].astype(int)
        X_test = test[feature_cols].replace([np.inf, -np.inf], np.nan)
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X_test)[:, 1]
        else:
            score = model.decision_function(X_test)
        pred = test[["timestamp", "date", target]].copy()
        pred["model"] = model_name
        pred["target"] = target
        pred["y_true"] = pred[target].astype(int)
        pred["score"] = score
        pred["predicted_positive"] = (score >= 0.5).astype(int)
        rows.append(pred)
        imp = extract_importance(model_name, model, feature_cols, target)
        if not imp.empty:
            imp["heldout_date"] = date
            importances.append(imp)
    if valid_folds < 2:
        return pd.DataFrame(), pd.DataFrame(), "NOT_AVAILABLE_FAIRLY"
    return pd.concat(rows, ignore_index=True), pd.concat(importances, ignore_index=True) if importances else pd.DataFrame(), "AVAILABLE"


def row_metrics(pred: pd.DataFrame) -> dict:
    return {
        "row_level_precision_diagnostic": precision_score(pred["y_true"], pred["predicted_positive"], zero_division=0),
        "row_level_recall_diagnostic": recall_score(pred["y_true"], pred["predicted_positive"], zero_division=0),
        "row_level_f1_diagnostic": f1_score(pred["y_true"], pred["predicted_positive"], zero_division=0),
    }


def write_report(comparison: pd.DataFrame, importance: pd.DataFrame, logistic_available: bool) -> None:
    best_rows = []
    for target, _, _, _ in TARGETS:
        sub = comparison[comparison["target"] == target].copy()
        sub = sub[pd.to_numeric(sub["f1"], errors="coerce").notna()]
        if sub.empty:
            continue
        best = sub.sort_values(["f1", "false_alerts_per_day"], ascending=[False, True]).iloc[0]
        best_rows.append(f"- `{target}`: {best['model']} (F1={best['f1']:.3f}, precision={best['precision']:.3f}, recall={best['recall_pod']:.3f}, false alerts/day={best['false_alerts_per_day']:.2f})")

    top = importance.sort_values("importance", ascending=False).head(15)
    top_lines = [f"- {row.feature}: {row.importance:.4f} ({row.model}, {row.target})" for row in top.itertuples()]

    report = f"""# Forecasting v3 Model Report

## Validation

Headline validation uses leave-one-date-out blocked validation on the v3 feature table. No random row split is used as a headline metric.

Prediction cadence is {PREDICTION_CADENCE_SECONDS} seconds, with active hard-score rows retained.

## Stable v1 Baseline Preserved

The current recommended production-facing prototype remains the v1 90-minute precursor-aware operational alert policy:

- Precision: {V1_BASELINE['precision']:.3f}
- Recall: {V1_BASELINE['recall']:.3f}
- F1: {V1_BASELINE['f1']:.3f}
- False alerts/day: {V1_BASELINE['false_alerts_per_day']:.2f}
- Valid alerted events: {V1_BASELINE['valid_alerted_events']}
- Mean lead time: {V1_BASELINE['mean_lead_time_min']:.2f} min
- Median lead time: {V1_BASELINE['median_lead_time_min']:.2f} min

v3 does not replace this baseline unless it actually outperforms it under blocked validation.

## Model Availability

- Logistic Regression L2: {'AVAILABLE_FAIRLY' if logistic_available else 'NOT_AVAILABLE_FAIRLY'}
- Extra Trees: diagnostic challenger
- Shallow Random Forest: diagnostic challenger
- Rule score baseline: fixed interpretable baseline

## Best v3 Model Per Target

{chr(10).join(best_rows)}

## Top 15 Diagnostic Feature Importances

{chr(10).join(top_lines) if top_lines else '- No feature importances available.'}

## Caveats

- Dataset is small and quality-gated.
- v3 models are diagnostic ML experiments, not replacements for the stable v1 90-minute policy.
- TSS/HSS are marked diagnostic/undefined where robust true-negative episode units are unavailable.
- Peak target results are diagnostic only; onset targets are the forecasting focus.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    full = load_dataset()
    df = modelling_frame(full)
    events = load_events()
    feature_cols = feature_columns(df)

    all_predictions = []
    all_metrics = []
    all_importances = []
    logistic_statuses = []

    for target, horizon, anchor, _role in TARGETS:
        rule_pred = run_rule_baseline(df, target)
        metrics, episodes = evaluate_episode_metrics(rule_pred, events, target, horizon, anchor, "rule_score_baseline")
        metrics.update(row_metrics(rule_pred))
        all_metrics.append(metrics)
        all_predictions.append(rule_pred)

        for model_name, model in make_models().items():
            pred, imp, status = run_model_blocked(df, target, feature_cols, model_name, model)
            if model_name == "logistic_regression_l2":
                logistic_statuses.append(status)
            if pred.empty:
                all_metrics.append(
                    {
                        "model": model_name,
                        "target": target,
                        "target_role": "diagnostic" if "peak" in target else ("primary" if "30min" in target else "secondary"),
                        "validation_method": status,
                        "precision": np.nan,
                        "recall_pod": np.nan,
                        "f1": np.nan,
                        "false_alerts_per_day": np.nan,
                        "valid_alerted_events": np.nan,
                        "total_events": len(events),
                        "useful_alert_episodes": np.nan,
                        "isolated_false_alerts": np.nan,
                        "notes": status,
                    }
                )
                continue
            metrics, episodes = evaluate_episode_metrics(pred, events, target, horizon, anchor, model_name)
            metrics.update(row_metrics(pred))
            all_metrics.append(metrics)
            all_predictions.append(pred)
            if not imp.empty:
                all_importances.append(imp)

    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    comparison = pd.DataFrame(all_metrics)
    importance = pd.concat(all_importances, ignore_index=True) if all_importances else pd.DataFrame()
    if not importance.empty:
        importance = (
            importance.groupby(["model", "target", "feature", "importance_type"], as_index=False)["importance"]
            .mean()
            .sort_values(["target", "model", "importance"], ascending=[True, True, False])
        )

    predictions.to_csv(PREDICTIONS_PATH, index=False)
    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)
    importance.to_csv(IMPORTANCE_PATH, index=False)
    logistic_available = any(status == "AVAILABLE" for status in logistic_statuses)
    write_report(comparison, importance, logistic_available)

    best = (
        comparison[pd.to_numeric(comparison["f1"], errors="coerce").notna()]
        .sort_values(["target", "f1", "false_alerts_per_day"], ascending=[True, False, True])
        .groupby("target")
        .head(1)
    )
    top15 = importance.sort_values("importance", ascending=False).head(15) if not importance.empty else pd.DataFrame()

    print("models trained: rule_score_baseline, logistic_regression_l2, extra_trees_challenger, shallow_random_forest_diagnostic")
    print("validation method: leave-one-date-out blocked validation; no random row split")
    print("best model per target:")
    print(best[["target", "model", "precision", "recall_pod", "f1", "false_alerts_per_day", "valid_alerted_events", "mean_lead_time_min"]].to_string(index=False))
    print(f"whether Logistic Regression was available fairly: {'yes' if logistic_available else 'no'}")
    print("feature importance top 15:")
    if top15.empty:
        print("none")
    else:
        print(top15[["model", "target", "feature", "importance"]].to_string(index=False))
    print("comparison with v1 baseline:")
    print(f"v1 90-min policy precision={V1_BASELINE['precision']:.3f}, recall={V1_BASELINE['recall']:.3f}, F1={V1_BASELINE['f1']:.3f}, false alerts/day={V1_BASELINE['false_alerts_per_day']:.2f}, valid alerted events={V1_BASELINE['valid_alerted_events']}, mean lead={V1_BASELINE['mean_lead_time_min']:.2f} min")
    print("caveats: v3 is diagnostic; small data; blocked validation only; v1 recommendation unchanged unless v3 truly outperforms it")


if __name__ == "__main__":
    main()
