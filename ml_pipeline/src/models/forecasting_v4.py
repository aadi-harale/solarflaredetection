from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


OUT_DIR = Path("results") / "forecasting_v4"
DATASET_PATH = OUT_DIR / "forecasting_v4_dataset.csv"
MASTER_CLASSIFIED_PATH = Path("results") / "master_flare_catalogue_classified_v2.csv"
MASTER_PATH = Path("results") / "master_flare_catalogue.csv"
MODEL_COMPARISON_PATH = OUT_DIR / "forecasting_v4_model_comparison.csv"
PREDICTIONS_PATH = OUT_DIR / "forecasting_v4_predictions.csv"
IMPORTANCE_PATH = OUT_DIR / "forecasting_v4_feature_importance.csv"
REPORT_PATH = OUT_DIR / "forecasting_v4_model_report.md"

TARGETS = [
    ("flare_onset_next_30min", 30, "primary"),
    ("flare_onset_next_60min", 60, "secondary"),
    ("high_class_flare_next_60min", 60, "class_specific"),
    ("low_class_flare_next_60min", 60, "class_specific"),
    ("m_or_x_class_like_next_60min", 60, "diagnostic"),
]
LABEL_COLUMNS = {target for target, _, _ in TARGETS}
PREDICTION_CADENCE_SECONDS = 60
EPISODE_GAP_SECONDS = 60
MAX_TRAIN_NEGATIVES_PER_FOLD = 3000
NEGATIVE_TO_POSITIVE_RATIO = 2
MAX_FEATURES = 70
V3_POLICY = {
    "precision": 0.5151515151515151,
    "recall": 0.8235294117647058,
    "f1": 0.6338215712383488,
    "false_alerts_per_day": 1.3333333333333333,
    "valid_alerted_events": "14 / 17",
    "mean_lead_time_min": 39.44404761904762,
    "median_lead_time_min": 40.18333333333334,
}


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path.resolve()}")


def load_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    require(path)
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed", errors="coerce")
    df = df.dropna(subset=["timestamp"]).copy()
    df["date"] = df["date"].astype(str).str.replace(r"\.0$", "", regex=True)
    return df.sort_values(["date", "timestamp"]).reset_index(drop=True)


def load_events() -> pd.DataFrame:
    path = MASTER_CLASSIFIED_PATH if MASTER_CLASSIFIED_PATH.exists() else MASTER_PATH
    require(path)
    events = pd.read_csv(path)
    events["date"] = events["date"].astype(str).str.replace(r"\.0$", "", regex=True)
    onset_col = "combined_start" if "combined_start" in events.columns else "soft_start"
    peak_col = "soft_peak" if "soft_peak" in events.columns else "combined_peak"
    events["event_onset_time"] = pd.to_datetime(events[onset_col], utc=True, format="mixed", errors="coerce")
    events["event_peak_time"] = pd.to_datetime(events[peak_col], utc=True, format="mixed", errors="coerce")
    events["event_id"] = events["event_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    if "quality_label" not in events.columns:
        events["quality_label"] = ""
    if "surya_estimated_class_group" not in events.columns:
        events["surya_estimated_class_group"] = ""
    if "goes_class_group" not in events.columns:
        events["goes_class_group"] = ""
    return events.dropna(subset=["event_onset_time"]).copy()


def modelling_frame(df: pd.DataFrame) -> pd.DataFrame:
    sampled = df.groupby("date").cumcount().mod(PREDICTION_CADENCE_SECONDS).eq(0)
    active = pd.Series(False, index=df.index)
    if "hard_score" in df.columns:
        active |= pd.to_numeric(df["hard_score"], errors="coerce").fillna(0).ge(6)
    if "soft_score" in df.columns:
        active |= pd.to_numeric(df["soft_score"], errors="coerce").fillna(0).ge(6)
    if "precursor_fusion_score_v4" in df.columns:
        active |= pd.to_numeric(df["precursor_fusion_score_v4"], errors="coerce").fillna(0).ge(0.65)
    if "hard_nonthermal_precursor_score" in df.columns:
        active |= pd.to_numeric(df["hard_nonthermal_precursor_score"], errors="coerce").fillna(0).ge(0.65)
    return df[sampled | active].copy().reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = {"timestamp", "date", "quality_label", "algorithm_name"} | LABEL_COLUMNS
    preferred_fragments = (
        "score",
        "flux",
        "ratio",
        "rise",
        "impulse",
        "burst",
        "qpp",
        "lag_correlation",
        "integral",
        "derivative",
        "median",
        "slope",
        "percentile",
        "background",
        "persistence",
        "acceleration",
    )
    v4_priority = (
        "precursor_fusion_score_v4",
        "soft_gradual_plus_hard_impulsive_score",
        "hard_nonthermal_precursor_score",
        "hard_impulsive_enhancement_score",
        "soft_gradual_enhancement_score",
        "combined_dynamic_range_score",
        "hard_qpp_score",
        "hard_oscillation_persistence_score",
        "hard_before_soft_enhancement_score",
        "soft_small_flare_sensitive_score",
        "hard_impulsive_dynamic_score",
        "hard_to_soft_percentile_ratio",
    )
    cols = []
    for col in df.columns:
        if col in blocked:
            continue
        if not any(fragment in col for fragment in preferred_fragments):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any():
            cols.append(col)
    priority = [col for col in v4_priority if col in cols]
    remaining = [col for col in cols if col not in priority]
    return (priority + remaining)[:MAX_FEATURES]


def deterministic_balanced_train(train: pd.DataFrame, target: str) -> pd.DataFrame:
    positives = train[train[target].astype(int).eq(1)]
    negatives = train[train[target].astype(int).eq(0)]
    if positives.empty or negatives.empty:
        return train
    max_negatives = min(len(negatives), max(MAX_TRAIN_NEGATIVES_PER_FOLD, NEGATIVE_TO_POSITIVE_RATIO * len(positives)))
    if len(negatives) <= max_negatives:
        return train
    step = max(1, len(negatives) // max_negatives)
    sampled_negatives = negatives.iloc[::step].head(max_negatives)
    return pd.concat([positives, sampled_negatives], ignore_index=True).sort_values(["date", "timestamp"])


def make_models() -> dict[str, object]:
    return {
        "logistic_regression_l2": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                penalty="l2",
                C=0.7,
                class_weight="balanced",
                max_iter=1000,
                solver="liblinear",
                random_state=42,
            ),
        ),
        "extra_trees_challenger": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=35,
                max_depth=5,
                min_samples_leaf=35,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        "shallow_random_forest_diagnostic": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(
                n_estimators=35,
                max_depth=5,
                min_samples_leaf=35,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        "gradient_boosting_challenger": make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingClassifier(
                n_estimators=20,
                max_depth=2,
                min_samples_leaf=40,
                learning_rate=0.05,
                random_state=42,
            ),
        ),
    }


def extract_importance(model_name: str, estimator: object, feature_cols: list[str], target: str) -> pd.DataFrame:
    final = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
    if hasattr(final, "feature_importances_"):
        values = final.feature_importances_
        kind = "tree_feature_importance"
    elif hasattr(final, "coef_"):
        values = np.abs(final.coef_[0])
        kind = "absolute_standardized_coefficient"
    else:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "model": model_name,
            "target": target,
            "feature": feature_cols,
            "importance": values,
            "importance_type": kind,
        }
    )


def physics_rule_score(df: pd.DataFrame) -> pd.Series:
    parts = []
    for col in [
        "precursor_fusion_score_v4",
        "soft_gradual_plus_hard_impulsive_score",
        "hard_nonthermal_precursor_score",
        "hard_qpp_score",
        "combined_dynamic_range_score",
    ]:
        if col in df.columns:
            parts.append(pd.to_numeric(df[col], errors="coerce").fillna(0).clip(0, 1))
    if not parts:
        return pd.Series(0.0, index=df.index)
    return pd.concat(parts, axis=1).mean(axis=1).clip(0, 1)


def run_rule_baseline(df: pd.DataFrame, target: str) -> pd.DataFrame:
    score = physics_rule_score(df)
    pred = df[["timestamp", "date", "quality_label", target]].copy()
    pred["model"] = "physics_rule_score_v4"
    pred["target"] = target
    pred["y_true"] = pred[target].astype(int)
    pred["score"] = score
    pred["predicted_positive"] = score.ge(0.35).astype(int)
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
        train = deterministic_balanced_train(train, target)
        valid_folds += 1
        X_train = train[feature_cols].replace([np.inf, -np.inf], np.nan)
        y_train = train[target].astype(int)
        X_test = test[feature_cols].replace([np.inf, -np.inf], np.nan)
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X_test)[:, 1]
        else:
            score = model.decision_function(X_test)
            score = (score - np.nanmin(score)) / (np.nanmax(score) - np.nanmin(score) + 1e-9)
        pred = test[["timestamp", "date", "quality_label", target]].copy()
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
    return (
        pd.concat(rows, ignore_index=True),
        pd.concat(importances, ignore_index=True) if importances else pd.DataFrame(),
        "AVAILABLE",
    )


def merge_alert_episodes(pred: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    positives = pred[pred["predicted_positive"].eq(1)].copy()
    if positives.empty:
        return pd.DataFrame(columns=["date", "alert_start", "alert_end", "max_score", "row_count", "quality_label"])
    rows = []
    for date, group in positives.sort_values(["date", "timestamp"]).groupby("date"):
        start = end = previous = None
        max_score = -np.inf
        count = 0
        quality = ""
        for _, row in group.iterrows():
            ts = row["timestamp"]
            if start is None or (ts - previous).total_seconds() > EPISODE_GAP_SECONDS:
                if start is not None:
                    rows.append({"date": date, "alert_start": start, "alert_end": end, "max_score": max_score, "row_count": count, "quality_label": quality})
                start = ts
                max_score = float(row[score_col])
                count = 1
                quality = row.get("quality_label", "")
            else:
                max_score = max(max_score, float(row[score_col]))
                count += 1
            end = ts
            previous = ts
        if start is not None:
            rows.append({"date": date, "alert_start": start, "alert_end": end, "max_score": max_score, "row_count": count, "quality_label": quality})
    return pd.DataFrame(rows)


def evaluate_episode_metrics(pred: pd.DataFrame, events: pd.DataFrame, target: str, horizon_min: int, model_name: str) -> tuple[dict, pd.DataFrame]:
    episodes = merge_alert_episodes(pred)
    event_hits: dict[str, dict] = {}
    episode_rows = []
    for _, episode in episodes.iterrows():
        same_date = events[events["date"].eq(str(episode["date"]))].copy()
        candidates = same_date[
            (same_date["event_onset_time"] > episode["alert_start"])
            & (same_date["event_onset_time"] <= episode["alert_start"] + pd.Timedelta(minutes=horizon_min))
        ].copy()
        if candidates.empty:
            episode_type = "ISOLATED_FALSE_ALERT"
            matched_event_id = ""
            lead_time = np.nan
        else:
            candidates["lead_time_min"] = (candidates["event_onset_time"] - episode["alert_start"]).dt.total_seconds() / 60.0
            match = candidates.sort_values("lead_time_min").iloc[0]
            episode_type = "USEFUL_ALERT_EPISODE"
            matched_event_id = str(match["event_id"])
            lead_time = float(match["lead_time_min"])
            if matched_event_id not in event_hits or episode["alert_start"] < event_hits[matched_event_id]["first_alert_time"]:
                event_hits[matched_event_id] = {
                    "event_id": matched_event_id,
                    "first_alert_time": episode["alert_start"],
                    "lead_time_min": lead_time,
                    "goes_class_group": match.get("goes_class_group", ""),
                    "surya_estimated_class_group": match.get("surya_estimated_class_group", ""),
                }
        episode_rows.append({**episode.to_dict(), "model": model_name, "target": target, "episode_type": episode_type, "matched_event_id": matched_event_id, "lead_time_min": lead_time})

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
    class_recall = {}
    if not hit_df.empty:
        for cls in ["LOW_OR_MODERATE", "HIGH"]:
            class_events = events[
                events.get("surya_estimated_class_group", pd.Series("", index=events.index)).astype(str).eq(cls)
            ]
            if not class_events.empty:
                hit_ids = set(hit_df["event_id"].astype(str))
                class_recall[f"{cls.lower()}_event_recall"] = len(set(class_events["event_id"].astype(str)) & hit_ids) / len(class_events)
    metrics = {
        "model": model_name,
        "target": target,
        "target_role": "diagnostic" if "m_or_x" in target else ("class_specific" if "class" in target else ("primary" if "30min" in target else "secondary")),
        "validation_method": "leave-one-date-out blocked validation" if model_name != "physics_rule_score_v4" else "physics rule score evaluated on blocked-cadence v4 frame",
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
        "tss_hss_reliability_flag": "COARSE_EPISODE_DIAGNOSTIC_ONLY",
        "notes": "TSS/HSS diagnostic only; robust true-negative episode units are not available in this small sample.",
        **class_recall,
    }
    return metrics, episode_eval


def row_metrics(pred: pd.DataFrame) -> dict:
    y_true = pred["y_true"].astype(int)
    y_pred = pred["predicted_positive"].astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tss_den = (tp + fn) * (fp + tn)
    hss_den = ((tp + fn) * (fn + tn)) + ((tp + fp) * (fp + tn))
    tss = (tp / (tp + fn) - fp / (fp + tn)) if tss_den and (tp + fn) and (fp + tn) else "UNDEFINED_SMALL_SAMPLE"
    hss = (2 * (tp * tn - fp * fn) / hss_den) if hss_den else "UNDEFINED_SMALL_SAMPLE"
    return {
        "row_level_precision_diagnostic": precision_score(y_true, y_pred, zero_division=0),
        "row_level_recall_diagnostic": recall_score(y_true, y_pred, zero_division=0),
        "row_level_f1_diagnostic": f1_score(y_true, y_pred, zero_division=0),
        "row_level_tss_diagnostic": tss,
        "row_level_hss_diagnostic": hss,
        "row_level_units": len(pred),
        "row_level_positive_units": int(y_true.sum()),
        "row_level_negative_units": int((1 - y_true).sum()),
    }


def write_report(comparison: pd.DataFrame, importance: pd.DataFrame, logistic_available: bool, gradient_available: bool) -> None:
    usable = comparison[pd.to_numeric(comparison["f1"], errors="coerce").notna()].copy()
    best_rows = []
    for target, _, _ in TARGETS:
        sub = usable[usable["target"].eq(target)]
        if sub.empty:
            continue
        best = sub.sort_values(["f1", "false_alerts_per_day"], ascending=[False, True]).iloc[0]
        best_rows.append(f"- `{target}`: {best['model']} (F1={best['f1']:.3f}, precision={best['precision']:.3f}, recall={best['recall_pod']:.3f}, false alerts/day={best['false_alerts_per_day']:.2f})")
    top = importance.sort_values("importance", ascending=False).head(15) if not importance.empty else pd.DataFrame()
    top_lines = [f"- {row.feature}: {row.importance:.4f} ({row.model}, {row.target})" for row in top.itertuples()]
    report = f"""# Forecasting v4 Model Report

## Core Algorithm

SuryaAlert-XF: Aditya-L1 Soft-Hard X-ray Fusion Forecasting Algorithm.

## Validation

Headline validation uses leave-one-date-out blocked validation on the v4 feature table. No random row split is used.

Prediction cadence is {PREDICTION_CADENCE_SECONDS} seconds, with active physics-score rows retained.

## v3 Baseline Preserved

Best current Phase 3 state-machine policy:

- Precision: {V3_POLICY['precision']:.3f}
- Recall: {V3_POLICY['recall']:.3f}
- F1: {V3_POLICY['f1']:.3f}
- False alerts/day: {V3_POLICY['false_alerts_per_day']:.2f}
- Valid alerted events: {V3_POLICY['valid_alerted_events']}
- Mean lead time: {V3_POLICY['mean_lead_time_min']:.2f} min
- Median lead time: {V3_POLICY['median_lead_time_min']:.2f} min

v4 only replaces v3 if it beats the explicit Phase 4B rule.

## Model Availability

- Physics rule score v4: available
- Logistic Regression L2: {'AVAILABLE_FAIRLY' if logistic_available else 'NOT_AVAILABLE_FAIRLY'}
- Extra Trees: available diagnostic challenger
- Shallow Random Forest: available diagnostic challenger
- Gradient Boosting: {'AVAILABLE_FAIRLY' if gradient_available else 'NOT_AVAILABLE_FAIRLY'}

## Best v4 Model Per Target

{chr(10).join(best_rows)}

## Top 15 Diagnostic Feature Importances

{chr(10).join(top_lines) if top_lines else '- No feature importances available.'}

## Caveats

- v4 is an advanced research extension unless the policy layer beats v3 under the predefined rule.
- TSS/HSS are diagnostic only due to small sample size and limited true-negative episode units.
- Class-specific targets are useful diagnostics, not standalone operational class forecasts.
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
    statuses: dict[str, list[str]] = {}

    for target, horizon, _role in TARGETS:
        rule_pred = run_rule_baseline(df, target)
        metrics, _episodes = evaluate_episode_metrics(rule_pred, events, target, horizon, "physics_rule_score_v4")
        metrics.update(row_metrics(rule_pred))
        all_metrics.append(metrics)
        all_predictions.append(rule_pred)

        for model_name, model in make_models().items():
            pred, imp, status = run_model_blocked(df, target, feature_cols, model_name, model)
            statuses.setdefault(model_name, []).append(status)
            if pred.empty:
                all_metrics.append(
                    {
                        "model": model_name,
                        "target": target,
                        "target_role": _role,
                        "validation_method": status,
                        "precision": np.nan,
                        "recall_pod": np.nan,
                        "f1": np.nan,
                        "false_alerts_per_day": np.nan,
                        "valid_alerted_events": np.nan,
                        "total_events": len(events),
                        "notes": status,
                    }
                )
                continue
            metrics, _episodes = evaluate_episode_metrics(pred, events, target, horizon, model_name)
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
    logistic_available = any(s == "AVAILABLE" for s in statuses.get("logistic_regression_l2", []))
    gradient_available = any(s == "AVAILABLE" for s in statuses.get("gradient_boosting_challenger", []))
    write_report(comparison, importance, logistic_available, gradient_available)

    best = (
        comparison[pd.to_numeric(comparison["f1"], errors="coerce").notna()]
        .sort_values(["target", "f1", "false_alerts_per_day"], ascending=[True, False, True])
        .groupby("target")
        .head(1)
    )
    top15 = importance.sort_values("importance", ascending=False).head(15) if not importance.empty else pd.DataFrame()

    print("models trained: physics_rule_score_v4, logistic_regression_l2, extra_trees_challenger, shallow_random_forest_diagnostic, gradient_boosting_challenger")
    print("validation method: leave-one-date-out blocked validation; no random row split")
    print("best v4 model per target:")
    print(best[["target", "model", "precision", "recall_pod", "f1", "false_alerts_per_day", "valid_alerted_events", "mean_lead_time_min"]].to_string(index=False))
    print(f"whether Logistic Regression was available fairly: {'yes' if logistic_available else 'no'}")
    print(f"whether Gradient Boosting was available fairly: {'yes' if gradient_available else 'no'}")
    print("feature importance top 15:")
    if top15.empty:
        print("none")
    else:
        print(top15[["model", "target", "feature", "importance"]].to_string(index=False))
    print("comparison with v3:")
    print(f"v3 policy precision={V3_POLICY['precision']:.3f}, recall={V3_POLICY['recall']:.3f}, F1={V3_POLICY['f1']:.3f}, false alerts/day={V3_POLICY['false_alerts_per_day']:.2f}, valid alerted events={V3_POLICY['valid_alerted_events']}, mean lead={V3_POLICY['mean_lead_time_min']:.2f} min")
    print("caveats: v4 model scores feed a separate policy layer; replacement is decided only by scripts/24_forecasting_v4_policy_sweep.py")


if __name__ == "__main__":
    main()
