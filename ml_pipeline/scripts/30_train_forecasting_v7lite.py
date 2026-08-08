from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.forecasting_v7lite_xray_features import V7LITE_FEATURE_COLUMNS


OUT_DIR = PROJECT_ROOT / "results" / "forecasting_v7lite"
DATASET_PATH = OUT_DIR / "forecasting_v7lite_dataset.csv"
CATALOGUE_PATH = PROJECT_ROOT / "results" / "forecasting_v6" / "combined_nowcast_catalogue_clean.csv"
MODEL_COMPARISON_PATH = OUT_DIR / "forecasting_v7lite_model_comparison.csv"
POLICY_SWEEP_PATH = OUT_DIR / "forecasting_v7lite_policy_sweep.csv"
IMPORTANCE_PATH = OUT_DIR / "forecasting_v7lite_feature_importance.csv"
REPORT_PATH = OUT_DIR / "forecasting_v7lite_report.md"
COMPARISON_PATH = OUT_DIR / "v3_v6_v7lite_comparison.csv"
COMPARISON_MD_PATH = OUT_DIR / "v3_v6_v7lite_comparison.md"
PREDICTIONS_PATH = OUT_DIR / "forecasting_v7lite_predictions.csv"

V3 = {"system": "v3_high_recall", "precision": 0.515, "recall": 0.824, "f1": 0.634, "false_alerts_per_day": 1.33, "valid_alerted_events": 14, "total_events": 17, "mean_lead_time_min": 39.44, "median_lead_time_min": 40.18}
V6 = {"system": "v6_low_false_alert", "precision": 0.600, "recall": 0.7333333333333333, "f1": 0.660, "false_alerts_per_day": 1.1111111111111112, "valid_alerted_events": 11, "total_events": 15, "mean_lead_time_min": 40.1939393939394, "median_lead_time_min": 32.083333333333336}


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path.resolve()}")


def load_dataset() -> pd.DataFrame:
    require(DATASET_PATH)
    df = pd.read_csv(DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed", errors="coerce")
    df["date"] = df["date"].astype(str).str.replace(r"\.0$", "", regex=True)
    return df.dropna(subset=["timestamp"]).sort_values(["date", "timestamp"]).reset_index(drop=True)


def load_events() -> pd.DataFrame:
    require(CATALOGUE_PATH)
    events = pd.read_csv(CATALOGUE_PATH)
    if events.empty:
        return pd.DataFrame(columns=["event_id", "date", "event_onset_time"])
    events["date"] = events["source_date"].astype(str).str.replace(r"\.0$", "", regex=True)
    events["event_id"] = events.get("v6_event_uid", events.get("event_id", "")).astype(str)
    events["event_onset_time"] = pd.to_datetime(events["event_start"], utc=True, format="mixed", errors="coerce")
    return events.dropna(subset=["event_onset_time"]).copy()


def modelling_frame(df: pd.DataFrame) -> pd.DataFrame:
    sampled = df.groupby("date").cumcount().mod(60).eq(0)
    active = pd.to_numeric(df.get("hard_score", 0), errors="coerce").fillna(0).ge(4)
    fusion = pd.to_numeric(df.get("soft_hard_precursor_fusion_score", 0), errors="coerce").fillna(0).ge(3.5)
    return df[sampled | active | fusion].copy().reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    blocked = {"timestamp", "source_date", "date", "quality_label", "is_quiet_day", "inside_detected_event"}
    cols = []
    for col in df.columns:
        if col in blocked or col.startswith("flare_next_") or col.startswith("time_to_"):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any():
            cols.append(col)
    return cols


def make_models() -> dict[str, object]:
    return {
        "v6_best_extra_trees_baseline": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(n_estimators=40, max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=42, n_jobs=-1),
        ),
        "extra_trees_v7lite": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(n_estimators=60, max_depth=7, min_samples_leaf=18, class_weight="balanced", random_state=43, n_jobs=-1),
        ),
        "gradient_boosting_v7lite": make_pipeline(
            SimpleImputer(strategy="median"),
            GradientBoostingClassifier(n_estimators=40, max_depth=2, learning_rate=0.05, random_state=42),
        ),
        "logistic_regression_l2": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=1000, solver="liblinear", random_state=42),
        ),
    }


def extract_importance(model_name: str, model: object, features: list[str], target: str) -> pd.DataFrame:
    final = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(final, "feature_importances_"):
        vals = final.feature_importances_
        kind = "tree_feature_importance"
    elif hasattr(final, "coef_"):
        vals = np.abs(final.coef_[0])
        kind = "absolute_standardized_coefficient"
    else:
        return pd.DataFrame()
    return pd.DataFrame({"model": model_name, "target": target, "feature": features, "importance": vals, "importance_type": kind})


def run_blocked_models(df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = [c for c in ["flare_next_30min", "flare_next_60min"] if c in df.columns]
    predictions = []
    importances = []
    for target in targets:
        for model_name, model in make_models().items():
            rows = []
            valid_folds = 0
            for date in sorted(df["date"].unique()):
                train = df[df["date"] != date].copy()
                test = df[df["date"] == date].copy()
                if train[target].nunique() < 2 or test.empty:
                    continue
                valid_folds += 1
                X_train = train[features].replace([np.inf, -np.inf], np.nan)
                X_test = test[features].replace([np.inf, -np.inf], np.nan)
                y_train = train[target].astype(int)
                model.fit(X_train, y_train)
                score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
                pred = test[["timestamp", "date", target, "hard_score", "soft_hard_precursor_fusion_score"]].copy()
                pred["model"] = model_name
                pred["target"] = target
                pred["y_true"] = pred[target].astype(int)
                pred["score"] = score
                pred["predicted_positive"] = (score >= 0.5).astype(int)
                rows.append(pred)
                imp = extract_importance(model_name, model, features, target)
                if not imp.empty:
                    imp["heldout_date"] = date
                    importances.append(imp)
            if valid_folds >= 2 and rows:
                predictions.append(pd.concat(rows, ignore_index=True))
    pred = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    imp = pd.concat(importances, ignore_index=True) if importances else pd.DataFrame()
    if not imp.empty:
        imp = imp.groupby(["model", "target", "feature", "importance_type"], as_index=False)["importance"].mean()
    pred.to_csv(PREDICTIONS_PATH, index=False)
    imp.sort_values("importance", ascending=False).to_csv(IMPORTANCE_PATH, index=False)
    return pred, imp


def infer_episode_gap_seconds(pred: pd.DataFrame, gap_seconds: int | None = None) -> int:
    if gap_seconds is not None:
        return int(gap_seconds)
    if pred.empty or "timestamp" not in pred.columns:
        return 120
    ts = pd.to_datetime(pred["timestamp"], utc=True, format="mixed", errors="coerce").dropna().sort_values()
    cadence = ts.diff().dt.total_seconds().dropna()
    cadence = cadence[cadence > 0]
    if cadence.empty:
        return 120
    return int(max(120, 2 * float(cadence.median())))


def merge_episodes(pred: pd.DataFrame, gap_seconds: int | None = None) -> pd.DataFrame:
    gap_seconds = infer_episode_gap_seconds(pred, gap_seconds)
    positives = pred[pred["predicted_positive"].eq(1)].copy()
    if positives.empty:
        return pd.DataFrame(columns=["date", "alert_start", "alert_end", "max_score"])
    rows = []
    for date, group in positives.sort_values(["date", "timestamp"]).groupby("date"):
        start = end = prev = None
        max_score = -np.inf
        for _, row in group.iterrows():
            ts = row["timestamp"]
            if start is None or (ts - prev).total_seconds() > gap_seconds:
                if start is not None:
                    rows.append({"date": date, "alert_start": start, "alert_end": end, "max_score": max_score})
                start = ts
                max_score = float(row["score"])
            else:
                max_score = max(max_score, float(row["score"]))
            end = ts
            prev = ts
        if start is not None:
            rows.append({"date": date, "alert_start": start, "alert_end": end, "max_score": max_score})
    return pd.DataFrame(rows)


def evaluate(pred: pd.DataFrame, events: pd.DataFrame, model: str, target: str) -> dict:
    episodes = merge_episodes(pred)
    event_hits = {}
    useful = false = 0
    for _, ep in episodes.iterrows():
        same = events[events["date"].eq(str(ep["date"]))]
        candidates = same[
            (same["event_onset_time"] > ep["alert_start"])
            & (same["event_onset_time"] <= ep["alert_start"] + pd.Timedelta(minutes=90))
        ].copy()
        if candidates.empty:
            false += 1
        else:
            candidates["lead_time_min"] = (candidates["event_onset_time"] - ep["alert_start"]).dt.total_seconds() / 60.0
            match = candidates.sort_values("lead_time_min").iloc[0]
            useful += 1
            eid = str(match["event_id"])
            if eid not in event_hits or ep["alert_start"] < event_hits[eid]["alert_start"]:
                event_hits[eid] = {"alert_start": ep["alert_start"], "lead_time_min": float(match["lead_time_min"])}
    total = len(episodes)
    total_events = len(events)
    valid = len(event_hits)
    precision = useful / total if total else 0.0
    recall = valid / total_events if total_events else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    lead = pd.Series([v["lead_time_min"] for v in event_hits.values()], dtype=float)
    return {
        "model": model,
        "target": target,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alerts_per_day": false / max(1, pred["date"].nunique()),
        "valid_alerted_events": valid,
        "total_events": total_events,
        "useful_alert_episodes": useful,
        "isolated_false_alerts": false,
        "total_alert_episodes": total,
        "mean_lead_time_min": float(lead.mean()) if not lead.empty else np.nan,
        "median_lead_time_min": float(lead.median()) if not lead.empty else np.nan,
    }


def model_comparison(pred: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, target), group in pred.groupby(["model", "target"]):
        rows.append(evaluate(group, events, model, target))
    out = pd.DataFrame(rows).sort_values(["f1", "false_alerts_per_day"], ascending=[False, True])
    out.to_csv(MODEL_COMPARISON_PATH, index=False)
    return out


def run_state_machine(frame: pd.DataFrame, p30: float, p60: float, bins: int, cooldown_min: int) -> pd.DataFrame:
    rows = []
    cooldown = pd.Timedelta(minutes=cooldown_min)
    for date, group in frame.sort_values(["date", "timestamp"]).groupby("date"):
        watch = 0
        cooldown_until = pd.Timestamp.min.tz_localize("UTC")
        for row in group.itertuples():
            ts = row.timestamp
            fusion = float(getattr(row, "soft_hard_precursor_fusion_score", 0.0) or 0.0)
            if ts < cooldown_until:
                pos = 0
                state = "COOLDOWN"
                watch = 0
            elif float(row.p30) >= p30 or (float(row.p60) >= p60 and fusion >= 1.0) or float(getattr(row, "hard_score", 0.0) or 0.0) >= 4.0:
                pos = 1
                state = "ALERT"
                cooldown_until = ts + cooldown
                watch = 0
            elif float(row.p60) >= p60 * 0.85 or fusion >= 1.5:
                watch += 1
                pos = 0
                state = "WATCH" if watch >= bins else "CLEAR"
            else:
                watch = 0
                pos = 0
                state = "CLEAR"
            rows.append({"timestamp": ts, "date": date, "model": row.model, "state": state, "predicted_positive": pos, "score": max(float(row.p30), float(row.p60))})
    return pd.DataFrame(rows)


def policy_sweep(pred: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        pred.pivot_table(
            index=["timestamp", "date", "model", "hard_score", "soft_hard_precursor_fusion_score"],
            columns="target",
            values="score",
            aggfunc="max",
        )
        .reset_index()
        .rename(columns={"flare_next_30min": "p30", "flare_next_60min": "p60"})
    )
    pivot["p30"] = pd.to_numeric(pivot.get("p30", 0), errors="coerce").fillna(0.0)
    pivot["p60"] = pd.to_numeric(pivot.get("p60", 0), errors="coerce").fillna(0.0)
    rows = []
    for model in sorted(pivot["model"].unique()):
        sub = pivot[pivot["model"].eq(model)].copy()
        for p30 in [0.5, 0.6, 0.7, 0.8]:
            for p60 in [0.3, 0.4, 0.5, 0.6]:
                for bins in [2, 3, 6]:
                    for cooldown in [30, 45, 60, 90]:
                        sm = run_state_machine(sub, p30, p60, bins, cooldown)
                        row = evaluate(sm, events, model, "v7lite_state_machine_90min")
                        row.update({"p30_threshold": p30, "p60_threshold": p60, "consecutive_bins": bins, "cooldown_min": cooldown})
                        rows.append(row)
    out = pd.DataFrame(rows).sort_values(["f1", "false_alerts_per_day", "recall"], ascending=[False, True, False])
    out.to_csv(POLICY_SWEEP_PATH, index=False)
    return out


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            value = row[col]
            vals.append(f"{value:.4g}" if isinstance(value, float) and not pd.isna(value) else str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def decide(best: pd.Series) -> tuple[str, str]:
    precision = float(best["precision"])
    recall = float(best["recall"])
    f1 = float(best["f1"])
    far = float(best["false_alerts_per_day"])
    if f1 > 0.660 and far <= 1.33 and recall >= 0.733:
        return "v7-Lite balanced mode", "v7-Lite beats the v6 balanced-mode rule."
    if recall > 0.824 and far <= 1.50:
        return "v7-Lite high-recall mode", "v7-Lite beats the v3 high-recall rule."
    return "v6 balanced low-false-alert mode and v3 high-recall mode", "v7-Lite does not beat the predefined v6/v3 replacement rules."


def write_reports(best: pd.Series, comparison: pd.DataFrame, imp: pd.DataFrame, dataset_rows: int, event_count: int, feature_count: int, recommendation: str, reason: str) -> None:
    final_rows = pd.DataFrame([V3, V6, {
        "system": "v7lite_xray_precursor",
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "f1": float(best["f1"]),
        "false_alerts_per_day": float(best["false_alerts_per_day"]),
        "valid_alerted_events": int(best["valid_alerted_events"]),
        "total_events": int(best["total_events"]),
        "mean_lead_time_min": float(best["mean_lead_time_min"]),
        "median_lead_time_min": float(best["median_lead_time_min"]),
    }])
    final_rows["final_recommendation"] = recommendation
    final_rows.to_csv(COMPARISON_PATH, index=False)
    COMPARISON_MD_PATH.write_text("# v3 vs v6 vs v7-Lite Comparison\n\n" + md_table(final_rows) + "\n", encoding="utf-8")
    top = imp.sort_values("importance", ascending=False).head(20) if not imp.empty else pd.DataFrame()
    report = f"""# Forecasting v7-Lite Report

v7-Lite adds physics-guided Aditya-L1 X-ray precursor features inspired by observed soft X-ray preflare enhancement and hard X-ray impulsive/oscillatory behavior. These features are evaluated using the same blocked date-wise validation as earlier models.

## Decision

Final recommended mode: **{recommendation}**.

Reason: {reason}

## Best v7-Lite Policy

- Model: `{best['model']}`
- Precision: {float(best['precision']):.3f}
- Recall/POD: {float(best['recall']):.3f}
- F1: {float(best['f1']):.3f}
- False alerts/day: {float(best['false_alerts_per_day']):.2f}
- Valid alerted events: {int(best['valid_alerted_events'])} / {int(best['total_events'])}
- Mean lead time: {float(best['mean_lead_time_min']):.2f} min
- Median lead time: {float(best['median_lead_time_min']):.2f} min

## Dataset

- Rows: {dataset_rows:,}
- Events: {event_count}
- Numeric features used: {feature_count}
- Added v7-Lite feature columns: {len(V7LITE_FEATURE_COLUMNS)}

## Caveats

- No nowcasting logic was changed.
- QPP/oscillation features are proxy scores only, not statistically proven QPP detections.
- No magnetogram branch, 3h/6h labels, or operational-readiness claim is introduced.
- Scores are not claimed as fully calibrated probabilities.
"""
    if not top.empty:
        report += "\n## Top Diagnostic Feature Importances\n\n" + md_table(top[["model", "target", "feature", "importance", "importance_type"]].head(20)) + "\n"
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_dataset()
    events = load_events()
    frame = modelling_frame(data)
    features = feature_columns(frame)
    pred, imp = run_blocked_models(frame, features)
    comparison = model_comparison(pred, events)
    sweep = policy_sweep(pred, events)
    best = sweep.iloc[0]
    recommendation, reason = decide(best)
    write_reports(best, comparison, imp, len(data), len(events), len(features), recommendation, reason)

    print(f"dataset rows: {len(data):,}")
    print(f"event count: {len(events)}")
    print(f"feature count: {len(features)}")
    print(f"best v7-Lite model: {best['model']}")
    print(f"v7-Lite precision: {float(best['precision']):.3f}")
    print(f"v7-Lite recall: {float(best['recall']):.3f}")
    print(f"v7-Lite F1: {float(best['f1']):.3f}")
    print(f"v7-Lite false alerts/day: {float(best['false_alerts_per_day']):.2f}")
    print(f"lead time: mean={float(best['mean_lead_time_min']):.2f} min, median={float(best['median_lead_time_min']):.2f} min")
    print(f"whether v7-Lite beats v6: {'yes' if recommendation.startswith('v7-Lite balanced') else 'no'}")
    print(f"final recommended mode: {recommendation}")


if __name__ == "__main__":
    main()
