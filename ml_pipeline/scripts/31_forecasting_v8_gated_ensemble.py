from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.forecasting_v8_model_zoo import (
    blocked_oof_predictions,
    feature_columns,
    model_specs,
    modelling_frame,
    rule_score_predictions,
)


OUT_DIR = PROJECT_ROOT / "results" / "forecasting_v8"
DATASET_PATH = PROJECT_ROOT / "results" / "forecasting_v7lite" / "forecasting_v7lite_dataset.csv"
CATALOGUE_PATH = PROJECT_ROOT / "results" / "forecasting_v6" / "combined_nowcast_catalogue_clean.csv"
MODEL_ZOO_PATH = OUT_DIR / "v8_model_zoo_comparison.csv"
MODEL_ZOO_REPORT_PATH = OUT_DIR / "v8_model_zoo_report.md"
CALIBRATION_REPORT_PATH = OUT_DIR / "v8_calibrated_probability_report.md"
RELIABILITY_PATH = OUT_DIR / "v8_reliability_table.csv"
PREDICTIONS_PATH = OUT_DIR / "v8_oof_predictions.csv"
V8_COMPARISON_PATH = OUT_DIR / "v8_vs_v3_v6_v7lite_comparison.csv"

V3 = {"system": "v3_high_recall_baseline", "precision": 0.515, "recall": 0.824, "f1": 0.634, "false_alerts_per_day": 1.33, "valid_alerted_events": 14, "total_events": 17, "mean_lead_time_min": 39.44, "median_lead_time_min": 40.18}
V6 = {"system": "v6_best_policy_baseline", "precision": 0.600, "recall": 0.7333333333333333, "f1": 0.660, "false_alerts_per_day": 1.1111111111111112, "valid_alerted_events": 11, "total_events": 15, "mean_lead_time_min": 40.1939393939394, "median_lead_time_min": 32.083333333333336}
V7 = {"system": "v7lite_physics_baseline", "precision": 0.45283, "recall": 0.866667, "f1": 0.594852, "false_alerts_per_day": 3.222222, "valid_alerted_events": 13, "total_events": 15, "mean_lead_time_min": 33.326923, "median_lead_time_min": 33.116667}


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path.resolve()}")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    require(DATASET_PATH)
    require(CATALOGUE_PATH)
    df = pd.read_csv(DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed", errors="coerce")
    df["date"] = df["date"].astype(str).str.replace(r"\.0$", "", regex=True)
    df = df.dropna(subset=["timestamp"]).sort_values(["date", "timestamp"]).reset_index(drop=True)
    events = pd.read_csv(CATALOGUE_PATH)
    events["date"] = events["source_date"].astype(str).str.replace(r"\.0$", "", regex=True)
    events["event_id"] = events.get("v6_event_uid", events.get("event_id", "")).astype(str)
    events["event_onset_time"] = pd.to_datetime(events["event_start"], utc=True, format="mixed", errors="coerce")
    events = events.dropna(subset=["event_onset_time"]).copy()
    return df, events


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


def merge_episodes(pred: pd.DataFrame, threshold: float, gap_seconds: int | None = None) -> pd.DataFrame:
    gap_seconds = infer_episode_gap_seconds(pred, gap_seconds)
    positives = pred[pred["score"].ge(threshold)].copy()
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


def evaluate_at_threshold(pred: pd.DataFrame, events: pd.DataFrame, threshold: float) -> dict:
    episodes = merge_episodes(pred, threshold)
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
        "threshold": threshold,
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


def best_threshold_metrics(pred: pd.DataFrame, events: pd.DataFrame) -> dict:
    rows = [evaluate_at_threshold(pred, events, thr) for thr in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]]
    return sorted(rows, key=lambda r: (r["f1"], -r["false_alerts_per_day"], r["recall"], r["precision"]), reverse=True)[0]


def build_predictions(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[dict], list[str], list[str]]:
    target = "flare_next_60min" if "flare_next_60min" in frame.columns else "flare_next_30min"
    predictions = [rule_score_predictions(frame, target)]
    tried = ["rule_score_baseline"]
    skipped = []
    for spec in model_specs():
        if spec.estimator is None:
            skipped.append(f"{spec.name}: {spec.skipped_reason}")
            continue
        pred, status = blocked_oof_predictions(frame, target, features, spec)
        if pred.empty:
            skipped.append(f"{spec.name}: {status}")
        else:
            predictions.append(pred)
            tried.append(spec.name)
    all_pred = pd.concat(predictions, ignore_index=True)
    return all_pred, [], tried, skipped


def make_stacker_predictions(pred: pd.DataFrame, frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pivot = pred.pivot_table(index=["timestamp", "date"], columns="model", values="score", aggfunc="max").reset_index()
    labels = frame[["timestamp", "date", target]].drop_duplicates(["timestamp", "date"])
    pivot = pivot.merge(labels, on=["timestamp", "date"], how="inner")
    model_cols = [c for c in pivot.columns if c not in {"timestamp", "date", target}]
    rows = []
    for date in sorted(pivot["date"].unique()):
        train = pivot[pivot["date"] != date]
        test = pivot[pivot["date"] == date]
        if train[target].nunique() < 2 or len(model_cols) < 2:
            continue
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="liblinear")
        clf.fit(train[model_cols].fillna(0), train[target].astype(int))
        score = clf.predict_proba(test[model_cols].fillna(0))[:, 1]
        out = test[["timestamp", "date", target]].copy()
        out["model"] = "l2_logistic_oof_stacker"
        out["target"] = target
        out["y_true"] = out[target].astype(int)
        out["score"] = score
        rows.append(out)
    logistic = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    rows = []
    if pivot["date"].nunique() >= 6 and len(model_cols) >= 3:
        for date in sorted(pivot["date"].unique()):
            train = pivot[pivot["date"] != date]
            test = pivot[pivot["date"] == date]
            if train[target].nunique() < 2:
                continue
            clf = ExtraTreesClassifier(n_estimators=40, max_depth=4, min_samples_leaf=20, class_weight="balanced", random_state=42, n_jobs=-1)
            clf.fit(train[model_cols].fillna(0), train[target].astype(int))
            score = clf.predict_proba(test[model_cols].fillna(0))[:, 1]
            out = test[["timestamp", "date", target]].copy()
            out["model"] = "extra_trees_oof_stacker"
            out["target"] = target
            out["y_true"] = out[target].astype(int)
            out["score"] = score
            rows.append(out)
    extra = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return logistic, extra


def reliability_bins(pred: pd.DataFrame, model: str, score_col: str = "score") -> pd.DataFrame:
    sub = pred[pred["model"].eq(model)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["score_bin"] = pd.cut(sub[score_col].clip(0, 1), bins=np.linspace(0, 1, 6), include_lowest=True)
    return (
        sub.groupby("score_bin", observed=False)
        .agg(sample_count=("y_true", "size"), mean_score=(score_col, "mean"), observed_event_rate=("y_true", "mean"))
        .reset_index()
        .assign(model=model, score_type=score_col)
    )


def calibrate_top_models(pred: pd.DataFrame, top_models: list[str]) -> tuple[pd.DataFrame, str]:
    rel_rows = []
    lines = ["# v8 Calibrated Probability Report", "", "Current probabilities are diagnostic; blocked OOF scores are used for calibration checks.", ""]
    calibrated_frames = []
    for model in top_models:
        sub = pred[pred["model"].eq(model)].copy().reset_index(drop=True)
        if sub.empty or sub["y_true"].nunique() < 2:
            continue
        rel_rows.append(reliability_bins(sub, model))
        brier_raw = brier_score_loss(sub["y_true"], sub["score"].clip(0, 1))
        cal_scores = np.zeros(len(sub))
        iso_scores = np.full(len(sub), np.nan)
        for date in sorted(sub["date"].unique()):
            train = sub[sub["date"] != date]
            test = sub[sub["date"] == date]
            if train["y_true"].nunique() < 2:
                cal_scores[test.index.to_numpy()] = test["score"].clip(0, 1)
                continue
            lr = LogisticRegression(C=1.0, solver="liblinear")
            lr.fit(train[["score"]], train["y_true"])
            cal_scores[test.index.to_numpy()] = lr.predict_proba(test[["score"]])[:, 1]
            if train["y_true"].sum() >= 30:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(train["score"], train["y_true"])
                iso_scores[test.index.to_numpy()] = iso.predict(test["score"])
        sub["platt_score"] = cal_scores
        calibrated_frames.append(sub)
        brier_platt = brier_score_loss(sub["y_true"], sub["platt_score"].clip(0, 1))
        rel_rows.append(reliability_bins(sub.rename(columns={"platt_score": "score"}), model + "_platt"))
        iso_note = "isotonic attempted where training folds had >=30 positives" if np.isfinite(iso_scores).any() else "isotonic skipped: too few positive examples per blocked fold"
        lines.extend([f"## {model}", f"- Raw Brier score: {brier_raw:.4f}", f"- Platt/sigmoid Brier score: {brier_platt:.4f}", f"- {iso_note}", ""])
    reliability = pd.concat([r for r in rel_rows if not r.empty], ignore_index=True) if rel_rows else pd.DataFrame()
    report = "\n".join(lines)
    return reliability, report


def static_baselines() -> list[dict]:
    return [
        {**V3, "model_type": "baseline", "target": "published_reference", "threshold": np.nan, "notes": "Reference high-recall v3 state-machine baseline."},
        {**V6, "model_type": "baseline", "target": "published_reference", "threshold": np.nan, "notes": "Reference v6 low-false-alert baseline."},
        {**V7, "model_type": "baseline", "target": "published_reference", "threshold": np.nan, "notes": "Reference v7-Lite physics-feature baseline."},
    ]


def decide(best: pd.Series) -> tuple[bool, bool, str, str]:
    f1 = float(best["f1"])
    far = float(best["false_alerts_per_day"])
    recall = float(best["recall"])
    beats_v6 = f1 > 0.660 and far <= 1.33 and recall >= 0.733
    beats_v3 = recall > 0.824 and far <= 1.50
    balanced = "v8 gated ensemble" if beats_v6 else "v6 balanced low-false-alert mode"
    high = "v8 gated ensemble" if beats_v3 else "v3 high-recall mode"
    return beats_v6, beats_v3, balanced, high


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            vals.append(f"{val:.4g}" if isinstance(val, float) and not pd.isna(val) else str(val).replace("|", "/"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data, events = load_data()
    frame = modelling_frame(data)
    features = feature_columns(frame)
    target = "flare_next_60min" if "flare_next_60min" in frame.columns else "flare_next_30min"
    if PREDICTIONS_PATH.exists():
        predictions = pd.read_csv(PREDICTIONS_PATH)
        predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], utc=True, format="mixed", errors="coerce")
        tried = sorted(predictions["model"].dropna().astype(str).unique().tolist())
        skipped = [f"{spec.name}: {spec.skipped_reason}" for spec in model_specs() if spec.estimator is None]
    else:
        predictions, _metrics, tried, skipped = build_predictions(frame, features)
        stack_lr, stack_et = make_stacker_predictions(predictions, frame, target)
        for extra in [stack_lr, stack_et]:
            if not extra.empty:
                predictions = pd.concat([predictions, extra], ignore_index=True)
                tried.append(extra["model"].iloc[0])
        predictions.to_csv(PREDICTIONS_PATH, index=False)
    rows = []
    for (model, target_name), group in predictions.groupby(["model", "target"]):
        row = best_threshold_metrics(group, events)
        row.update({"system": model, "model_type": "model_zoo_or_meta", "target": target_name, "notes": "Blocked date-wise OOF predictions; threshold tuned on OOF operating points."})
        rows.append(row)
    rows.extend(static_baselines())
    comparison = pd.DataFrame(rows)
    comparison = comparison.sort_values(["f1", "false_alerts_per_day", "recall"], ascending=[False, True, False])
    comparison.to_csv(MODEL_ZOO_PATH, index=False)
    comparison.to_csv(V8_COMPARISON_PATH, index=False)

    candidate_rows = comparison[comparison["model_type"].eq("model_zoo_or_meta")].copy()
    best_raw = candidate_rows.iloc[0]
    top_models = candidate_rows.head(3)["system"].astype(str).tolist()
    reliability, cal_report = calibrate_top_models(predictions, top_models)
    reliability.to_csv(RELIABILITY_PATH, index=False)
    CALIBRATION_REPORT_PATH.write_text(cal_report, encoding="utf-8")

    beats_v6, beats_v3, balanced, high = decide(best_raw)
    event_dates = events["date"].nunique()
    deep_learning_note = "Deep learning skipped because dataset is too small for reliable blocked validation."
    if event_dates >= 50 and len(events) >= 100:
        deep_learning_note = "Deep learning still skipped in v8-Lite because the user requested no risky model complexity."

    report = f"""# Forecasting v8 Model Zoo Report

## Validation

All train/evaluate steps use blocked date-wise out-of-fold predictions. No random row split is used. Features are inherited from the SoLEXS + HEL1OS v7-Lite/v6 tables and use current/past timestamps only.

## Models Tried

{chr(10).join(f"- {m}" for m in tried)}

## Models Skipped

{chr(10).join(f"- {s}" for s in skipped) if skipped else "- None"}
- SMOTE skipped by default: time-series row synthesis can leak temporal structure if not designed carefully.
- {deep_learning_note}

## Best Raw Model

- Model: `{best_raw['system']}`
- Precision: {float(best_raw['precision']):.3f}
- Recall/POD: {float(best_raw['recall']):.3f}
- F1: {float(best_raw['f1']):.3f}
- False alerts/day: {float(best_raw['false_alerts_per_day']):.2f}

## Selection Decision

- Beats v6 balanced mode: {'yes' if beats_v6 else 'no'}
- Beats v3 high-recall mode: {'yes' if beats_v3 else 'no'}
- Final recommended balanced mode: **{balanced}**
- Final high-recall mode: **{high}**

If a model improves F1 but raises false alerts/day too much, it is kept as research mode only.

## Model Zoo Table

{md_table(comparison.head(20))}
"""
    MODEL_ZOO_REPORT_PATH.write_text(report, encoding="utf-8")

    print(f"models tried: {', '.join(tried)}")
    print(f"models skipped: {'; '.join(skipped) if skipped else 'none'}; SMOTE skipped; deep learning skipped")
    print(f"best raw model: {best_raw['system']}")
    print(f"best calibrated model: {top_models[0]} with Platt diagnostic report written")
    print(f"best gated ensemble: {best_raw['system'] if 'stacker' in str(best_raw['system']) else 'rule-gated ensemble fallback not selected'}")
    print(f"v8 precision: {float(best_raw['precision']):.3f}")
    print(f"v8 recall: {float(best_raw['recall']):.3f}")
    print(f"v8 F1: {float(best_raw['f1']):.3f}")
    print(f"v8 false alerts/day: {float(best_raw['false_alerts_per_day']):.2f}")
    print(f"whether v8 beats v6: {'yes' if beats_v6 else 'no'}")
    print(f"whether v8 beats v3 high-recall mode: {'yes' if beats_v3 else 'no'}")
    print(f"final recommended balanced mode: {balanced}")
    print(f"final high-recall mode: {high}")


if __name__ == "__main__":
    main()
