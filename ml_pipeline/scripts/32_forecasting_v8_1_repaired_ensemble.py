from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OUT_DIR = PROJECT_ROOT / "results" / "forecasting_v8_1"
V6_PRED_PATH = PROJECT_ROOT / "results" / "forecasting_v6" / "forecasting_v6_predictions.csv"
V6_DATASET_PATH = PROJECT_ROOT / "results" / "forecasting_v6" / "forecasting_v6_dataset.csv"
V6_SWEEP_PATH = PROJECT_ROOT / "results" / "forecasting_v6" / "forecasting_v6_policy_sweep.csv"
V6_CAT_PATH = PROJECT_ROOT / "results" / "forecasting_v6" / "combined_nowcast_catalogue_clean.csv"
V3_PRED_PATH = PROJECT_ROOT / "results" / "forecasting_v3" / "forecasting_v3_predictions.csv"
V7_PRED_PATH = PROJECT_ROOT / "results" / "forecasting_v7lite" / "forecasting_v7lite_predictions.csv"
V7_DATASET_PATH = PROJECT_ROOT / "results" / "forecasting_v7lite" / "forecasting_v7lite_dataset.csv"
V8_PRED_PATH = PROJECT_ROOT / "results" / "forecasting_v8" / "v8_oof_predictions.csv"

ALIGNED_PATH = OUT_DIR / "aligned_predictions.csv"
POLICY_SWEEP_PATH = OUT_DIR / "v8_1_policy_sweep.csv"
COMPARISON_PATH = OUT_DIR / "v3_v6_v7lite_v8_1_comparison.csv"
COMPARISON_MD_PATH = OUT_DIR / "v3_v6_v7lite_v8_1_comparison.md"
RECOMMENDATION_PATH = OUT_DIR / "v8_1_recommendation.md"
REPAIR_AUDIT_PATH = OUT_DIR / "evaluation_repair_audit.md"
ALIGNMENT_AUDIT_PATH = OUT_DIR / "alignment_audit.md"
FAILURE_PATH = OUT_DIR / "v6_reproduction_failure.md"

V3_REF = {"system": "v3_high_recall", "precision": 0.515, "recall": 0.824, "f1": 0.634, "false_alerts_per_day": 1.33, "valid_alerted_events": 14, "total_events": 17, "mean_lead_time_min": 39.44, "median_lead_time_min": 40.18}
V6_REF = {"system": "v6_balanced", "precision": 0.600, "recall": 0.7333333333333333, "f1": 0.660, "false_alerts_per_day": 1.1111111111111112, "valid_alerted_events": 11, "total_events": 15, "mean_lead_time_min": 40.1939393939394, "median_lead_time_min": 32.083333333333336}
V7_REF = {"system": "v7lite_physics", "precision": 0.45283, "recall": 0.866667, "f1": 0.594852, "false_alerts_per_day": 3.222222, "valid_alerted_events": 13, "total_events": 15, "mean_lead_time_min": 33.326923, "median_lead_time_min": 33.116667}


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path.resolve()}")


def load_events() -> pd.DataFrame:
    require(V6_CAT_PATH)
    events = pd.read_csv(V6_CAT_PATH)
    events["date"] = events["source_date"].astype(str).str.replace(r"\.0$", "", regex=True)
    events["event_id"] = events.get("v6_event_uid", events.get("event_id", "")).astype(str)
    events["event_onset_time"] = pd.to_datetime(events["event_start"], utc=True, format="mixed", errors="coerce")
    return events.dropna(subset=["event_onset_time"]).copy()


def pivot_scores(path: Path, model: str, target_map: dict[str, str], prefix: str) -> pd.DataFrame:
    require(path)
    pred = pd.read_csv(path)
    pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True, format="mixed", errors="coerce")
    pred["date"] = pred["date"].astype(str).str.replace(r"\.0$", "", regex=True)
    pred = pred[pred["model"].astype(str).eq(model) & pred["target"].isin(target_map.keys())].copy()
    if pred.empty:
        return pd.DataFrame(columns=["timestamp", "date"])
    piv = pred.pivot_table(index=["timestamp", "date"], columns="target", values="score", aggfunc="max").reset_index()
    piv = piv.rename(columns={old: f"{prefix}_{new}" for old, new in target_map.items()})
    return piv


def v6_policy_frame() -> pd.DataFrame:
    v6 = pivot_scores(V6_PRED_PATH, "extra_trees_challenger", {"flare_next_30min": "p30", "flare_next_60min": "p60"}, "v6")
    data = pd.read_csv(V6_DATASET_PATH, usecols=["timestamp", "date", "hard_score", "quality_label"])
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, format="mixed", errors="coerce")
    data["date"] = data["date"].astype(str).str.replace(r"\.0$", "", regex=True)
    out = v6.merge(data, on=["timestamp", "date"], how="left")
    for col in ["v6_p30", "v6_p60", "hard_score"]:
        out[col] = pd.to_numeric(out.get(col, 0), errors="coerce").fillna(0.0)
    out["quality_label"] = out["quality_label"].fillna("UNKNOWN")
    return out.sort_values(["date", "timestamp"]).reset_index(drop=True)


def run_v6_state_machine(df: pd.DataFrame, p30: float = 0.6, p60: float = 0.3, bins: int = 2, cooldown_min: int = 60) -> pd.DataFrame:
    rows = []
    cooldown = pd.Timedelta(minutes=cooldown_min)
    for date, group in df.sort_values(["date", "timestamp"]).groupby("date"):
        watch_count = 0
        cooldown_until = pd.Timestamp.min.tz_localize("UTC")
        for row in group.itertuples():
            ts = row.timestamp
            alert_signal = float(row.v6_p30) >= p30 or float(row.hard_score) >= 4.0
            watch_signal = float(row.v6_p60) >= p60
            if ts < cooldown_until:
                state = "COOLDOWN"
                positive = 0
                watch_count = 0
            elif alert_signal:
                state = "ALERT"
                positive = 1
                cooldown_until = ts + cooldown
                watch_count = 0
            elif watch_signal:
                watch_count += 1
                state = "WATCH" if watch_count >= bins else "CLEAR"
                positive = 0
            else:
                state = "CLEAR"
                positive = 0
                watch_count = 0
            rows.append({"timestamp": ts, "date": date, "state": state, "predicted_positive": positive, "score": max(float(row.v6_p30), float(row.v6_p60)), "policy_name": "v6_reproduced"})
    return pd.DataFrame(rows)


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
                max_score = float(row.get("score", 0))
            else:
                max_score = max(max_score, float(row.get("score", 0)))
            end = ts
            prev = ts
        if start is not None:
            rows.append({"date": date, "alert_start": start, "alert_end": end, "max_score": max_score})
    return pd.DataFrame(rows)


def evaluate_policy(pred: pd.DataFrame, events: pd.DataFrame, policy_name: str) -> dict:
    episodes = merge_episodes(pred)
    event_hits = {}
    useful = false = 0
    for _, ep in episodes.iterrows():
        same = events[events["date"].eq(str(ep["date"]))]
        candidates = same[(same["event_onset_time"] > ep["alert_start"]) & (same["event_onset_time"] <= ep["alert_start"] + pd.Timedelta(minutes=90))].copy()
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
    valid = len(event_hits)
    total_events = len(events)
    precision = useful / total if total else 0.0
    recall = valid / total_events if total_events else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    lead = pd.Series([v["lead_time_min"] for v in event_hits.values()], dtype=float)
    return {
        "policy_name": policy_name,
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


def reproduction_ok(metrics: dict) -> bool:
    return (
        abs(metrics["precision"] - 0.600) <= 0.02
        and abs(metrics["recall"] - 0.7333333333333333) <= 0.02
        and abs(metrics["f1"] - 0.660) <= 0.02
        and abs(metrics["false_alerts_per_day"] - 1.1111111111111112) <= 0.05
    )


def write_failure(metrics: dict) -> None:
    FAILURE_PATH.write_text(
        "# v6 Reproduction Failure\n\n"
        "v8.1 stopped before ensemble evaluation because the v6 baseline could not be reproduced with comparable episode-level metrics.\n\n"
        f"Observed metrics: {metrics}\n\n"
        "Likely cause: column, timestamp cadence, target, or state-machine evaluation mismatch.\n",
        encoding="utf-8",
    )


def align_predictions() -> pd.DataFrame:
    base = v6_policy_frame()
    v3 = pivot_scores(V3_PRED_PATH, "shallow_random_forest_diagnostic", {"flare_onset_next_30min": "p30", "flare_onset_next_60min": "p60"}, "v3")
    v7 = pivot_scores(V7_PRED_PATH, "extra_trees_v7lite", {"flare_next_30min": "p30", "flare_next_60min": "p60"}, "v7lite")
    v8 = pivot_scores(V8_PRED_PATH, "extra_trees", {"flare_next_60min": "p60"}, "v8_extra_trees")
    out = base.merge(v3, on=["timestamp", "date"], how="left").merge(v7, on=["timestamp", "date"], how="left").merge(v8, on=["timestamp", "date"], how="left")
    feature_cols = ["timestamp", "date", "soft_preflare_enhancement_score", "hard_impulsive_precursor_score", "hard_oscillation_proxy_score", "soft_hard_precursor_fusion_score"]
    features = pd.read_csv(V7_DATASET_PATH, usecols=lambda c: c in feature_cols)
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True, format="mixed", errors="coerce")
    features["date"] = features["date"].astype(str).str.replace(r"\.0$", "", regex=True)
    out = out.merge(features, on=["timestamp", "date"], how="left")
    out["v3_score"] = out[["v3_p30", "v3_p60"]].max(axis=1, skipna=True).fillna(0.0)
    out["v6_score"] = out[["v6_p30", "v6_p60"]].max(axis=1, skipna=True).fillna(0.0)
    out["v7lite_score"] = out[["v7lite_p30", "v7lite_p60"]].max(axis=1, skipna=True).fillna(0.0)
    out["soft_preflare_score"] = pd.to_numeric(out.get("soft_preflare_enhancement_score", 0), errors="coerce").fillna(0.0)
    out["hard_impulsive_score"] = pd.to_numeric(out.get("hard_impulsive_precursor_score", 0), errors="coerce").fillna(0.0)
    out["hard_oscillation_proxy_score"] = pd.to_numeric(out.get("hard_oscillation_proxy_score", 0), errors="coerce").fillna(0.0)
    out["soft_hard_fusion_score"] = pd.to_numeric(out.get("soft_hard_precursor_fusion_score", 0), errors="coerce").fillna(0.0)
    out["target_label"] = (pd.to_numeric(out.get("v6_p60", 0), errors="coerce").fillna(0) > -1).astype(int)
    out["matched_event_id"] = ""
    keep = ["timestamp", "date", "v3_score", "v6_score", "v7lite_score", "v8_extra_trees_p60", "soft_preflare_score", "hard_impulsive_score", "hard_oscillation_proxy_score", "soft_hard_fusion_score", "quality_label", "hard_score", "target_label", "matched_event_id", "v6_p30", "v6_p60"]
    out = out[[c for c in keep if c in out.columns]].sort_values(["date", "timestamp"])
    out.to_csv(ALIGNED_PATH, index=False)
    ALIGNMENT_AUDIT_PATH.write_text(
        "# v8.1 Alignment Audit\n\n"
        f"- Rows aligned on v6 prediction frame: {len(out):,}\n"
        f"- Dates: {out['date'].nunique()}\n"
        "- v3, v6, v7-Lite, and v8 scores are joined by exact timestamp/date where available.\n"
        "- v6 remains the primary frame and v7-Lite never triggers alone in repaired policies.\n",
        encoding="utf-8",
    )
    return out


def apply_cooldown_signal(df: pd.DataFrame, signal: pd.Series, policy_name: str, cooldown_min: int = 60) -> pd.DataFrame:
    rows = []
    cooldown = pd.Timedelta(minutes=cooldown_min)
    temp = df.copy()
    temp["policy_signal"] = signal.astype(bool).to_numpy()
    for date, group in temp.sort_values(["date", "timestamp"]).groupby("date"):
        cooldown_until = pd.Timestamp.min.tz_localize("UTC")
        for row in group.itertuples():
            ts = row.timestamp
            if ts < cooldown_until:
                pos = 0
            elif bool(row.policy_signal):
                pos = 1
                cooldown_until = ts + cooldown
            else:
                pos = 0
            rows.append({"timestamp": ts, "date": date, "predicted_positive": pos, "score": float(getattr(row, "v6_score", 0.0)), "policy_name": policy_name})
    return pd.DataFrame(rows)


def physics_confirm(df: pd.DataFrame) -> pd.Series:
    soft = df["soft_preflare_score"] >= df["soft_preflare_score"].quantile(0.80)
    hard = df["hard_impulsive_score"] >= df["hard_impulsive_score"].quantile(0.80)
    osc = df["hard_oscillation_proxy_score"] >= df["hard_oscillation_proxy_score"].quantile(0.80)
    fusion = df["soft_hard_fusion_score"] >= df["soft_hard_fusion_score"].quantile(0.80)
    return (soft.astype(int) + hard.astype(int) + osc.astype(int) + fusion.astype(int)) >= 2


def _past_date_quantile(df: pd.DataFrame, column: str, q: float = 0.80, min_periods: int = 30) -> pd.Series:
    thresholds = pd.Series(np.nan, index=df.index, dtype=float)
    if column not in df.columns:
        return thresholds
    ordered = df.sort_values(["date", "timestamp"])
    for _, group in ordered.groupby("date", sort=False):
        values = pd.to_numeric(group[column], errors="coerce").shift(1)
        thresholds.loc[group.index] = values.expanding(min_periods=min_periods).quantile(q)
    return thresholds


def physics_confirm_rigorous(df: pd.DataFrame) -> pd.Series:
    required = [
        "soft_preflare_score",
        "hard_impulsive_score",
        "hard_oscillation_proxy_score",
        "soft_hard_fusion_score",
    ]
    if not {"date", "timestamp", *required}.issubset(df.columns):
        return pd.Series(False, index=df.index)

    confirmations = []
    for column in required:
        threshold = _past_date_quantile(df, column)
        values = pd.to_numeric(df[column], errors="coerce")
        confirmations.append(values.ge(threshold).where(threshold.notna(), False).astype(int))
    return sum(confirmations) >= 2


def sweep_repaired_policies(aligned: pd.DataFrame, events: pd.DataFrame, reproduced_metrics: dict) -> pd.DataFrame:
    rows = [{**reproduced_metrics, "policy_family": "v6_reproduced", "details": "v6-only reproduced baseline"}]
    physics = physics_confirm_rigorous(aligned)
    for v6_strong in [0.6, 0.7]:
        for v6_mod in [0.3, 0.4, 0.5]:
            for v3_strong in [0.5, 0.6, 0.7]:
                signal = (aligned["v6_score"] >= v6_strong) | ((aligned["v3_score"] >= v3_strong) & (aligned["v6_score"] >= v6_mod) & aligned["quality_label"].astype(str).str.upper().eq("GOOD"))
                pred = apply_cooldown_signal(aligned, signal, "v6_v3_rescue", 60)
                rows.append({**evaluate_policy(pred, events, "v6_v3_rescue"), "policy_family": "v6_v3_rescue", "details": f"v6_strong={v6_strong}, v6_mod={v6_mod}, v3_strong={v3_strong}"})
                signal = (aligned["v6_score"] >= v6_strong) | ((aligned["v3_score"] >= v3_strong) & (aligned["v6_score"] >= v6_mod) & physics)
                pred = apply_cooldown_signal(aligned, signal, "v6_v3_v7_physics_rescue", 60)
                rows.append({**evaluate_policy(pred, events, "v6_v3_v7_physics_rescue"), "policy_family": "v6_v3_v7_physics_rescue", "details": f"v6_strong={v6_strong}, v6_mod={v6_mod}, v3_strong={v3_strong}, physics>=2"})
        for support in [0.4, 0.5, 0.6]:
            signal = (aligned["v6_score"] >= 0.4) & ((aligned["v3_score"] >= support) | (aligned["v7lite_score"] >= support))
            pred = apply_cooldown_signal(aligned, signal, "agreement_mode", 60)
            rows.append({**evaluate_policy(pred, events, "agreement_mode"), "policy_family": "agreement_mode", "details": f"v6>=0.4 and support>={support}"})
    weights = [(0.6, 0.25, 0.15), (0.7, 0.2, 0.1), (0.8, 0.15, 0.05)]
    for w6, w3, w7 in weights:
        score = w6 * aligned["v6_score"] + w3 * aligned["v3_score"] + w7 * aligned["v7lite_score"]
        for thr in [0.4, 0.5, 0.6]:
            signal = (score >= thr) & (aligned["v6_score"] >= 0.25)
            pred = apply_cooldown_signal(aligned.assign(v6_score=score), signal, "weighted_ensemble", 60)
            rows.append({**evaluate_policy(pred, events, "weighted_ensemble"), "policy_family": "weighted_ensemble", "details": f"w6={w6}, w3={w3}, w7={w7}, thr={thr}"})
    if "v8_extra_trees_p60" in aligned.columns:
        for thr in [0.5, 0.6, 0.7]:
            signal = (aligned["v8_extra_trees_p60"].fillna(0) >= thr) & (aligned["v6_score"] >= 0.25)
            pred = apply_cooldown_signal(aligned, signal, "model_zoo_episode_repaired", 60)
            rows.append({**evaluate_policy(pred, events, "model_zoo_episode_repaired"), "policy_family": "model_zoo_episode_repaired", "details": f"v8_extra_trees>={thr} and v6>=0.25"})
    sweep = pd.DataFrame(rows).sort_values(["f1", "false_alerts_per_day", "recall"], ascending=[False, True, False])
    sweep.to_csv(POLICY_SWEEP_PATH, index=False)
    return sweep


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


def write_outputs(best: pd.Series, reproduced: dict, beats_v6: bool, beats_v3: bool) -> tuple[str, str]:
    balanced = "v8.1 repaired gated ensemble" if beats_v6 else "v6 balanced mode"
    high = "v8.1 repaired gated ensemble" if beats_v3 else "v3 high-recall mode"
    comparison = pd.DataFrame([V3_REF, V6_REF, V7_REF, {
        "system": "v8_1_repaired_best",
        "precision": float(best["precision"]),
        "recall": float(best["recall"]),
        "f1": float(best["f1"]),
        "false_alerts_per_day": float(best["false_alerts_per_day"]),
        "valid_alerted_events": int(best["valid_alerted_events"]),
        "total_events": int(best["total_events"]),
        "mean_lead_time_min": float(best["mean_lead_time_min"]),
        "median_lead_time_min": float(best["median_lead_time_min"]),
    }])
    comparison.to_csv(COMPARISON_PATH, index=False)
    COMPARISON_MD_PATH.write_text("# v3/v6/v7-Lite/v8.1 Comparison\n\n" + md_table(comparison) + "\n", encoding="utf-8")
    REPAIR_AUDIT_PATH.write_text(
        "# v8.1 Evaluation Repair Audit\n\n"
        "- v6 baseline is reproduced before ensemble evaluation.\n"
        "- All policies are evaluated as alert episodes.\n"
        "- Consecutive/cooldown state-machine logic is used for v6 reproduction.\n"
        "- Repaired ensemble policies apply cooldown and merge duplicate positives into alert episodes.\n"
        "- FAR/day is false alert episodes divided by evaluated days.\n"
        "- Event matching uses the v6 90-minute future event window.\n"
        f"- Reproduced v6 metrics: precision={reproduced['precision']:.3f}, recall={reproduced['recall']:.3f}, F1={reproduced['f1']:.3f}, FAR/day={reproduced['false_alerts_per_day']:.2f}.\n",
        encoding="utf-8",
    )
    RECOMMENDATION_PATH.write_text(
        "# v8.1 Repaired Gated Ensemble Recommendation\n\n"
        f"Best policy: `{best['policy_name']}` / `{best['policy_family']}`\n\n"
        f"- Precision: {float(best['precision']):.3f}\n"
        f"- Recall/POD: {float(best['recall']):.3f}\n"
        f"- F1: {float(best['f1']):.3f}\n"
        f"- FAR/day: {float(best['false_alerts_per_day']):.2f}\n"
        f"- Valid alerted events: {int(best['valid_alerted_events'])} / {int(best['total_events'])}\n"
        f"- Mean lead time: {float(best['mean_lead_time_min']):.2f} min\n\n"
        f"Beats v6 balanced rule: {'yes' if beats_v6 else 'no'}.\n\n"
        f"Beats v3 high-recall rule: {'yes' if beats_v3 else 'no'}.\n\n"
        f"Final recommended balanced mode: **{balanced}**.\n\n"
        f"Final high-recall mode: **{high}**.\n",
        encoding="utf-8",
    )
    return balanced, high


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    events = load_events()
    v6_frame = v6_policy_frame()
    reproduced_pred = run_v6_state_machine(v6_frame)
    reproduced = evaluate_policy(reproduced_pred, events, "v6_reproduced")
    if not reproduction_ok(reproduced):
        write_failure(reproduced)
        print("whether v6 was reproduced: no")
        print(f"reproduced v6 metrics: {reproduced}")
        print(f"failure report: {FAILURE_PATH}")
        return
    aligned = align_predictions()
    sweep = sweep_repaired_policies(aligned, events, reproduced)
    best = sweep.iloc[0]
    beats_v6 = float(best["f1"]) > 0.660 and float(best["false_alerts_per_day"]) <= 1.33 and float(best["recall"]) >= 0.733
    beats_v3 = float(best["recall"]) > 0.824 and float(best["false_alerts_per_day"]) <= 1.50
    balanced, high = write_outputs(best, reproduced, beats_v6, beats_v3)
    print("whether v6 was reproduced: yes")
    print(f"reproduced v6 metrics: precision={reproduced['precision']:.3f}, recall={reproduced['recall']:.3f}, F1={reproduced['f1']:.3f}, FAR/day={reproduced['false_alerts_per_day']:.2f}")
    print(f"best v8.1 policy: {best['policy_name']} / {best['policy_family']} ({best['details']})")
    print(f"v8.1 precision: {float(best['precision']):.3f}")
    print(f"v8.1 recall: {float(best['recall']):.3f}")
    print(f"v8.1 F1: {float(best['f1']):.3f}")
    print(f"v8.1 FAR/day: {float(best['false_alerts_per_day']):.2f}")
    print(f"whether v8.1 beats v6: {'yes' if beats_v6 else 'no'}")
    print(f"whether v8.1 beats v3: {'yes' if beats_v3 else 'no'}")
    print(f"final recommended balanced mode: {balanced}")
    print(f"final high-recall mode: {high}")


if __name__ == "__main__":
    main()
