from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path("results") / "forecasting_v4"
PRED_PATH = OUT_DIR / "forecasting_v4_predictions.csv"
MODEL_COMPARISON_PATH = OUT_DIR / "forecasting_v4_model_comparison.csv"
DATASET_PATH = OUT_DIR / "forecasting_v4_dataset.csv"
MASTER_CLASSIFIED_PATH = Path("results") / "master_flare_catalogue_classified_v2.csv"
MASTER_PATH = Path("results") / "master_flare_catalogue.csv"

SWEEP_PATH = OUT_DIR / "forecasting_v4_policy_sweep.csv"
RECOMMENDATIONS_PATH = OUT_DIR / "forecasting_v4_policy_recommendations.csv"
RECOMMENDATIONS_MD_PATH = OUT_DIR / "forecasting_v4_policy_recommendations.md"

P30_THRESHOLDS = [0.30, 0.50, 0.70]
P60_THRESHOLDS = [0.30, 0.60]
HIGH_CLASS_THRESHOLDS = [0.30, 0.60]
FUSION_THRESHOLDS = [0.15, 0.30]
QPP_THRESHOLDS = [0.00]
CONSECUTIVE_BINS = [2, 3]
COOLDOWN_MIN = [30, 60]
QUALITY_MODES = ["GOOD_ONLY", "GOOD_PLUS_QUESTIONABLE"]
EPISODE_GAP_SECONDS = 60
MAX_POLICY_MODELS = 1
V3 = {
    "precision": 0.5151515151515151,
    "recall": 0.8235294117647058,
    "f1": 0.6338215712383488,
    "false_alerts_per_day": 1.3333333333333333,
    "valid_alerted_events": 14,
    "total_events": 17,
    "mean_lead_time_min": 39.44404761904762,
    "median_lead_time_min": 40.18333333333334,
}


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path.resolve()}")


def load_events() -> pd.DataFrame:
    path = MASTER_CLASSIFIED_PATH if MASTER_CLASSIFIED_PATH.exists() else MASTER_PATH
    require(path)
    events = pd.read_csv(path)
    events["date"] = events["date"].astype(str).str.replace(r"\.0$", "", regex=True)
    onset_col = "combined_start" if "combined_start" in events.columns else "soft_start"
    events["event_onset_time"] = pd.to_datetime(events[onset_col], utc=True, format="mixed", errors="coerce")
    events["event_id"] = events["event_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    if "quality_label" not in events.columns:
        events["quality_label"] = ""
    if "surya_estimated_class_group" not in events.columns:
        events["surya_estimated_class_group"] = ""
    return events.dropna(subset=["event_onset_time"]).copy()


def policy_model_names() -> list[str]:
    require(MODEL_COMPARISON_PATH)
    comp = pd.read_csv(MODEL_COMPARISON_PATH)
    targets_needed = {"flare_onset_next_30min", "flare_onset_next_60min", "high_class_flare_next_60min"}
    pred = pd.read_csv(PRED_PATH, usecols=["model", "target"])
    eligible = []
    for model, group in pred.groupby("model"):
        if targets_needed.issubset(set(group["target"])):
            eligible.append(model)
    if not eligible:
        return []
    sub = comp[comp["model"].isin(eligible) & comp["target"].isin(targets_needed)].copy()
    sub["f1_num"] = pd.to_numeric(sub["f1"], errors="coerce").fillna(-1)
    ranked = sub.groupby("model", as_index=False)["f1_num"].mean().sort_values("f1_num", ascending=False)
    return ranked["model"].head(MAX_POLICY_MODELS).tolist()


def load_policy_frame(model_name: str) -> pd.DataFrame:
    require(PRED_PATH)
    require(DATASET_PATH)
    pred = pd.read_csv(PRED_PATH)
    pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True, format="mixed", errors="coerce")
    pred["date"] = pred["date"].astype(str).str.replace(r"\.0$", "", regex=True)
    targets = ["flare_onset_next_30min", "flare_onset_next_60min", "high_class_flare_next_60min"]
    pred = pred[pred["model"].eq(model_name) & pred["target"].isin(targets)]
    if pred.empty:
        return pd.DataFrame()
    pivot = (
        pred.pivot_table(index=["timestamp", "date", "model"], columns="target", values="score", aggfunc="max")
        .reset_index()
        .rename(
            columns={
                "flare_onset_next_30min": "p30",
                "flare_onset_next_60min": "p60",
                "high_class_flare_next_60min": "p_high",
            }
        )
    )
    dataset_cols = [
        "timestamp",
        "date",
        "quality_label",
        "precursor_fusion_score_v4",
        "hard_qpp_score",
        "hard_nonthermal_precursor_score",
        "soft_gradual_enhancement_score",
        "hard_score",
        "soft_score",
        "post_flare_decay_state",
    ]
    data = pd.read_csv(DATASET_PATH, usecols=lambda c: c in dataset_cols)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, format="mixed", errors="coerce")
    data["date"] = data["date"].astype(str).str.replace(r"\.0$", "", regex=True)
    out = pivot.merge(data, on=["timestamp", "date"], how="left")
    for col in ["p30", "p60", "p_high", "precursor_fusion_score_v4", "hard_qpp_score", "hard_nonthermal_precursor_score", "soft_gradual_enhancement_score", "hard_score", "soft_score"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    if "post_flare_decay_state" in out.columns:
        out["post_flare_decay_state"] = pd.to_numeric(out["post_flare_decay_state"], errors="coerce").fillna(0).astype(int)
    return out.sort_values(["date", "timestamp"]).reset_index(drop=True)


def run_state_machine(
    df: pd.DataFrame,
    p30_thr: float,
    p60_thr: float,
    high_thr: float,
    fusion_thr: float,
    qpp_thr: float,
    bins: int,
    cooldown_min: int,
) -> pd.DataFrame:
    rows = []
    cooldown_delta = pd.Timedelta(minutes=cooldown_min)
    for date, group in df.groupby("date", sort=False):
        watch_count = 0
        cooldown_until = pd.Timestamp.min.tz_localize("UTC")
        cols = [
            "timestamp",
            "quality_label",
            "model",
            "p30",
            "p60",
            "p_high",
            "precursor_fusion_score_v4",
            "hard_qpp_score",
            "hard_score",
            "soft_score",
            "post_flare_decay_state",
        ]
        for src in group[cols].itertuples(index=False):
            ts = src.timestamp
            p30 = float(src.p30 or 0.0)
            p60 = float(src.p60 or 0.0)
            p_high = float(src.p_high or 0.0)
            fusion = float(src.precursor_fusion_score_v4 or 0.0)
            qpp = float(src.hard_qpp_score or 0.0)
            hard_score = float(src.hard_score or 0.0)
            soft_score = float(src.soft_score or 0.0)
            precursor_confirmed = fusion >= fusion_thr and qpp >= qpp_thr
            watch_signal = p60 >= p60_thr and precursor_confirmed
            forecast_signal = (p30 >= p30_thr and precursor_confirmed) or (p_high >= high_thr and precursor_confirmed)
            nowcast_signal = hard_score >= 4.0 or soft_score >= 6.0
            if ts < cooldown_until:
                watch_count = 0
                continue
            elif nowcast_signal:
                state = "NOWCAST_CONFIRMED"
                positive = 1
                cooldown_until = ts + cooldown_delta
                watch_count = 0
            elif forecast_signal:
                state = "FORECAST_ALERT"
                positive = 1
                cooldown_until = ts + cooldown_delta
                watch_count = 0
            elif watch_signal:
                watch_count += 1
                if watch_count >= bins:
                    state = "FORECAST_ALERT"
                    positive = 1
                    cooldown_until = ts + cooldown_delta
                    watch_count = 0
                else:
                    continue
            else:
                watch_count = 0
                continue
            rows.append(
                {
                    "timestamp": ts,
                    "date": date,
                    "quality_label": src.quality_label,
                    "model": src.model,
                    "p30": p30,
                    "p60": p60,
                    "p_high": p_high,
                    "precursor_fusion_score_v4": fusion,
                    "hard_qpp_score": qpp,
                    "hard_score": hard_score,
                    "soft_score": soft_score,
                    "post_flare_decay_state": src.post_flare_decay_state,
                    "state": state,
                    "predicted_positive": positive,
                }
            )
    return pd.DataFrame(rows)


def merge_episodes(pred: pd.DataFrame) -> pd.DataFrame:
    pos = pred[pred["predicted_positive"].eq(1)].copy()
    if pos.empty:
        return pd.DataFrame(columns=["date", "alert_start", "alert_end", "max_p30", "max_p60", "max_p_high"])
    rows = []
    for date, group in pos.sort_values(["date", "timestamp"]).groupby("date"):
        start = end = prev = None
        max_p30 = max_p60 = max_p_high = 0.0
        first = None
        states = []
        for _, row in group.iterrows():
            ts = row["timestamp"]
            if start is None or (ts - prev).total_seconds() > EPISODE_GAP_SECONDS:
                if start is not None:
                    rows.append({**first, "alert_start": start, "alert_end": end, "max_p30": max_p30, "max_p60": max_p60, "max_p_high": max_p_high, "states_seen": ";".join(sorted(set(states)))})
                start = ts
                first = row.to_dict()
                max_p30 = float(row["p30"])
                max_p60 = float(row["p60"])
                max_p_high = float(row["p_high"])
                states = [str(row.get("state", ""))]
            else:
                max_p30 = max(max_p30, float(row["p30"]))
                max_p60 = max(max_p60, float(row["p60"]))
                max_p_high = max(max_p_high, float(row["p_high"]))
                states.append(str(row.get("state", "")))
            end = ts
            prev = ts
        if start is not None:
            rows.append({**first, "alert_start": start, "alert_end": end, "max_p30": max_p30, "max_p60": max_p60, "max_p_high": max_p_high, "states_seen": ";".join(sorted(set(states)))})
    return pd.DataFrame(rows)


def evaluate(
    pred: pd.DataFrame,
    events: pd.DataFrame,
    model: str,
    p30: float,
    p60: float,
    high_thr: float,
    fusion_thr: float,
    qpp_thr: float,
    bins: int,
    cooldown: int,
    quality_mode: str,
) -> dict:
    episodes = merge_episodes(pred)
    event_hits = {}
    useful = 0
    false = 0
    for _, ep in episodes.iterrows():
        same_date = events[events["date"].eq(str(ep["date"]))]
        candidates = same_date[
            (same_date["event_onset_time"] > ep["alert_start"])
            & (same_date["event_onset_time"] <= ep["alert_start"] + pd.Timedelta(minutes=90))
        ].copy()
        if candidates.empty:
            false += 1
            continue
        candidates["lead_time_min"] = (candidates["event_onset_time"] - ep["alert_start"]).dt.total_seconds() / 60.0
        match = candidates.sort_values("lead_time_min").iloc[0]
        useful += 1
        eid = str(match["event_id"])
        lead = float(match["lead_time_min"])
        if eid not in event_hits or ep["alert_start"] < event_hits[eid]["first_alert_time"]:
            event_hits[eid] = {"event_id": eid, "lead_time_min": lead, "first_alert_time": ep["alert_start"]}
    total = len(episodes)
    valid = len(event_hits)
    total_events = len(events)
    precision = useful / total if total else 0.0
    recall = valid / total_events if total_events else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    lead = pd.Series([v["lead_time_min"] for v in event_hits.values()], dtype=float)
    return {
        "model": model,
        "p30_threshold": p30,
        "p60_threshold": p60,
        "high_class_probability_threshold": high_thr,
        "precursor_fusion_score_v4_threshold": fusion_thr,
        "hard_qpp_score_threshold": qpp_thr,
        "consecutive_bins": bins,
        "cooldown_min": cooldown,
        "quality_mode": quality_mode,
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
        "iqr_lead_time_min": float(lead.quantile(0.75) - lead.quantile(0.25)) if not lead.empty else np.nan,
        "positive_lead_time_percent": float((lead > 0).mean() * 100.0) if not lead.empty else np.nan,
        "validation_method": "leave-one-date-out blocked v4 prediction streams; state-machine policy post-processing",
    }


def v4_beats_v3(row: pd.Series) -> bool:
    return bool(
        row["f1"] > V3["f1"]
        or (row["false_alerts_per_day"] < V3["false_alerts_per_day"] and row["recall"] >= V3["recall"])
        or (row["precision"] > V3["precision"] and row["recall"] >= 0.70)
    )


def sweep_policies() -> tuple[pd.DataFrame, pd.DataFrame]:
    events_all = load_events()
    rows = []
    for model in policy_model_names():
        frame_all = load_policy_frame(model)
        if frame_all.empty:
            continue
        for quality_mode in QUALITY_MODES:
            if quality_mode == "GOOD_ONLY":
                frame = frame_all[frame_all["quality_label"].eq("GOOD")].copy()
                events = events_all[events_all["quality_label"].eq("GOOD")].copy()
            else:
                frame = frame_all[frame_all["quality_label"].isin(["GOOD", "QUESTIONABLE"])].copy()
                events = events_all[events_all["quality_label"].isin(["GOOD", "QUESTIONABLE"])].copy()
            if frame.empty or events.empty:
                continue
            for p30 in P30_THRESHOLDS:
                for p60 in P60_THRESHOLDS:
                    for high_thr in HIGH_CLASS_THRESHOLDS:
                        for fusion_thr in FUSION_THRESHOLDS:
                            for qpp_thr in QPP_THRESHOLDS:
                                for bins in CONSECUTIVE_BINS:
                                    for cooldown in COOLDOWN_MIN:
                                        states = run_state_machine(frame, p30, p60, high_thr, fusion_thr, qpp_thr, bins, cooldown)
                                        rows.append(evaluate(states, events, model, p30, p60, high_thr, fusion_thr, qpp_thr, bins, cooldown, quality_mode))
    sweep = pd.DataFrame(rows)
    if sweep.empty:
        return sweep, pd.DataFrame()
    rec = sweep.sort_values(
        ["f1", "false_alerts_per_day", "recall", "precision", "mean_lead_time_min"],
        ascending=[False, True, False, False, False],
    ).head(10)
    return sweep, rec


def write_report(rec: pd.DataFrame) -> None:
    if rec.empty:
        RECOMMENDATIONS_MD_PATH.write_text("# Forecasting v4 Policy Recommendations\n\nNo policy rows available.\n", encoding="utf-8")
        return
    best = rec.iloc[0]
    beats = v4_beats_v3(best)
    decision = (
        "Forecasting v4 beats the predefined v3 replacement rule and becomes the current recommended forecasting research mode."
        if beats
        else "Forecasting v4 is retained as an advanced research extension. The Phase 3 v3 state-machine policy remains the final forecasting mode because v4 did not beat the predefined replacement rule."
    )
    lines = ["| metric | value |", "| --- | --- |"]
    for key, value in best.to_dict().items():
        lines.append(f"| {key} | {value} |")
    report = f"""# Forecasting v4 Policy Recommendations

## Best v4 State-Machine Policy

{chr(10).join(lines)}

## Decision

{decision}

## v3 Reference

- Precision: {V3['precision']:.3f}
- Recall: {V3['recall']:.3f}
- F1: {V3['f1']:.3f}
- False alerts/day: {V3['false_alerts_per_day']:.2f}
- Valid alerted events: {V3['valid_alerted_events']} / {V3['total_events']}
- Mean lead time: {V3['mean_lead_time_min']:.2f} min

## Caveats

- v4 uses blocked/date-wise prediction streams and a post-processing state-machine policy.
- Nowcasting logic and v1/v2/v3 outputs are not modified.
- Small-sample metrics remain diagnostic until more flare and quiet/control days are added.
"""
    RECOMMENDATIONS_MD_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep, rec = sweep_policies()
    sweep.to_csv(SWEEP_PATH, index=False)
    rec.to_csv(RECOMMENDATIONS_PATH, index=False)
    write_report(rec)
    best = rec.iloc[0] if not rec.empty else pd.Series(dtype=object)
    beats = bool(not rec.empty and v4_beats_v3(best))
    print("best v4 policy:")
    print(best.to_string() if not rec.empty else "none")
    print(f"whether v4 beats v3: {'yes' if beats else 'no'}")
    print("comparison with v3:")
    print(f"v3 precision={V3['precision']:.3f}, recall={V3['recall']:.3f}, F1={V3['f1']:.3f}, false alerts/day={V3['false_alerts_per_day']:.2f}, mean lead={V3['mean_lead_time_min']:.2f} min")
    print(f"final forecasting mode: {'v4 best state-machine policy' if beats else 'v3 state-machine policy'}")
    print("caveats: v4 is isolated under results/forecasting_v4; no nowcasting/v1/v2/v3 outputs modified")


if __name__ == "__main__":
    main()
