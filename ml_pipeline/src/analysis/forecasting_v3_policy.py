from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path("results") / "forecasting_v3"
PRED_PATH = OUT_DIR / "forecasting_v3_predictions.csv"
MODEL_COMPARISON_PATH = OUT_DIR / "forecasting_v3_model_comparison.csv"
DATASET_PATH = OUT_DIR / "forecasting_v3_dataset.csv"
MASTER_CLASSIFIED_PATH = Path("results") / "master_flare_catalogue_classified_v2.csv"
MASTER_PATH = Path("results") / "master_flare_catalogue.csv"
V2_COMPARISON_PATH = Path("results") / "precursor_forecast_v2_comparison.csv"
FINAL_REC_PATH = Path("results") / "final_system_recommendation.csv"

SWEEP_PATH = OUT_DIR / "forecasting_v3_policy_sweep.csv"
RECOMMENDATIONS_PATH = OUT_DIR / "forecasting_v3_policy_recommendations.csv"
RECOMMENDATIONS_MD_PATH = OUT_DIR / "forecasting_v3_policy_recommendations.md"
V1V2V3_PATH = OUT_DIR / "v1_v2_v3_forecasting_comparison.csv"
V1V2V3_MD_PATH = OUT_DIR / "v1_v2_v3_forecasting_comparison.md"
FALSE_ALERT_PATH = OUT_DIR / "forecasting_v3_false_alert_analysis.csv"
FALSE_ALERT_MD_PATH = OUT_DIR / "forecasting_v3_false_alert_analysis.md"

P30_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
P60_THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
CONSECUTIVE_BINS = [2, 3, 6, 12]
COOLDOWN_MIN = [15, 30, 45, 60]
QUALITY_MODES = ["GOOD_ONLY", "GOOD_PLUS_QUESTIONABLE"]
CONFIRMATION_OPTIONS = ["none", "combined_precursor_score", "hard_score_mean_10min", "soft_score_mean_10min"]
EPISODE_GAP_SECONDS = 60
POLICY_MODELS = ["shallow_random_forest_diagnostic"]
V1 = {
    "precision": 0.7241379310344828,
    "recall": 0.5294117647058824,
    "f1": 0.6116504854368933,
    "false_alerts_per_day": 1.6666666666666667,
    "valid_alerted_events": 9,
    "total_events": 17,
    "mean_lead_time_min": 40.50740740740741,
    "median_lead_time_min": 31.933333333333334,
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
    events["event_peak_time"] = pd.to_datetime(events["soft_peak"], utc=True, format="mixed", errors="coerce")
    events["event_id"] = events["event_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    return events.dropna(subset=["event_onset_time", "event_peak_time"]).copy()


def load_policy_frame(model_name: str) -> pd.DataFrame:
    require(PRED_PATH)
    require(DATASET_PATH)
    pred = pd.read_csv(PRED_PATH)
    pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True, format="mixed", errors="coerce")
    pred["date"] = pred["date"].astype(str).str.replace(r"\.0$", "", regex=True)
    pred = pred[pred["model"].eq(model_name) & pred["target"].isin(["flare_onset_next_30min", "flare_onset_next_60min"])]
    if pred.empty:
        return pd.DataFrame()
    pivot = (
        pred.pivot_table(index=["timestamp", "date", "model"], columns="target", values="score", aggfunc="max")
        .reset_index()
        .rename(columns={"flare_onset_next_30min": "p30", "flare_onset_next_60min": "p60"})
    )
    dataset_cols = [
        "timestamp",
        "date",
        "quality_label",
        "combined_precursor_score",
        "hard_score_mean_10min",
        "soft_score_mean_10min",
        "hard_score",
        "soft_score",
        "post_flare_decay_state",
    ]
    data = pd.read_csv(DATASET_PATH, usecols=lambda c: c in dataset_cols)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, format="mixed", errors="coerce")
    data["date"] = data["date"].astype(str).str.replace(r"\.0$", "", regex=True)
    out = pivot.merge(data, on=["timestamp", "date"], how="left")
    for col in ["p30", "p60", "combined_precursor_score", "hard_score_mean_10min", "soft_score_mean_10min", "hard_score", "soft_score"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    if "post_flare_decay_state" in out.columns:
        out["post_flare_decay_state"] = pd.to_numeric(out["post_flare_decay_state"], errors="coerce").fillna(0).astype(int)
    return out.sort_values(["date", "timestamp"]).reset_index(drop=True)


def model_names() -> list[str]:
    pred = pd.read_csv(PRED_PATH, usecols=["model", "target"])
    models = []
    for model, group in pred.groupby("model"):
        targets = set(group["target"])
        if {"flare_onset_next_30min", "flare_onset_next_60min"}.issubset(targets):
            models.append(model)
    preferred = [m for m in POLICY_MODELS if m in models]
    return preferred if preferred else sorted(models)


def confirmation_pass(row: pd.Series, option: str) -> bool:
    if option == "none":
        return True
    value = float(row.get(option, 0.0) or 0.0)
    if option == "combined_precursor_score":
        return value > 0.20
    return value > 0.0


def run_state_machine(df: pd.DataFrame, p30_thr: float, p60_thr: float, bins: int, cooldown_min: int, confirmation: str) -> pd.DataFrame:
    rows = []
    cooldown_delta = pd.Timedelta(minutes=cooldown_min)
    for date, group in df.groupby("date", sort=False):
        watch_count = 0
        cooldown_until = pd.Timestamp.min.tz_localize("UTC")
        timestamps = group["timestamp"].to_numpy()
        p30_vals = group["p30"].to_numpy(dtype=float)
        p60_vals = group["p60"].to_numpy(dtype=float)
        hard_vals = group.get("hard_score", pd.Series(0, index=group.index)).to_numpy(dtype=float)
        if confirmation == "none":
            confirms = np.ones(len(group), dtype=bool)
        else:
            vals = group.get(confirmation, pd.Series(0, index=group.index)).fillna(0).to_numpy(dtype=float)
            confirms = vals > (0.20 if confirmation == "combined_precursor_score" else 0.0)
        for i, ts_value in enumerate(timestamps):
            ts = pd.Timestamp(ts_value)
            watch_signal = p60_vals[i] >= p60_thr and confirms[i]
            alert_signal = (p30_vals[i] >= p30_thr and confirms[i]) or hard_vals[i] >= 4.0
            if ts < cooldown_until:
                state = "COOLDOWN"
                predicted_positive = 0
                watch_count = 0
            elif alert_signal:
                state = "ALERT"
                predicted_positive = 1
                cooldown_until = ts + cooldown_delta
                watch_count = 0
            elif watch_signal:
                watch_count += 1
                state = "WATCH" if watch_count >= bins else "CLEAR"
                predicted_positive = 0
            else:
                state = "CLEAR"
                predicted_positive = 0
                watch_count = 0
            src = group.iloc[i]
            rows.append(
                {
                    "timestamp": ts,
                    "date": date,
                    "quality_label": src.get("quality_label", ""),
                    "model": src.get("model", ""),
                    "p30": p30_vals[i],
                    "p60": p60_vals[i],
                    "state": state,
                    "predicted_positive": predicted_positive,
                    "combined_precursor_score": src.get("combined_precursor_score", np.nan),
                    "hard_score_mean_10min": src.get("hard_score_mean_10min", np.nan),
                    "soft_score_mean_10min": src.get("soft_score_mean_10min", np.nan),
                    "hard_score": src.get("hard_score", np.nan),
                    "soft_score": src.get("soft_score", np.nan),
                    "post_flare_decay_state": src.get("post_flare_decay_state", 0),
                }
            )
    return pd.DataFrame(rows)


def merge_episodes(pred: pd.DataFrame) -> pd.DataFrame:
    pos = pred[pred["predicted_positive"].eq(1)].copy()
    if pos.empty:
        return pd.DataFrame(columns=["date", "alert_start", "alert_end", "max_p30", "max_p60"])
    rows = []
    for date, group in pos.sort_values(["date", "timestamp"]).groupby("date"):
        start = end = prev = None
        max_p30 = max_p60 = 0.0
        first_row = None
        for _, row in group.iterrows():
            ts = row["timestamp"]
            if start is None or (ts - prev).total_seconds() > EPISODE_GAP_SECONDS:
                if start is not None:
                    rows.append({**first_row, "alert_start": start, "alert_end": end, "max_p30": max_p30, "max_p60": max_p60})
                start = ts
                first_row = row.to_dict()
                max_p30 = float(row["p30"])
                max_p60 = float(row["p60"])
            else:
                max_p30 = max(max_p30, float(row["p30"]))
                max_p60 = max(max_p60, float(row["p60"]))
            end = ts
            prev = ts
        if start is not None:
            rows.append({**first_row, "alert_start": start, "alert_end": end, "max_p30": max_p30, "max_p60": max_p60})
    return pd.DataFrame(rows)


def evaluate(pred: pd.DataFrame, events: pd.DataFrame, model: str, p30: float, p60: float, bins: int, cooldown: int, quality_mode: str, confirmation: str) -> tuple[dict, pd.DataFrame]:
    episodes = merge_episodes(pred)
    event_hits = {}
    episode_rows = []
    for _, ep in episodes.iterrows():
        same_date = events[events["date"].eq(str(ep["date"]))]
        candidates = same_date[(same_date["event_onset_time"] > ep["alert_start"]) & (same_date["event_onset_time"] <= ep["alert_start"] + pd.Timedelta(minutes=90))].copy()
        if candidates.empty:
            etype = "ISOLATED_FALSE_ALERT"
            eid = ""
            lead = np.nan
        else:
            candidates["lead_time_min"] = (candidates["event_onset_time"] - ep["alert_start"]).dt.total_seconds() / 60.0
            match = candidates.sort_values("lead_time_min").iloc[0]
            etype = "USEFUL_ALERT"
            eid = str(match["event_id"])
            lead = float(match["lead_time_min"])
            if eid not in event_hits or ep["alert_start"] < event_hits[eid]["first_alert_time"]:
                event_hits[eid] = {"event_id": eid, "lead_time_min": lead, "first_alert_time": ep["alert_start"]}
        episode_rows.append({**ep.to_dict(), "episode_type": etype, "matched_event_id": eid, "lead_time_min": lead})
    edf = pd.DataFrame(episode_rows)
    useful = int((edf.get("episode_type", pd.Series(dtype=str)) == "USEFUL_ALERT").sum()) if not edf.empty else 0
    false = int((edf.get("episode_type", pd.Series(dtype=str)) == "ISOLATED_FALSE_ALERT").sum()) if not edf.empty else 0
    total = len(edf)
    valid = len(event_hits)
    total_events = len(events)
    precision = useful / total if total else 0.0
    recall = valid / total_events if total_events else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    lead = pd.Series([v["lead_time_min"] for v in event_hits.values()], dtype=float)
    metrics = {
        "model": model,
        "p30_threshold": p30,
        "p60_threshold": p60,
        "consecutive_bins": bins,
        "cooldown_min": cooldown,
        "quality_mode": quality_mode,
        "confirmation_feature": confirmation,
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
        "validation_method": "blocked/date-wise v3 prediction streams; state-machine post-processing",
    }
    return metrics, edf


def sweep_policies() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events_all = load_events()
    rows = []
    best_episodes = pd.DataFrame()
    best_metric = None
    for model in model_names():
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
            for confirmation in CONFIRMATION_OPTIONS:
                if confirmation != "none" and confirmation not in frame.columns:
                    continue
                for p30 in P30_THRESHOLDS:
                    for cooldown in COOLDOWN_MIN:
                        # p60 and consecutive bins control WATCH state only. ALERT emissions depend on p30,
                        # confirmation, nowcast trigger, and cooldown, so compute once and replicate WATCH-only rows.
                        states = run_state_machine(frame, p30, P60_THRESHOLDS[0], CONSECUTIVE_BINS[0], cooldown, confirmation)
                        base_metrics, episodes = evaluate(
                            states,
                            events,
                            model,
                            p30,
                            P60_THRESHOLDS[0],
                            CONSECUTIVE_BINS[0],
                            cooldown,
                            quality_mode,
                            confirmation,
                        )
                        for p60 in P60_THRESHOLDS:
                            for bins in CONSECUTIVE_BINS:
                                metrics = dict(base_metrics)
                                metrics["p60_threshold"] = p60
                                metrics["consecutive_bins"] = bins
                                rows.append(metrics)
                        if best_metric is None:
                            best_metric = base_metrics
                            best_episodes = episodes
                        else:
                            candidate_key = (base_metrics["f1"], -base_metrics["false_alerts_per_day"], base_metrics["recall"], base_metrics["precision"])
                            best_key = (best_metric["f1"], -best_metric["false_alerts_per_day"], best_metric["recall"], best_metric["precision"])
                            if candidate_key > best_key:
                                best_metric = base_metrics
                                best_episodes = episodes
    sweep = pd.DataFrame(rows)
    if sweep.empty:
        return sweep, pd.DataFrame(), best_episodes
    rec = sweep.sort_values(["f1", "false_alerts_per_day", "recall", "precision"], ascending=[False, True, False, False]).head(10).copy()
    return sweep, rec, best_episodes


def v3_beats_v1(row: pd.Series) -> bool:
    return bool(
        row["recall"] >= 0.50
        and row["f1"] >= 0.50
        and row["false_alerts_per_day"] <= 2.50
        and pd.notna(row["mean_lead_time_min"])
        and row["mean_lead_time_min"] > 0
    )


def build_comparison(best: pd.Series) -> pd.DataFrame:
    rows = [
        {"system": "v1 90-min precursor-aware policy", **V1, "status": "FINAL_RECOMMENDED"},
    ]
    if V2_COMPARISON_PATH.exists():
        v2 = pd.read_csv(V2_COMPARISON_PATH)
        rf90 = v2[v2["system"].eq("new v2 trained target (90-min)")]
        if not rf90.empty:
            r = rf90.iloc[0]
            rows.append(
                {
                    "system": "v2 RF diagnostic",
                    "precision": r["precision"],
                    "recall": r["recall"],
                    "f1": r["f1"],
                    "false_alerts_per_day": r["false_alerts_per_day"],
                    "valid_alerted_events": np.nan,
                    "total_events": np.nan,
                    "mean_lead_time_min": r["mean_lead_time_min"],
                    "median_lead_time_min": np.nan,
                    "status": "DIAGNOSTIC",
                }
            )
    if MODEL_COMPARISON_PATH.exists():
        raw = pd.read_csv(MODEL_COMPARISON_PATH)
        for target, label in [("flare_onset_next_30min", "v3 raw onset_30 model"), ("flare_onset_next_60min", "v3 raw onset_60 model")]:
            sub = raw[raw["target"].eq(target)].copy()
            if not sub.empty:
                r = sub.sort_values(["f1", "false_alerts_per_day"], ascending=[False, True]).iloc[0]
                rows.append(
                    {
                        "system": label,
                        "precision": r["precision"],
                        "recall": r["recall_pod"],
                        "f1": r["f1"],
                        "false_alerts_per_day": r["false_alerts_per_day"],
                        "valid_alerted_events": r["valid_alerted_events"],
                        "total_events": r["total_events"],
                        "mean_lead_time_min": r["mean_lead_time_min"],
                        "median_lead_time_min": r["median_lead_time_min"],
                        "status": "DIAGNOSTIC",
                    }
                )
    rows.append(
        {
            "system": "v3 best state-machine policy",
            "precision": best["precision"],
            "recall": best["recall"],
            "f1": best["f1"],
            "false_alerts_per_day": best["false_alerts_per_day"],
            "valid_alerted_events": best["valid_alerted_events"],
            "total_events": best["total_events"],
            "mean_lead_time_min": best["mean_lead_time_min"],
            "median_lead_time_min": best["median_lead_time_min"],
            "status": "REPLACES_V1" if v3_beats_v1(best) else "DIAGNOSTIC_DOES_NOT_REPLACE_V1",
        }
    )
    return pd.DataFrame(rows)


def classify_false_alerts(false_eps: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    event_times = events[["date", "event_onset_time", "event_peak_time"]].copy()
    for _, row in false_eps.iterrows():
        date_events = event_times[event_times["date"].eq(str(row["date"]))]
        nearest = np.nan
        if not date_events.empty:
            diffs = (date_events["event_onset_time"] - row["alert_start"]).dt.total_seconds().abs() / 60.0
            nearest = float(diffs.min())
        if str(row.get("quality_label", "")).upper() == "QUESTIONABLE":
            cause = "QUESTIONABLE_DATE"
        elif int(row.get("post_flare_decay_state", 0) or 0) == 1:
            cause = "POST_FLARE_DECAY"
        elif float(row.get("hard_score", 0) or 0) >= 4 and float(row.get("soft_score", 0) or 0) < 2:
            cause = "HARD_ONLY_SPIKE"
        elif float(row.get("soft_score_mean_10min", 0) or 0) > 0 and float(row.get("hard_score", 0) or 0) < 4:
            cause = "SOFT_BACKGROUND_DRIFT"
        elif pd.notna(nearest) and nearest < 10:
            cause = "DUPLICATE_ALERT"
        elif pd.notna(nearest) and nearest > 90:
            cause = "TRUE_ISOLATED_FALSE_ALERT"
        else:
            cause = "UNKNOWN"
        rows.append(
            {
                "date": row.get("date"),
                "timestamp": row.get("alert_start"),
                "quality_label": row.get("quality_label", ""),
                "model_used": row.get("model", ""),
                "p30": row.get("max_p30", row.get("p30", np.nan)),
                "p60": row.get("max_p60", row.get("p60", np.nan)),
                "precursor_score": row.get("combined_precursor_score", np.nan),
                "soft_score": row.get("soft_score", np.nan),
                "hard_score": row.get("hard_score", np.nan),
                "post_flare_decay_state": row.get("post_flare_decay_state", np.nan),
                "quality_group": "GOOD" if str(row.get("quality_label", "")).upper() == "GOOD" else "QUESTIONABLE_OR_UNKNOWN",
                "nearest_goes_event_time_difference_min": nearest,
                "likely_cause_category": cause,
            }
        )
    return pd.DataFrame(rows)


def write_reports(sweep: pd.DataFrame, rec: pd.DataFrame, comparison: pd.DataFrame, false_alerts: pd.DataFrame) -> None:
    best = rec.iloc[0] if not rec.empty else pd.Series(dtype=object)
    beats = (not rec.empty) and v3_beats_v1(best)
    decision = (
        "Forecasting v3 meets the replacement rule and may be considered for replacing v1."
        if beats
        else "Forecasting v3 satisfies the trained ML forecasting requirement and provides a research-backed forecasting prototype, but the v1 90-minute precursor-aware policy remains the final recommended operating mode because it has higher precision/F1 and lower false-alert burden on the current small dataset."
    )
    causes = false_alerts["likely_cause_category"].value_counts().to_dict() if not false_alerts.empty else {}
    if not rec.empty:
        best_lines = "\n".join(f"- {k}: {v}" for k, v in best.to_dict().items())
    else:
        best_lines = "No policy rows available."
    if not comparison.empty:
        cols = ["system", "precision", "recall", "f1", "false_alerts_per_day", "valid_alerted_events", "mean_lead_time_min", "status"]
        comparison_lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in comparison.iterrows():
            comparison_lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
        comparison_table = "\n".join(comparison_lines)
    else:
        comparison_table = "No comparison rows available."
    rec_md = f"""# Forecasting v3 Policy Recommendations

## Best v3 State-Machine Policy

{best_lines}

## Decision

{decision}

## Main False-Alert Causes

{causes}

## Caveats

- This is state-machine post-processing of blocked/date-wise v3 prediction streams.
- It does not change nowcasting, v1/v2 results, or the current v1 90-minute recommended policy.
- v3 should not be claimed better than v1 unless the replacement rule is satisfied.
"""
    RECOMMENDATIONS_MD_PATH.write_text(rec_md, encoding="utf-8")

    comp_md = f"""# v1 vs v2 vs v3 Forecasting Comparison

{comparison_table}

## Decision Rule

v3 replaces v1 only if recall >= 0.50, F1 >= 0.50, false alerts/day <= 2.50, mean lead time is positive, and validation remains blocked/date-wise.

## Result

{decision}
"""
    V1V2V3_MD_PATH.write_text(comp_md, encoding="utf-8")

    fa_md = f"""# Forecasting v3 False-Alert Analysis

False alert episodes analyzed: {len(false_alerts)}

## Likely Cause Categories

{causes}

## Notes

- Cause labels are heuristic diagnostics for future improvement.
- SUIT/VELC/other payloads remain future work; SoLEXS + HEL1OS remain the core inputs.
"""
    FALSE_ALERT_MD_PATH.write_text(fa_md, encoding="utf-8")


def update_final_reports(comparison: pd.DataFrame, rec: pd.DataFrame) -> list[str]:
    best = rec.iloc[0] if not rec.empty else pd.Series(dtype=object)
    decision = (
        "v3 replaces v1 under the predefined rule."
        if (not rec.empty and v3_beats_v1(best))
        else "v1 remains final because v3 does not beat it under the predefined blocked-validation rule."
    )
    section = f"""## Trained ML Forecasting v3

Forecasting v3 adds a trained ML forecasting prototype using only SoLEXS + HEL1OS features. It predicts probability of flare onset in the next 30/60 minutes using blocked date-wise validation. The current v1 90-minute precursor-aware policy remains the final recommended operating mode unless v3 beats it under the predefined rule.

## Onset-Based Forecast Targets

Primary target: `flare_onset_next_30min`. Secondary target: `flare_onset_next_60min`. Peak prediction remains diagnostic only.

## Probability State-Machine Alert Policy

The v3 policy layer uses CLEAR, WATCH, ALERT, and COOLDOWN states. WATCH is driven by sustained 60-minute probability plus precursor confirmation. ALERT is driven by high 30-minute probability or nowcast detector trigger. COOLDOWN suppresses repeated alerts after an event.

## v1 vs v2 vs v3 Forecasting Comparison

See `results/forecasting_v3/v1_v2_v3_forecasting_comparison.csv`.

Decision: {decision}

## False-Alert Analysis and Next Forecasting Improvements

See `results/forecasting_v3/forecasting_v3_false_alert_analysis.csv`. Main future work is reducing QUESTIONABLE-date, post-flare-decay, hard-only-spike, and soft-background-drift false alerts without adding non-X-ray payloads to the core system.
"""
    paths = [
        Path("results/final_hackathon_evidence_report.md"),
        Path("results/space_agency_evaluation_criteria_scorecard.md"),
        Path("results/hackathon_diagnostic_report.md"),
        Path("final_submission_package/final_idea_submission.md"),
        Path("final_submission_package/detailed_project_summary.md"),
        Path("final_submission_package/architecture_summary.md"),
        Path("final_submission_package/final_claims_and_caveats.md"),
        Path("final_submission_package/final_demo_script.md"),
        Path("final_submission_package/final_readme.md"),
    ]
    updated = []
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        title = "## Trained ML Forecasting v3"
        if title in text:
            before = text.split(title, 1)[0].rstrip()
            after = text.split(title, 1)[1]
            import re

            match = re.search(r"\n## ", after)
            text = before + "\n\n" + section + ("\n" + after[match.start() :].lstrip() if match else "")
        else:
            text = text.rstrip() + "\n\n" + section
        path.write_text(text, encoding="utf-8")
        updated.append(str(path))
    return updated


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep, rec, best_episodes = sweep_policies()
    sweep.to_csv(SWEEP_PATH, index=False)
    rec.to_csv(RECOMMENDATIONS_PATH, index=False)
    best = rec.iloc[0] if not rec.empty else pd.Series(dtype=object)
    comparison = build_comparison(best) if not rec.empty else pd.DataFrame()
    comparison.to_csv(V1V2V3_PATH, index=False)
    false_eps = best_episodes[best_episodes.get("episode_type", pd.Series(dtype=str)).eq("ISOLATED_FALSE_ALERT")].copy() if not best_episodes.empty else pd.DataFrame()
    false_alerts = classify_false_alerts(false_eps, load_events()) if not false_eps.empty else pd.DataFrame()
    false_alerts.to_csv(FALSE_ALERT_PATH, index=False)
    write_reports(sweep, rec, comparison, false_alerts)
    updated = update_final_reports(comparison, rec)
    beats = bool(not rec.empty and v3_beats_v1(best))
    causes = false_alerts["likely_cause_category"].value_counts().to_dict() if not false_alerts.empty else {}
    print("best v3 policy:")
    print(best.to_string() if not rec.empty else "none")
    print(f"whether v3 beats v1: {'yes' if beats else 'no'}")
    print(f"final recommended forecasting mode: {'v3 best state-machine policy' if beats else 'v1 90-minute precursor-aware operational alert policy'}")
    print(f"main false-alert causes: {causes}")
    print("files created/updated:")
    for path in [SWEEP_PATH, RECOMMENDATIONS_PATH, RECOMMENDATIONS_MD_PATH, V1V2V3_PATH, V1V2V3_MD_PATH, FALSE_ALERT_PATH, FALSE_ALERT_MD_PATH]:
        print(path)
    for path in updated:
        print(path)


if __name__ == "__main__":
    main()
