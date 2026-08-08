from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

APPROVED_FLARES_PATH = PROJECT_ROOT / "results" / "expanded_80day_review" / "approved_flare_events_CANDIDATE.csv"
APPROVED_QUIET_PATH = PROJECT_ROOT / "results" / "expanded_80day_review" / "approved_quiet_days_CANDIDATE.csv"
SHADOW_PATH = PROJECT_ROOT / "results" / "shadow_holdout" / "shadow_holdout_dates.csv"
EXPANDED_DIR = PROJECT_ROOT / "results" / "expanded_80day_ingestion"
OUT_DIR = PROJECT_ROOT / "results" / "expanded_approved_model_recheck"

FROZEN_V8_1 = {
    "precision": 0.556,
    "recall": 0.933,
    "f1": 0.697,
    "far_per_day": 1.78,
    "valid_alerted_events": "14/15",
    "mean_lead_time_min": 51.20,
}


def ensure_out() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    if df.empty:
        return "_No rows._"
    view = df[columns].copy() if columns else df.copy()
    view = view.fillna("")
    headers = list(view.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("\n", " ") for col in headers) + " |")
    return "\n".join(lines)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [p for p in [APPROVED_FLARES_PATH, APPROVED_QUIET_PATH, SHADOW_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing approved review inputs: " + ", ".join(str(p) for p in missing))
    flares = pd.read_csv(APPROVED_FLARES_PATH)
    quiet = pd.read_csv(APPROVED_QUIET_PATH)
    holdout = pd.read_csv(SHADOW_PATH)
    for df in [flares, quiet, holdout]:
        df["date"] = df["date"].astype(str)
    return flares, quiet, holdout


def date_split(flares: pd.DataFrame, quiet: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    holdout_dates = set(holdout["date"].astype(str))
    flare_dates = set(flares["date"].astype(str))
    quiet_dates = set(quiet["date"].astype(str))
    rows = []
    for date in sorted(flare_dates | quiet_dates | holdout_dates):
        rows.append(
            {
                "date": date,
                "contains_approved_flare": date in flare_dates,
                "contains_approved_quiet": date in quiet_dates,
                "split": "SHADOW_HOLDOUT" if date in holdout_dates else "TRAIN_DEV",
            }
        )
    split = pd.DataFrame(rows)
    split.to_csv(OUT_DIR / "approved_date_split.csv", index=False)
    return split


def read_scored_timeseries(date: str) -> pd.DataFrame:
    path = EXPANDED_DIR / f"{date}_scored_timeseries.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    first = df.columns[0]
    if first not in {"timestamp", "time_utc"}:
        df = df.rename(columns={first: "timestamp"})
    else:
        df = df.rename(columns={first: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed", errors="coerce")
    df["date"] = date
    for col in [
        "soft_solexs_2_22",
        "hard_cdte_5_20",
        "hard_czt_20_40",
        "soft_score",
        "hard_score",
        "cdte_score",
        "czt_score",
        "czt_minus_cdte_score",
        "czt_to_cdte_ratio_safe",
    ]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "hard_band_dominance" not in df.columns:
        df["hard_band_dominance"] = "UNKNOWN"
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df


def add_labels(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["flare_onset_next_30min"] = False
    out["flare_onset_next_60min"] = False
    out["event_id_next_90min"] = ""
    if out.empty or events.empty:
        return out
    events = events.copy()
    events["event_start"] = pd.to_datetime(events["event_start"], utc=True, format="mixed", errors="coerce")
    for _, event in events.dropna(subset=["event_start"]).iterrows():
        start = event["event_start"]
        eid = str(event.get("candidate_global_event_id", event.get("event_id", "")))
        mask30 = (out["timestamp"] <= start) & (out["timestamp"] >= start - pd.Timedelta(minutes=30))
        mask60 = (out["timestamp"] <= start) & (out["timestamp"] >= start - pd.Timedelta(minutes=60))
        mask90 = (out["timestamp"] <= start) & (out["timestamp"] >= start - pd.Timedelta(minutes=90))
        out.loc[mask30, "flare_onset_next_30min"] = True
        out.loc[mask60, "flare_onset_next_60min"] = True
        out.loc[mask90 & out["event_id_next_90min"].eq(""), "event_id_next_90min"] = eid
    return out


def causal_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    g = df.copy()
    hard_total = g[["hard_cdte_5_20", "hard_czt_20_40"]].fillna(0).sum(axis=1)
    soft = g["soft_solexs_2_22"].fillna(0)
    g["hard_total_proxy"] = hard_total
    g["hard_to_soft_ratio_safe"] = hard_total / (soft.abs() + 1e-6)
    g["hard_score_max_component"] = g[["cdte_score", "czt_score"]].max(axis=1)
    g["hel1os_burst_support"] = ((g["cdte_score"] >= 50).astype(int) + (g["czt_score"] >= 50).astype(int))
    g["soft_support"] = g["soft_score"] >= 10
    g["hard_support"] = g["hard_score"] >= 50
    g["fusion_support"] = g["soft_support"] | g["hard_support"]
    return g


def build_dataset(flares: pd.DataFrame, quiet: pd.DataFrame, split: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    train_dates = split.loc[split["split"].eq("TRAIN_DEV"), "date"].astype(str).tolist()
    holdout_dates = split.loc[split["split"].eq("SHADOW_HOLDOUT"), "date"].astype(str).tolist()
    frames = []
    for date in train_dates:
        scored = read_scored_timeseries(date)
        if scored.empty:
            continue
        events = flares[flares["date"].astype(str).eq(date)]
        scored = add_labels(scored, events)
        scored = causal_feature_frame(scored)
        scored["evaluation_unit"] = np.where(scored["date"].isin(quiet["date"].astype(str)), "APPROVED_QUIET_DAY", "APPROVED_FLARE_DAY")
        scored["split"] = "TRAIN_DEV"
        keep = [
            "timestamp",
            "date",
            "split",
            "evaluation_unit",
            "soft_solexs_2_22",
            "hard_cdte_5_20",
            "hard_czt_20_40",
            "soft_score",
            "hard_score",
            "cdte_score",
            "czt_score",
            "czt_minus_cdte_score",
            "czt_to_cdte_ratio_safe",
            "hard_band_dominance",
            "hard_total_proxy",
            "hard_to_soft_ratio_safe",
            "hard_score_max_component",
            "hel1os_burst_support",
            "soft_support",
            "hard_support",
            "fusion_support",
            "flare_onset_next_30min",
            "flare_onset_next_60min",
            "event_id_next_90min",
        ]
        frames.append(scored[keep])
    dataset = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    dataset.to_csv(OUT_DIR / "approved_expanded_forecasting_dataset.csv", index=False)
    return dataset, train_dates, holdout_dates


def merge_alerts(pos: pd.DataFrame, gap_seconds: int = 120) -> pd.DataFrame:
    if pos.empty:
        return pd.DataFrame(columns=["date", "alert_start", "alert_end", "duration_sec", "max_score"])
    rows = []
    for date, group in pos.sort_values("timestamp").groupby("date"):
        current_start = None
        current_end = None
        current_max = -np.inf
        for _, row in group.iterrows():
            ts = row["timestamp"]
            score = float(row.get("policy_score", 0))
            if current_start is None:
                current_start = current_end = ts
                current_max = score
                continue
            if (ts - current_end).total_seconds() <= gap_seconds:
                current_end = ts
                current_max = max(current_max, score)
            else:
                rows.append(
                    {
                        "date": date,
                        "alert_start": current_start,
                        "alert_end": current_end,
                        "duration_sec": (current_end - current_start).total_seconds(),
                        "max_score": current_max,
                    }
                )
                current_start = current_end = ts
                current_max = score
        if current_start is not None:
            rows.append(
                {
                    "date": date,
                    "alert_start": current_start,
                    "alert_end": current_end,
                    "duration_sec": (current_end - current_start).total_seconds(),
                    "max_score": current_max,
                }
            )
    return pd.DataFrame(rows)


def policy_mask(df: pd.DataFrame, policy: str) -> tuple[pd.Series, pd.Series]:
    soft10 = df["soft_score"].fillna(0) >= 10
    soft20 = df["soft_score"].fillna(0) >= 20
    soft50 = df["soft_score"].fillna(0) >= 50
    hard35 = df["hard_score"].fillna(0) >= 35
    hard50 = df["hard_score"].fillna(0) >= 50
    hard100 = df["hard_score"].fillna(0) >= 100
    burst = df["hel1os_burst_support"].fillna(0) >= 1
    physics_support_count = (
        (df["soft_score"].fillna(0) >= 20).astype(int)
        + (df["hard_score"].fillna(0) >= 50).astype(int)
        + (df["hel1os_burst_support"].fillna(0) >= 1).astype(int)
        + (df["hard_to_soft_ratio_safe"].replace([np.inf, -np.inf], np.nan).fillna(0) >= 0.25).astype(int)
    )
    if policy == "simple_threshold_baseline":
        mask = soft20
        score = df["soft_score"].fillna(0)
    elif policy == "solexs_only":
        mask = soft10
        score = df["soft_score"].fillna(0)
    elif policy == "hel1os_only":
        mask = hard50
        score = df["hard_score"].fillna(0)
    elif policy == "soft_hard_fusion":
        mask = soft10 | hard50
        score = df[["soft_score", "hard_score"]].fillna(0).max(axis=1)
    elif policy == "v6_conservative_fallback_proxy":
        mask = soft50 | hard100
        score = df[["soft_score", "hard_score"]].fillna(0).max(axis=1)
    elif policy == "v7lite_physics_branch_proxy":
        mask = soft10 | hard35 | ((physics_support_count >= 2) & (soft20 | hard35))
        score = df[["soft_score", "hard_score"]].fillna(0).max(axis=1) + physics_support_count
    elif policy == "v8_1_repaired_gated_proxy":
        v6_strong = soft50 | hard100
        v3_strong = soft10 | hard35
        v6_moderate = soft20 | hard50
        mask = v6_strong | (v3_strong & v6_moderate & (physics_support_count >= 2))
        score = df[["soft_score", "hard_score"]].fillna(0).max(axis=1) + physics_support_count
    elif policy == "neupert_feature_branch_proxy":
        mask = (df["hard_to_soft_ratio_safe"].replace([np.inf, -np.inf], np.nan).fillna(0) >= 0.25) & (soft20 | hard35)
        score = df["hard_to_soft_ratio_safe"].replace([np.inf, -np.inf], np.nan).fillna(0)
    elif policy == "hel1os_burst_augmented_proxy":
        mask = (soft10 | hard35) & (burst | hard50)
        score = df[["soft_score", "hard_score"]].fillna(0).max(axis=1) + (burst.astype(int) * 5)
    else:
        raise ValueError(f"Unknown policy: {policy}")
    return mask.fillna(False), score


def evaluate_policy(dataset: pd.DataFrame, events: pd.DataFrame, train_dates: list[str], policy: str) -> dict:
    if dataset.empty:
        return {"policy": policy, "precision": np.nan, "recall": np.nan, "f1": np.nan}
    mask, score = policy_mask(dataset, policy)
    positives = dataset.loc[mask, ["timestamp", "date"]].copy()
    positives["policy_score"] = score.loc[mask].to_numpy()
    alerts = merge_alerts(positives)
    events = events[events["date"].astype(str).isin(train_dates)].copy()
    events["event_start"] = pd.to_datetime(events["event_start"], utc=True, format="mixed", errors="coerce")

    useful = 0
    false_alerts = 0
    matched_event_ids: set[str] = set()
    lead_times = []
    quiet_false = 0
    event_dates = set(events["date"].astype(str))
    for _, alert in alerts.iterrows():
        same = events[events["date"].astype(str).eq(str(alert["date"]))]
        match_event = None
        best_lead = None
        for _, event in same.dropna(subset=["event_start"]).iterrows():
            lead = (event["event_start"] - alert["alert_start"]).total_seconds() / 60
            overlap = alert["alert_start"] <= event["event_start"] <= alert["alert_end"] + pd.Timedelta(minutes=5)
            if (0 <= lead <= 90) or overlap:
                if best_lead is None or lead > best_lead:
                    best_lead = lead
                    match_event = event
        if match_event is not None:
            useful += 1
            eid = str(match_event.get("candidate_global_event_id", match_event.get("event_id", "")))
            if eid not in matched_event_ids:
                matched_event_ids.add(eid)
                lead_times.append(best_lead)
        else:
            false_alerts += 1
            if str(alert["date"]) not in event_dates:
                quiet_false += 1
    total_events = len(events)
    precision = useful / len(alerts) if len(alerts) else 0.0
    recall = len(matched_event_ids) / total_events if total_events else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    missed = total_events - len(matched_event_ids)
    return {
        "policy": policy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "far_per_day": false_alerts / max(len(train_dates), 1),
        "valid_alerted_events": len(matched_event_ids),
        "total_events": total_events,
        "false_alerts": false_alerts,
        "missed_events": missed,
        "mean_lead_time_min": float(np.nanmean(lead_times)) if lead_times else np.nan,
        "median_lead_time_min": float(np.nanmedian(lead_times)) if lead_times else np.nan,
        "quiet_day_false_alerts": quiet_false,
        "total_alert_episodes": len(alerts),
        "notes": "Expanded approved TRAIN_DEV policy recheck; not frozen model replacement.",
    }


def write_reports(
    dataset: pd.DataFrame,
    flares: pd.DataFrame,
    quiet: pd.DataFrame,
    holdout: pd.DataFrame,
    train_dates: list[str],
    holdout_dates: list[str],
    model_results: pd.DataFrame,
    baseline_results: pd.DataFrame,
    burst_ablation: pd.DataFrame,
) -> None:
    positives30 = int(dataset["flare_onset_next_30min"].sum()) if not dataset.empty else 0
    positives60 = int(dataset["flare_onset_next_60min"].sum()) if not dataset.empty else 0
    summary = [
        "# Approved Expanded Forecasting Dataset Summary",
        "",
        f"- Rows: {len(dataset)}",
        f"- Columns: {len(dataset.columns) if not dataset.empty else 0}",
        f"- TRAIN_DEV dates: {len(train_dates)}",
        f"- SHADOW_HOLDOUT dates excluded: {len(holdout_dates)}",
        f"- Approved flare events on TRAIN_DEV: {len(flares[flares['date'].astype(str).isin(train_dates)])}",
        f"- Approved quiet days on TRAIN_DEV: {len(quiet[quiet['date'].astype(str).isin(train_dates)])}",
        f"- Positive rows, onset next 30 min: {positives30}",
        f"- Positive rows, onset next 60 min: {positives60}",
        "",
        "Forbidden future/catalogue fields are not included as model-input columns: GOES times/classes, future peak times, hard-to-soft lead time, and validation match fields are excluded.",
    ]
    (OUT_DIR / "approved_dataset_summary.md").write_text("\n".join(summary), encoding="utf-8")

    protocol = [
        "# Shadow Holdout Protocol",
        "",
        "The following dates are excluded from training, threshold tuning, and model selection. Run the final selected model on them once near the end.",
        "",
        markdown_table(holdout),
        "",
        "Rules:",
        "- Do not train on these dates.",
        "- Do not tune thresholds on these dates.",
        "- Do not inspect repeated metric variants on these dates.",
        "- Use the same event-level alert episode logic as TRAIN_DEV.",
    ]
    (OUT_DIR / "shadow_holdout_protocol.md").write_text("\n".join(protocol), encoding="utf-8")

    v8 = model_results[model_results["policy"].eq("v8_1_repaired_gated_proxy")]
    best = model_results.sort_values(["f1", "recall", "precision"], ascending=False).head(1)
    recommendation = "Deep learning is not the next best step yet; first validate these expanded approved policies on the locked shadow holdout and review NO_MATCH events."
    if not best.empty and best.iloc[0]["policy"] == "v8_1_repaired_gated_proxy":
        hold = "The v8.1-style gated proxy is the best expanded TRAIN_DEV policy recheck row."
    else:
        hold = "The v8.1-style gated proxy does not dominate every expanded TRAIN_DEV policy row; treat this as diagnostic, not a frozen-result change."
    report = [
        "# Expanded Approved Model Recheck Report",
        "",
        f"- Approved flare events total: {len(flares)}",
        f"- Approved quiet days total: {len(quiet)}",
        f"- TRAIN_DEV dates used: {', '.join(train_dates)}",
        f"- SHADOW_HOLDOUT dates excluded: {', '.join(holdout_dates)}",
        "",
        "## Existing Model / Rule Recheck",
        "",
        markdown_table(model_results),
        "",
        "## Soft-only vs Hard-only vs Fusion",
        "",
        markdown_table(baseline_results),
        "",
        "## HEL1OS Burst Ablation",
        "",
        markdown_table(burst_ablation),
        "",
        "## v8.1 Expanded-Data Interpretation",
        "",
        hold,
        "",
        "Frozen v8.1 headline metrics remain unchanged:",
        f"- Precision {FROZEN_V8_1['precision']}, recall {FROZEN_V8_1['recall']}, F1 {FROZEN_V8_1['f1']}, FAR/day {FROZEN_V8_1['far_per_day']}, mean lead {FROZEN_V8_1['mean_lead_time_min']} min.",
        "",
        "## Deep Learning Recommendation",
        "",
        recommendation,
    ]
    (OUT_DIR / "EXPANDED_APPROVED_MODEL_RECHECK_REPORT.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    ensure_out()
    flares, quiet, holdout = load_inputs()
    split = date_split(flares, quiet, holdout)
    dataset, train_dates, holdout_dates = build_dataset(flares, quiet, split)

    train_flares = flares[flares["date"].astype(str).isin(train_dates)].copy()
    policies = [
        "v6_conservative_fallback_proxy",
        "v7lite_physics_branch_proxy",
        "v8_1_repaired_gated_proxy",
        "neupert_feature_branch_proxy",
        "hel1os_burst_augmented_proxy",
    ]
    model_results = pd.DataFrame([evaluate_policy(dataset, train_flares, train_dates, p) for p in policies])
    model_results.to_csv(OUT_DIR / "existing_model_recheck_results.csv", index=False)

    baselines = ["simple_threshold_baseline", "solexs_only", "hel1os_only", "soft_hard_fusion"]
    baseline_results = pd.DataFrame([evaluate_policy(dataset, train_flares, train_dates, p) for p in baselines])
    baseline_results.to_csv(OUT_DIR / "soft_hard_fusion_rigorous_baseline.csv", index=False)

    burst_ablation = pd.DataFrame(
        [
            {**evaluate_policy(dataset, train_flares, train_dates, "soft_hard_fusion"), "ablation": "without_HEL1OS_burst_features"},
            {**evaluate_policy(dataset, train_flares, train_dates, "hel1os_burst_augmented_proxy"), "ablation": "with_HEL1OS_burst_features"},
        ]
    )
    burst_ablation.to_csv(OUT_DIR / "hel1os_burst_ablation.csv", index=False)

    write_reports(dataset, flares, quiet, holdout, train_dates, holdout_dates, model_results, baseline_results, burst_ablation)

    best = model_results.sort_values(["f1", "recall", "precision"], ascending=False).head(1)
    v8 = model_results[model_results["policy"].eq("v8_1_repaired_gated_proxy")].head(1)
    fusion = baseline_results[baseline_results["policy"].eq("soft_hard_fusion")].head(1)
    print(f"approved forecasting rows: {len(dataset)}")
    print(f"train/dev dates: {', '.join(train_dates)}")
    print(f"shadow holdout dates: {', '.join(holdout_dates)}")
    print(f"best expanded-data model: {best.iloc[0]['policy'] if not best.empty else 'none'}")
    print(f"v8.1 expanded-data result: {v8.to_dict('records')[0] if not v8.empty else 'missing'}")
    print(f"fusion vs soft-only vs hard-only result: {baseline_results[['policy','precision','recall','f1','far_per_day']].to_dict('records')}")
    print(f"HEL1OS burst ablation result: {burst_ablation[['ablation','precision','recall','f1','far_per_day']].to_dict('records')}")
    print("final recommendation: Keep frozen metrics unchanged; use this as TRAIN_DEV recheck and reserve shadow holdout for one final evaluation.")
    print(f"report path: {OUT_DIR / 'EXPANDED_APPROVED_MODEL_RECHECK_REPORT.md'}")


if __name__ == "__main__":
    main()
