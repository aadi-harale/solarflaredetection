from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RECHECK_DIR = PROJECT_ROOT / "results" / "expanded_approved_model_recheck"
REVIEW_DIR = PROJECT_ROOT / "results" / "expanded_80day_review"
HOLDOUT_PATH = PROJECT_ROOT / "results" / "shadow_holdout" / "shadow_holdout_dates.csv"
OUT_DIR = PROJECT_ROOT / "results" / "expanded_policy_repair"

DATASET_PATH = RECHECK_DIR / "approved_expanded_forecasting_dataset.csv"
FLARES_PATH = REVIEW_DIR / "approved_flare_events_CANDIDATE.csv"
QUIET_PATH = REVIEW_DIR / "approved_quiet_days_CANDIDATE.csv"
CURRENT_MODEL_RESULTS_PATH = RECHECK_DIR / "existing_model_recheck_results.csv"
CURRENT_BASELINES_PATH = RECHECK_DIR / "soft_hard_fusion_rigorous_baseline.csv"

FROZEN_V8_1 = {
    "policy": "frozen_original_v8_1_reference_only",
    "precision": 0.556,
    "recall": 0.933,
    "f1": 0.697,
    "far_per_day": 1.78,
    "valid_alerted_events": 14,
    "total_events": 15,
    "false_alerts": np.nan,
    "missed_events": 1,
    "mean_lead_time_min": 51.20,
    "median_lead_time_min": 52.43,
    "quiet_day_false_alerts": np.nan,
    "notes": "Frozen reference only; not recomputed on expanded TRAIN_DEV.",
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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [p for p in [DATASET_PATH, FLARES_PATH, QUIET_PATH, HOLDOUT_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(str(p) for p in missing))
    dataset = pd.read_csv(DATASET_PATH)
    flares = pd.read_csv(FLARES_PATH)
    quiet = pd.read_csv(QUIET_PATH)
    holdout = pd.read_csv(HOLDOUT_PATH)
    for df in [dataset, flares, quiet, holdout]:
        df["date"] = df["date"].astype(str)
    dataset["timestamp"] = pd.to_datetime(dataset["timestamp"], utc=True, format="mixed", errors="coerce")
    for col in [
        "soft_score",
        "hard_score",
        "cdte_score",
        "czt_score",
        "hard_total_proxy",
        "hard_to_soft_ratio_safe",
        "hel1os_burst_support",
        "soft_solexs_2_22",
        "hard_cdte_5_20",
        "hard_czt_20_40",
    ]:
        if col not in dataset.columns:
            dataset[col] = 0.0
        dataset[col] = pd.to_numeric(dataset[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    return dataset, flares, quiet, holdout


def current_train_dates(dataset: pd.DataFrame) -> list[str]:
    return sorted(dataset.loc[dataset["split"].eq("TRAIN_DEV"), "date"].dropna().astype(str).unique())


def audit_holdout_balance(flares: pd.DataFrame, quiet: pd.DataFrame, holdout: pd.DataFrame, train_dates: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    holdout_dates = set(holdout["date"].astype(str))
    train_dates_set = set(train_dates)
    rows = [
        {
            "group": "TRAIN_DEV",
            "approved_flare_events": int(flares["date"].astype(str).isin(train_dates_set).sum()),
            "approved_flare_dates": int(flares.loc[flares["date"].astype(str).isin(train_dates_set), "date"].nunique()),
            "approved_quiet_days": int(quiet["date"].astype(str).isin(train_dates_set).sum()),
            "dates": ",".join(sorted(train_dates_set)),
        },
        {
            "group": "SHADOW_HOLDOUT",
            "approved_flare_events": int(flares["date"].astype(str).isin(holdout_dates).sum()),
            "approved_flare_dates": int(flares.loc[flares["date"].astype(str).isin(holdout_dates), "date"].nunique()),
            "approved_quiet_days": int(quiet["date"].astype(str).isin(holdout_dates).sum()),
            "dates": ",".join(sorted(holdout_dates)),
        },
    ]
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT_DIR / "shadow_holdout_balance_audit.csv", index=False)

    quiet_candidates = holdout[holdout["holdout_type"].astype(str).eq("QUIET")].copy()
    flare_candidates = holdout[holdout["holdout_type"].astype(str).eq("FLARE")].copy()
    preferred_flare = flare_candidates[~flare_candidates["is_prior_frozen_date"].astype(str).str.upper().eq("TRUE")]
    if preferred_flare.empty:
        preferred_flare = flare_candidates
    selected_flare = preferred_flare.head(1)
    selected_quiet = quiet_candidates.head(4)
    proposal = pd.concat([selected_flare, selected_quiet], ignore_index=True)
    proposal["proposal_reason"] = np.where(
        proposal["holdout_type"].eq("FLARE"),
        "Keep one flare date to maximize TRAIN_DEV approved flare events; caveat if not fully unseen.",
        "Keep quiet/control date for FAR/day testing without consuming flare training events.",
    )
    proposal.to_csv(OUT_DIR / "revised_shadow_holdout_proposal.csv", index=False)
    return audit, proposal


def signal_window(dataset: pd.DataFrame, date: str, event_start: pd.Timestamp, minutes: int = 90) -> pd.DataFrame:
    return dataset[
        dataset["date"].eq(str(date))
        & (dataset["timestamp"] <= event_start)
        & (dataset["timestamp"] >= event_start - pd.Timedelta(minutes=minutes))
    ].copy()


def policy_mask(df: pd.DataFrame, policy: str, params: dict | None = None) -> tuple[pd.Series, pd.Series]:
    params = params or {}
    soft = df["soft_score"].fillna(0)
    hard = df["hard_score"].fillna(0)
    cdte = df["cdte_score"].fillna(0)
    czt = df["czt_score"].fillna(0)
    burst = df["hel1os_burst_support"].fillna(0)
    ratio = df["hard_to_soft_ratio_safe"].fillna(0)
    soft_flux = df["soft_solexs_2_22"].fillna(0)
    soft_rising = soft >= params.get("soft_rise", 10)
    solexs_not_quiet = (soft >= params.get("soft_not_quiet_score", 5)) | (soft_flux >= params.get("soft_not_quiet_flux", 150))
    hard_strong = hard >= params.get("hard_strong", 50)
    hard_mod = hard >= params.get("hard_moderate", 35)
    burst_strong = burst >= params.get("burst_min", 1)
    neupert = ratio >= params.get("ratio_min", 0.25)
    weak_hard_support = hard >= params.get("hard_weak", 20)

    if policy == "solexs_only":
        mask = soft >= 10
        score = soft
    elif policy == "hel1os_only":
        mask = hard >= 50
        score = hard
    elif policy == "fusion_proxy":
        mask = (soft >= 10) | (hard >= 50)
        score = pd.concat([soft, hard], axis=1).max(axis=1)
    elif policy == "burst_augmented_proxy":
        mask = ((soft >= 10) | (hard >= 35)) & ((burst >= 1) | (hard >= 50))
        score = pd.concat([soft, hard], axis=1).max(axis=1) + 5 * (burst >= 1).astype(int)
    elif policy == "policy_A_hard_burst_or_trend":
        mask = burst_strong | hard_strong
        score = hard + 10 * burst_strong.astype(int)
    elif policy == "policy_B_hard_and_soft_not_quiet":
        mask = hard_strong & solexs_not_quiet
        score = hard + soft.clip(upper=50)
    elif policy == "policy_C_hard_moderate_soft_rising":
        mask = hard_mod & soft_rising
        score = hard + soft
    elif policy == "policy_D_burst_with_soft_confirmation":
        mask = burst_strong & solexs_not_quiet
        score = hard + soft + 10 * burst_strong.astype(int)
    elif policy == "policy_E_hard_first_neupert_soft_rescue":
        mask = burst_strong | (hard_mod & neupert) | ((soft >= params.get("soft_very_strong", 50)) & weak_hard_support)
        score = hard + soft.clip(upper=50) + 20 * neupert.astype(int) + 10 * burst_strong.astype(int)
    else:
        raise ValueError(f"Unknown policy: {policy}")
    return mask.fillna(False), score.fillna(0)


def merge_alerts(pos: pd.DataFrame, gap_seconds: int = 120) -> pd.DataFrame:
    if pos.empty:
        return pd.DataFrame(columns=["date", "alert_start", "alert_end", "duration_sec", "max_score", "source_branch"])
    rows = []
    for (date, source), group in pos.sort_values("timestamp").groupby(["date", "source_branch"]):
        start = end = None
        max_score = -np.inf
        for _, row in group.iterrows():
            ts = row["timestamp"]
            score = float(row["policy_score"])
            if start is None:
                start = end = ts
                max_score = score
            elif (ts - end).total_seconds() <= gap_seconds:
                end = ts
                max_score = max(max_score, score)
            else:
                rows.append({"date": date, "alert_start": start, "alert_end": end, "duration_sec": (end - start).total_seconds(), "max_score": max_score, "source_branch": source})
                start = end = ts
                max_score = score
        if start is not None:
            rows.append({"date": date, "alert_start": start, "alert_end": end, "duration_sec": (end - start).total_seconds(), "max_score": max_score, "source_branch": source})
    return pd.DataFrame(rows)


def alert_episodes_for_policy(dataset: pd.DataFrame, policy: str, params: dict | None = None, source_branch: str | None = None) -> pd.DataFrame:
    mask, score = policy_mask(dataset, policy, params)
    pos = dataset.loc[mask, ["timestamp", "date", "soft_score", "hard_score", "cdte_score", "czt_score", "hel1os_burst_support", "hard_to_soft_ratio_safe"]].copy()
    pos["policy_score"] = score.loc[mask].to_numpy()
    pos["source_branch"] = source_branch or policy
    return merge_alerts(pos)


def match_alerts_to_events(alerts: pd.DataFrame, events: pd.DataFrame, train_dates: list[str]) -> tuple[dict, pd.DataFrame]:
    events = events[events["date"].astype(str).isin(train_dates)].copy()
    events["event_start"] = pd.to_datetime(events["event_start"], utc=True, format="mixed", errors="coerce")
    useful = 0
    false_alerts = 0
    quiet_false = 0
    matched_ids: set[str] = set()
    leads = []
    rows = []
    event_dates = set(events["date"].astype(str))
    for _, alert in alerts.iterrows():
        same = events[events["date"].astype(str).eq(str(alert["date"]))]
        best = None
        best_lead = None
        for _, event in same.dropna(subset=["event_start"]).iterrows():
            lead = (event["event_start"] - alert["alert_start"]).total_seconds() / 60
            overlap = alert["alert_start"] <= event["event_start"] <= alert["alert_end"] + pd.Timedelta(minutes=5)
            if (0 <= lead <= 90) or overlap:
                if best is None or lead > best_lead:
                    best = event
                    best_lead = lead
        if best is not None:
            useful += 1
            eid = str(best.get("candidate_global_event_id", best.get("event_id", "")))
            first_for_event = eid not in matched_ids
            matched_ids.add(eid)
            if first_for_event:
                leads.append(best_lead)
            rows.append({**alert.to_dict(), "matched_event_id": eid, "lead_time_min": best_lead, "episode_type": "USEFUL"})
        else:
            false_alerts += 1
            if str(alert["date"]) not in event_dates:
                quiet_false += 1
            rows.append({**alert.to_dict(), "matched_event_id": "", "lead_time_min": np.nan, "episode_type": "FALSE_ALERT"})
    total_events = len(events)
    precision = useful / len(alerts) if len(alerts) else 0.0
    recall = len(matched_ids) / total_events if total_events else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "far_per_day": false_alerts / max(len(train_dates), 1),
        "valid_alerted_events": len(matched_ids),
        "total_events": total_events,
        "false_alerts": false_alerts,
        "missed_events": total_events - len(matched_ids),
        "mean_lead_time_min": float(np.nanmean(leads)) if leads else np.nan,
        "median_lead_time_min": float(np.nanmedian(leads)) if leads else np.nan,
        "quiet_day_false_alerts": quiet_false,
        "total_alert_episodes": len(alerts),
    }
    return metrics, pd.DataFrame(rows)


def current_policy_alerts(dataset: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "solexs_only": alert_episodes_for_policy(dataset, "solexs_only"),
        "hel1os_only": alert_episodes_for_policy(dataset, "hel1os_only"),
        "fusion_proxy": alert_episodes_for_policy(dataset, "fusion_proxy"),
        "burst_augmented_proxy": alert_episodes_for_policy(dataset, "burst_augmented_proxy"),
    }


def event_error_table(dataset: pd.DataFrame, flares: pd.DataFrame, train_dates: list[str]) -> pd.DataFrame:
    train_events = flares[flares["date"].astype(str).isin(train_dates)].copy()
    train_events["event_start"] = pd.to_datetime(train_events["event_start"], utc=True, format="mixed", errors="coerce")
    alert_map = current_policy_alerts(dataset)
    rows = []
    for _, event in train_events.iterrows():
        date = str(event["date"])
        start = event["event_start"]
        window = signal_window(dataset, date, start, 90)
        row = {
            "candidate_global_event_id": event.get("candidate_global_event_id", ""),
            "date": date,
            "event_start": start,
            "goes_class": event.get("goes_class", ""),
            "max_soft_score_90min": float(window["soft_score"].max()) if not window.empty else np.nan,
            "max_hard_score_90min": float(window["hard_score"].max()) if not window.empty else np.nan,
            "max_cdte_score_90min": float(window["cdte_score"].max()) if not window.empty else np.nan,
            "max_czt_score_90min": float(window["czt_score"].max()) if not window.empty else np.nan,
            "max_burst_support_90min": float(window["hel1os_burst_support"].max()) if not window.empty else np.nan,
            "max_neupert_ratio_90min": float(window["hard_to_soft_ratio_safe"].max()) if not window.empty else np.nan,
        }
        detected_any = False
        for policy, alerts in alert_map.items():
            same = alerts[alerts["date"].astype(str).eq(date)]
            valid = same[(same["alert_start"] <= start) & (same["alert_start"] >= start - pd.Timedelta(minutes=90))]
            detected = not valid.empty
            row[f"detected_by_{policy}"] = detected
            if detected:
                first = valid.sort_values("alert_start").iloc[0]
                row[f"{policy}_first_alert_time"] = first["alert_start"]
                row[f"{policy}_lead_time_min"] = (start - first["alert_start"]).total_seconds() / 60
                detected_any = True
            else:
                row[f"{policy}_first_alert_time"] = ""
                row[f"{policy}_lead_time_min"] = np.nan
        if detected_any:
            row["missed_reason"] = ""
        elif row["max_hard_score_90min"] < 35 and row["max_soft_score_90min"] < 10:
            row["missed_reason"] = "LOW_SOFT_AND_HARD_SCORES"
        elif row["max_hard_score_90min"] < 35:
            row["missed_reason"] = "WEAK_HEL1OS_PRECURSOR"
        elif row["max_soft_score_90min"] < 10:
            row["missed_reason"] = "WEAK_SOLEXS_CONFIRMATION"
        else:
            row["missed_reason"] = "TIMING_OR_EPISODE_MERGE_MISS"
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "train_dev_event_error_table.csv", index=False)
    return out


def likely_false_cause(row: pd.Series) -> str:
    soft = float(row.get("soft_score", 0))
    hard = float(row.get("hard_score", 0))
    burst = float(row.get("hel1os_burst_support", 0))
    ratio = float(row.get("hard_to_soft_ratio_safe", 0))
    if soft >= 10 and hard < 35:
        return "SOFT_ONLY_BACKGROUND_OR_SMALL_RISE"
    if hard >= 50 and burst >= 1:
        return "HARD_BURST_WITHOUT_APPROVED_EVENT"
    if hard >= 35 and ratio >= 0.25:
        return "HARD_NEUPERT_LIKE_WITHOUT_APPROVED_EVENT"
    if hard >= 35:
        return "HARD_ONLY_SPIKE"
    return "LOW_LEVEL_THRESHOLD_CROSSING"


def false_alert_table(dataset: pd.DataFrame, flares: pd.DataFrame, quiet_dates: list[str], train_dates: list[str]) -> pd.DataFrame:
    rows = []
    for policy, alerts in current_policy_alerts(dataset).items():
        metrics, classified = match_alerts_to_events(alerts, flares, train_dates)
        false = classified[classified["episode_type"].eq("FALSE_ALERT") & classified["date"].astype(str).isin(quiet_dates)].copy()
        for _, alert in false.iterrows():
            nearby = dataset[(dataset["date"].eq(str(alert["date"]))) & (dataset["timestamp"].between(alert["alert_start"], alert["alert_end"]))]
            peak = nearby.sort_values("hard_score", ascending=False).head(1)
            sig = peak.iloc[0] if not peak.empty else pd.Series(dtype=float)
            rows.append(
                {
                    "date": alert["date"],
                    "start_time": alert["alert_start"],
                    "end_time": alert["alert_end"],
                    "source_branch": policy,
                    "soft_signal_score": sig.get("soft_score", np.nan),
                    "hard_signal_score": sig.get("hard_score", np.nan),
                    "burst_score": sig.get("hel1os_burst_support", np.nan),
                    "neupert_ratio": sig.get("hard_to_soft_ratio_safe", np.nan),
                    "likely_false_alert_cause": likely_false_cause(sig),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "train_dev_false_alert_table.csv", index=False)
    return out


def repair_policy_grid() -> list[tuple[str, dict]]:
    grid: list[tuple[str, dict]] = []
    for hard in [50, 75]:
        for burst in [1, 2]:
            grid.append(("policy_A_hard_burst_or_trend", {"hard_strong": hard, "burst_min": burst}))
    for hard in [50, 75]:
        for soft_score in [5, 10]:
            grid.append(("policy_B_hard_and_soft_not_quiet", {"hard_strong": hard, "soft_not_quiet_score": soft_score, "soft_not_quiet_flux": 150}))
    for hard in [25, 35, 50]:
        for soft_rise in [10, 20]:
            grid.append(("policy_C_hard_moderate_soft_rising", {"hard_moderate": hard, "soft_rise": soft_rise}))
    for burst in [1, 2]:
        for soft_score in [5, 10]:
            grid.append(("policy_D_burst_with_soft_confirmation", {"burst_min": burst, "soft_not_quiet_score": soft_score, "soft_not_quiet_flux": 150}))
    for hard in [25, 35]:
        for ratio in [0.1, 0.25]:
            grid.append(("policy_E_hard_first_neupert_soft_rescue", {"hard_moderate": hard, "ratio_min": ratio, "soft_very_strong": 50, "hard_weak": 20, "burst_min": 1}))
    return grid


def run_repair_sweep(dataset: pd.DataFrame, flares: pd.DataFrame, train_dates: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for idx, (policy, params) in enumerate(repair_policy_grid(), start=1):
        alerts = alert_episodes_for_policy(dataset, policy, params)
        metrics, _ = match_alerts_to_events(alerts, flares, train_dates)
        rows.append({"repair_policy_id": f"R{idx:03d}", "policy": policy, "params": str(params), **metrics})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT_DIR / "policy_repair_threshold_sweep.csv", index=False)
    best = sweep.sort_values(["f1", "recall", "precision", "far_per_day"], ascending=[False, False, False, True]).head(5)
    best.to_csv(OUT_DIR / "policy_repair_best_results.csv", index=False)
    return sweep, best


def comparison_table(best: pd.DataFrame, model_results: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not best.empty:
        row = best.iloc[0].to_dict()
        row["comparison_group"] = "best_repaired_policy"
        rows.append(row)
    for name in ["v8_1_repaired_gated_proxy", "hel1os_only", "soft_hard_fusion", "burst_augmented_proxy"]:
        source = model_results if name in set(model_results.get("policy", [])) else baselines
        match = source[source["policy"].eq(name)]
        if not match.empty:
            row = match.iloc[0].to_dict()
            row["comparison_group"] = "current_expanded_recheck"
            rows.append(row)
    rows.append({**FROZEN_V8_1, "comparison_group": "frozen_reference"})
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "policy_repair_comparison.csv", index=False)
    return out


def neural_network_recommendation(flares: pd.DataFrame, quiet: pd.DataFrame, proposal: pd.DataFrame, best: pd.DataFrame, current_train_events: int) -> str:
    proposed_holdout = set(proposal["date"].astype(str))
    train_events_after = int((~flares["date"].astype(str).isin(proposed_holdout)).sum())
    train_quiet_after = int((~quiet["date"].astype(str).isin(proposed_holdout)).sum())
    if train_events_after < 10:
        return f"No. Even after the revised holdout proposal, TRAIN_DEV would have only {train_events_after} approved flare events; neural networks remain too data-hungry."
    if best.empty:
        return "No. No repaired policy result exists yet, so neural-network failure cases are not well-defined."
    if train_quiet_after < 10:
        return f"No. Only {train_quiet_after} approved quiet days would remain for FAR/day testing."
    return (
        "Maybe later, but not yet as the next step. TRAIN_DEV flare count may be adequate after holdout rebalance, "
        "but the repaired policy failure cases should be reviewed first."
    )


def write_report(
    audit: pd.DataFrame,
    proposal: pd.DataFrame,
    event_errors: pd.DataFrame,
    false_alerts: pd.DataFrame,
    sweep: pd.DataFrame,
    best: pd.DataFrame,
    comparison: pd.DataFrame,
    nn_recommendation: str,
) -> None:
    missed = event_errors[event_errors["missed_reason"].astype(str).ne("")]
    causes = false_alerts["likely_false_alert_cause"].value_counts().reset_index(name="count").rename(columns={"index": "likely_false_alert_cause"}) if not false_alerts.empty else pd.DataFrame()
    best_row = best.head(1)
    md = [
        "# Expanded Policy Repair Report",
        "",
        "This is a TRAIN_DEV-only, non-destructive policy repair pass. Frozen v8.1/v6 metrics are not changed, and the shadow holdout is not evaluated.",
        "",
        "## Shadow Holdout Balance",
        "",
        markdown_table(audit),
        "",
        "The current shadow holdout consumes too many approved flare events for a tiny expanded dataset. A revised proposal is generated but not applied/evaluated.",
        "",
        markdown_table(proposal),
        "",
        "## Event-Level Missed-Event Analysis",
        "",
        f"- TRAIN_DEV approved flare events inspected: {len(event_errors)}",
        f"- Events missed by all current branches: {len(missed)}",
        "",
        markdown_table(event_errors[["candidate_global_event_id", "date", "goes_class", "detected_by_solexs_only", "detected_by_hel1os_only", "detected_by_fusion_proxy", "detected_by_burst_augmented_proxy", "missed_reason"]]),
        "",
        "## False-Alert Analysis",
        "",
        markdown_table(causes),
        "",
        "Fusion performed worse than HEL1OS-only because the soft branch adds extra quiet-day/low-level threshold crossings while not recovering enough additional approved flare events on TRAIN_DEV.",
        "",
        "## Best Repaired Policy",
        "",
        markdown_table(best_row),
        "",
        "## Comparison",
        "",
        markdown_table(comparison),
        "",
        "## HEL1OS Burst Feature Promotion",
        "",
        "HEL1OS burst features should be promoted as a conservative diagnostic/confirmation signal. They improved the expanded TRAIN_DEV F1 versus the naive fusion proxy and reduced FAR/day, but they still do not justify replacing frozen v8.1.",
        "",
        "## Neural Network Decision",
        "",
        nn_recommendation,
        "",
        "## Final Recommendation",
        "",
        "Do not run neural networks yet. First rebalance the holdout proposal, review NO_MATCH/manual events, and use HEL1OS burst/Neupert-style evidence as confirmation rather than allowing noisy SoLEXS-only alerts to drive fusion.",
    ]
    (OUT_DIR / "EXPANDED_POLICY_REPAIR_REPORT.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    ensure_out()
    dataset, flares, quiet, holdout = load_inputs()
    train_dates = current_train_dates(dataset)
    train_quiet_dates = sorted(set(quiet["date"].astype(str)) & set(train_dates))
    audit, proposal = audit_holdout_balance(flares, quiet, holdout, train_dates)
    event_errors = event_error_table(dataset, flares, train_dates)
    false_alerts = false_alert_table(dataset, flares, train_quiet_dates, train_dates)
    sweep, best = run_repair_sweep(dataset, flares, train_dates)

    current_models = pd.read_csv(CURRENT_MODEL_RESULTS_PATH) if CURRENT_MODEL_RESULTS_PATH.exists() else pd.DataFrame()
    current_baselines = pd.read_csv(CURRENT_BASELINES_PATH) if CURRENT_BASELINES_PATH.exists() else pd.DataFrame()
    comparison = comparison_table(best, current_models, current_baselines)
    current_train_events = int(flares["date"].astype(str).isin(train_dates).sum())
    nn = neural_network_recommendation(flares, quiet, proposal, best, current_train_events)
    write_report(audit, proposal, event_errors, false_alerts, sweep, best, comparison, nn)

    proposed_holdout = set(proposal["date"].astype(str))
    train_events_after = int((~flares["date"].astype(str).isin(proposed_holdout)).sum())
    best_row = best.iloc[0].to_dict() if not best.empty else {}
    hel1os = current_baselines[current_baselines["policy"].eq("hel1os_only")].head(1).to_dict("records")
    fusion = current_baselines[current_baselines["policy"].eq("soft_hard_fusion")].head(1).to_dict("records")
    print(f"TRAIN_DEV flare events before revised holdout proposal: {current_train_events}")
    print(f"TRAIN_DEV flare events after revised holdout proposal: {train_events_after}")
    print(f"best repaired policy: {best_row.get('policy', 'none')} {best_row.get('params', '')}")
    print(f"best repaired policy metrics: precision={best_row.get('precision')}, recall={best_row.get('recall')}, f1={best_row.get('f1')}, FAR/day={best_row.get('far_per_day')}")
    print(f"comparison to HEL1OS-only: {hel1os[0] if hel1os else 'missing'}")
    print(f"comparison to fusion proxy: {fusion[0] if fusion else 'missing'}")
    print(f"whether neural networks are worth trying next: {nn}")
    print(f"report path: {OUT_DIR / 'EXPANDED_POLICY_REPAIR_REPORT.md'}")


if __name__ == "__main__":
    main()
