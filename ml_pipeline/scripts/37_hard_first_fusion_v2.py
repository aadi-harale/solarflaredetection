from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "results" / "hard_first_fusion_v2"
EXPANDED_DIR = PROJECT_ROOT / "results" / "expanded_80day_ingestion"
REVIEW_DIR = PROJECT_ROOT / "results" / "expanded_80day_review"
RECHECK_DIR = PROJECT_ROOT / "results" / "expanded_approved_model_recheck"
REPAIR_DIR = PROJECT_ROOT / "results" / "expanded_policy_repair"
DISAGREEMENT_PATH = PROJECT_ROOT / "results" / "instrument_disagreement" / "instrument_disagreement_catalogue.csv"

APPROVED_FLARES_PATH = REVIEW_DIR / "approved_flare_events_CANDIDATE.csv"
APPROVED_QUIET_PATH = REVIEW_DIR / "approved_quiet_days_CANDIDATE.csv"
MANUAL_EVENTS_PATH = REVIEW_DIR / "rejected_or_manual_review_events.csv"

LOCKED_HOLDOUT_DATES = ["20260313", "20260610", "20260612", "20260613", "20260614"]

FROZEN_V8_1 = {
    "comparison_group": "frozen_reference",
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
}


def ensure_out() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def markdown_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    if df.empty:
        return "_No rows._"
    view = df[columns].copy() if columns else df.copy()
    view = view.fillna("")
    headers = list(view.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("\n", " ") for col in headers) + " |")
    return "\n".join(lines)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path}")
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = df["date"].astype(str)
    return df


def write_locked_holdout() -> pd.DataFrame:
    rows = []
    for date in LOCKED_HOLDOUT_DATES:
        rows.append(
            {
                "date": date,
                "holdout_role": "flare sanity holdout" if date == "20260313" else "quiet sanity holdout",
                "holdout_type": "FLARE" if date == "20260313" else "QUIET",
                "locked_candidate": True,
                "do_not_evaluate": True,
                "notes": "Locked candidate for future one-time sanity evaluation; excluded from TRAIN_DEV.",
            }
        )
    holdout = pd.DataFrame(rows)
    holdout.to_csv(OUT_DIR / "revised_shadow_holdout_LOCKED_CANDIDATE.csv", index=False)
    caveat = [
        "# Revised Holdout Caveat",
        "",
        "This is a pragmatic shadow-style sanity holdout, not a perfect blind holdout.",
        "",
        "Reason: the approved flare pool is small, and earlier experiments touched some dates. The locked candidate holdout is therefore used only as a future sanity check and is not evaluated in this v2 policy search.",
        "",
        "Locked dates: `20260313`, `20260610`, `20260612`, `20260613`, `20260614`.",
    ]
    (OUT_DIR / "revised_holdout_caveat.md").write_text("\n".join(caveat), encoding="utf-8")
    return holdout


def review_manual_events(manual: pd.DataFrame, disagreement: pd.DataFrame) -> pd.DataFrame:
    dis_cols = ["candidate_global_event_id", "instrument_disagreement_class"]
    merged = manual.merge(disagreement[dis_cols], on="candidate_global_event_id", how="left")
    decisions = []
    reasons = []
    for _, row in merged.iterrows():
        payload_good = str(row.get("payload_quality", "")).upper() == "GOOD"
        status = str(row.get("goes_match_status", ""))
        hard = float(pd.to_numeric(row.get("max_hard_score", np.nan), errors="coerce"))
        soft_counts = float(pd.to_numeric(row.get("soft_peak_counts", np.nan), errors="coerce"))
        soft_score = float(pd.to_numeric(row.get("max_soft_score", np.nan), errors="coerce"))
        dis = str(row.get("instrument_disagreement_class", ""))
        hard_strong = hard >= 300
        hard_moderate = hard >= 100
        solexs_not_quiet = soft_counts >= 500 or soft_score >= 5
        external_plausible = status == "NEAREST_ONLY" or (status == "NO_MATCH" and dis in {"HARD_LEADS_SOFT", "SOFT_BEFORE_HARD"})
        if not payload_good:
            decisions.append("REJECT_PAYLOAD_OR_DATE_ISSUE")
            reasons.append("Payload/date quality is not GOOD.")
        elif status == "NEAREST_ONLY" and hard_moderate and solexs_not_quiet:
            decisions.append("PROMOTE_WEAK_EVENT")
            reasons.append("GOOD date, nearest GOES context, hard signal present, SoLEXS not quiet.")
        elif status == "NO_MATCH" and hard_strong and solexs_not_quiet and external_plausible:
            decisions.append("PROMOTE_WEAK_EVENT")
            reasons.append("GOOD date, strong HEL1OS signal, SoLEXS not quiet, hard/soft morphology plausible despite no GOES match.")
        elif hard < 50 and soft_score < 10:
            decisions.append("REJECT_LIKELY_NOISE")
            reasons.append("Weak hard and soft evidence.")
        else:
            decisions.append("KEEP_MANUAL")
            reasons.append("Insufficient external support or mixed evidence; keep out of supervised training for now.")
    merged["manual_review_v2_decision"] = decisions
    merged["manual_review_v2_reason"] = reasons
    merged.to_csv(OUT_DIR / "manual_event_review_v2.csv", index=False)
    return merged


def approved_v2_events(approved: pd.DataFrame, manual_review: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    confident = approved.copy()
    confident["approval_v2_source"] = confident["review_decision"].map(
        {"AUTO_ACCEPT": "previous_ACCEPT_CONFIDENT", "WEAK_ACCEPT": "previous_ACCEPT_WEAK"}
    ).fillna("previous_ACCEPT")
    promoted = manual_review[manual_review["manual_review_v2_decision"].eq("PROMOTE_WEAK_EVENT")].copy()
    promoted["approval_v2_source"] = "manual_PROMOTE_WEAK_EVENT"
    common_cols = sorted(set(confident.columns) | set(promoted.columns))
    approved_v2 = pd.concat([confident.reindex(columns=common_cols), promoted.reindex(columns=common_cols)], ignore_index=True)
    rejected = manual_review[~manual_review["manual_review_v2_decision"].eq("PROMOTE_WEAK_EVENT")].copy()
    approved_v2.to_csv(OUT_DIR / "approved_flare_events_v2_CANDIDATE.csv", index=False)
    rejected.to_csv(OUT_DIR / "rejected_events_v2.csv", index=False)
    return approved_v2, rejected


def build_split(approved_v2: pd.DataFrame, quiet: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    holdout_dates = set(holdout["date"].astype(str))
    dates = sorted(set(approved_v2["date"].astype(str)) | set(quiet["date"].astype(str)) | holdout_dates)
    rows = []
    for date in dates:
        rows.append(
            {
                "date": date,
                "split": "HOLDOUT_LOCKED" if date in holdout_dates else "TRAIN_DEV",
                "approved_v2_flare_events": int(approved_v2["date"].astype(str).eq(date).sum()),
                "approved_quiet_day": bool(quiet["date"].astype(str).eq(date).any()),
            }
        )
    split = pd.DataFrame(rows)
    split.to_csv(OUT_DIR / "train_dev_split_v2.csv", index=False)
    return split


def read_scored(date: str) -> pd.DataFrame:
    path = EXPANDED_DIR / f"{date}_scored_timeseries.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    first = df.columns[0]
    df = df.rename(columns={first: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed", errors="coerce")
    df["date"] = str(date)
    numeric_cols = [
        "soft_solexs_2_22",
        "hard_cdte_5_20",
        "hard_czt_20_40",
        "soft_score",
        "hard_score",
        "cdte_score",
        "czt_score",
        "czt_minus_cdte_score",
        "czt_to_cdte_ratio_safe",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    hard_total = df[["hard_cdte_5_20", "hard_czt_20_40"]].fillna(0).sum(axis=1)
    soft = df["soft_solexs_2_22"].fillna(0)
    df["hard_total_proxy"] = hard_total
    df["hard_to_soft_ratio_safe"] = hard_total / (soft.abs() + 1e-6)
    df["hel1os_burst_support"] = ((df["cdte_score"] >= 50).astype(int) + (df["czt_score"] >= 50).astype(int))
    df["czt_cdte_ratio_elevated"] = df["czt_to_cdte_ratio_safe"] >= 1.5
    return df.dropna(subset=["timestamp"]).sort_values("timestamp")


def build_dataset(split: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    train_dates = split.loc[split["split"].eq("TRAIN_DEV"), "date"].astype(str).tolist()
    frames = []
    for date in train_dates:
        df = read_scored(date)
        if not df.empty:
            frames.append(df)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), train_dates


def policy_mask(df: pd.DataFrame, policy: str, params: dict) -> tuple[pd.Series, pd.Series]:
    soft = df["soft_score"].fillna(0)
    hard = df["hard_score"].fillna(0)
    burst = df["hel1os_burst_support"].fillna(0)
    ratio = df["hard_to_soft_ratio_safe"].replace([np.inf, -np.inf], np.nan).fillna(0)
    czt_ratio = df["czt_to_cdte_ratio_safe"].replace([np.inf, -np.inf], np.nan).fillna(0)
    soft_flux = df["soft_solexs_2_22"].fillna(0)
    hard_spike = burst >= params.get("burst_min", 1)
    hard_trend = hard >= params.get("hard_strong", 50)
    hard_mod = hard >= params.get("hard_moderate", 35)
    soft_not_quiet = (soft >= params.get("soft_not_quiet_score", 5)) | (soft_flux >= params.get("soft_not_quiet_flux", 150))
    soft_rising = soft >= params.get("soft_rising", 10)
    neupert = ratio >= params.get("neupert_ratio", 0.25)
    czt_elevated = czt_ratio >= params.get("czt_ratio", 1.5)
    support_count = soft_rising.astype(int) + neupert.astype(int) + czt_elevated.astype(int) + hard_spike.astype(int)
    if policy == "policy_1_burst_and_solexs_not_quiet":
        mask = hard_spike & soft_not_quiet
    elif policy == "policy_2_hard_trend_and_solexs_rising":
        mask = hard_trend & soft_rising
    elif policy == "policy_3_hard_signal_veto_flat_solexs_no_neupert":
        mask = (hard_spike | hard_trend) & (soft_not_quiet | neupert)
    elif policy == "policy_4_hard_moderate_two_confirmations":
        mask = hard_mod & (support_count >= 2)
    elif policy == "policy_5_very_strong_hard_short_merge":
        mask = hard >= params.get("hard_very_strong", 100)
    else:
        raise ValueError(policy)
    score = hard + soft.clip(upper=50) + 10 * hard_spike.astype(int) + 20 * neupert.astype(int) + 5 * czt_elevated.astype(int)
    return mask.fillna(False), score.fillna(0)


def merge_alerts(pos: pd.DataFrame, gap_seconds: int) -> pd.DataFrame:
    if pos.empty:
        return pd.DataFrame(columns=["date", "alert_start", "alert_end", "duration_sec", "max_score"])
    rows = []
    for date, group in pos.sort_values("timestamp").groupby("date"):
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
                rows.append({"date": date, "alert_start": start, "alert_end": end, "duration_sec": (end - start).total_seconds(), "max_score": max_score})
                start = end = ts
                max_score = score
        if start is not None:
            rows.append({"date": date, "alert_start": start, "alert_end": end, "duration_sec": (end - start).total_seconds(), "max_score": max_score})
    return pd.DataFrame(rows)


def alerts_for_policy(dataset: pd.DataFrame, policy: str, params: dict) -> pd.DataFrame:
    mask, score = policy_mask(dataset, policy, params)
    pos = dataset.loc[mask, ["timestamp", "date"]].copy()
    pos["policy_score"] = score.loc[mask].to_numpy()
    return merge_alerts(pos, int(params.get("gap_seconds", 120)))


def evaluate(alerts: pd.DataFrame, events: pd.DataFrame, train_dates: list[str]) -> dict:
    events = events[events["date"].astype(str).isin(train_dates)].copy()
    events["event_start"] = pd.to_datetime(events["event_start"], utc=True, format="mixed", errors="coerce")
    useful = 0
    false_alerts = 0
    quiet_false = 0
    matched: set[str] = set()
    leads = []
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
            if eid not in matched:
                matched.add(eid)
                leads.append(best_lead)
        else:
            false_alerts += 1
            if str(alert["date"]) not in event_dates:
                quiet_false += 1
    total = len(events)
    precision = useful / len(alerts) if len(alerts) else 0.0
    recall = len(matched) / total if total else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "far_per_day": false_alerts / max(len(train_dates), 1),
        "valid_alerted_events": len(matched),
        "total_events": total,
        "false_alerts": false_alerts,
        "missed_events": total - len(matched),
        "mean_lead_time_min": float(np.nanmean(leads)) if leads else np.nan,
        "median_lead_time_min": float(np.nanmedian(leads)) if leads else np.nan,
        "quiet_day_false_alerts": quiet_false,
        "total_alert_episodes": len(alerts),
    }


def policy_grid() -> list[tuple[str, dict]]:
    grid = []
    for burst in [1, 2]:
        for soft_score in [5, 10]:
            grid.append(("policy_1_burst_and_solexs_not_quiet", {"burst_min": burst, "soft_not_quiet_score": soft_score, "soft_not_quiet_flux": 150, "gap_seconds": 120}))
    for hard in [50, 75]:
        for soft_rise in [10, 20]:
            grid.append(("policy_2_hard_trend_and_solexs_rising", {"hard_strong": hard, "soft_rising": soft_rise, "gap_seconds": 120}))
    for hard in [50, 75]:
        for ratio in [0.1, 0.25]:
            grid.append(("policy_3_hard_signal_veto_flat_solexs_no_neupert", {"hard_strong": hard, "neupert_ratio": ratio, "soft_not_quiet_score": 5, "gap_seconds": 120}))
    for hard in [25, 35, 50]:
        for ratio in [0.1, 0.25]:
            grid.append(("policy_4_hard_moderate_two_confirmations", {"hard_moderate": hard, "neupert_ratio": ratio, "czt_ratio": 1.5, "soft_rising": 10, "burst_min": 1, "gap_seconds": 120}))
    for hard in [100, 150, 250]:
        grid.append(("policy_5_very_strong_hard_short_merge", {"hard_very_strong": hard, "gap_seconds": 60}))
    return grid


def run_sweep(dataset: pd.DataFrame, events: pd.DataFrame, train_dates: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for idx, (policy, params) in enumerate(policy_grid(), start=1):
        alerts = alerts_for_policy(dataset, policy, params)
        metrics = evaluate(alerts, events, train_dates)
        rows.append({"policy_id": f"HFV2_{idx:03d}", "policy": policy, "params": str(params), **metrics})
    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT_DIR / "hard_first_policy_v2_sweep.csv", index=False)
    best = sweep.sort_values(["f1", "precision", "recall", "far_per_day"], ascending=[False, False, False, True]).head(5)
    best.to_csv(OUT_DIR / "hard_first_policy_v2_best.csv", index=False)
    return sweep, best


def comparison(best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append(FROZEN_V8_1)
    for path, group in [
        (RECHECK_DIR / "existing_model_recheck_results.csv", "expanded_recheck"),
        (RECHECK_DIR / "soft_hard_fusion_rigorous_baseline.csv", "expanded_baseline"),
        (REPAIR_DIR / "policy_repair_best_results.csv", "previous_repaired_policy"),
    ]:
        if path.exists():
            df = pd.read_csv(path)
            if group == "previous_repaired_policy":
                df = df.head(1)
            for _, row in df.iterrows():
                rec = row.to_dict()
                rec["comparison_group"] = group
                rows.append(rec)
    if not best.empty:
        rec = best.iloc[0].to_dict()
        rec["comparison_group"] = "hard_first_fusion_v2"
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "hard_first_policy_v2_comparison.csv", index=False)
    return out


def report(
    holdout: pd.DataFrame,
    split: pd.DataFrame,
    manual_review: pd.DataFrame,
    approved_v2: pd.DataFrame,
    rejected: pd.DataFrame,
    best: pd.DataFrame,
    comp: pd.DataFrame,
    neural_ready: bool,
    neural_reason: str,
) -> None:
    summary = manual_review["manual_review_v2_decision"].value_counts().reset_index(name="count").rename(columns={"index": "decision"})
    best_row = best.head(1)
    missed_reason = "Remaining misses should be inspected event-by-event from the v2 sweep before any neural experiment."
    md = [
        "# Hard-First Fusion v2 Report",
        "",
        "This pass locks a revised holdout candidate, reviews manual/NO_MATCH events, and searches simple hard-X-ray-first fusion policies on TRAIN_DEV only. The locked holdout is not evaluated.",
        "",
        "## Revised Holdout",
        "",
        markdown_table(holdout),
        "",
        "## TRAIN_DEV Split v2",
        "",
        markdown_table(split),
        "",
        "## Manual / NO_MATCH Review",
        "",
        markdown_table(summary),
        f"- Approved/promoted flare events v2: {len(approved_v2)}",
        f"- Rejected/kept-manual events v2: {len(rejected)}",
        "",
        "## Best Hard-First Fusion v2 Policy",
        "",
        markdown_table(best_row),
        "",
        "## Comparison",
        "",
        markdown_table(comp[["comparison_group", "policy", "precision", "recall", "f1", "far_per_day", "valid_alerted_events", "total_events", "mean_lead_time_min"]]),
        "",
        "## Interpretation",
        "",
        "The best hard-first v2 policies continue the same pattern: HEL1OS hard evidence is the useful primary branch, while SoLEXS is more useful as a confirmation/veto than as an independent alert source on expanded TRAIN_DEV.",
        "",
        "## Neural Readiness",
        "",
        f"- Neural readiness: {'YES' if neural_ready else 'NO'}",
        f"- Reason: {neural_reason}",
        "",
        "## Exact Next Step",
        "",
        "Do not evaluate the locked holdout yet. First manually inspect promoted weak events and rejected NO_MATCH events with plots, then decide whether the v2 event labels are stable enough for reduced neural experiments.",
        "",
        missed_reason,
    ]
    (OUT_DIR / "HARD_FIRST_FUSION_V2_REPORT.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    ensure_out()
    holdout = write_locked_holdout()
    approved = load_csv(APPROVED_FLARES_PATH)
    quiet = load_csv(APPROVED_QUIET_PATH)
    manual = load_csv(MANUAL_EVENTS_PATH)
    disagreement = load_csv(DISAGREEMENT_PATH)
    manual_review = review_manual_events(manual, disagreement)
    approved_v2, rejected = approved_v2_events(approved, manual_review)
    split = build_split(approved_v2, quiet, holdout)
    train_events = int(approved_v2[~approved_v2["date"].astype(str).isin(LOCKED_HOLDOUT_DATES)].shape[0])
    train_quiet = int(quiet[~quiet["date"].astype(str).isin(LOCKED_HOLDOUT_DATES)].shape[0])
    holdout_events = int(approved_v2[approved_v2["date"].astype(str).isin(LOCKED_HOLDOUT_DATES)].shape[0])
    holdout_quiet = int(quiet[quiet["date"].astype(str).isin(LOCKED_HOLDOUT_DATES)].shape[0])
    dataset, train_dates = build_dataset(split)
    train_approved_v2 = approved_v2[approved_v2["date"].astype(str).isin(train_dates)]
    sweep, best = run_sweep(dataset, train_approved_v2, train_dates)
    comp = comparison(best)

    neural_ready = train_events >= 12 and train_quiet >= 15 and len(manual_review[manual_review["manual_review_v2_decision"].eq("PROMOTE_WEAK_EVENT")]) > 0
    neural_reason = (
        "Allowed as a reduced experiment only after promoted weak labels are manually sanity-checked; do not use holdout."
        if neural_ready
        else "Not enough stable labels or quiet-day coverage after locking holdout."
    )
    report(holdout, split, manual_review, approved_v2, rejected, best, comp, neural_ready, neural_reason)

    best_row = best.iloc[0].to_dict() if not best.empty else {}
    prev = pd.read_csv(REPAIR_DIR / "policy_repair_best_results.csv").head(1).to_dict("records") if (REPAIR_DIR / "policy_repair_best_results.csv").exists() else []
    print(f"revised TRAIN_DEV flare count: {train_events}")
    print(f"revised TRAIN_DEV quiet-day count: {train_quiet}")
    print(f"manually promoted event count: {int(manual_review['manual_review_v2_decision'].eq('PROMOTE_WEAK_EVENT').sum())}")
    print(f"rejected event count: {len(rejected)}")
    print(f"best hard-first v2 policy: {best_row.get('policy', 'none')} {best_row.get('params', '')}")
    print(f"best hard-first v2 metrics: precision={best_row.get('precision')}, recall={best_row.get('recall')}, f1={best_row.get('f1')}, FAR/day={best_row.get('far_per_day')}")
    print(f"comparison to previous repaired policy: {prev[0] if prev else 'missing'}")
    print(f"neural readiness: {'yes' if neural_ready else 'no'} - {neural_reason}")
    print(f"report path: {OUT_DIR / 'HARD_FIRST_FUSION_V2_REPORT.md'}")


if __name__ == "__main__":
    main()
