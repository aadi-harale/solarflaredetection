from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "results" / "manual_event_review_plots"
PLOT_DIR = OUT_DIR / "event_plots"
EXPANDED_DIR = PROJECT_ROOT / "results" / "expanded_80day_ingestion"
V2_DIR = PROJECT_ROOT / "results" / "hard_first_fusion_v2"
RECHECK_DIR = PROJECT_ROOT / "results" / "expanded_approved_model_recheck"
DISAGREEMENT_PATH = PROJECT_ROOT / "results" / "instrument_disagreement" / "instrument_disagreement_catalogue.csv"

MANUAL_PATH = V2_DIR / "manual_event_review_v2.csv"
APPROVED_V2_PATH = V2_DIR / "approved_flare_events_v2_CANDIDATE.csv"
REJECTED_V2_PATH = V2_DIR / "rejected_events_v2.csv"
SPLIT_PATH = V2_DIR / "train_dev_split_v2.csv"
DATASET_PATH = RECHECK_DIR / "approved_expanded_forecasting_dataset.csv"
HOLDOUT_DATES = {"20260313", "20260610", "20260612", "20260613", "20260614"}


def ensure_out() -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = [p for p in [MANUAL_PATH, APPROVED_V2_PATH, REJECTED_V2_PATH, SPLIT_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing manual review inputs: " + ", ".join(str(p) for p in missing))
    manual = pd.read_csv(MANUAL_PATH)
    approved = pd.read_csv(APPROVED_V2_PATH)
    rejected = pd.read_csv(REJECTED_V2_PATH)
    split = pd.read_csv(SPLIT_PATH)
    for df in [manual, approved, rejected, split]:
        if "date" in df.columns:
            df["date"] = df["date"].astype(str)
    return manual, approved, rejected, split


def read_scored(date: str) -> pd.DataFrame:
    path = EXPANDED_DIR / f"{date}_scored_timeseries.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed", errors="coerce")
    df["date"] = str(date)
    for col in [
        "soft_solexs_2_22",
        "hard_cdte_5_20",
        "hard_czt_20_40",
        "soft_score",
        "hard_score",
        "cdte_score",
        "czt_score",
        "czt_to_cdte_ratio_safe",
    ]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df["hel1os_burst_support"] = ((df["cdte_score"].fillna(0) >= 50).astype(int) + (df["czt_score"].fillna(0) >= 50).astype(int))
    hard_total = df[["hard_cdte_5_20", "hard_czt_20_40"]].fillna(0).sum(axis=1)
    soft = df["soft_solexs_2_22"].fillna(0)
    df["hard_to_soft_ratio_safe"] = hard_total / (soft.abs() + 1e-6)
    return df.dropna(subset=["timestamp"]).sort_values("timestamp")


def hard_first_v2_alerts(date_df: pd.DataFrame) -> pd.DataFrame:
    if date_df.empty:
        return pd.DataFrame(columns=["alert_start", "alert_end"])
    mask = (date_df["hel1os_burst_support"].fillna(0) >= 1) & (
        (date_df["soft_score"].fillna(0) >= 5) | (date_df["soft_solexs_2_22"].fillna(0) >= 150)
    )
    pos = date_df.loc[mask, ["timestamp"]].copy()
    rows = []
    start = end = None
    for ts in pos["timestamp"]:
        if start is None:
            start = end = ts
        elif (ts - end).total_seconds() <= 120:
            end = ts
        else:
            rows.append({"alert_start": start, "alert_end": end})
            start = end = ts
    if start is not None:
        rows.append({"alert_start": start, "alert_end": end})
    return pd.DataFrame(rows)


def evidence_labels(window: pd.DataFrame, event: pd.Series) -> dict:
    hard = float(pd.to_numeric(event.get("max_hard_score", np.nan), errors="coerce"))
    soft_counts = float(pd.to_numeric(event.get("soft_peak_counts", np.nan), errors="coerce"))
    soft_score = float(pd.to_numeric(event.get("max_soft_score", np.nan), errors="coerce"))
    burst_max = float(window["hel1os_burst_support"].max()) if not window.empty else 0.0
    max_cdte = float(window["cdte_score"].max()) if not window.empty else np.nan
    max_czt = float(window["czt_score"].max()) if not window.empty else np.nan
    max_ratio = float(window["hard_to_soft_ratio_safe"].replace([np.inf, -np.inf], np.nan).max()) if not window.empty else np.nan
    solexs = "STRONG" if soft_counts >= 500 or soft_score >= 20 else ("NOT_QUIET" if soft_counts >= 150 or soft_score >= 5 else "QUIET")
    hel1os = "STRONG" if hard >= 300 or max(max_cdte, max_czt) >= 100 else ("MODERATE" if hard >= 50 or max(max_cdte, max_czt) >= 50 else "WEAK")
    burst = "YES" if burst_max >= 1 else "NO"
    status = str(event.get("goes_match_status", ""))
    if status == "NEAREST_ONLY":
        timing = "WEAK"
    elif status in {"EXACT_PEAK_MATCH", "WINDOW_OVERLAP_MATCH"}:
        timing = "GOOD"
    else:
        timing = "NONE"
    return {
        "SoLEXS_evidence": solexs,
        "HEL1OS_evidence": hel1os,
        "burst_evidence": burst,
        "timing_support": timing,
        "max_cdte_score_window": max_cdte,
        "max_czt_score_window": max_czt,
        "max_hard_to_soft_ratio_window": max_ratio,
    }


def suggested_label(event: pd.Series, ev: dict, questionable_dates: set[str]) -> tuple[str, str]:
    payload_good = str(event.get("payload_quality", "")).upper() == "GOOD"
    date = str(event.get("date", ""))
    if not payload_good:
        return "REJECT_PAYLOAD_ISSUE", "Payload quality is not GOOD."
    if date in questionable_dates:
        return "REJECT_QUESTIONABLE_DATE", "Date is marked questionable in split/review context."
    if (
        ev["HEL1OS_evidence"] == "STRONG"
        and ev["burst_evidence"] == "YES"
        and ev["SoLEXS_evidence"] in {"NOT_QUIET", "STRONG"}
        and ev["timing_support"] in {"GOOD", "WEAK"}
    ):
        return "PROMOTE_WEAK_EVENT", "Meets conservative promotion rule; still requires human visual sanity check before training use."
    if ev["HEL1OS_evidence"] == "WEAK" and ev["SoLEXS_evidence"] == "QUIET":
        return "REJECT_LIKELY_NOISE", "Weak hard evidence and quiet SoLEXS."
    return "KEEP_MANUAL", "Mixed evidence or missing external timing support."


def make_plot(date_df: pd.DataFrame, event: pd.Series, alerts: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    start = pd.to_datetime(event["event_start"], utc=True, format="mixed", errors="coerce")
    end = pd.to_datetime(event["event_end"], utc=True, format="mixed", errors="coerce")
    x0 = start - pd.Timedelta(minutes=30)
    x1 = end + pd.Timedelta(minutes=30)
    view = date_df[date_df["timestamp"].between(x0, x1)].copy()
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(view["timestamp"], view["soft_solexs_2_22"], lw=0.9, color="#1f77b4")
    axes[0].set_ylabel("SoLEXS")
    axes[1].plot(view["timestamp"], view["hard_cdte_5_20"], lw=0.8, color="#d62728", label="CdTe 5-20")
    axes[1].plot(view["timestamp"], view["hard_czt_20_40"], lw=0.8, color="#9467bd", label="CZT 20-40")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_ylabel("HEL1OS")
    axes[2].plot(view["timestamp"], view["soft_score"], lw=0.8, color="#1f77b4", label="soft_score")
    axes[2].plot(view["timestamp"], view["hard_score"], lw=0.8, color="#d62728", label="hard_score")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].set_ylabel("Scores")
    axes[3].plot(view["timestamp"], view["hel1os_burst_support"], lw=0.8, color="#2ca02c", label="burst support")
    axes[3].plot(view["timestamp"], view["hard_to_soft_ratio_safe"].clip(upper=10), lw=0.8, color="#ff7f0e", label="hard/soft ratio clipped")
    axes[3].legend(loc="upper right", fontsize=8)
    axes[3].set_ylabel("Burst/Ratio")
    for ax in axes:
        ax.axvspan(start, end, color="#fdae61", alpha=0.25, label="event window")
        if pd.notna(pd.to_datetime(event.get("soft_peak_time", pd.NaT), utc=True, errors="coerce")):
            ax.axvline(pd.to_datetime(event["soft_peak_time"], utc=True, format="mixed"), color="#1f77b4", ls="--", lw=1)
        if pd.notna(pd.to_datetime(event.get("hard_trigger_time", pd.NaT), utc=True, errors="coerce")):
            ax.axvline(pd.to_datetime(event["hard_trigger_time"], utc=True, format="mixed"), color="#d62728", ls=":", lw=1)
        for _, alert in alerts.iterrows():
            if alert["alert_end"] >= x0 and alert["alert_start"] <= x1:
                ax.axvspan(alert["alert_start"], alert["alert_end"], color="#2ca02c", alpha=0.18)
        ax.grid(alpha=0.2)
    title = f"Manual event {event.get('candidate_global_event_id')} {event.get('date')} GOES={event.get('goes_match_status')}"
    axes[0].set_title(title)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_review() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manual, approved_v2, rejected_v2, split = load_inputs()
    manual = manual[
        ~manual["date"].astype(str).isin(HOLDOUT_DATES)
        & (
            manual["manual_review_v2_decision"].astype(str).eq("KEEP_MANUAL")
            | manual["goes_match_status"].astype(str).isin(["NO_MATCH", "NEAREST_ONLY"])
            | manual["instrument_disagreement_class"].astype(str).isin(["HARD_ONLY", "HARD_LEADS_SOFT", "NO_MATCH"])
        )
    ].copy()
    date_cache: dict[str, pd.DataFrame] = {}
    rows = []
    for _, event in manual.iterrows():
        date = str(event["date"])
        if date not in date_cache:
            date_cache[date] = read_scored(date)
        date_df = date_cache[date]
        start = pd.to_datetime(event["event_start"], utc=True, format="mixed", errors="coerce")
        window = date_df[date_df["timestamp"].between(start - pd.Timedelta(minutes=90), start)]
        alerts = hard_first_v2_alerts(date_df)
        ev = evidence_labels(window, event)
        label, reason = suggested_label(event, ev, questionable_dates=set())
        plot_path = PLOT_DIR / f"event_{event.get('candidate_global_event_id')}_{date}.png"
        make_plot(date_df, event, alerts, plot_path)
        rows.append(
            {
                "event_id": event.get("candidate_global_event_id"),
                "date": date,
                "start_time": event.get("event_start"),
                "end_time": event.get("event_end"),
                "GOES_status": event.get("goes_match_status"),
                "disagreement_type": event.get("instrument_disagreement_class"),
                "SoLEXS_evidence": ev["SoLEXS_evidence"],
                "HEL1OS_evidence": ev["HEL1OS_evidence"],
                "burst_evidence": ev["burst_evidence"],
                "timing_support": ev["timing_support"],
                "payload_quality": event.get("payload_quality"),
                "plot_path": str(plot_path),
                "final_suggested_label": label,
                "label_reason": reason,
                "max_cdte_score_window": ev["max_cdte_score_window"],
                "max_czt_score_window": ev["max_czt_score_window"],
                "max_hard_to_soft_ratio_window": ev["max_hard_to_soft_ratio_window"],
            }
        )
    review = pd.DataFrame(rows)
    review.to_csv(OUT_DIR / "manual_event_visual_review_table.csv", index=False)

    train_dates = set(split.loc[split["split"].eq("TRAIN_DEV"), "date"].astype(str))
    promoted_ids = set(review.loc[review["final_suggested_label"].eq("PROMOTE_WEAK_EVENT"), "event_id"].astype(str))
    manual_ids = set(review["event_id"].astype(str))
    manual_full = manual.copy()
    manual_full["candidate_global_event_id"] = manual_full["candidate_global_event_id"].astype(str)
    promoted_events = manual_full[manual_full["candidate_global_event_id"].isin(promoted_ids)].copy()
    promoted_events["approval_v3_source"] = "visual_review_PROMOTE_WEAK_EVENT"
    approved_v3 = pd.concat([approved_v2, promoted_events], ignore_index=True, sort=False).drop_duplicates("candidate_global_event_id")
    rejected_labels = {"REJECT_LIKELY_NOISE", "REJECT_PAYLOAD_ISSUE", "REJECT_QUESTIONABLE_DATE"}
    rejected_ids = set(review.loc[review["final_suggested_label"].isin(rejected_labels), "event_id"].astype(str))
    rejected_v3 = pd.concat([rejected_v2, manual_full[manual_full["candidate_global_event_id"].isin(rejected_ids)]], ignore_index=True, sort=False).drop_duplicates("candidate_global_event_id")
    remaining = manual_full[manual_full["candidate_global_event_id"].isin(manual_ids - promoted_ids - rejected_ids)].copy()
    approved_v3.to_csv(OUT_DIR / "approved_flare_events_v3_CANDIDATE.csv", index=False)
    rejected_v3.to_csv(OUT_DIR / "rejected_events_v3.csv", index=False)
    remaining.to_csv(OUT_DIR / "remaining_manual_events_v3.csv", index=False)

    quiet_days = int(split[(split["split"].eq("TRAIN_DEV")) & (split["approved_quiet_day"].astype(str).str.upper().eq("TRUE"))].shape[0])
    stable_events = int(approved_v3[approved_v3["date"].astype(str).isin(train_dates)].shape[0])
    unresolved = len(remaining)
    neural_ready = stable_events >= 14 and quiet_days >= 20 and unresolved == 0
    stability = pd.DataFrame(
        [
            {"metric": "train_dev_stable_flare_events", "value": stable_events},
            {"metric": "train_dev_quiet_days", "value": quiet_days},
            {"metric": "manual_events_reviewed", "value": len(review)},
            {"metric": "promoted_events", "value": len(promoted_events)},
            {"metric": "rejected_events_added", "value": len(rejected_ids)},
            {"metric": "remaining_manual_events", "value": unresolved},
            {"metric": "holdout_evaluated", "value": "NO"},
            {"metric": "neural_readiness", "value": "YES" if neural_ready else "NO"},
        ]
    )
    stability.to_csv(OUT_DIR / "label_stability_summary.csv", index=False)
    return review, approved_v3, rejected_v3, remaining


def write_reports(review: pd.DataFrame, approved_v3: pd.DataFrame, rejected_v3: pd.DataFrame, remaining: pd.DataFrame) -> bool:
    split = pd.read_csv(SPLIT_PATH)
    train_dates = set(split.loc[split["split"].eq("TRAIN_DEV"), "date"].astype(str))
    quiet_days = int(split[(split["split"].eq("TRAIN_DEV")) & (split["approved_quiet_day"].astype(str).str.upper().eq("TRUE"))].shape[0])
    stable_events = int(approved_v3[approved_v3["date"].astype(str).isin(train_dates)].shape[0])
    neural_ready = stable_events >= 14 and quiet_days >= 20 and len(remaining) == 0
    decision = [
        "# Neural Readiness Decision",
        "",
        f"- TRAIN_DEV stable flare events: {stable_events}",
        f"- TRAIN_DEV quiet days: {quiet_days}",
        f"- Remaining unresolved manual events: {len(remaining)}",
        "- Holdout evaluated: NO",
        "",
        f"Neural readiness: {'YES' if neural_ready else 'NO'}",
        "",
    ]
    if neural_ready:
        decision.append("Recommendation: run only reduced neural experiments next: MLP, 1D CNN, optional LSTM later. Do not use holdout for tuning.")
    else:
        decision.append("Recommendation: no neural networks yet. Resolve remaining manual events or keep them excluded before training data-hungry models.")
    (OUT_DIR / "neural_readiness_decision.md").write_text("\n".join(decision), encoding="utf-8")

    counts = review["final_suggested_label"].value_counts().reset_index(name="count").rename(columns={"index": "final_suggested_label"})
    report = [
        "# Manual Event Review Report",
        "",
        "This pass generated plots and conservative label suggestions for unresolved manual/NO_MATCH events on TRAIN_DEV only. The locked holdout was not evaluated.",
        "",
        "## Review Counts",
        "",
        markdown_table(counts),
        "",
        f"- Manual events reviewed: {len(review)}",
        f"- Promoted weak events: {int((review['final_suggested_label'] == 'PROMOTE_WEAK_EVENT').sum()) if not review.empty else 0}",
        f"- Rejected events added: {int(review['final_suggested_label'].isin(['REJECT_LIKELY_NOISE','REJECT_PAYLOAD_ISSUE','REJECT_QUESTIONABLE_DATE']).sum()) if not review.empty else 0}",
        f"- Remaining manual events: {len(remaining)}",
        f"- Final TRAIN_DEV flare count: {stable_events}",
        f"- Final TRAIN_DEV quiet-day count: {quiet_days}",
        "",
        "## Hard-First v2 Status",
        "",
        "Hard-first v2 remains a valid TRAIN_DEV policy candidate. This review does not change frozen v8.1/v6 metrics and does not evaluate holdout dates.",
        "",
        "## Neural Readiness",
        "",
        f"Neural readiness: {'YES' if neural_ready else 'NO'}",
        "",
        "## Next Exact Step",
        "",
        "Inspect the generated PNG plots manually. Only after human confirmation should any PROMOTE_WEAK_EVENT suggestions be considered stable labels for training.",
    ]
    (OUT_DIR / "MANUAL_EVENT_REVIEW_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return neural_ready


def main() -> None:
    ensure_out()
    review, approved_v3, rejected_v3, remaining = run_review()
    neural_ready = write_reports(review, approved_v3, rejected_v3, remaining)
    split = pd.read_csv(SPLIT_PATH)
    quiet_days = int(split[(split["split"].eq("TRAIN_DEV")) & (split["approved_quiet_day"].astype(str).str.upper().eq("TRUE"))].shape[0])
    train_dates = set(split.loc[split["split"].eq("TRAIN_DEV"), "date"].astype(str))
    train_flares = int(approved_v3[approved_v3["date"].astype(str).isin(train_dates)].shape[0])
    promoted = int((review["final_suggested_label"] == "PROMOTE_WEAK_EVENT").sum()) if not review.empty else 0
    rejected = int(review["final_suggested_label"].isin(["REJECT_LIKELY_NOISE", "REJECT_PAYLOAD_ISSUE", "REJECT_QUESTIONABLE_DATE"]).sum()) if not review.empty else 0
    print(f"manual events reviewed: {len(review)}")
    print(f"promoted weak events: {promoted}")
    print(f"rejected events: {rejected}")
    print(f"remaining manual events: {len(remaining)}")
    print(f"final TRAIN_DEV flare count: {train_flares}")
    print(f"final TRAIN_DEV quiet-day count: {quiet_days}")
    print(f"neural readiness: {'yes' if neural_ready else 'no'}")
    print(f"report path: {OUT_DIR / 'MANUAL_EVENT_REVIEW_REPORT.md'}")


if __name__ == "__main__":
    main()
