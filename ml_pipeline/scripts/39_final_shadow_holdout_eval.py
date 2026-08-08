from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "results" / "final_shadow_holdout_eval"
PLOT_DIR = OUT_DIR / "holdout_plots"
EXPANDED_DIR = PROJECT_ROOT / "results" / "expanded_80day_ingestion"
V2_DIR = PROJECT_ROOT / "results" / "hard_first_fusion_v2"
MANUAL_DIR = PROJECT_ROOT / "results" / "manual_event_review_plots"

LOCKED_HOLDOUT_PATH = V2_DIR / "revised_shadow_holdout_LOCKED_CANDIDATE.csv"
APPROVED_V3_PATH = MANUAL_DIR / "approved_flare_events_v3_CANDIDATE.csv"
APPROVED_V2_PATH = V2_DIR / "approved_flare_events_v2_CANDIDATE.csv"
GOES_VALIDATION_PATH = EXPANDED_DIR / "expanded_goes_validation.csv"

TRAIN_DEV_HARD_FIRST_V2 = {
    "evaluation_group": "TRAIN_DEV hard-first v2",
    "precision": 0.219,
    "recall": 0.357,
    "f1": 0.271,
    "far_per_day": 0.926,
    "valid_alerted_events": "5/14",
    "false_alerts": np.nan,
    "missed_events": 9,
    "mean_lead_time_min": np.nan,
    "median_lead_time_min": np.nan,
    "quiet_day_false_alerts": np.nan,
    "notes": "TRAIN_DEV selected policy result from hard-first fusion v2.",
}

FROZEN_V8_1 = {
    "evaluation_group": "Frozen v8.1 reference only",
    "precision": 0.556,
    "recall": 0.933,
    "f1": 0.697,
    "far_per_day": 1.78,
    "valid_alerted_events": "14/15",
    "false_alerts": np.nan,
    "missed_events": 1,
    "mean_lead_time_min": 51.20,
    "median_lead_time_min": 52.43,
    "quiet_day_false_alerts": np.nan,
    "notes": "Historical frozen reference; not recomputed on this tiny holdout.",
}


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


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not LOCKED_HOLDOUT_PATH.exists():
        raise FileNotFoundError(f"Missing locked holdout file: {LOCKED_HOLDOUT_PATH}")
    holdout = pd.read_csv(LOCKED_HOLDOUT_PATH)
    holdout["date"] = holdout["date"].astype(str)
    events_path = APPROVED_V3_PATH if APPROVED_V3_PATH.exists() else APPROVED_V2_PATH
    if not events_path.exists():
        raise FileNotFoundError("Missing approved flare events v3/v2 candidate file.")
    events = pd.read_csv(events_path)
    events["date"] = events["date"].astype(str)
    goes = pd.read_csv(GOES_VALIDATION_PATH) if GOES_VALIDATION_PATH.exists() else pd.DataFrame()
    if not goes.empty and "date" in goes.columns:
        goes["date"] = goes["date"].astype(str)
    return holdout, events, goes


def read_scored(date: str) -> pd.DataFrame:
    path = EXPANDED_DIR / f"{date}_scored_timeseries.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df.rename(columns={df.columns[0]: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed", errors="coerce")
    df["date"] = str(date)
    for col in ["soft_solexs_2_22", "hard_cdte_5_20", "hard_czt_20_40", "soft_score", "hard_score", "cdte_score", "czt_score"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    df["hel1os_burst_support"] = ((df["cdte_score"] >= 50).astype(int) + (df["czt_score"] >= 50).astype(int))
    return df.dropna(subset=["timestamp"]).sort_values("timestamp")


def selected_policy_alerts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "alert_start", "alert_end", "duration_sec", "max_score"])
    # Selected TRAIN_DEV policy: policy_1_burst_and_solexs_not_quiet
    # params: burst_min=1, soft_not_quiet_score=5, soft_not_quiet_flux=150, gap_seconds=120
    mask = (df["hel1os_burst_support"].fillna(0) >= 1) & (
        (df["soft_score"].fillna(0) >= 5) | (df["soft_solexs_2_22"].fillna(0) >= 150)
    )
    pos = df.loc[mask, ["timestamp", "date", "soft_score", "hard_score", "hel1os_burst_support"]].copy()
    pos["policy_score"] = df.loc[mask, ["soft_score", "hard_score"]].max(axis=1).to_numpy() + 10 * df.loc[mask, "hel1os_burst_support"].to_numpy()
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
            elif (ts - end).total_seconds() <= 120:
                end = ts
                max_score = max(max_score, score)
            else:
                rows.append({"date": date, "alert_start": start, "alert_end": end, "duration_sec": (end - start).total_seconds(), "max_score": max_score})
                start = end = ts
                max_score = score
        if start is not None:
            rows.append({"date": date, "alert_start": start, "alert_end": end, "duration_sec": (end - start).total_seconds(), "max_score": max_score})
    return pd.DataFrame(rows)


def evaluate(holdout: pd.DataFrame, events: pd.DataFrame) -> tuple[dict, pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    holdout_dates = set(holdout["date"].astype(str))
    holdout_events = events[events["date"].astype(str).isin(holdout_dates)].copy()
    holdout_events["event_start"] = pd.to_datetime(holdout_events["event_start"], utc=True, format="mixed", errors="coerce")
    date_frames = {date: read_scored(date) for date in sorted(holdout_dates)}
    alerts = pd.concat([selected_policy_alerts(df) for df in date_frames.values() if not df.empty], ignore_index=True)
    if alerts.empty:
        alerts = pd.DataFrame(columns=["date", "alert_start", "alert_end", "duration_sec", "max_score"])

    useful = 0
    false_alerts = 0
    quiet_false = 0
    matched_ids: set[str] = set()
    leads = []
    event_rows = []
    quiet_dates = set(holdout.loc[holdout["holdout_type"].astype(str).eq("QUIET"), "date"].astype(str))

    for _, event in holdout_events.iterrows():
        same_alerts = alerts[alerts["date"].astype(str).eq(str(event["date"]))]
        valid = []
        for _, alert in same_alerts.iterrows():
            lead = (event["event_start"] - alert["alert_start"]).total_seconds() / 60
            overlap = alert["alert_start"] <= event["event_start"] <= alert["alert_end"] + pd.Timedelta(minutes=5)
            if (0 <= lead <= 90) or overlap:
                valid.append((lead, alert))
        if valid:
            valid = sorted(valid, key=lambda item: item[0], reverse=True)
            lead, alert = valid[0]
            eid = str(event.get("candidate_global_event_id", event.get("event_id", "")))
            matched_ids.add(eid)
            leads.append(lead)
            detected = True
            first_alert = alert["alert_start"]
            missed_reason = ""
        else:
            detected = False
            first_alert = ""
            lead = np.nan
            missed_reason = "No selected-policy alert in 90-minute precursor/event window."
        event_rows.append(
            {
                "date": event["date"],
                "expected_label": "flare",
                "candidate_global_event_id": event.get("candidate_global_event_id", ""),
                "goes_class": event.get("goes_class", ""),
                "goes_match_status": event.get("goes_match_status", ""),
                "event_start": event.get("event_start", ""),
                "detected_alert": detected,
                "first_alert_time": first_alert,
                "lead_time_min": lead,
                "false_alert_count": 0,
                "missed_reason": missed_reason,
                "notes": "Holdout flare event evaluated once with fixed hard-first v2 policy.",
            }
        )

    # Classify alert episodes as useful/false after event matching.
    for _, alert in alerts.iterrows():
        same_events = holdout_events[holdout_events["date"].astype(str).eq(str(alert["date"]))]
        matched = False
        for _, event in same_events.iterrows():
            lead = (event["event_start"] - alert["alert_start"]).total_seconds() / 60
            overlap = alert["alert_start"] <= event["event_start"] <= alert["alert_end"] + pd.Timedelta(minutes=5)
            if (0 <= lead <= 90) or overlap:
                matched = True
                break
        if matched:
            useful += 1
        else:
            false_alerts += 1
            if str(alert["date"]) in quiet_dates:
                quiet_false += 1

    for date in sorted(quiet_dates):
        false_count = int(alerts[alerts["date"].astype(str).eq(date)].shape[0])
        event_rows.append(
            {
                "date": date,
                "expected_label": "quiet",
                "candidate_global_event_id": "",
                "goes_class": "",
                "goes_match_status": "",
                "event_start": "",
                "detected_alert": false_count > 0,
                "first_alert_time": alerts.loc[alerts["date"].astype(str).eq(date), "alert_start"].min() if false_count else "",
                "lead_time_min": np.nan,
                "false_alert_count": false_count,
                "missed_reason": "",
                "notes": "Quiet holdout date; any alert is a false alert episode.",
            }
        )

    total_events = len(holdout_events)
    precision = useful / len(alerts) if len(alerts) else 0.0
    recall = len(matched_ids) / total_events if total_events else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "policy": "policy_1_burst_and_solexs_not_quiet",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "far_per_day": false_alerts / max(len(holdout_dates), 1),
        "valid_alerted_events": len(matched_ids),
        "total_events": total_events,
        "false_alerts": false_alerts,
        "missed_events": total_events - len(matched_ids),
        "mean_lead_time_min": float(np.nanmean(leads)) if leads else np.nan,
        "median_lead_time_min": float(np.nanmedian(leads)) if leads else np.nan,
        "quiet_day_false_alerts": quiet_false,
        "total_alert_episodes": len(alerts),
        "notes": "Locked holdout evaluated once; no threshold tuning after this result.",
    }
    return metrics, pd.DataFrame(event_rows), date_frames, alerts


def make_plots(holdout: pd.DataFrame, events: pd.DataFrame, date_frames: dict[str, pd.DataFrame], alerts: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    events = events.copy()
    events["event_start"] = pd.to_datetime(events.get("event_start"), utc=True, format="mixed", errors="coerce")
    events["event_end"] = pd.to_datetime(events.get("event_end"), utc=True, format="mixed", errors="coerce")
    for date, df in date_frames.items():
        if df.empty:
            continue
        day_events = events[events["date"].astype(str).eq(date)]
        day_alerts = alerts[alerts["date"].astype(str).eq(date)]
        fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
        axes[0].plot(df["timestamp"], df["soft_solexs_2_22"], lw=0.8, color="#1f77b4")
        axes[0].set_ylabel("SoLEXS")
        axes[1].plot(df["timestamp"], df["hard_cdte_5_20"], lw=0.7, color="#d62728", label="CdTe")
        axes[1].plot(df["timestamp"], df["hard_czt_20_40"], lw=0.7, color="#9467bd", label="CZT")
        axes[1].legend(fontsize=8)
        axes[1].set_ylabel("HEL1OS")
        axes[2].plot(df["timestamp"], df["soft_score"], lw=0.7, color="#1f77b4", label="soft_score")
        axes[2].plot(df["timestamp"], df["hard_score"], lw=0.7, color="#d62728", label="hard_score")
        axes[2].legend(fontsize=8)
        axes[2].set_ylabel("Scores")
        axes[3].plot(df["timestamp"], df["hel1os_burst_support"], lw=0.7, color="#2ca02c")
        axes[3].set_ylabel("Burst")
        for ax in axes:
            for _, event in day_events.iterrows():
                if pd.notna(event["event_start"]) and pd.notna(event["event_end"]):
                    ax.axvspan(event["event_start"], event["event_end"], color="#fdae61", alpha=0.25)
            for _, alert in day_alerts.iterrows():
                ax.axvspan(alert["alert_start"], alert["alert_end"], color="#2ca02c", alpha=0.18)
            ax.grid(alpha=0.2)
        axes[0].set_title(f"Locked holdout {date}: hard-first v2 fixed policy")
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"{date}_holdout_hard_first_v2.png", dpi=150)
        plt.close(fig)


def write_outputs(metrics: dict, event_table: pd.DataFrame, holdout: pd.DataFrame) -> None:
    pd.DataFrame([metrics]).to_csv(OUT_DIR / "hard_first_v2_holdout_results.csv", index=False)
    event_table.to_csv(OUT_DIR / "holdout_event_level_table.csv", index=False)
    comparison = pd.DataFrame(
        [
            TRAIN_DEV_HARD_FIRST_V2,
            {
                "evaluation_group": "HOLDOUT hard-first v2",
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "far_per_day": metrics["far_per_day"],
                "valid_alerted_events": f"{metrics['valid_alerted_events']}/{metrics['total_events']}",
                "false_alerts": metrics["false_alerts"],
                "missed_events": metrics["missed_events"],
                "mean_lead_time_min": metrics["mean_lead_time_min"],
                "median_lead_time_min": metrics["median_lead_time_min"],
                "quiet_day_false_alerts": metrics["quiet_day_false_alerts"],
                "notes": "Tiny locked sanity holdout; not a replacement for frozen v8.1.",
            },
            FROZEN_V8_1,
        ]
    )
    comparison.to_csv(OUT_DIR / "train_dev_vs_holdout_comparison.csv", index=False)
    holdout_dates = ", ".join(holdout["date"].astype(str))
    recommendation = (
        "Hard-first v2 remains an exploratory expanded-data research branch; it should not replace frozen v8.1."
        if metrics["f1"] < FROZEN_V8_1["f1"]
        else "Hard-first v2 is promising on this tiny holdout, but still should not replace frozen v8.1 without larger validation."
    )
    md = [
        "# Final Shadow Holdout Evaluation Report",
        "",
        "This locked holdout was evaluated once with the selected hard-first v2 policy. No thresholds were changed after seeing these results.",
        "",
        f"- Holdout dates: {holdout_dates}",
        "- Caveat: this is a shadow-style sanity holdout, not a perfect blind benchmark, because the approved flare pool is small and earlier analysis touched some dates.",
        "- Selected policy: `policy_1_burst_and_solexs_not_quiet`",
        "",
        "## Holdout Metrics",
        "",
        markdown_table(pd.DataFrame([metrics])),
        "",
        "## TRAIN_DEV vs Holdout vs Frozen Reference",
        "",
        markdown_table(comparison),
        "",
        "## Event-Level Holdout Table",
        "",
        markdown_table(event_table),
        "",
        "## Final Decision",
        "",
        recommendation,
        "",
        "Neural networks remain deferred unless more validated flare labels are available. Do not use this holdout for further threshold tuning.",
    ]
    (OUT_DIR / "FINAL_SHADOW_HOLDOUT_EVAL_REPORT.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    ensure_out()
    holdout, events, _goes = load_inputs()
    metrics, event_table, date_frames, alerts = evaluate(holdout, events)
    make_plots(holdout, events[events["date"].astype(str).isin(set(holdout["date"].astype(str)))], date_frames, alerts)
    write_outputs(metrics, event_table, holdout)
    flare_dates = holdout.loc[holdout["holdout_type"].astype(str).eq("FLARE"), "date"].astype(str).tolist()
    quiet_dates = holdout.loc[holdout["holdout_type"].astype(str).eq("QUIET"), "date"].astype(str).tolist()
    print(f"holdout flare dates: {', '.join(flare_dates)}")
    print(f"holdout quiet dates: {', '.join(quiet_dates)}")
    print(f"holdout metrics: {metrics}")
    print(f"quiet-day false alerts: {metrics['quiet_day_false_alerts']}")
    print(f"comparison to TRAIN_DEV: TRAIN_DEV F1={TRAIN_DEV_HARD_FIRST_V2['f1']}, holdout F1={metrics['f1']}; TRAIN_DEV FAR/day={TRAIN_DEV_HARD_FIRST_V2['far_per_day']}, holdout FAR/day={metrics['far_per_day']}")
    print("final recommendation: hard-first v2 remains an exploratory expanded-data research branch; do not replace frozen v8.1 or run neural nets yet.")
    print(f"report path: {OUT_DIR / 'FINAL_SHADOW_HOLDOUT_EVAL_REPORT.md'}")


if __name__ == "__main__":
    main()
