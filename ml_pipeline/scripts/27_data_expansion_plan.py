from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(".")
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "data_expansion"

V3_FALSE_ALERTS = RESULTS / "forecasting_v3" / "forecasting_v3_false_alert_analysis.csv"
V5_RULES_AUDIT = RESULTS / "forecasting_v5" / "forecasting_v5_false_alert_rules_audit.csv"
V5_COMPARISON = RESULTS / "forecasting_v5" / "v1_v2_v3_v4_v5_comparison.csv"
QUALITY_GROUP = RESULTS / "quality_group_metric_comparison.csv"
MASTER = RESULTS / "master_flare_catalogue_classified_v2.csv"
QUALITY = RESULTS / "data_quality_report.csv"
QUIET = RESULTS / "quiet_day_validation.csv"
MATCHING = RESULTS / "date_matching_report.csv"
GOES = ROOT / "data" / "external" / "goes_flare_events.csv"

ERROR_PROFILE_CSV = OUT_DIR / "current_error_profile.csv"
ERROR_PROFILE_MD = OUT_DIR / "current_error_profile.md"
PLAN_CSV = OUT_DIR / "aditya_l1_data_expansion_plan.csv"
PLAN_MD = OUT_DIR / "aditya_l1_data_expansion_plan.md"
TARGET_FLARE_CSV = OUT_DIR / "target_event_dates_for_download.csv"
TARGET_QUIET_CSV = OUT_DIR / "target_quiet_dates_for_download.csv"
RETRAINING_MD = OUT_DIR / "retraining_plan_v6.md"

V3_FINAL = {
    "precision": 0.515,
    "recall": 0.824,
    "f1": 0.634,
    "false_alerts_per_day": 1.33,
    "valid_alerted_events": "14/17",
    "mean_lead_time_min": 39.44,
    "median_lead_time_min": 40.18,
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def norm_date(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    out = df.copy()
    out[col] = out[col].astype(str).str.replace(r"\.0$", "", regex=True)
    return out


def goes_group(goes_class: object) -> str:
    text = str(goes_class).strip().upper()
    return text[:1] if text else "UNKNOWN"


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "false": norm_date(read_csv(V3_FALSE_ALERTS)),
        "v5_rules": norm_date(read_csv(V5_RULES_AUDIT)),
        "v5_comparison": read_csv(V5_COMPARISON),
        "quality_group": read_csv(QUALITY_GROUP),
        "master": norm_date(read_csv(MASTER)),
        "quality": norm_date(read_csv(QUALITY)),
        "quiet": norm_date(read_csv(QUIET)),
        "matching": norm_date(read_csv(MATCHING)),
        "goes": read_csv(GOES),
    }


def build_error_profile(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    false = data["false"].copy()
    master = data["master"].copy()
    quality = data["quality"].copy()
    quiet = data["quiet"].copy()
    dates = set()
    for df in [false, master, quality, quiet]:
        if not df.empty and "date" in df.columns:
            dates |= set(df["date"].astype(str))
    rows = []
    for date in sorted(dates):
        f = false[false["date"].astype(str).eq(date)] if not false.empty else pd.DataFrame()
        m = master[master["date"].astype(str).eq(date)] if not master.empty else pd.DataFrame()
        q = quality[quality["date"].astype(str).eq(date)] if not quality.empty else pd.DataFrame()
        quiet_row = quiet[quiet["date"].astype(str).eq(date)] if not quiet.empty else pd.DataFrame()
        cause_counts = f.get("likely_cause_category", pd.Series(dtype=str)).value_counts().to_dict()
        quality_label = ""
        if not q.empty:
            quality_label = str(q.iloc[0].get("overlap_quality_label", ""))
        elif not quiet_row.empty:
            quality_label = str(quiet_row.iloc[0].get("quality_label", ""))
        cleaned_events = len(m)
        valid_tp = int(m.get("valid_forecast_alerted", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not m.empty else 0
        rows.append(
            {
                "date": date,
                "quality_label": quality_label,
                "v3_false_alert_count": len(f),
                "questionable_date_false_alerts": int(cause_counts.get("QUESTIONABLE_DATE", 0)),
                "post_flare_decay_false_alerts": int(cause_counts.get("POST_FLARE_DECAY", 0)),
                "duplicate_false_alerts": int(cause_counts.get("DUPLICATE_ALERT", 0)),
                "true_isolated_false_alerts": int(cause_counts.get("TRUE_ISOLATED_FALSE_ALERT", 0)),
                "other_false_alerts": int(len(f) - sum(cause_counts.get(k, 0) for k in ["QUESTIONABLE_DATE", "POST_FLARE_DECAY", "DUPLICATE_ALERT", "TRUE_ISOLATED_FALSE_ALERT"])),
                "cleaned_event_count": cleaned_events,
                "valid_forecast_true_positive_count": valid_tp,
                "solexs_finite_percent": q.iloc[0].get("solexs_finite_percent", pd.NA) if not q.empty else pd.NA,
                "cdte_finite_percent": q.iloc[0].get("cdte_finite_percent", pd.NA) if not q.empty else pd.NA,
                "czt_finite_percent": q.iloc[0].get("czt_finite_percent", pd.NA) if not q.empty else pd.NA,
                "problem_score": len(f) + (2 if quality_label == "QUESTIONABLE" else 0) + (5 if quality_label == "BROKEN" else 0),
                "notes": classify_date_notes(len(f), quality_label, cleaned_events, valid_tp, cause_counts),
            }
        )
    profile = pd.DataFrame(rows).sort_values(["problem_score", "v3_false_alert_count", "cleaned_event_count"], ascending=[False, False, False])
    profile.to_csv(ERROR_PROFILE_CSV, index=False)
    return profile


def classify_date_notes(false_count: int, quality_label: str, events: int, tp: int, causes: dict) -> str:
    parts = []
    if quality_label == "BROKEN":
        parts.append("BROKEN date; exclude from supervised evaluation")
    elif quality_label == "QUESTIONABLE":
        parts.append("QUESTIONABLE telemetry; prioritize improved coverage if scientifically important")
    if false_count:
        parts.append(f"{false_count} v3 false-alert diagnostics")
    if causes.get("POST_FLARE_DECAY", 0):
        parts.append("post-flare decay false-alert pattern present")
    if causes.get("DUPLICATE_ALERT", 0):
        parts.append("duplicate alert pattern present")
    if events:
        parts.append(f"{events} cleaned events, {tp} valid forecast true positives")
    if not parts:
        parts.append("quiet/control or low-error date")
    return "; ".join(parts)


def build_goes_daily(goes: pd.DataFrame) -> pd.DataFrame:
    if goes.empty:
        return pd.DataFrame(columns=["date", "c_count", "m_count", "x_count", "total_goes_events", "max_class_group"])
    out = goes.copy()
    out["peak_time_utc"] = pd.to_datetime(out["peak_time_utc"], utc=True, errors="coerce")
    out = out.dropna(subset=["peak_time_utc"])
    out["date"] = out["peak_time_utc"].dt.strftime("%Y%m%d")
    out["class_group"] = out["goes_class"].map(goes_group)
    rows = []
    for date, group in out.groupby("date"):
        counts = group["class_group"].value_counts().to_dict()
        max_group = "X" if counts.get("X", 0) else ("M" if counts.get("M", 0) else ("C" if counts.get("C", 0) else "UNKNOWN"))
        rows.append(
            {
                "date": date,
                "c_count": int(counts.get("C", 0)),
                "m_count": int(counts.get("M", 0)),
                "x_count": int(counts.get("X", 0)),
                "total_goes_events": len(group),
                "max_class_group": max_group,
            }
        )
    return pd.DataFrame(rows)


def add_plan_row(rows: list[dict], date: str, target_type: str, priority: str, reason: str, benefit: str, notes: str) -> None:
    rows.append(
        {
            "date": date,
            "target_type": target_type,
            "priority": priority,
            "reason": reason,
            "expected_benefit": benefit,
            "required_payloads": "SoLEXS + HEL1OS",
            "external_validation_source": "GOES/SWPC",
            "notes": notes,
        }
    )


def build_download_plan(data: dict[str, pd.DataFrame], profile: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    goes_daily = build_goes_daily(data["goes"])
    matching = data["matching"]
    quiet = data["quiet"]
    quality_by_date = dict(zip(profile["date"], profile["quality_label"])) if not profile.empty else {}
    matched_by_date = dict(zip(matching["date"], matching["matched"])) if not matching.empty else {}
    skip_by_date = dict(zip(matching["date"], matching.get("skip_reason", pd.Series("", index=matching.index)))) if not matching.empty else {}

    rows = []
    for _, row in goes_daily.sort_values(["x_count", "m_count", "total_goes_events"], ascending=False).iterrows():
        date = str(row["date"])
        if row["x_count"] > 0 or row["m_count"] > 0:
            matched = bool(matched_by_date.get(date, False))
            quality = quality_by_date.get(date, "UNPROCESSED_OR_UNMATCHED")
            priority = "HIGH" if row["x_count"] > 0 else "MEDIUM"
            reason = f"GOES/SWPC file has C={row['c_count']}, M={row['m_count']}, X={row['x_count']} events; max group {row['max_class_group']}."
            if matched and quality == "QUESTIONABLE":
                notes = "Already processed but QUESTIONABLE; prioritize better/complete Level-1 coverage if available."
            elif matched and quality == "GOOD":
                notes = "Already processed; useful benchmark date for regression comparison."
            elif matched and quality == "BROKEN":
                notes = "Matched but BROKEN; reacquire SoLEXS/HEL1OS if possible."
            else:
                notes = f"Not currently usable or unmatched; {skip_by_date.get(date, 'check payload availability')}."
            add_plan_row(
                rows,
                date,
                "FLARE",
                priority,
                reason,
                "Improves event diversity, high-class recall calibration, and blocked validation robustness.",
                notes,
            )

    # Payload-completion targets adjacent to existing activity/controls.
    for date, skip in sorted(skip_by_date.items()):
        if date in set(goes_daily["date"].astype(str)):
            continue
        if "missing" in str(skip).lower():
            priority = "MEDIUM" if date.startswith(("202602", "202603", "202606")) else "LOW"
            add_plan_row(
                rows,
                date,
                "FLARE",
                priority,
                "Date appears in local matching report but lacks one required payload.",
                "Can expand blocked date-wise validation if GOES/SWPC confirms flare activity or transition behavior.",
                f"Payload completion target; skip reason: {skip}. Verify GOES/SWPC event activity before using as flare training date.",
            )

    # Quiet/control targets from current confirmed controls and nearby unmatched dates.
    if not quiet.empty:
        for _, row in quiet[quiet["quiet_status"].astype(str).isin(["CONFIRMED_QUIET", "LOW_ACTIVITY_CONTROL", "QUESTIONABLE_CONTROL"])].iterrows():
            priority = "HIGH" if row["quiet_status"] == "CONFIRMED_QUIET" else "MEDIUM"
            add_plan_row(
                rows,
                str(row["date"]),
                "QUIET",
                priority,
                f"Current control label: {row['quiet_status']}; cleaned events={row['cleaned_event_count']}, skipped candidates={row['skipped_candidate_count']}.",
                "Adds/retains negative control examples to reduce false alerts and calibrate quiet-day thresholds.",
                "Already processed control; keep as benchmark and add similar adjacent quiet days.",
            )
    for date, skip in sorted(skip_by_date.items()):
        if date not in set(goes_daily["date"].astype(str)) and "missing" in str(skip).lower():
            add_plan_row(
                rows,
                date,
                "QUIET",
                "LOW",
                "Unmatched date near current observation periods; may be useful as quiet/control if GOES/SWPC shows no major flares.",
                "Potential negative/control expansion after payload completion and GOES/SWPC quiet validation.",
                f"Do not label quiet until GOES/SWPC external check confirms no major events; skip reason: {skip}.",
            )

    plan = pd.DataFrame(rows).drop_duplicates(subset=["date", "target_type", "reason"]).sort_values(["target_type", "priority", "date"])
    flare = plan[plan["target_type"].eq("FLARE")].copy()
    quiet_targets = plan[plan["target_type"].eq("QUIET")].copy()
    plan.to_csv(PLAN_CSV, index=False)
    flare.to_csv(TARGET_FLARE_CSV, index=False)
    quiet_targets.to_csv(TARGET_QUIET_CSV, index=False)
    return plan, flare, quiet_targets


def write_markdown(profile: pd.DataFrame, plan: pd.DataFrame, flare: pd.DataFrame, quiet: pd.DataFrame, data: dict[str, pd.DataFrame]) -> None:
    top_problem = profile.head(8)
    false = data["false"]
    qgroup = data["quality_group"]
    top_tp = profile.sort_values(["valid_forecast_true_positive_count", "cleaned_event_count"], ascending=False).head(8)
    lines = [
        "# Current Error Profile",
        "",
        f"Current final mode remains Forecasting v3 state-machine ML policy: precision {V3_FINAL['precision']}, recall {V3_FINAL['recall']}, F1 {V3_FINAL['f1']}, false alerts/day {V3_FINAL['false_alerts_per_day']}.",
        "",
        "## False Alerts by Cause",
        "",
        str(false.get("likely_cause_category", pd.Series(dtype=str)).value_counts().to_dict() if not false.empty else {}),
        "",
        "## False Alerts by Quality Label",
        "",
        str(false.get("quality_label", pd.Series(dtype=str)).value_counts().to_dict() if not false.empty else {}),
        "",
        "## GOOD vs QUESTIONABLE Performance",
        "",
        qgroup.to_csv(index=False) if not qgroup.empty else "Missing quality-group comparison.",
        "",
        "## Most Problematic Dates",
        "",
        top_problem.to_csv(index=False),
        "",
        "## Dates Contributing True Positives",
        "",
        top_tp.to_csv(index=False),
    ]
    ERROR_PROFILE_MD.write_text("\n".join(lines), encoding="utf-8")

    plan_lines = [
        "# Aditya-L1 Data Expansion Plan",
        "",
        "The next forecasting improvement should come from more matched SoLEXS + HEL1OS dates, not from adding model complexity.",
        "",
        f"- Recommended flare/payload-completion targets: {len(flare)}",
        f"- Recommended quiet/control targets: {len(quiet)}",
        "",
        "## Flare-Date Strategy",
        "",
        "- Prioritize GOES/SWPC M/X dates from the existing official label file.",
        "- Re-download or improve QUESTIONABLE active dates when better SoLEXS/HEL1OS coverage exists.",
        "- Complete unmatched payload dates only when both SoLEXS and HEL1OS Level-1 light curves can be obtained.",
        "",
        "## Quiet-Control Strategy",
        "",
        "- Add GOOD quiet days with both payloads as negative/control examples.",
        "- Keep QUESTIONABLE controls marked; do not use BROKEN dates for supervised forecasting.",
        "- Validate quiet/control status against GOES/SWPC before using as negative labels.",
        "",
        "## Top Targets",
        "",
        plan.head(20).to_csv(index=False),
    ]
    PLAN_MD.write_text("\n".join(plan_lines), encoding="utf-8")


def write_retraining_plan() -> None:
    text = f"""# Retraining Plan v6

Current final forecasting mode remains the Forecasting v3 state-machine ML policy:

- Precision: {V3_FINAL['precision']}
- Recall/POD: {V3_FINAL['recall']}
- F1: {V3_FINAL['f1']}
- False alerts/day: {V3_FINAL['false_alerts_per_day']}
- Valid alerted events: {V3_FINAL['valid_alerted_events']}
- Mean lead time: {V3_FINAL['mean_lead_time_min']} min
- Median lead time: {V3_FINAL['median_lead_time_min']} min

v4 and v5 do not replace v3 on the current small dataset. v6 should improve the data, not add more model complexity.

## v6 Data and Retraining Steps

1. Add new SoLEXS + HEL1OS ZIP files to `data/zips`.
2. Re-run extraction and nested ZIP extraction.
3. Re-run FITS/lightcurve inspection for new dates.
4. Rebuild combined SoLEXS + HEL1OS light curves for matched dates only.
5. Re-run nowcasting without changing nowcast logic.
6. Rebuild cleaned nowcast and master catalogues.
7. Re-run GOES/SWPC external validation for new event dates.
8. Re-run data-quality audit and exclude BROKEN dates.
9. Rebuild forecasting v3/v4 datasets using GOOD and QUESTIONABLE dates only.
10. Re-run blocked date-wise validation.
11. Compare original v3 versus expanded-data v6.
12. Keep old v3 final until v6 beats it.

## v6 Target Improvement Rule

Expanded-data v6 should replace current v3 only if:

- F1 > 0.634, and
- false alerts/day < 1.33, and
- recall >= 0.824 or precision improves meaningfully with recall >= 0.75.

## Caveats

- GOES/SWPC is for external validation and label construction only, never as an input feature.
- SoLEXS + HEL1OS remain the core data sources.
- BROKEN dates remain excluded from supervised forecasting.
- Quiet/control days must stay in the dataset as negative examples when quality is GOOD.
"""
    RETRAINING_MD.write_text(text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_inputs()
    profile = build_error_profile(data)
    plan, flare, quiet = build_download_plan(data, profile)
    write_markdown(profile, plan, flare, quiet, data)
    write_retraining_plan()
    top_problem = profile.head(5)["date"].tolist()
    print(f"top problematic dates: {', '.join(top_problem)}")
    print(f"recommended flare dates count: {len(flare)}")
    print(f"recommended quiet dates count: {len(quiet)}")
    print(f"retraining plan path: {RETRAINING_MD}")
    print("final recommendation: v3 remains final until expanded-data v6 retraining beats it")


if __name__ == "__main__":
    main()
