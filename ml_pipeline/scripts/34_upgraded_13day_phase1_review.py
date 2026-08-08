from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPANDED_DIR = PROJECT_ROOT / "results" / "expanded_80day_ingestion"
REVIEW_DIR = PROJECT_ROOT / "results" / "expanded_80day_review"
SHADOW_DIR = PROJECT_ROOT / "results" / "shadow_holdout"
SPECTRAL_DIR = PROJECT_ROOT / "results" / "solexs_spectral_feasibility"
BURST_DIR = PROJECT_ROOT / "results" / "hel1os_burst_features"
FUSION_DIR = PROJECT_ROOT / "results" / "fusion_ablation"
DISAGREEMENT_DIR = PROJECT_ROOT / "results" / "instrument_disagreement"
BACKUP_DIR = PROJECT_ROOT / "backup" / "LATEST_STABLE_BEFORE_EXPERIMENTS"
EXPERIMENT_DIR = PROJECT_ROOT / "space_agency_experiment"
PRIOR_FROZEN_DATES = {
    "20260201",
    "20260202",
    "20260204",
    "20260222",
    "20260224",
    "20260310",
    "20260311",
    "20260603",
    "20260605",
}


def ensure_dirs() -> None:
    for path in [REVIEW_DIR, SHADOW_DIR, SPECTRAL_DIR, BURST_DIR, FUSION_DIR, DISAGREEMENT_DIR, BACKUP_DIR, EXPERIMENT_DIR]:
        path.mkdir(parents=True, exist_ok=True)


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


def create_lightweight_backup() -> None:
    copied = []
    skipped = []

    def ignore_heavy(dir_path: str, names: list[str]) -> set[str]:
        ignored = {
            "__pycache__",
            ".pytest_cache",
            "combined_lightcurves",
            "raw_expanded_80day",
            "zips",
        }
        return {name for name in names if name in ignored or name.endswith(".zip")}

    for folder in ["app", "src", "scripts", "final_submission_package"]:
        src = PROJECT_ROOT / folder
        dst = BACKUP_DIR / folder
        if not src.exists():
            skipped.append(folder)
            continue
        if dst.exists():
            copied.append(f"{folder} (already present)")
            continue
        shutil.copytree(src, dst, ignore=ignore_heavy)
        copied.append(folder)

    results_src = PROJECT_ROOT / "results"
    results_dst = BACKUP_DIR / "results"
    if results_src.exists() and not results_dst.exists():
        def ignore_results(dir_path: str, names: list[str]) -> set[str]:
            ignored = {"combined_lightcurves", "__pycache__"}
            return {name for name in names if name in ignored or name.endswith("_scored_timeseries.csv")}

        shutil.copytree(results_src, results_dst, ignore=ignore_results)
        copied.append("results")
    elif results_dst.exists():
        copied.append("results (already present)")

    for name in ["README.md", "requirements.txt"]:
        src = PROJECT_ROOT / name
        dst = BACKUP_DIR / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied.append(name)

    skipped_lines = [f"- {item}" for item in skipped] if skipped else ["- None"]
    manifest = [
        "# Latest Stable Before Experiments",
        "",
        f"- Created/verified: {datetime.now().isoformat(timespec='seconds')}",
        "- Purpose: lightweight safety copy before upgraded 13-day research-plan experiments.",
        "- Heavy raw ZIPs and expanded combined lightcurve CSVs are intentionally excluded.",
        "",
        "## Copied",
        *[f"- {item}" for item in copied],
        "",
        "## Skipped/Missing",
        *skipped_lines,
        "",
        "Main frozen metrics remain unchanged; this backup is not a new model result.",
    ]
    (BACKUP_DIR / "BACKUP_MANIFEST.md").write_text("\n".join(manifest), encoding="utf-8")


def require_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    master_path = EXPANDED_DIR / "expanded_master_flare_catalogue_CANDIDATE.csv"
    quiet_path = EXPANDED_DIR / "expanded_quiet_day_validation.csv"
    availability_path = EXPANDED_DIR / "date_payload_availability.csv"
    missing = [p for p in [master_path, quiet_path, availability_path] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing expanded ingestion inputs: " + ", ".join(str(p) for p in missing))
    return pd.read_csv(master_path), pd.read_csv(quiet_path), pd.read_csv(availability_path)


def review_expanded_data(master: pd.DataFrame, quiet: pd.DataFrame, availability: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    date_quality = availability.set_index("date")["quality"].astype(str).to_dict()
    reviewed = master.copy()
    reviewed["payload_quality"] = reviewed["date"].map(date_quality).fillna("UNKNOWN")
    reviewed["review_decision"] = "MANUAL_REVIEW"
    reviewed["review_reason"] = "Default manual review."

    exact_good = reviewed["payload_quality"].eq("GOOD") & reviewed["goes_match_status"].eq("EXACT_PEAK_MATCH")
    window_good = reviewed["payload_quality"].eq("GOOD") & reviewed["goes_match_status"].eq("WINDOW_OVERLAP_MATCH")
    reviewed.loc[exact_good, "review_decision"] = "AUTO_ACCEPT"
    reviewed.loc[exact_good, "review_reason"] = "GOOD date with both payloads and GOES peak match within tolerance."
    reviewed.loc[window_good, "review_decision"] = "WEAK_ACCEPT"
    reviewed.loc[window_good, "review_reason"] = "GOOD date with both payloads and GOES event-window overlap."

    no_match = reviewed["goes_match_status"].isin(["NO_MATCH", "NEAREST_ONLY"])
    reviewed.loc[no_match, "review_reason"] = "GOES match is nearest-only or absent; do not train until manually reviewed."
    reviewed.loc[~reviewed["payload_quality"].eq("GOOD"), "review_reason"] = "Date quality is not GOOD; manual review required."

    approved_flare = reviewed[reviewed["review_decision"].isin(["AUTO_ACCEPT", "WEAK_ACCEPT"])].copy()
    manual = reviewed[~reviewed["review_decision"].isin(["AUTO_ACCEPT", "WEAK_ACCEPT"])].copy()

    approved_quiet = quiet[
        quiet["date_classification"].eq("GOOD_QUIET") & quiet["payload_quality"].eq("GOOD") & quiet["cleaned_event_count"].astype(int).eq(0)
    ].copy()
    approved_quiet["review_decision"] = "AUTO_ACCEPT_QUIET"
    approved_quiet["review_reason"] = "GOOD date, both payloads, no cleaned nowcast events, no GOES event on date in current label file."

    approved_flare.to_csv(REVIEW_DIR / "approved_flare_events_CANDIDATE.csv", index=False)
    approved_quiet.to_csv(REVIEW_DIR / "approved_quiet_days_CANDIDATE.csv", index=False)
    manual.to_csv(REVIEW_DIR / "rejected_or_manual_review_events.csv", index=False)

    report = [
        "# Expanded 80-Day Candidate Review",
        "",
        "Acceptance rules used:",
        "- Auto-accept flare event: GOOD date + both payloads + `EXACT_PEAK_MATCH`.",
        "- Weak-accept flare event: GOOD date + both payloads + `WINDOW_OVERLAP_MATCH`.",
        "- Manual review: `NEAREST_ONLY`, `NO_MATCH`, QUESTIONABLE/PARTIAL dates, or non-GOOD payload quality.",
        "- Auto-accept quiet day: GOOD payload quality + zero cleaned events + no GOES event on date in the current label file.",
        "",
        f"- Candidate flare events reviewed: {len(reviewed)}",
        f"- Approved flare events: {len(approved_flare)}",
        f"- Manual/rejected flare events: {len(manual)}",
        f"- Approved quiet days: {len(approved_quiet)}",
        "",
        "## Flare Decision Counts",
        "",
        markdown_table(reviewed["review_decision"].value_counts().reset_index(name="count").rename(columns={"index": "review_decision"})),
        "",
        "## Manual Review Event Breakdown",
        "",
        markdown_table(manual["goes_match_status"].value_counts().reset_index(name="count").rename(columns={"index": "goes_match_status"})),
        "",
        "No `NO_MATCH` or `NEAREST_ONLY` events are approved for training by this review pass.",
    ]
    (REVIEW_DIR / "EXPANDED_80DAY_REVIEW_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return approved_flare, approved_quiet, manual


def create_shadow_holdout(approved_flare: pd.DataFrame, approved_quiet: pd.DataFrame) -> pd.DataFrame:
    flare_dates = (
        approved_flare.assign(class_group=approved_flare["goes_class"].astype(str).str[0].where(approved_flare["goes_class"].astype(str).str[0].isin(["C", "M", "X"]), "UNKNOWN"))
        .groupby("date")
        .agg(event_count=("candidate_global_event_id", "count"), class_groups=("class_group", lambda x: ",".join(sorted(set(x)))))
        .reset_index()
        .sort_values(["class_groups", "date"])
    )
    quiet_dates = approved_quiet[["date"]].drop_duplicates().sort_values("date")

    flare_dates["is_prior_frozen_date"] = flare_dates["date"].astype(str).isin(PRIOR_FROZEN_DATES)
    preferred_flare = flare_dates[~flare_dates["is_prior_frozen_date"]]
    fallback_flare = flare_dates[flare_dates["is_prior_frozen_date"]]
    selected_flare = pd.concat([preferred_flare, fallback_flare], ignore_index=True).head(3)

    selected_rows = []
    for _, row in selected_flare.iterrows():
        prior = bool(row.get("is_prior_frozen_date", False))
        selected_rows.append(
            {
                "date": row["date"],
                "holdout_type": "FLARE",
                "event_count": int(row["event_count"]),
                "class_groups": row["class_groups"],
                "is_prior_frozen_date": prior,
                "selection_reason": (
                    "Approved flare date selected for shadow holdout; do not tune on it. "
                    + ("Caveat: this date overlaps earlier frozen experiments." if prior else "Preferred newly expanded approved flare date.")
                ),
            }
        )
    for _, row in quiet_dates.tail(4).iterrows():
        selected_rows.append(
            {
                "date": row["date"],
                "holdout_type": "QUIET",
                "event_count": 0,
                "class_groups": "",
                "is_prior_frozen_date": str(row["date"]) in PRIOR_FROZEN_DATES,
                "selection_reason": "Approved quiet/control date selected for shadow holdout; do not tune on it.",
            }
        )
    holdout = pd.DataFrame(selected_rows).drop_duplicates("date").sort_values("date")
    holdout.to_csv(SHADOW_DIR / "shadow_holdout_dates.csv", index=False)

    md = [
        "# Shadow Holdout Dates",
        "",
        "These dates are locked for future final validation. Do not train or tune thresholds on them.",
        "",
        "Caveat: the approved expanded flare pool is still small, so some flare holdout dates may overlap earlier frozen experiments. Quiet holdout dates are prioritized from newly expanded approved quiet/control days when available.",
        "",
        markdown_table(holdout),
    ]
    (SHADOW_DIR / "shadow_holdout_dates.md").write_text("\n".join(md), encoding="utf-8")
    return holdout


def inspect_solexs_spectral() -> pd.DataFrame:
    try:
        from astropy.io import fits
    except Exception as exc:
        (SPECTRAL_DIR / "SOLEXS_SPECTRAL_FEASIBILITY_REPORT.md").write_text(
            f"# SoLEXS Spectral Feasibility\n\nAstropy import failed: {exc}\n", encoding="utf-8"
        )
        return pd.DataFrame()

    solexs_files = sorted((PROJECT_ROOT / "data" / "raw_expanded_80day" / "solexs").rglob("AL1_SOLEXS_*_L1.lc.gz"))
    rows = []
    for path in solexs_files[:80]:
        date_match = pd.Series([path.name]).str.extract(r"(20\d{6})").iloc[0, 0]
        try:
            with fits.open(path) as hdul:
                for idx, hdu in enumerate(hdul):
                    columns = []
                    rows_count = 0
                    if hasattr(hdu, "columns") and hdu.columns is not None:
                        columns = list(hdu.columns.names)
                        rows_count = len(hdu.data) if hdu.data is not None else 0
                    rows.append(
                        {
                            "date": date_match,
                            "file_path": str(path),
                            "hdu_index": idx,
                            "hdu_name": hdu.name,
                            "rows": rows_count,
                            "columns": ";".join(columns),
                            "has_energy_columns": any("ENER" in str(c).upper() or "E_" in str(c).upper() for c in columns),
                            "has_multiple_count_columns": sum("COUNT" in str(c).upper() or "RATE" in str(c).upper() for c in columns) > 1,
                            "readable": True,
                            "error": "",
                        }
                    )
        except Exception as exc:
            rows.append(
                {
                    "date": date_match,
                    "file_path": str(path),
                    "hdu_index": "",
                    "hdu_name": "",
                    "rows": 0,
                    "columns": "",
                    "has_energy_columns": False,
                    "has_multiple_count_columns": False,
                    "readable": False,
                    "error": str(exc),
                }
            )
    inventory = pd.DataFrame(rows)
    inventory.to_csv(SPECTRAL_DIR / "solexs_channel_inventory.csv", index=False)
    has_spectral = bool(inventory["has_energy_columns"].any() or inventory["has_multiple_count_columns"].any()) if not inventory.empty else False
    conclusion = (
        "Potential multi-channel/spectral columns were found; targeted feature extraction should be prototyped in `space_agency_experiment` before model use."
        if has_spectral
        else "Current downloaded SoLEXS L1 light-curve files appear to expose summed/light-curve level columns only, so spectral-temperature features remain future work for this dataset."
    )
    report = [
        "# SoLEXS Spectral Feasibility Report",
        "",
        f"- SoLEXS files inspected: {len(set(inventory['file_path'])) if not inventory.empty else 0}",
        f"- HDU/table rows inspected: {len(inventory)}",
        f"- Energy/multi-count evidence found: {'yes' if has_spectral else 'no'}",
        "",
        conclusion,
        "",
        "This check does not change any model inputs or frozen metrics.",
    ]
    (SPECTRAL_DIR / "SOLEXS_SPECTRAL_FEASIBILITY_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return inventory


def hel1os_burst_statistics() -> pd.DataFrame:
    combined_dir = EXPANDED_DIR / "combined_lightcurves"
    rows = []
    for csv_path in sorted(combined_dir.glob("*_combined_lightcurves_long.csv")):
        date = csv_path.name[:8]
        try:
            df = pd.read_csv(csv_path, usecols=["time_utc", "instrument", "detector", "band", "count_rate"])
        except Exception as exc:
            rows.append({"date": date, "detector_group": "ERROR", "error": str(exc)})
            continue
        df = df[df["instrument"].astype(str).str.upper().eq("HEL1OS")].copy()
        if df.empty:
            continue
        df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, format="mixed", errors="coerce")
        df["count_rate"] = pd.to_numeric(df["count_rate"], errors="coerce")
        df["detector_group"] = np.where(df["detector"].astype(str).str.upper().str.contains("CZT"), "CZT", "CdTe")
        for group, part in df.dropna(subset=["time_utc"]).groupby("detector_group"):
            series = part.groupby("time_utc")["count_rate"].mean().sort_index()
            if len(series) < 10:
                continue
            baseline = series.rolling("5min", min_periods=10).median()
            resid = series - baseline
            mad = resid.abs().rolling("5min", min_periods=10).median().replace(0, np.nan)
            z = 0.6745 * resid / mad
            spikes = z > 6
            spike_times = series.index[spikes.fillna(False)]
            gaps = pd.Series(spike_times).diff().dt.total_seconds().dropna() if len(spike_times) else pd.Series(dtype=float)
            duration_hours = max((series.index.max() - series.index.min()).total_seconds() / 3600, 1 / 3600)
            rows.append(
                {
                    "date": date,
                    "detector_group": group,
                    "samples": len(series),
                    "duration_hours": duration_hours,
                    "burst_count_1min_proxy": int(spikes.rolling("60s").sum().max()) if len(spikes) else 0,
                    "burst_count_3min_proxy": int(spikes.rolling("180s").sum().max()) if len(spikes) else 0,
                    "burst_rate_5min_max": float(spikes.rolling("300s").sum().max() / 5) if len(spikes) else 0.0,
                    "peak_impulse_amplitude": float(resid.max()) if len(resid) else np.nan,
                    "inter_burst_gap_mean_sec": float(gaps.mean()) if len(gaps) else np.nan,
                    "inter_burst_gap_std_sec": float(gaps.std()) if len(gaps) else np.nan,
                    "hard_spike_density_per_hour": float(spikes.sum() / duration_hours),
                    "error": "",
                }
            )
    stats = pd.DataFrame(rows)
    if not stats.empty:
        pivot = stats.pivot_table(index="date", columns="detector_group", values="hard_spike_density_per_hour", aggfunc="mean")
        stats = stats.merge(
            pivot.assign(
                czt_spike_fraction=lambda x: x.get("CZT", 0) / (x.get("CZT", 0) + x.get("CdTe", 0) + 1e-9),
                czt_cdte_spike_ratio=lambda x: x.get("CZT", 0) / (x.get("CdTe", 0) + 1e-9),
            )[["czt_spike_fraction", "czt_cdte_spike_ratio"]].reset_index(),
            on="date",
            how="left",
        )
    stats.to_csv(BURST_DIR / "hel1os_burst_feature_summary.csv", index=False)
    report = [
        "# HEL1OS Burst / Spike Feature Report",
        "",
        "This report extracts causal, rolling-window burst/spike diagnostics from expanded candidate light curves. These are research features only and are not used to change frozen v8.1/v6 metrics.",
        "",
        f"- Date-detector rows summarized: {len(stats)}",
        f"- Dates covered: {stats['date'].nunique() if not stats.empty else 0}",
        "",
        "The hard oscillatory/spike proxies should be described as burst-count and short-timescale variability proxies, not validated QPP detection.",
    ]
    (BURST_DIR / "HEL1OS_BURST_FEATURE_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    return stats


def method_triggers(df: pd.DataFrame, method: str) -> pd.Series:
    soft = pd.to_numeric(df.get("soft_peak_counts", pd.Series(index=df.index, dtype=float)), errors="coerce").fillna(0) >= 500
    hard = pd.to_numeric(df.get("max_hard_score", pd.Series(index=df.index, dtype=float)), errors="coerce").fillna(0) >= 50
    if method == "threshold_soft_only":
        return soft
    if method == "hel1os_hard_only":
        return hard
    return soft | hard


def fusion_baseline_comparison(approved_flare: pd.DataFrame, approved_quiet: pd.DataFrame) -> pd.DataFrame:
    quiet_dates = sorted(approved_quiet["date"].astype(str).unique())
    quiet_raw_frames = []
    for date in quiet_dates:
        raw_path = EXPANDED_DIR / f"{date}_nowcast_catalogue.csv"
        if raw_path.exists():
            raw = pd.read_csv(raw_path)
            if not raw.empty:
                raw["date"] = date
                quiet_raw_frames.append(raw)
    quiet_raw = pd.concat(quiet_raw_frames, ignore_index=True) if quiet_raw_frames else pd.DataFrame()

    rows = []
    methods = [
        ("threshold_soft_only", "SoLEXS-only", "Soft X-ray thermal threshold from existing nowcast fields."),
        ("hel1os_hard_only", "HEL1OS-only", "Hard X-ray impulsive threshold from existing nowcast fields."),
        ("soft_hard_fusion", "SoLEXS + HEL1OS fusion", "Existing unified clean-catalogue logic: soft OR hard evidence."),
    ]
    total_events = len(approved_flare)
    quiet_days = max(len(quiet_dates), 1)
    for method, label, notes in methods:
        inputs = {
            "threshold_soft_only": "soft only",
            "hel1os_hard_only": "hard only",
            "soft_hard_fusion": "soft + hard",
        }[method]
        detected = int(method_triggers(approved_flare, method).sum()) if not approved_flare.empty else 0
        false_alerts = int(method_triggers(quiet_raw, method).sum()) if not quiet_raw.empty else 0
        precision = detected / (detected + false_alerts) if detected + false_alerts else np.nan
        recall = detected / total_events if total_events else np.nan
        f1 = 2 * precision * recall / (precision + recall) if precision == precision and recall == recall and precision + recall else np.nan
        triggered_events = approved_flare[method_triggers(approved_flare, method)].copy() if not approved_flare.empty else pd.DataFrame()
        lead = pd.to_numeric(triggered_events.get("lead_time_min", pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "method": label,
                "inputs": inputs,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "far_per_day": false_alerts / quiet_days,
                "mean_lead_time_min": float(lead.mean()) if lead.notna().any() else np.nan,
                "median_lead_time_min": float(lead.median()) if lead.notna().any() else np.nan,
                "valid_alerted_events": detected,
                "total_approved_flare_events": total_events,
                "quiet_day_false_alerts": false_alerts,
                "approved_quiet_days_used": len(quiet_dates),
                "notes": notes + " Diagnostic only; not a retrained model.",
            }
        )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(FUSION_DIR / "soft_hard_fusion_baseline_comparison.csv", index=False)
    md = [
        "# Soft-Only vs Hard-Only vs Fusion Baseline Comparison",
        "",
        "This diagnostic uses approved expanded candidate flare events and approved quiet days. It does not retrain any model and does not replace frozen v8.1/v6 metrics.",
        "",
        markdown_table(comparison),
        "",
        "The purpose is to answer why Aditya-L1 fusion matters: soft-only captures thermal response, hard-only captures impulsive evidence, and the unified system combines both.",
    ]
    (FUSION_DIR / "soft_hard_fusion_baseline_comparison.md").write_text("\n".join(md), encoding="utf-8")
    return comparison


def instrument_disagreement_catalogue(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    soft = pd.to_numeric(out.get("soft_peak_counts", pd.Series(index=out.index, dtype=float)), errors="coerce").fillna(0) >= 500
    hard = pd.to_numeric(out.get("max_hard_score", pd.Series(index=out.index, dtype=float)), errors="coerce").fillna(0) >= 50
    lead = pd.to_numeric(out.get("lead_time_min", pd.Series(index=out.index, dtype=float)), errors="coerce")
    labels = []
    for s, h, l, status in zip(soft, hard, lead, out.get("goes_match_status", pd.Series([""] * len(out)))):
        if status in {"NO_MATCH", "NEAREST_ONLY"}:
            labels.append("NO_MATCH")
        elif s and h and pd.notna(l) and l >= 1:
            labels.append("HARD_LEADS_SOFT")
        elif s and h and pd.notna(l) and l < 0:
            labels.append("SOFT_BEFORE_HARD")
        elif s and h:
            labels.append("SOFT_HARD_CONFIRMED")
        elif h:
            labels.append("HARD_ONLY")
        elif s:
            labels.append("SOFT_ONLY")
        else:
            labels.append("NO_MATCH")
    out["instrument_disagreement_class"] = labels
    cols = [
        "candidate_global_event_id",
        "date",
        "event_id",
        "goes_match_status",
        "goes_class",
        "soft_peak_counts",
        "max_hard_score",
        "max_soft_score",
        "lead_time_min",
        "instrument_disagreement_class",
    ]
    out.to_csv(DISAGREEMENT_DIR / "instrument_disagreement_catalogue.csv", index=False)
    counts = out["instrument_disagreement_class"].value_counts().reset_index(name="count").rename(columns={"index": "instrument_disagreement_class"})
    md = [
        "# Instrument Disagreement Report",
        "",
        "This catalogue classifies candidate events by whether SoLEXS soft X-ray evidence, HEL1OS hard X-ray evidence, or both are present. It is a research diagnostic and does not alter nowcast semantics.",
        "",
        markdown_table(counts),
    ]
    (DISAGREEMENT_DIR / "instrument_disagreement_report.md").write_text("\n".join(md), encoding="utf-8")
    return out


def main() -> None:
    ensure_dirs()
    create_lightweight_backup()
    master, quiet, availability = require_inputs()
    approved_flare, approved_quiet, manual = review_expanded_data(master, quiet, availability)
    holdout = create_shadow_holdout(approved_flare, approved_quiet)
    spectral = inspect_solexs_spectral()
    burst = hel1os_burst_statistics()
    fusion = fusion_baseline_comparison(approved_flare, approved_quiet)
    disagreement = instrument_disagreement_catalogue(master)

    print(f"backup path: {BACKUP_DIR}")
    print(f"experiment workspace: {EXPERIMENT_DIR}")
    print(f"approved flare events: {len(approved_flare)}")
    print(f"approved quiet days: {len(approved_quiet)}")
    print(f"manual/review events: {len(manual)}")
    print(f"shadow holdout dates: {', '.join(holdout['date'].astype(str)) if not holdout.empty else 'none'}")
    print(f"solexs spectral evidence rows: {len(spectral)}")
    print(f"hel1os burst summary rows: {len(burst)}")
    print(f"fusion baseline rows: {len(fusion)}")
    print(f"instrument disagreement rows: {len(disagreement)}")
    print(f"review report: {REVIEW_DIR / 'EXPANDED_80DAY_REVIEW_REPORT.md'}")


if __name__ == "__main__":
    main()
