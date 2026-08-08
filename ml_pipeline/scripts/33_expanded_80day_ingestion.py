from __future__ import annotations

import math
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.extract import safe_extract_zip
from src.io.read_lightcurves import (
    candidate_quality,
    coverage_seconds,
    date_window,
    read_hel1os_lc,
    read_solexs_lc,
    select_best_lightcurve_files,
)
from src.processing.nowcast import clean_nowcast_catalogue, run_nowcast


ZIPS_DIR = PROJECT_ROOT / "data" / "zips"
RAW_EXPANDED_DIR = PROJECT_ROOT / "data" / "raw_expanded_80day"
SOLEXS_RAW_DIR = RAW_EXPANDED_DIR / "solexs"
HEL1OS_RAW_DIR = RAW_EXPANDED_DIR / "hel1os"
UNKNOWN_RAW_DIR = RAW_EXPANDED_DIR / "unknown"
RESULTS_DIR = PROJECT_ROOT / "results" / "expanded_80day_ingestion"
COMBINED_DIR = RESULTS_DIR / "combined_lightcurves"
GOES_PATH = PROJECT_ROOT / "data" / "external" / "goes_flare_events.csv"

TARGET_RANGES = [
    ("2026-02-01", "2026-02-06"),
    ("2026-02-22", "2026-02-28"),
    ("2026-03-01", "2026-03-15"),
    ("2026-06-03", "2026-06-14"),
    ("2026-01-15", "2026-01-31"),
    ("2026-03-16", "2026-03-31"),
    ("2026-04-01", "2026-04-07"),
]
DATE_RE = re.compile(r"20\d{6}")
SCIENCE_SUFFIXES = {".fits", ".fit", ".fts", ".lc", ".pha", ".csv", ".txt", ".gz"}


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
        values = [str(row[col]).replace("\n", " ") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def ensure_dirs() -> None:
    for path in [RESULTS_DIR, COMBINED_DIR, SOLEXS_RAW_DIR, HEL1OS_RAW_DIR, UNKNOWN_RAW_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def target_dates() -> list[str]:
    dates: list[str] = []
    for start, end in TARGET_RANGES:
        for ts in pd.date_range(start, end, freq="D", tz="UTC"):
            dates.append(ts.strftime("%Y%m%d"))
    return sorted(set(dates))


def guess_payload(text: str) -> str:
    upper = text.upper()
    if "SOLEXS" in upper:
        return "SoLEXS"
    if "HEL1OS" in upper or "HLS_" in upper or "HLS-" in upper:
        return "HEL1OS"
    return "UNKNOWN"


def zip_member_summary(zip_path: Path) -> tuple[list[str], bool, str]:
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
        dates = sorted(set(DATE_RE.findall(" ".join(names[:500]))))
        nested = any(name.lower().endswith(".zip") for name in names)
        payload = guess_payload(zip_path.name + " " + " ".join(names[:50]))
        return dates, nested, payload
    except Exception:
        return [], False, guess_payload(zip_path.name)


def scan_zip_inventory() -> pd.DataFrame:
    targets = set(target_dates())
    rows = []
    for zip_path in sorted(ZIPS_DIR.glob("*.zip")):
        member_dates, nested, member_payload = zip_member_summary(zip_path)
        name_dates = sorted(set(DATE_RE.findall(zip_path.name)))
        all_dates = sorted(set(name_dates) | set(member_dates))
        payload = member_payload if member_payload != "UNKNOWN" else guess_payload(zip_path.name)
        intersects = any(d in targets for d in all_dates)
        rows.append(
            {
                "zip_path": str(zip_path),
                "file_name": zip_path.name,
                "size_bytes": zip_path.stat().st_size,
                "guessed_payload": payload,
                "guessed_date_range": f"{min(all_dates)}..{max(all_dates)}" if all_dates else "",
                "guessed_dates": ";".join(all_dates),
                "nested_zip": nested,
                "within_requested_ranges": intersects or not all_dates,
            }
        )
    inventory = pd.DataFrame(rows)
    inventory.to_csv(RESULTS_DIR / "zip_inventory.csv", index=False)
    return inventory


def destination_for_zip(zip_path: Path, payload: str) -> Path:
    if payload == "SoLEXS":
        base = SOLEXS_RAW_DIR
    elif payload == "HEL1OS":
        base = HEL1OS_RAW_DIR
    else:
        base = UNKNOWN_RAW_DIR
    return base / zip_path.stem


def extract_relevant_zips(inventory: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rec in inventory.to_dict("records"):
        if not bool(rec.get("within_requested_ranges", False)):
            rows.append({**rec, "destination": "", "status": "skipped_not_in_requested_ranges", "error": ""})
            continue
        zip_path = Path(rec["zip_path"])
        dest = destination_for_zip(zip_path, str(rec["guessed_payload"]))
        status = "extracted"
        error = ""
        try:
            if dest.exists() and any(dest.iterdir()):
                status = "already_exists_skipped"
            else:
                safe_extract_zip(zip_path, dest)
        except Exception as exc:
            status = "extract_failed"
            error = str(exc)
        rows.append({**rec, "destination": str(dest), "status": status, "error": error})

    while True:
        pending = []
        for nested in RAW_EXPANDED_DIR.rglob("*.zip"):
            out_dir = nested.with_suffix("")
            if not out_dir.exists() or not any(out_dir.iterdir()):
                pending.append((nested, out_dir))
        if not pending:
            break
        for nested, out_dir in pending:
            status = "nested_extracted"
            error = ""
            try:
                safe_extract_zip(nested, out_dir)
            except Exception as exc:
                status = "nested_extract_failed"
                error = str(exc)
            rows.append(
                {
                    "zip_path": str(nested),
                    "file_name": nested.name,
                    "size_bytes": nested.stat().st_size,
                    "guessed_payload": guess_payload(str(nested)),
                    "guessed_date_range": "",
                    "guessed_dates": "",
                    "nested_zip": True,
                    "within_requested_ranges": True,
                    "destination": str(out_dir),
                    "status": status,
                    "error": error,
                }
            )

    manifest = pd.DataFrame(rows)
    manifest.to_csv(RESULTS_DIR / "extraction_manifest.csv", index=False)
    return manifest


def write_raw_tree() -> None:
    lines = [str(RAW_EXPANDED_DIR)]
    for path in sorted(RAW_EXPANDED_DIR.rglob("*")):
        depth = len(path.relative_to(RAW_EXPANDED_DIR).parts)
        lines.append(f"{'  ' * depth}{path.name}")
    (RESULTS_DIR / "raw_tree.txt").write_text("\n".join(lines), encoding="utf-8")


def detect_date(path: Path) -> str:
    match = DATE_RE.search(str(path))
    return match.group(0) if match else ""


def inspect_science_file(path: Path) -> dict:
    payload = guess_payload(str(path))
    ext = "".join(path.suffixes[-2:]) if path.name.lower().endswith(".lc.gz") else path.suffix.lower()
    row = {
        "file_path": str(path),
        "payload": payload,
        "detected_date": detect_date(path),
        "extension": ext,
        "readable": False,
        "rows": 0,
        "time_range": "",
        "columns": "",
        "errors": "",
    }
    try:
        if path.name.startswith("AL1_SOLEXS") and path.name.endswith("_L1.lc.gz"):
            df = read_solexs_lc(path)
        elif path.name.lower().startswith("lightcurve_") and path.suffix.lower() == ".fits":
            df = read_hel1os_lc(path)
        elif path.suffix.lower() in {".csv", ".txt"}:
            df = pd.read_csv(path, nrows=10000)
        else:
            from astropy.io import fits

            with fits.open(path) as hdul:
                table = next((hdu for hdu in hdul if getattr(hdu, "data", None) is not None and hasattr(hdu, "columns")), None)
                if table is None:
                    raise ValueError("no readable table HDU found")
                data = table.data
                df = pd.DataFrame(np.asarray(data).byteswap().newbyteorder() if hasattr(np.asarray(data), "byteswap") else data)

        row["readable"] = True
        row["rows"] = int(len(df))
        row["columns"] = ";".join(map(str, df.columns[:60]))
        if "time_utc" in df.columns:
            times = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
            if times.notna().any():
                row["time_range"] = f"{times.min().isoformat()}..{times.max().isoformat()}"
    except Exception as exc:
        row["errors"] = str(exc)
    return row


def inspect_files() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW_EXPANDED_DIR.rglob("*")):
        if not path.is_file():
            continue
        suffixes = {s.lower() for s in path.suffixes}
        if suffixes & SCIENCE_SUFFIXES or path.name.lower().endswith(".lc.gz"):
            rows.append(inspect_science_file(path))
    report = pd.DataFrame(rows)
    report.to_csv(RESULTS_DIR / "file_inspection_report.csv", index=False)
    return report


def files_for_date(date: str) -> tuple[list[Path], list[Path]]:
    solexs = sorted(SOLEXS_RAW_DIR.rglob(f"AL1_SOLEXS_{date}_*_L1.lc.gz"))
    hel1os: list[Path] = []
    for detector in ["cdte1", "cdte2", "czt1", "czt2"]:
        hel1os += sorted(HEL1OS_RAW_DIR.rglob(f"HLS_{date}*/**/lightcurve_{detector}.fits"))
    return solexs, sorted(set(hel1os))


def quality_label(duration_hours: float, soft_finite: float, hard_finite: float, has_both: bool) -> str:
    if not has_both:
        return "PAYLOAD_MISSING"
    if duration_hours >= 20 and soft_finite >= 80 and hard_finite >= 80:
        return "GOOD"
    if duration_hours < 4 or soft_finite < 20 or hard_finite < 20:
        return "BROKEN"
    return "PARTIAL"


def build_for_date(date: str, solexs_files: list[Path], hel1os_files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_solexs, selected_hel1os, duplicate_report = select_best_lightcurve_files(date, solexs_files, hel1os_files)
    if not selected_solexs or not selected_hel1os:
        return pd.DataFrame(), duplicate_report
    frames = [read_solexs_lc(p) for p in selected_solexs]
    for path in selected_hel1os:
        df = read_hel1os_lc(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(), duplicate_report
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["time_utc", "count_rate"]).sort_values("time_utc")
    start, end = date_window(date)
    out = out[(out["time_utc"] >= start) & (out["time_utc"] < end)]
    return out, duplicate_report


def series_from_long(df: pd.DataFrame, instrument: str, band_contains: str) -> pd.Series:
    temp = df.copy()
    temp["time_utc"] = pd.to_datetime(temp["time_utc"], utc=True, format="mixed", errors="coerce")
    temp["count_rate"] = pd.to_numeric(temp["count_rate"], errors="coerce")
    mask = temp["instrument"].astype(str).str.upper().eq(instrument.upper())
    mask &= temp["band"].astype(str).str.contains(band_contains, case=False, na=False)
    series = temp.loc[mask].dropna(subset=["time_utc"]).set_index("time_utc")["count_rate"].sort_index()
    if series.index.has_duplicates:
        series = series.groupby(level=0).mean()
    return series


def finite_percent(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    values = pd.to_numeric(series, errors="coerce")
    return float(np.isfinite(values).sum() / len(values) * 100.0)


def combined_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "duration_hours": 0.0,
            "solexs_coverage": "",
            "hel1os_coverage": "",
            "soft_finite": 0.0,
            "hard_finite": 0.0,
        }
    times = pd.to_datetime(df["time_utc"], utc=True, errors="coerce")
    start = times.min()
    end = times.max()
    duration = (end - start).total_seconds() / 3600 if pd.notna(start) and pd.notna(end) else 0.0
    soft = series_from_long(df, "SoLEXS", "2-22")
    cdte = series_from_long(df, "HEL1OS", "5-20")
    czt = series_from_long(df, "HEL1OS", "20-40")
    soft_fin = finite_percent(soft)
    hard_fin = max(finite_percent(cdte), finite_percent(czt))
    return {
        "duration_hours": duration,
        "solexs_coverage": f"{soft.index.min().isoformat()}..{soft.index.max().isoformat()}" if len(soft) else "",
        "hel1os_coverage": f"{min([s.index.min() for s in [cdte, czt] if len(s)]).isoformat()}..{max([s.index.max() for s in [cdte, czt] if len(s)]).isoformat()}" if len(cdte) or len(czt) else "",
        "soft_finite": soft_fin,
        "hard_finite": hard_fin,
    }


def make_quicklook(df: pd.DataFrame, date: str, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt

        temp = df.copy()
        temp["time_utc"] = pd.to_datetime(temp["time_utc"], utc=True, errors="coerce")
        temp["count_rate"] = pd.to_numeric(temp["count_rate"], errors="coerce")
        temp["label"] = temp["instrument"].astype(str) + " " + temp["detector"].astype(str) + " " + temp["band"].astype(str)
        fig, ax = plt.subplots(figsize=(12, 5))
        for label, group in temp.groupby("label"):
            ax.plot(group["time_utc"], group["count_rate"], lw=0.8, label=label)
        ax.set_title(f"Expanded candidate light curves {date}")
        ax.set_xlabel("UTC")
        ax.set_ylabel("Count rate")
        ax.legend(fontsize=7, loc="best")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[WARN] quicklook skipped for {date}: {exc}")


def build_availability_and_lightcurves() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    availability_rows = []
    duplicate_rows = []
    processed_dates: list[str] = []

    for date in target_dates():
        solexs_files, hel1os_files = files_for_date(date)
        has_both = bool(solexs_files and hel1os_files)
        long_df = pd.DataFrame()
        duplicate_report = pd.DataFrame()
        error = ""
        csv_path = COMBINED_DIR / f"{date}_combined_lightcurves_long.csv"
        if has_both:
            try:
                if csv_path.exists():
                    long_df = pd.read_csv(csv_path)
                else:
                    long_df, duplicate_report = build_for_date(date, solexs_files, hel1os_files)
            except Exception as exc:
                error = str(exc)
        if not duplicate_report.empty:
            duplicate_rows.append(duplicate_report)
        metrics = combined_metrics(long_df)
        label = quality_label(metrics["duration_hours"], metrics["soft_finite"], metrics["hard_finite"], has_both and not long_df.empty)
        notes = error
        if not has_both:
            missing = []
            if not solexs_files:
                missing.append("SoLEXS")
            if not hel1os_files:
                missing.append("HEL1OS")
            notes = "missing " + " and ".join(missing)
        elif long_df.empty:
            notes = notes or "matched files found but combined lightcurve is empty/unreadable"

        availability_rows.append(
            {
                "date": date,
                "solexs_available": bool(solexs_files),
                "hel1os_available": bool(hel1os_files),
                "solexs_coverage": metrics["solexs_coverage"],
                "hel1os_coverage": metrics["hel1os_coverage"],
                "quality": label,
                "notes": notes,
            }
        )

        if label in {"GOOD", "PARTIAL"} and not long_df.empty:
            png_path = COMBINED_DIR / f"{date}_solexs_hel1os_quicklook.png"
            if not csv_path.exists():
                long_df.to_csv(csv_path, index=False)
            if not png_path.exists():
                make_quicklook(long_df, date, png_path)
            processed_dates.append(date)

    availability = pd.DataFrame(availability_rows)
    availability.to_csv(RESULTS_DIR / "date_payload_availability.csv", index=False)
    duplicate_report_all = pd.concat(duplicate_rows, ignore_index=True) if duplicate_rows else pd.DataFrame()
    if not duplicate_report_all.empty:
        duplicate_report_all.to_csv(RESULTS_DIR / "duplicate_file_selection_report.csv", index=False)
    return availability, duplicate_report_all, processed_dates


def run_expanded_nowcast(processed_dates: list[str]) -> pd.DataFrame:
    catalogues = []
    summary_rows = []
    for date in processed_dates:
        input_csv = COMBINED_DIR / f"{date}_combined_lightcurves_long.csv"
        try:
            raw_path = RESULTS_DIR / f"{date}_nowcast_catalogue.csv"
            scored_path = RESULTS_DIR / f"{date}_scored_timeseries.csv"
            clean_path = RESULTS_DIR / f"{date}_nowcast_catalogue_clean.csv"
            if raw_path.exists() and scored_path.exists() and clean_path.exists():
                raw_cat = pd.read_csv(raw_path)
                clean = pd.read_csv(clean_path)
            else:
                raw_cat, scored = run_nowcast(input_csv=input_csv, out_dir=RESULTS_DIR, date=date)
                clean = clean_nowcast_catalogue(
                    input_path=raw_path,
                    output_path=clean_path,
                )
            clean = clean.copy()
            clean.insert(0, "date", date)
            if not clean.empty:
                catalogues.append(clean)
            summary_rows.append(
                {
                    "date": date,
                    "raw_candidates": len(raw_cat),
                    "cleaned_events": len(clean),
                    "solexs_triggered_candidates": int((raw_cat.get("max_soft_score", pd.Series(dtype=float)) > 8).sum()) if not raw_cat.empty else 0,
                    "hel1os_triggered_candidates": int((raw_cat.get("max_hard_score", pd.Series(dtype=float)) > 8).sum()) if not raw_cat.empty else 0,
                    "status": "processed",
                    "error": "",
                }
            )
        except Exception as exc:
            summary_rows.append(
                {
                    "date": date,
                    "raw_candidates": 0,
                    "cleaned_events": 0,
                    "solexs_triggered_candidates": 0,
                    "hel1os_triggered_candidates": 0,
                    "status": "failed",
                    "error": str(exc),
                }
            )
    combined = pd.concat(catalogues, ignore_index=True) if catalogues else pd.DataFrame()
    if not combined.empty:
        combined.insert(0, "candidate_global_event_id", range(1, len(combined) + 1))
    combined.to_csv(RESULTS_DIR / "expanded_nowcast_catalogue_CANDIDATE.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "expanded_nowcast_summary.csv", index=False)
    md = ["# Expanded Nowcast Summary", ""]
    md.append(f"- Processed dates: {len(processed_dates)}")
    md.append(f"- Cleaned candidate events: {len(combined)}")
    md.append("")
    if not summary.empty:
        md.append(markdown_table(summary))
    (RESULTS_DIR / "expanded_nowcast_summary.md").write_text("\n".join(md), encoding="utf-8")
    return combined


def load_goes() -> pd.DataFrame:
    if not GOES_PATH.exists():
        return pd.DataFrame()
    goes = pd.read_csv(GOES_PATH)
    for col in ["start_time_utc", "peak_time_utc", "end_time_utc"]:
        if col in goes.columns:
            goes[col] = pd.to_datetime(goes[col], utc=True, errors="coerce")
    if "date" not in goes.columns and "peak_time_utc" in goes.columns:
        goes["date"] = goes["peak_time_utc"].dt.strftime("%Y%m%d")
    return goes


def overlap(a_start: pd.Timestamp, a_end: pd.Timestamp, b_start: pd.Timestamp, b_end: pd.Timestamp) -> bool:
    if pd.isna(a_start) or pd.isna(a_end) or pd.isna(b_start) or pd.isna(b_end):
        return False
    return max(a_start, b_start) <= min(a_end, b_end)


def validate_goes(events: pd.DataFrame) -> pd.DataFrame:
    goes = load_goes()
    rows = []
    if events.empty:
        out = pd.DataFrame()
        out.to_csv(RESULTS_DIR / "expanded_goes_validation.csv", index=False)
        return out
    events = events.copy()
    for col in ["event_start", "event_end", "soft_peak_time"]:
        events[col] = pd.to_datetime(events[col], utc=True, errors="coerce")
    for _, event in events.iterrows():
        same_date = goes[goes.get("date", pd.Series(dtype=str)).astype(str).eq(str(event["date"]))] if not goes.empty else pd.DataFrame()
        status = "NO_MATCH"
        matched = None
        nearest_delta = np.nan
        if not same_date.empty and "peak_time_utc" in same_date.columns:
            deltas = (same_date["peak_time_utc"] - event["soft_peak_time"]).abs()
            nearest_idx = deltas.idxmin()
            nearest_delta = deltas.loc[nearest_idx].total_seconds() / 60 if pd.notna(deltas.loc[nearest_idx]) else np.nan
            nearest = same_date.loc[nearest_idx]
            peak_match = pd.notna(event["soft_peak_time"]) and nearest_delta <= 10
            window_matches = same_date.apply(
                lambda g: overlap(event["event_start"], event["event_end"], g.get("start_time_utc"), g.get("end_time_utc")),
                axis=1,
            )
            if peak_match:
                status = "EXACT_PEAK_MATCH"
                matched = nearest
            elif window_matches.any():
                status = "WINDOW_OVERLAP_MATCH"
                matched = same_date.loc[window_matches.idxmax()]
            else:
                status = "NEAREST_ONLY"
                matched = nearest
        rows.append(
            {
                "candidate_global_event_id": event.get("candidate_global_event_id", ""),
                "date": event["date"],
                "surya_event_id": event.get("event_id", ""),
                "surya_event_start": event["event_start"],
                "surya_event_end": event["event_end"],
                "surya_soft_peak_time": event["soft_peak_time"],
                "goes_match_status": status,
                "goes_event_id": matched.get("event_id", "") if matched is not None else "",
                "goes_peak_time_utc": matched.get("peak_time_utc", pd.NaT) if matched is not None else pd.NaT,
                "goes_start_time_utc": matched.get("start_time_utc", pd.NaT) if matched is not None else pd.NaT,
                "goes_end_time_utc": matched.get("end_time_utc", pd.NaT) if matched is not None else pd.NaT,
                "goes_class": matched.get("goes_class", "") if matched is not None else "",
                "active_region": matched.get("active_region", "") if matched is not None else "",
                "peak_time_difference_min": nearest_delta,
            }
        )
    validation = pd.DataFrame(rows)
    validation.to_csv(RESULTS_DIR / "expanded_goes_validation.csv", index=False)
    return validation


def quiet_day_validation(availability: pd.DataFrame, events: pd.DataFrame, goes: pd.DataFrame) -> pd.DataFrame:
    event_counts = events.groupby("date").size().to_dict() if not events.empty else {}
    goes_dates = set(goes.get("date", pd.Series(dtype=str)).dropna().astype(str)) if not goes.empty else set()
    rows = []
    for _, row in availability.iterrows():
        date = str(row["date"])
        quality = row["quality"]
        count = int(event_counts.get(date, 0))
        has_goes = date in goes_dates
        if quality == "PAYLOAD_MISSING":
            classification = "PAYLOAD_MISSING"
        elif quality == "BROKEN":
            classification = "BROKEN"
        elif count > 0:
            classification = "GOOD_FLARE" if quality == "GOOD" else "QUESTIONABLE"
        elif quality == "GOOD" and not has_goes:
            classification = "GOOD_QUIET"
        elif quality == "GOOD" and has_goes:
            classification = "QUESTIONABLE"
        else:
            classification = "QUESTIONABLE"
        rows.append(
            {
                "date": date,
                "payload_quality": quality,
                "cleaned_event_count": count,
                "goes_events_on_date": bool(has_goes),
                "date_classification": classification,
                "notes": "GOES/SWPC labels used only for validation/context; no manual quiet labels assumed.",
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "expanded_quiet_day_validation.csv", index=False)
    return out


def create_master_candidate(events: pd.DataFrame, goes_validation: pd.DataFrame, quiet: pd.DataFrame) -> None:
    if events.empty:
        master = pd.DataFrame()
    else:
        master = events.merge(
            goes_validation[
                [
                    "candidate_global_event_id",
                    "goes_match_status",
                    "goes_event_id",
                    "goes_class",
                    "active_region",
                    "peak_time_difference_min",
                ]
            ],
            on="candidate_global_event_id",
            how="left",
        )
    master.to_csv(RESULTS_DIR / "expanded_master_flare_catalogue_CANDIDATE.csv", index=False)

    old_count = 0
    for path in [
        PROJECT_ROOT / "results" / "master_flare_catalogue_classified_v2.csv",
        PROJECT_ROOT / "results" / "master_flare_catalogue.csv",
    ]:
        if path.exists():
            old_count = len(pd.read_csv(path))
            break
    matched = int(master.get("goes_match_status", pd.Series(dtype=str)).isin(["EXACT_PEAK_MATCH", "WINDOW_OVERLAP_MATCH"]).sum()) if not master.empty else 0
    quiet_days = int((quiet["date_classification"] == "GOOD_QUIET").sum()) if not quiet.empty else 0
    false_on_quiet = 0
    if not quiet.empty and not events.empty:
        quiet_dates = set(quiet.loc[quiet["date_classification"] == "GOOD_QUIET", "date"].astype(str))
        false_on_quiet = int(events["date"].astype(str).isin(quiet_dates).sum())
    skipped = quiet[quiet["date_classification"].isin(["BROKEN", "PAYLOAD_MISSING", "QUESTIONABLE"])] if not quiet.empty else pd.DataFrame()

    md = [
        "# Expanded Master Catalogue Candidate Summary",
        "",
        f"- Existing frozen master event count: {old_count}",
        f"- New candidate cleaned events: {len(master)}",
        f"- Total if merged after review: {old_count + len(master)}",
        f"- GOES/SWPC timing-supported candidate matches: {matched}",
        f"- Confirmed quiet candidate days: {quiet_days}",
        f"- False detections on confirmed quiet candidate days: {false_on_quiet}",
        "",
        "## Excluded / Review Dates",
        "",
    ]
    if skipped.empty:
        md.append("No excluded or questionable dates recorded.")
    else:
        md.append(markdown_table(skipped, ["date", "payload_quality", "date_classification", "notes"]))
    (RESULTS_DIR / "expanded_master_catalogue_summary.md").write_text("\n".join(md), encoding="utf-8")


def write_final_report(
    inventory: pd.DataFrame,
    manifest: pd.DataFrame,
    availability: pd.DataFrame,
    processed_dates: list[str],
    events: pd.DataFrame,
    goes_validation: pd.DataFrame,
    quiet: pd.DataFrame,
) -> str:
    both_dates = availability.loc[availability["solexs_available"] & availability["hel1os_available"], "date"].astype(str).tolist()
    broken_questionable = quiet.loc[quiet["date_classification"].isin(["BROKEN", "QUESTIONABLE"]), "date"].astype(str).tolist() if not quiet.empty else []
    matched_events = int(goes_validation.get("goes_match_status", pd.Series(dtype=str)).isin(["EXACT_PEAK_MATCH", "WINDOW_OVERLAP_MATCH"]).sum()) if not goes_validation.empty else 0
    safe_merge = bool(len(events) > 0 and len(broken_questionable) == 0)
    recommendation = (
        "Candidate outputs are suitable for manual review before merging. Do not retrain until questionable/broken dates are reviewed."
        if not safe_merge
        else "Candidate outputs look internally consistent, but merge/retraining should still be done in a separate explicit step."
    )
    md = [
        "# Expanded 80-Day Ingestion Report",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- ZIPs detected: {len(inventory)}",
        f"- ZIP extraction rows: {len(manifest)}",
        f"- Dates checked: {len(availability)}",
        f"- Dates with both SoLEXS and HEL1OS payloads: {len(both_dates)}",
        f"- Dates processed into combined light curves: {len(processed_dates)}",
        f"- New cleaned candidate events: {len(events)}",
        f"- GOES/SWPC timing-supported candidate events: {matched_events}",
        f"- GOOD quiet candidate days: {int((quiet['date_classification'] == 'GOOD_QUIET').sum()) if not quiet.empty else 0}",
        "",
        "## Dates With Both Payloads",
        "",
        ", ".join(both_dates) if both_dates else "None.",
        "",
        "## Dates Processed",
        "",
        ", ".join(processed_dates) if processed_dates else "None.",
        "",
        "## Date Classifications",
        "",
        markdown_table(quiet["date_classification"].value_counts().reset_index(name="count").rename(columns={"index": "date_classification"}))
        if not quiet.empty
        else "No quiet/date validation rows.",
        "",
        "## Recommendation",
        "",
        recommendation,
        "",
        "This run created candidate outputs only and did not overwrite frozen v8.1/v6 results or merge into the final master catalogue.",
    ]
    path = RESULTS_DIR / "EXPANDED_80DAY_INGESTION_REPORT.md"
    path.write_text("\n".join(md), encoding="utf-8")
    return recommendation


def main() -> None:
    ensure_dirs()
    inventory_path = RESULTS_DIR / "zip_inventory.csv"
    extraction_path = RESULTS_DIR / "extraction_manifest.csv"
    inspection_path = RESULTS_DIR / "file_inspection_report.csv"

    inventory = pd.read_csv(inventory_path) if inventory_path.exists() else scan_zip_inventory()
    manifest = (
        pd.read_csv(extraction_path)
        if extraction_path.exists() and RAW_EXPANDED_DIR.exists() and any(RAW_EXPANDED_DIR.iterdir())
        else extract_relevant_zips(inventory)
    )
    if not (RESULTS_DIR / "raw_tree.txt").exists():
        write_raw_tree()
    if not inspection_path.exists():
        inspect_files()
    availability, _, processed_dates = build_availability_and_lightcurves()
    events = run_expanded_nowcast(processed_dates)
    goes = load_goes()
    goes_validation = validate_goes(events)
    quiet = quiet_day_validation(availability, events, goes)
    create_master_candidate(events, goes_validation, quiet)
    recommendation = write_final_report(inventory, manifest, availability, processed_dates, events, goes_validation, quiet)

    both_dates = availability.loc[availability["solexs_available"] & availability["hel1os_available"], "date"].astype(str).tolist()
    new_quiet = int((quiet["date_classification"] == "GOOD_QUIET").sum()) if not quiet.empty else 0
    broken_questionable = quiet.loc[quiet["date_classification"].isin(["BROKEN", "QUESTIONABLE"]), "date"].astype(str).tolist() if not quiet.empty else []
    extracted = int(manifest["status"].astype(str).str.contains("extracted").sum()) if "status" in manifest.columns else 0

    print(f"total ZIPs detected: {len(inventory)}")
    print(f"total ZIPs extracted: {extracted}")
    print(f"total dates checked: {len(availability)}")
    print(f"dates with both SoLEXS and HEL1OS: {', '.join(both_dates) if both_dates else 'none'}")
    print(f"dates processed: {', '.join(processed_dates) if processed_dates else 'none'}")
    print(f"new flare events found: {len(events)}")
    print(f"new quiet days confirmed: {new_quiet}")
    print(f"broken/questionable dates: {', '.join(broken_questionable) if broken_questionable else 'none'}")
    print(f"report path: {RESULTS_DIR / 'EXPANDED_80DAY_INGESTION_REPORT.md'}")
    print(f"safe to proceed or not: {recommendation}")


if __name__ == "__main__":
    main()
