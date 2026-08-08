from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from astropy.io import fits


RAW_DIR = Path("data/raw")
RESULTS_DIR = Path("results")
DATE_PATTERN = re.compile(r"(20\d{6})")
DUPLICATE_REPORT_COLUMNS = [
    "date",
    "instrument",
    "candidate_path",
    "selected",
    "finite_percent",
    "row_count",
    "start_time",
    "end_time",
    "reason_selected_or_rejected",
]


def infer_detector(path: Path) -> str:
    name = path.name.lower()
    if "cdte1" in name:
        return "CdTe1"
    if "cdte2" in name:
        return "CdTe2"
    if "czt1" in name:
        return "CZT1"
    if "czt2" in name:
        return "CZT2"
    return "UNKNOWN"


def read_solexs_lc(path: Path) -> pd.DataFrame:
    with fits.open(path) as hdul:
        data = hdul["RATE"].data
        return pd.DataFrame(
            {
                "time_utc": pd.to_datetime(data["TIME"], unit="s", utc=True),
                "instrument": "SoLEXS",
                "detector": "SDD2",
                "band": "2-22 keV",
                "count_rate": np.asarray(data["COUNTS"], dtype=float),
                "stat_err": np.nan,
                "source_file": str(path),
            }
        )


def read_hel1os_lc(path: Path) -> pd.DataFrame:
    rows = []

    with fits.open(path) as hdul:
        for hdu in hdul:
            if not hasattr(hdu, "columns") or hdu.columns is None:
                continue

            cols = list(hdu.columns.names)
            if not {"ISOT", "CTR", "STAT_ERR"}.issubset(set(cols)):
                continue

            data = hdu.data
            header = hdu.header

            detector = header.get("DETNAM", infer_detector(path))
            elow = header.get("ELOW", "")
            ehigh = header.get("EHIGH", "")
            band = f"{elow:g}-{ehigh:g} keV" if elow != "" and ehigh != "" else hdu.name

            rows.append(
                pd.DataFrame(
                    {
                        "time_utc": pd.to_datetime(
                            [x.decode() if isinstance(x, bytes) else str(x) for x in data["ISOT"]],
                            utc=True,
                            errors="coerce",
                        ),
                        "instrument": "HEL1OS",
                        "detector": detector,
                        "band": band,
                        "count_rate": np.asarray(data["CTR"], dtype=float),
                        "stat_err": np.asarray(data["STAT_ERR"], dtype=float),
                        "source_file": str(path),
                    }
                )
            )

    if not rows:
        return pd.DataFrame()

    return pd.concat(rows, ignore_index=True)


def candidate_quality(path: Path, instrument: str) -> dict:
    try:
        if instrument == "SoLEXS":
            df = read_solexs_lc(path)
        else:
            df = read_hel1os_lc(path)
    except Exception as exc:
        return {
            "finite_percent": 0.0,
            "row_count": 0,
            "start_time": pd.NaT,
            "end_time": pd.NaT,
            "nan_count": 0,
            "max_count": np.nan,
            "error": str(exc),
        }

    if df.empty:
        return {
            "finite_percent": 0.0,
            "row_count": 0,
            "start_time": pd.NaT,
            "end_time": pd.NaT,
            "nan_count": 0,
            "max_count": np.nan,
            "error": "",
        }

    counts = pd.to_numeric(df["count_rate"], errors="coerce")
    finite = np.isfinite(counts)
    return {
        "finite_percent": float(finite.sum() / len(df) * 100) if len(df) else 0.0,
        "row_count": int(len(df)),
        "start_time": df["time_utc"].min(),
        "end_time": df["time_utc"].max(),
        "nan_count": int(counts.isna().sum()),
        "max_count": float(counts[finite].max()) if finite.any() else np.nan,
        "error": "",
    }


def coverage_seconds(start_time: object, end_time: object) -> float:
    start = pd.to_datetime(start_time, utc=True, errors="coerce")
    end = pd.to_datetime(end_time, utc=True, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return 0.0
    return max(0.0, (end - start).total_seconds())


def hel1os_duplicate_key(path: Path) -> tuple[str, str]:
    hls_match = re.search(r"(HLS_\d{8}_[^\\/]+)", str(path))
    hls_id = hls_match.group(1) if hls_match else path.parent.name
    return hls_id, path.name.lower()


def choose_best_candidate(date: str, instrument: str, paths: list[Path], group_label: str) -> tuple[list[Path], list[dict]]:
    rows = []
    scored = []
    for path in sorted(paths):
        metrics = candidate_quality(path, instrument)
        score = (
            metrics["finite_percent"],
            coverage_seconds(metrics["start_time"], metrics["end_time"]),
            metrics["row_count"],
        )
        scored.append((score, path, metrics))

    selected_path = max(scored, key=lambda item: item[0])[1] if scored else None
    duplicate_count = len(paths)

    for score, path, metrics in scored:
        selected = path == selected_path
        if metrics["error"]:
            reason = f"rejected from {group_label}: unreadable file ({metrics['error']})"
        elif selected and duplicate_count > 1:
            reason = (
                f"selected from {group_label}: highest finite percentage, longest coverage, "
                "then highest row count"
            )
        elif selected:
            reason = f"selected from {group_label}: only candidate"
        else:
            reason = (
                f"rejected from {group_label}: lower finite percentage, shorter coverage, "
                "or lower row count than selected candidate"
            )

        rows.append(
            {
                "date": date,
                "instrument": instrument,
                "candidate_path": str(path),
                "selected": selected,
                "finite_percent": metrics["finite_percent"],
                "row_count": metrics["row_count"],
                "start_time": metrics["start_time"],
                "end_time": metrics["end_time"],
                "reason_selected_or_rejected": reason,
            }
        )

    return ([selected_path] if selected_path is not None else []), rows


def select_best_lightcurve_files(
    date: str,
    solexs_files: list[Path],
    hel1os_files: list[Path],
) -> tuple[list[Path], list[Path], pd.DataFrame]:
    selected_solexs, rows = choose_best_candidate(date, "SoLEXS", solexs_files, "SoLEXS date group")

    selected_hel1os = []
    hel1os_groups: dict[tuple[str, str], list[Path]] = {}
    for path in hel1os_files:
        hel1os_groups.setdefault(hel1os_duplicate_key(path), []).append(path)

    for group_key, group_paths in sorted(hel1os_groups.items()):
        selected, group_rows = choose_best_candidate(
            date,
            "HEL1OS",
            group_paths,
            f"HEL1OS duplicate group {group_key[0]} {group_key[1]}",
        )
        selected_hel1os.extend(selected)
        rows.extend(group_rows)

    report = pd.DataFrame(rows, columns=DUPLICATE_REPORT_COLUMNS)
    return selected_solexs, sorted(selected_hel1os), report


def write_duplicate_selection_report(report: pd.DataFrame, append: bool = True) -> Path:
    out = RESULTS_DIR / "duplicate_file_selection_report.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if append and out.exists():
        existing = pd.read_csv(out)
        report = pd.concat([existing, report], ignore_index=True)
        report = report.drop_duplicates(subset=["date", "instrument", "candidate_path"], keep="last")
        report = report.sort_values(["date", "instrument", "candidate_path"])
    report.to_csv(out, index=False)
    return out


def date_to_display(date: str) -> str:
    return f"{date[:4]}-{date[4:6]}-{date[6:]}"


def date_window(date: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(date_to_display(date), tz="UTC")
    return start, start + pd.Timedelta(days=1)


def discover_available_dates(raw_dir: Path = RAW_DIR) -> tuple[list[str], pd.DataFrame]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw data directory: {raw_dir.resolve()}")

    solexs_dates = set()
    hel1os_dates = set()

    for path in raw_dir.rglob("AL1_SOLEXS_*_L1.lc.gz"):
        match = DATE_PATTERN.search(path.name)
        if match:
            solexs_dates.add(match.group(1))

    for path in raw_dir.rglob("lightcurve_*.fits"):
        match = DATE_PATTERN.search(str(path))
        if match and "HLS_" in str(path):
            hel1os_dates.add(match.group(1))

    all_dates = sorted(solexs_dates | hel1os_dates)
    rows = []
    matched = []

    for date in all_dates:
        has_solexs = date in solexs_dates
        has_hel1os = date in hel1os_dates
        if has_solexs and has_hel1os:
            status = "matched"
            reason = ""
            matched.append(date)
        elif has_solexs:
            status = "skipped"
            reason = "missing HEL1OS Level-1 lightcurve files"
        else:
            status = "skipped"
            reason = "missing SoLEXS Level-1 lightcurve files"

        rows.append(
            {
                "date": date,
                "has_solexs": has_solexs,
                "has_hel1os": has_hel1os,
                "status": status,
                "reason": reason,
            }
        )

    return matched, pd.DataFrame(rows)


def select_dates_for_mode(dates: list[str], dev_mode: bool) -> list[str]:
    if not dates:
        return []
    if dev_mode:
        selected = dates[:1]
        print(f"[DEV_MODE] Processing one matched date: {selected[0]}")
        return selected
    return dates


def find_files_for_date(date: str, raw_dir: Path = RAW_DIR) -> tuple[list[Path], list[Path]]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw data directory: {raw_dir.resolve()}")

    solexs_files = sorted(raw_dir.rglob(f"AL1_SOLEXS_{date}_*_L1.lc.gz"))

    hel1os_files = []
    hel1os_files += sorted(raw_dir.rglob(f"HLS_{date}*/**/lightcurve_cdte1.fits"))
    hel1os_files += sorted(raw_dir.rglob(f"HLS_{date}*/**/lightcurve_cdte2.fits"))
    hel1os_files += sorted(raw_dir.rglob(f"HLS_{date}*/**/lightcurve_czt1.fits"))
    hel1os_files += sorted(raw_dir.rglob(f"HLS_{date}*/**/lightcurve_czt2.fits"))

    selected_solexs, selected_hel1os, report = select_best_lightcurve_files(date, solexs_files, hel1os_files)
    if not report.empty:
        write_duplicate_selection_report(report, append=True)

    return selected_solexs, selected_hel1os


def build_lightcurves_for_date(date: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    solexs_files, hel1os_files = find_files_for_date(date, raw_dir)

    print(f"SoLEXS files for {date}:")
    for path in solexs_files:
        print(" ", path)

    print(f"\nHEL1OS files for {date}:")
    for path in hel1os_files:
        print(" ", path)

    if not solexs_files:
        raise FileNotFoundError(f"No SoLEXS {date} LC files found under {raw_dir.resolve()}")
    if not hel1os_files:
        raise FileNotFoundError(f"No HEL1OS {date} lightcurve files found under {raw_dir.resolve()}")

    frames = []
    for path in solexs_files:
        frames.append(read_solexs_lc(path))

    for path in hel1os_files:
        df = read_hel1os_lc(path)
        if not df.empty:
            frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df = all_df.dropna(subset=["time_utc", "count_rate"])
    all_df = all_df.sort_values("time_utc")

    start, end = date_window(date)
    all_df = all_df[(all_df["time_utc"] >= start) & (all_df["time_utc"] < end)]

    print("\nSummary:")
    print(all_df.groupby(["instrument", "detector", "band"]).size())

    return all_df


def find_june03_files(raw_dir: Path = RAW_DIR) -> tuple[list[Path], list[Path]]:
    return find_files_for_date("20260603", raw_dir)


def build_june03_lightcurves(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    return build_lightcurves_for_date("20260603", raw_dir)
