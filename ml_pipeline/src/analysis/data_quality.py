from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.io.read_lightcurves import discover_available_dates, select_dates_for_mode
from src.processing.nowcast import (
    add_hard_band_diagnostics,
    build_catalogue,
    detect_segments,
    make_series,
    merge_segments,
    robust_zscore,
)
from src.utils.config import is_dev_mode


PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")

REPORT_COLUMNS = [
    "date",
    "total_rows",
    "start_time_utc",
    "end_time_utc",
    "duration_hours",
    "solexs_rows",
    "hel1os_cdte_rows",
    "hel1os_czt_rows",
    "solexs_finite_percent",
    "cdte_finite_percent",
    "czt_finite_percent",
    "soft_max_count",
    "cdte_5_20_max_count",
    "czt_20_40_max_count",
    "soft_nan_count",
    "cdte_nan_count",
    "czt_nan_count",
    "soft_inf_count",
    "cdte_inf_count",
    "czt_inf_count",
    "overlap_quality_label",
]

SKIPPED_COLUMNS = [
    "date",
    "candidate_start",
    "candidate_end",
    "reason_skipped",
    "finite_soft_samples_in_peak_window",
    "finite_hard_samples_in_window",
    "max_soft_score",
    "max_hard_score",
    "max_soft_count",
    "max_cdte_count",
    "max_czt_count",
]


def finite_mask(series: pd.Series) -> pd.Series:
    return np.isfinite(series.astype(float))


def finite_percent(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(finite_mask(series).sum() / len(series) * 100)


def inf_count(series: pd.Series) -> int:
    x = series.astype(float)
    return int(np.isinf(x).sum())


def finite_max(series: pd.Series) -> float:
    x = series.astype(float)
    finite = x[np.isfinite(x)]
    if finite.empty:
        return np.nan
    return float(finite.max())


def classify_overlap_quality(duration_hours: float, soft_pct: float, cdte_pct: float, czt_pct: float) -> str:
    hard_pct = max(cdte_pct, czt_pct)
    min_signal_pct = min(soft_pct, hard_pct)

    if duration_hours >= 20 and soft_pct >= 80 and hard_pct >= 80:
        return "GOOD"
    if duration_hours < 6 or min_signal_pct < 20:
        return "BROKEN"
    return "QUESTIONABLE"


def read_combined_lightcurve(date: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{date}_combined_lightcurves_long.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing combined lightcurve CSV: {path.resolve()}")

    df = pd.read_csv(path)
    required = {"time_utc", "instrument", "detector", "band", "count_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, format="mixed", errors="coerce")
    df["instrument"] = df["instrument"].astype(str)
    df["detector"] = df["detector"].astype(str)
    df["band"] = df["band"].astype(str)
    df["count_rate"] = pd.to_numeric(df["count_rate"], errors="coerce")
    return df.dropna(subset=["time_utc"])


def audit_date(date: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    df = read_combined_lightcurve(date)
    ts = make_series(df, date)

    scored_ts = ts.copy()
    scored_ts["soft_score"] = robust_zscore(scored_ts["soft_solexs_2_22"])
    scored_ts["cdte_score"] = robust_zscore(scored_ts["hard_cdte_5_20"])
    scored_ts["czt_score"] = robust_zscore(scored_ts["hard_czt_20_40"])
    scored_ts["hard_score"] = scored_ts[["cdte_score", "czt_score"]].max(axis=1)
    scored_ts = add_hard_band_diagnostics(scored_ts)

    solexs_rows = df[df["instrument"] == "SoLEXS"]
    cdte_rows = df[(df["instrument"] == "HEL1OS") & df["detector"].str.contains("CdTe", case=False, regex=True)]
    czt_rows = df[(df["instrument"] == "HEL1OS") & df["detector"].str.contains("CZT", case=False, regex=True)]

    soft = scored_ts["soft_solexs_2_22"]
    cdte = scored_ts["hard_cdte_5_20"]
    czt = scored_ts["hard_czt_20_40"]

    start_time = df["time_utc"].min()
    end_time = df["time_utc"].max()
    duration_hours = (end_time - start_time).total_seconds() / 3600 if pd.notna(start_time) and pd.notna(end_time) else 0

    soft_pct = finite_percent(soft)
    cdte_pct = finite_percent(cdte)
    czt_pct = finite_percent(czt)

    row = {
        "date": date,
        "total_rows": int(len(df)),
        "start_time_utc": start_time,
        "end_time_utc": end_time,
        "duration_hours": duration_hours,
        "solexs_rows": int(len(solexs_rows)),
        "hel1os_cdte_rows": int(len(cdte_rows)),
        "hel1os_czt_rows": int(len(czt_rows)),
        "solexs_finite_percent": soft_pct,
        "cdte_finite_percent": cdte_pct,
        "czt_finite_percent": czt_pct,
        "soft_max_count": finite_max(soft),
        "cdte_5_20_max_count": finite_max(cdte),
        "czt_20_40_max_count": finite_max(czt),
        "soft_nan_count": int(soft.isna().sum()),
        "cdte_nan_count": int(cdte.isna().sum()),
        "czt_nan_count": int(czt.isna().sum()),
        "soft_inf_count": inf_count(soft),
        "cdte_inf_count": inf_count(cdte),
        "czt_inf_count": inf_count(czt),
        "overlap_quality_label": classify_overlap_quality(duration_hours, soft_pct, cdte_pct, czt_pct),
    }

    skipped = find_skipped_candidates(date, scored_ts)
    return row, skipped, scored_ts


def find_skipped_candidates(date: str, scored_ts: pd.DataFrame) -> pd.DataFrame:
    soft_trigger = scored_ts["soft_score"] > 8
    hard_trigger = scored_ts["hard_score"] > 8
    raw_segments = detect_segments(soft_trigger | hard_trigger, min_duration_seconds=5)
    candidates = merge_segments(raw_segments, merge_gap_minutes=12)

    rows = []
    for start, end in candidates:
        region_start = max(scored_ts.index.min(), start - pd.Timedelta(minutes=3))
        region_end = min(scored_ts.index.max(), end + pd.Timedelta(minutes=8))
        region = scored_ts.loc[region_start:region_end]

        if region.empty:
            rows.append(candidate_row(date, start, end, region, "empty expanded peak window"))
            continue

        if region["soft_solexs_2_22"].dropna().empty:
            rows.append(candidate_row(date, start, end, region, "no finite soft X-ray samples in expanded peak window"))
            continue

    return pd.DataFrame(rows, columns=SKIPPED_COLUMNS)


def candidate_row(
    date: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    region: pd.DataFrame,
    reason: str,
) -> dict:
    if region.empty:
        finite_soft = 0
        finite_hard = 0
        max_soft_score = np.nan
        max_hard_score = np.nan
        max_soft_count = np.nan
        max_cdte_count = np.nan
        max_czt_count = np.nan
    else:
        finite_soft = int(finite_mask(region["soft_solexs_2_22"]).sum())
        hard_finite_mask = finite_mask(region["hard_cdte_5_20"]) | finite_mask(region["hard_czt_20_40"])
        finite_hard = int(hard_finite_mask.sum())
        max_soft_score = finite_max(region["soft_score"])
        max_hard_score = finite_max(region["hard_score"])
        max_soft_count = finite_max(region["soft_solexs_2_22"])
        max_cdte_count = finite_max(region["hard_cdte_5_20"])
        max_czt_count = finite_max(region["hard_czt_20_40"])

    return {
        "date": date,
        "candidate_start": start,
        "candidate_end": end,
        "reason_skipped": reason,
        "finite_soft_samples_in_peak_window": finite_soft,
        "finite_hard_samples_in_window": finite_hard,
        "max_soft_score": max_soft_score,
        "max_hard_score": max_hard_score,
        "max_soft_count": max_soft_count,
        "max_cdte_count": max_cdte_count,
        "max_czt_count": max_czt_count,
    }


def load_clean_events(date: str) -> pd.DataFrame:
    path = RESULTS_DIR / f"{date}_nowcast_catalogue_clean.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    for col in ["event_start", "event_end"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def plot_audit(date: str, scored_ts: pd.DataFrame, skipped: pd.DataFrame) -> Path:
    clean = load_clean_events(date)

    fig, axes = plt.subplots(4, 1, figsize=(16, 11), sharex=True)

    axes[0].plot(scored_ts.index, scored_ts["soft_solexs_2_22"], linewidth=0.7, label="SoLEXS 2-22 keV")
    axes[0].set_ylabel("Soft counts")
    axes[0].legend(loc="upper right")

    axes[1].plot(scored_ts.index, scored_ts["hard_cdte_5_20"], linewidth=0.7, label="HEL1OS CdTe 5-20 keV")
    axes[1].plot(scored_ts.index, scored_ts["hard_czt_20_40"], linewidth=0.7, label="HEL1OS CZT 20-40 keV")
    axes[1].set_ylabel("Hard counts")
    axes[1].legend(loc="upper right")

    axes[2].plot(scored_ts.index, scored_ts["soft_score"], linewidth=0.7, label="Soft score")
    axes[2].axhline(8, linestyle="--", linewidth=1, color="gray")
    axes[2].set_ylabel("Soft score")
    axes[2].legend(loc="upper right")

    axes[3].plot(scored_ts.index, scored_ts["hard_score"], linewidth=0.7, label="Hard score")
    axes[3].axhline(8, linestyle="--", linewidth=1, color="gray")
    axes[3].set_ylabel("Hard score")
    axes[3].set_xlabel("UTC time")
    axes[3].legend(loc="upper right")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        for _, event in clean.iterrows():
            if pd.notna(event.get("event_start")) and pd.notna(event.get("event_end")):
                ax.axvspan(event["event_start"], event["event_end"], alpha=0.16, color="green")
        for _, candidate in skipped.iterrows():
            ax.axvspan(
                pd.to_datetime(candidate["candidate_start"], utc=True),
                pd.to_datetime(candidate["candidate_end"], utc=True),
                alpha=0.22,
                color="orangered",
            )

    fig.suptitle(f"Data Quality Audit - {date}", fontsize=14)
    fig.tight_layout()

    out = RESULTS_DIR / f"{date}_audit_lightcurves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def matched_processed_dates() -> list[str]:
    matched, _ = discover_available_dates()
    dates = []
    for date in matched:
        if (PROCESSED_DIR / f"{date}_combined_lightcurves_long.csv").exists():
            dates.append(date)
    return select_dates_for_mode(dates, is_dev_mode())


def run_audit() -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    dates = matched_processed_dates()
    if not dates:
        raise RuntimeError("No matched dates with combined lightcurve CSVs were found.")

    report_rows = []
    skipped_frames = []
    plot_paths = []

    for date in dates:
        print("\n" + "=" * 90)
        print(f"Auditing {date}")
        report_row, skipped, scored_ts = audit_date(date)
        report_rows.append(report_row)
        skipped_frames.append(skipped)
        plot_paths.append(plot_audit(date, scored_ts, skipped))

        print(f"Quality: {report_row['overlap_quality_label']}")
        print(f"Skipped candidates recorded: {len(skipped)}")

    report = pd.DataFrame(report_rows, columns=REPORT_COLUMNS)
    skipped_all = pd.concat(skipped_frames, ignore_index=True) if skipped_frames else pd.DataFrame(columns=SKIPPED_COLUMNS)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(RESULTS_DIR / "data_quality_report.csv", index=False)
    skipped_all.to_csv(RESULTS_DIR / "skipped_event_candidates.csv", index=False)

    print(f"\nSaved: {RESULTS_DIR / 'data_quality_report.csv'}")
    print(f"Saved: {RESULTS_DIR / 'skipped_event_candidates.csv'}")
    for path in plot_paths:
        print(f"Saved: {path}")

    return report, skipped_all, plot_paths


def main() -> None:
    run_audit()
