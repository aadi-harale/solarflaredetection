from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.io.read_lightcurves import date_window, discover_available_dates, select_dates_for_mode
from src.utils.config import is_dev_mode
from src.utils.plotting import plot_nowcast_events


INPUT_CSV = Path("data/processed/june03_combined_lightcurves_long.csv")
OUT_DIR = Path("results")
PROCESSED_DIR = Path("data/processed")

CATALOGUE_COLUMNS = [
    "event_id",
    "event_start",
    "event_end",
    "hard_trigger_time",
    "hard_peak_time",
    "soft_peak_time",
    "soft_peak_counts",
    "max_hard_score",
    "max_soft_score",
    "lead_time_min",
]

CLEAN_CATALOGUE_COLUMNS = [
    "event_id",
    "event_start",
    "event_end",
    "event_duration_sec",
    "hard_trigger_time",
    "hard_peak_time",
    "soft_peak_time",
    "soft_peak_counts",
    "max_hard_score",
    "max_soft_score",
    "lead_time_min",
    "alert_type",
]


def _cadence_aware_min_periods(series: pd.Series, window: str, fallback: int = 60) -> int:
    if not isinstance(series.index, pd.DatetimeIndex) or len(series.index) < 3:
        return fallback
    diffs = series.index.to_series().diff().dt.total_seconds().dropna()
    diffs = diffs[diffs > 0]
    if diffs.empty:
        return fallback
    window_seconds = pd.Timedelta(window).total_seconds()
    if window_seconds <= 0:
        return fallback
    expected_samples = window_seconds / float(diffs.median())
    return max(1, int(np.ceil(0.30 * expected_samples)))


def robust_zscore(series: pd.Series, window: str = "10min") -> pd.Series:
    """
    Causal robust z-score:
    Uses only previous data, not future data.
    score = (x - rolling_median) / rolling_MAD
    """
    x = series.astype(float)
    min_periods = _cadence_aware_min_periods(x, window)

    background = x.shift(1).rolling(window, min_periods=min_periods).median()
    abs_dev = (x.shift(1) - background).abs()
    mad = abs_dev.rolling(window, min_periods=min_periods).median()

    robust_sigma = 1.4826 * mad
    robust_sigma = robust_sigma.replace(0, np.nan)

    score = (x - background) / robust_sigma
    return score.replace([np.inf, -np.inf], np.nan).fillna(0)


def add_hard_band_diagnostics(ts: pd.DataFrame) -> pd.DataFrame:
    ts = ts.copy()
    cdte = pd.to_numeric(ts.get("cdte_score"), errors="coerce")
    czt = pd.to_numeric(ts.get("czt_score"), errors="coerce")
    ts["czt_minus_cdte_score"] = (czt - cdte).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    ts["czt_to_cdte_ratio_safe"] = (
        czt.clip(lower=0.0).add(1e-6) / cdte.clip(lower=0.0).add(1e-6)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(upper=1e6)

    diff = ts["czt_minus_cdte_score"]
    ts["hard_band_dominance"] = np.select(
        [
            cdte.isna() | czt.isna(),
            diff >= 1.0,
            diff <= -1.0,
        ],
        ["UNKNOWN", "CZT_DOMINANT", "CDTE_DOMINANT"],
        default="MIXED",
    )
    return ts


def make_series(df: pd.DataFrame, date: str = "20260603") -> pd.DataFrame:
    df = df.copy()

    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, format="mixed", errors="coerce")
    df = df.dropna(subset=["time_utc"])
    df["time_utc"] = df["time_utc"].dt.floor("s")

    df["band"] = df["band"].astype(str)
    df["detector"] = df["detector"].astype(str)

    start, end = date_window(date)
    df = df[(df["time_utc"] >= start) & (df["time_utc"] < end)]

    one_sec = pd.date_range(start=start, end=end - pd.Timedelta(seconds=1), freq="1s", tz="UTC")

    soft = df[(df["instrument"] == "SoLEXS") & (df["band"].str.contains("2-22", regex=False))]

    cdte_5_20 = df[
        (df["instrument"] == "HEL1OS")
        & (df["detector"].str.contains("CdTe", case=False, regex=True))
        & (df["band"].str.contains("5-20", regex=False))
    ]

    czt_20_40 = df[
        (df["instrument"] == "HEL1OS")
        & (df["detector"].str.contains("CZT", case=False, regex=True))
        & (df["band"].str.contains("20-40", regex=False))
    ]

    def aggregate(group: pd.DataFrame, name: str) -> pd.Series:
        if group.empty:
            print(f"[WARN] No data found for {name}")
            return pd.Series(index=one_sec, dtype=float, name=name)

        series = group.groupby("time_utc")["count_rate"].mean().sort_index().reindex(one_sec)
        series = series.interpolate(limit=2)
        series.name = name
        return series

    out = pd.DataFrame(index=one_sec)
    out["soft_solexs_2_22"] = aggregate(soft, "soft_solexs_2_22")
    out["hard_cdte_5_20"] = aggregate(cdte_5_20, "hard_cdte_5_20")
    out["hard_czt_20_40"] = aggregate(czt_20_40, "hard_czt_20_40")

    return out


def detect_segments(trigger: pd.Series, min_duration_seconds: int = 5) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    trigger = trigger.fillna(False).astype(bool)

    segments = []
    in_event = False
    start = None
    last = None

    for timestamp, val in trigger.items():
        if val and not in_event:
            in_event = True
            start = timestamp
            last = timestamp
        elif val and in_event:
            last = timestamp
        elif not val and in_event:
            if start is not None and last is not None:
                duration = (last - start).total_seconds() + 1
                if duration >= min_duration_seconds:
                    segments.append((start, last))
            in_event = False
            start = None
            last = None

    if in_event and start is not None and last is not None:
        duration = (last - start).total_seconds() + 1
        if duration >= min_duration_seconds:
            segments.append((start, last))

    return segments


def merge_segments(
    segments: list[tuple[pd.Timestamp, pd.Timestamp]],
    merge_gap_minutes: int = 12,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if not segments:
        return []

    segments = sorted(segments, key=lambda x: x[0])
    merged = [segments[0]]
    gap = pd.Timedelta(minutes=merge_gap_minutes)

    for start, end in segments[1:]:
        last_start, last_end = merged[-1]
        if start - last_end <= gap:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))

    return merged


def build_catalogue(ts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = ts.copy()

    ts["soft_score"] = robust_zscore(ts["soft_solexs_2_22"])
    ts["cdte_score"] = robust_zscore(ts["hard_cdte_5_20"])
    ts["czt_score"] = robust_zscore(ts["hard_czt_20_40"])

    ts["hard_score"] = ts[["cdte_score", "czt_score"]].max(axis=1)
    ts = add_hard_band_diagnostics(ts)

    soft_trigger = ts["soft_score"] > 8
    hard_trigger = ts["hard_score"] > 8
    combined_trigger = soft_trigger | hard_trigger

    raw_segments = detect_segments(combined_trigger, min_duration_seconds=5)
    events = merge_segments(raw_segments, merge_gap_minutes=12)

    rows = []
    for i, (start, end) in enumerate(events, start=1):
        region_start = max(ts.index.min(), start - pd.Timedelta(minutes=3))
        region_end = min(ts.index.max(), end + pd.Timedelta(minutes=8))
        region = ts.loc[region_start:region_end]

        if region.empty:
            continue

        if region["soft_solexs_2_22"].dropna().empty:
            print(f"[WARN] Skipping event candidate {i}: no finite soft X-ray samples in expanded region.")
            continue

        soft_peak_time = region["soft_solexs_2_22"].idxmax()
        soft_peak_value = region["soft_solexs_2_22"].max()

        hard_region = region[region["hard_score"] > 8]
        if not hard_region.empty:
            hard_trigger_time = hard_region.index.min()
            hard_peak_time = region["hard_score"].idxmax()
        else:
            hard_trigger_time = pd.NaT
            hard_peak_time = pd.NaT

        if pd.notna(hard_trigger_time):
            lead_time_min = (soft_peak_time - hard_trigger_time).total_seconds() / 60
        else:
            lead_time_min = np.nan

        rows.append(
            {
                "event_id": i,
                "event_start": start,
                "event_end": end,
                "hard_trigger_time": hard_trigger_time,
                "hard_peak_time": hard_peak_time,
                "soft_peak_time": soft_peak_time,
                "soft_peak_counts": float(soft_peak_value),
                "max_hard_score": float(region["hard_score"].max()),
                "max_soft_score": float(region["soft_score"].max()),
                "lead_time_min": float(lead_time_min) if pd.notna(lead_time_min) else np.nan,
            }
        )

    return pd.DataFrame(rows, columns=CATALOGUE_COLUMNS), ts


def run_nowcast(
    input_csv: Path = INPUT_CSV,
    out_dir: Path = OUT_DIR,
    date: str = "20260603",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv.resolve()}")

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    ts = make_series(df, date)
    catalogue, scored_ts = build_catalogue(ts)

    cat_path = out_dir / f"{date}_nowcast_catalogue.csv"
    scored_path = out_dir / f"{date}_scored_timeseries.csv"

    catalogue.to_csv(cat_path, index=False)
    scored_ts.to_csv(scored_path)

    print("\nDetected event catalogue:")
    if catalogue.empty:
        print("[BAD] No events detected. Thresholds may be too strict.")
    else:
        print(catalogue.to_string(index=False))

    print(f"\n[SAVED] {cat_path}")
    print(f"[SAVED] {scored_path}")

    plot_nowcast_events(scored_ts, catalogue, out_dir / f"{date}_nowcast_detected_events.png")
    return catalogue, scored_ts


def classify_event(row: pd.Series) -> str:
    hard_trigger_exists = pd.notna(row["hard_trigger_time"])
    lead_time_min = row["lead_time_min"]

    if hard_trigger_exists and pd.notna(lead_time_min) and lead_time_min < 0:
        return "ANOMALOUS_SOFT_BEFORE_HARD"
    if hard_trigger_exists and pd.notna(lead_time_min) and lead_time_min >= 1.0:
        return "FORECASTED_WITH_HARD_XRAY_LEAD"
    if hard_trigger_exists and pd.notna(lead_time_min) and 0 <= lead_time_min < 1.0:
        return "NOWCASTED_HARD_XRAY_NO_LEAD"
    return "SOFT_XRAY_ONLY"


def clean_nowcast_catalogue(
    input_path: Path = OUT_DIR / "june03_nowcast_catalogue.csv",
    output_path: Path = OUT_DIR / "june03_nowcast_catalogue_clean.csv",
) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Missing nowcast catalogue: {input_path.resolve()}")

    df = pd.read_csv(input_path)
    if df.empty:
        clean = pd.DataFrame(columns=CLEAN_CATALOGUE_COLUMNS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        clean.to_csv(output_path, index=False)
        print("\nCleaned catalogue:")
        print("[OK] No retained events.")
        print(f"\nSaved: {output_path}")
        return clean

    for col in ["event_start", "event_end", "hard_trigger_time", "hard_peak_time", "soft_peak_time"]:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    df["event_duration_sec"] = (df["event_end"] - df["event_start"]).dt.total_seconds()

    keep = (df["event_duration_sec"] >= 30) & ((df["soft_peak_counts"] >= 500) | (df["max_hard_score"] >= 50))
    clean = df[keep].copy()

    clean["alert_type"] = clean.apply(classify_event, axis=1)
    clean["event_id"] = range(1, len(clean) + 1)

    clean = clean[
        [
            "event_id",
            "event_start",
            "event_end",
            "event_duration_sec",
            "hard_trigger_time",
            "hard_peak_time",
            "soft_peak_time",
            "soft_peak_counts",
            "max_hard_score",
            "max_soft_score",
            "lead_time_min",
            "alert_type",
        ]
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output_path, index=False)

    print("\nCleaned catalogue:")
    print(clean.to_string(index=False))
    print(f"\nSaved: {output_path}")

    return clean


def run_nowcast_for_date(date: str) -> pd.DataFrame:
    input_csv = PROCESSED_DIR / f"{date}_combined_lightcurves_long.csv"
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing combined lightcurve CSV for {date}: {input_csv.resolve()}")

    run_nowcast(input_csv=input_csv, out_dir=OUT_DIR, date=date)
    clean = clean_nowcast_catalogue(
        input_path=OUT_DIR / f"{date}_nowcast_catalogue.csv",
        output_path=OUT_DIR / f"{date}_nowcast_catalogue_clean.csv",
    )
    clean.insert(0, "source_date", date)
    return clean


def write_combined_catalogue(catalogues: list[pd.DataFrame]) -> Path:
    if catalogues:
        combined = pd.concat(catalogues, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=["source_date", *CLEAN_CATALOGUE_COLUMNS])

    if not combined.empty:
        combined.insert(0, "global_event_id", range(1, len(combined) + 1))
    elif "global_event_id" not in combined.columns:
        combined.insert(0, "global_event_id", pd.Series(dtype=int))

    out = OUT_DIR / "combined_nowcast_catalogue_clean.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    print(f"\nSaved combined clean catalogue: {out}")
    return out


def main() -> None:
    matched_dates, _ = discover_available_dates()
    dates = select_dates_for_mode(matched_dates, is_dev_mode())
    if not dates:
        raise RuntimeError("No dates have both SoLEXS and HEL1OS Level-1 lightcurve data.")

    catalogues = []
    for date in dates:
        print("\n" + "=" * 90)
        print(f"Running nowcast for {date}")
        catalogues.append(run_nowcast_for_date(date))

    write_combined_catalogue(catalogues)
