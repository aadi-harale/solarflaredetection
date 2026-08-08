from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.io.read_lightcurves import discover_available_dates, select_dates_for_mode
from src.utils.config import is_dev_mode
from src.utils.quality_filter import (
    get_quality_label,
    is_usable_for_forecast,
    normalize_date,
    write_evaluation_dataset_summary,
)


TS_PATH = Path("results/june03_scored_timeseries.csv")
CAT_PATH = Path("results/june03_nowcast_catalogue_clean.csv")
OUT_PATH = Path("data/processed/june03_forecast_dataset.csv")
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")

HORIZONS_MIN = [5, 10, 30]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    base_cols = [
        "soft_solexs_2_22",
        "hard_cdte_5_20",
        "hard_czt_20_40",
        "soft_score",
        "hard_score",
    ]

    for col in base_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

        x = df[col].astype(float)

        for win in ["1min", "3min", "5min", "10min"]:
            df[f"{col}_mean_{win}"] = x.rolling(win, min_periods=10).mean()
            df[f"{col}_max_{win}"] = x.rolling(win, min_periods=10).max()
            df[f"{col}_std_{win}"] = x.rolling(win, min_periods=10).std()

        df[f"{col}_slope_60s"] = x - x.shift(60)
        df[f"{col}_slope_180s"] = x - x.shift(180)
        df[f"{col}_slope_300s"] = x - x.shift(300)

    df["hard_to_soft_ratio"] = (
        (df["hard_cdte_5_20"].fillna(0) + df["hard_czt_20_40"].fillna(0))
        / (df["soft_solexs_2_22"].fillna(0) + 1.0)
    )
    df["hard_score_minus_soft_score"] = df["hard_score"] - df["soft_score"]

    return df


def add_labels(df: pd.DataFrame, catalogue: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    peak_times = pd.to_datetime(catalogue["soft_peak_time"], utc=True, format="mixed", errors="coerce").dropna().sort_values()
    event_starts = pd.to_datetime(catalogue["event_start"], utc=True, format="mixed", errors="coerce").dropna().sort_values()
    event_ends = pd.to_datetime(catalogue["event_end"], utc=True, format="mixed", errors="coerce").dropna().sort_values()

    index = df.index

    for horizon in HORIZONS_MIN:
        horizon_delta = pd.Timedelta(minutes=horizon)

        labels = []
        lead_times = []

        for timestamp in index:
            future_peaks = peak_times[(peak_times > timestamp) & (peak_times <= timestamp + horizon_delta)]

            if len(future_peaks) > 0:
                labels.append(1)
                lead_times.append((future_peaks.iloc[0] - timestamp).total_seconds() / 60)
            else:
                labels.append(0)
                lead_times.append(np.nan)

        df[f"flare_next_{horizon}min"] = labels
        df[f"time_to_peak_within_{horizon}min"] = lead_times

    in_event = pd.Series(False, index=index)
    for start, end in zip(event_starts, event_ends):
        in_event |= (index >= start) & (index <= end)

    df["inside_detected_event"] = in_event.astype(int)
    return df


def build_forecast_dataset(
    ts_path: Path = TS_PATH,
    cat_path: Path = CAT_PATH,
    out_path: Path = OUT_PATH,
    source_date: str | None = None,
) -> pd.DataFrame:
    if not ts_path.exists():
        raise FileNotFoundError(f"Missing scored timeseries: {ts_path.resolve()}")
    if not cat_path.exists():
        raise FileNotFoundError(f"Missing clean nowcast catalogue: {cat_path.resolve()}")

    ts = pd.read_csv(ts_path)

    if "time_utc" in ts.columns:
        time_col = "time_utc"
    else:
        time_col = ts.columns[0]

    ts[time_col] = pd.to_datetime(ts[time_col], utc=True, format="mixed", errors="coerce")
    ts = ts.dropna(subset=[time_col])
    ts = ts.set_index(time_col).sort_index()
    ts.index.name = "time_utc"

    catalogue = pd.read_csv(cat_path)

    df = add_features(ts)
    df = add_labels(df, catalogue)
    df = df.dropna(subset=["soft_solexs_2_22_mean_10min"])
    if source_date is not None:
        df.insert(0, "source_date", source_date)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)

    print(f"Saved: {out_path}")
    print(f"Rows: {len(df):,}")

    print("\nLabel counts:")
    for horizon in HORIZONS_MIN:
        col = f"flare_next_{horizon}min"
        print(f"{col}:")
        print(df[col].value_counts().sort_index())

    print("\nInside detected event counts:")
    print(df["inside_detected_event"].value_counts().sort_index())
    return df


def build_forecast_dataset_for_date(date: str) -> pd.DataFrame:
    return build_forecast_dataset(
        ts_path=RESULTS_DIR / f"{date}_scored_timeseries.csv",
        cat_path=RESULTS_DIR / f"{date}_nowcast_catalogue_clean.csv",
        out_path=PROCESSED_DIR / f"{date}_forecast_dataset.csv",
        source_date=date,
    )


def write_combined_forecast_dataset(datasets: list[pd.DataFrame]) -> Path:
    cleaned_events_by_date = cleaned_event_counts()
    included_dates = []
    excluded_dates = {}
    quiet_dates = []
    questionable_dates = []
    gated_datasets = []

    for dataset in datasets:
        if dataset.empty or "source_date" not in dataset.columns:
            continue

        date = normalize_date(dataset["source_date"].iloc[0])
        quality_label = get_quality_label(date)

        if not is_usable_for_forecast(date):
            excluded_dates[date] = f"quality={quality_label}"
            continue

        event_count = cleaned_events_by_date.get(date, 0)
        temp = dataset.copy()
        temp["source_date"] = date
        temp["quality_label"] = quality_label
        temp["is_quiet_day"] = event_count == 0
        gated_datasets.append(temp)

        included_dates.append(date)
        if event_count == 0:
            quiet_dates.append(date)
        if quality_label == "QUESTIONABLE":
            questionable_dates.append(date)

    if gated_datasets:
        combined = pd.concat(gated_datasets, ignore_index=False).sort_index()
    else:
        combined = pd.DataFrame()

    out = PROCESSED_DIR / "combined_forecast_dataset.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out)
    print(f"\nSaved combined forecast dataset: {out}")
    print(f"Combined rows: {len(combined):,}")
    write_quality_gated_combined_catalogue(included_dates)
    summary_path = write_evaluation_dataset_summary(
        included_dates=included_dates,
        excluded_dates=excluded_dates,
        cleaned_events_by_date=cleaned_events_by_date,
        quiet_dates=quiet_dates,
        questionable_dates=questionable_dates,
    )
    print(f"Saved evaluation dataset summary: {summary_path}")
    return out


def cleaned_event_counts() -> dict[str, int]:
    counts = {}
    for path in sorted(RESULTS_DIR.glob("*_nowcast_catalogue_clean.csv")):
        if path.name.startswith("combined_") or path.name.startswith("june03_"):
            continue
        date = normalize_date(path.name.split("_", 1)[0])
        df = pd.read_csv(path)
        counts[date] = len(df)
    return counts


def write_quality_gated_combined_catalogue(included_dates: list[str]) -> Path:
    frames = []
    for date in included_dates:
        path = RESULTS_DIR / f"{date}_nowcast_catalogue_clean.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df.insert(0, "source_date", date)
        df.insert(1, "quality_label", get_quality_label(date))
        frames.append(df)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined.insert(0, "global_event_id", range(1, len(combined) + 1))
    else:
        combined = pd.DataFrame()

    out = RESULTS_DIR / "combined_nowcast_catalogue_clean.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    print(f"Saved quality-gated combined clean catalogue: {out}")
    return out


def main() -> None:
    matched_dates, _ = discover_available_dates()
    dates = select_dates_for_mode(matched_dates, is_dev_mode())
    if not dates:
        raise RuntimeError("No dates have both SoLEXS and HEL1OS Level-1 lightcurve data.")

    datasets = []
    for date in dates:
        print("\n" + "=" * 90)
        print(f"Building forecast dataset for {date}")
        datasets.append(build_forecast_dataset_for_date(date))

    write_combined_forecast_dataset(datasets)
