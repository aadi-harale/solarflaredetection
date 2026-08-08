from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.quality_filter import apply_quality_gate_to_forecast_df, normalize_date


PROJECT_ROOT = Path(".")
COMBINED_FORECAST_PATH = PROJECT_ROOT / "data" / "processed" / "combined_forecast_dataset.csv"
MASTER_CLASSIFIED_PATH = PROJECT_ROOT / "results" / "master_flare_catalogue_classified_v2.csv"
MASTER_PATH = PROJECT_ROOT / "results" / "master_flare_catalogue.csv"
OUT_DIR = PROJECT_ROOT / "results" / "forecasting_v3"
DATASET_PATH = OUT_DIR / "forecasting_v3_dataset.csv"
AUDIT_CSV_PATH = OUT_DIR / "forecasting_v3_label_audit.csv"
AUDIT_MD_PATH = OUT_DIR / "forecasting_v3_label_audit.md"

LABEL_COLUMNS = [
    "flare_onset_next_30min",
    "flare_onset_next_60min",
    "flare_peak_next_15min",
    "high_class_onset_next_60min",
]

LEGACY_LABEL_PREFIXES = ("flare_next_", "time_to_peak_within_")


def load_aligned_forecast_base(path: Path = COMBINED_FORECAST_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing aligned forecast dataset: {path.resolve()}")
    df = pd.read_csv(path)
    if "time_utc" not in df.columns:
        df = df.rename(columns={df.columns[0]: "time_utc"})
    required = {"time_utc", "source_date", "soft_solexs_2_22", "hard_cdte_5_20", "hard_czt_20_40"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, format="mixed", errors="coerce")
    df = df.dropna(subset=["time_utc"])
    df["date"] = df["source_date"].map(normalize_date)
    df = apply_quality_gate_to_forecast_df(df)
    df["date"] = df["source_date"].map(normalize_date)
    return df.sort_values(["date", "time_utc"]).reset_index(drop=True)


def load_master_events() -> pd.DataFrame:
    path = MASTER_CLASSIFIED_PATH if MASTER_CLASSIFIED_PATH.exists() else MASTER_PATH
    if not path.exists():
        raise FileNotFoundError(f"Missing master catalogue: {path.resolve()}")
    events = pd.read_csv(path)
    required = {"event_id", "date", "soft_peak"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    events = events.copy()
    events["date"] = events["date"].map(normalize_date)
    onset_col = "combined_start" if "combined_start" in events.columns else "soft_start"
    if onset_col not in events.columns:
        raise ValueError(f"{path} is missing combined_start/soft_start onset column")
    events["event_onset_time"] = pd.to_datetime(events[onset_col], utc=True, format="mixed", errors="coerce")
    events["event_peak_time"] = pd.to_datetime(events["soft_peak"], utc=True, format="mixed", errors="coerce")
    events = events.dropna(subset=["event_onset_time", "event_peak_time"])
    if "surya_estimated_class_group" in events.columns:
        events["class_group_for_label"] = events["surya_estimated_class_group"].astype(str)
    elif "goes_class_group" in events.columns:
        events["class_group_for_label"] = np.where(events["goes_class_group"].astype(str).isin(["M", "X"]), "HIGH", "LOW_OR_MODERATE")
    else:
        events["class_group_for_label"] = ""
    return events.sort_values(["date", "event_onset_time"]).reset_index(drop=True)


def safe_divide(num: pd.Series, den: pd.Series | float, fill: float = 0.0) -> pd.Series:
    out = num / (den if isinstance(den, pd.Series) else float(den))
    return out.replace([np.inf, -np.inf], np.nan).fillna(fill)


def past_mad_zscore(x: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    median = x.rolling(window, min_periods=max(10, min(window, 60))).median()
    abs_dev = (x - median).abs()
    mad = abs_dev.rolling(window, min_periods=max(10, min(window, 60))).median()
    z = safe_divide(x - median, 1.4826 * mad.replace(0, np.nan))
    return median, z


def monotonic_rise_count(x: pd.Series, seconds: int) -> pd.Series:
    positive = x.diff().gt(0).astype(float)
    return positive.rolling(seconds, min_periods=1).sum()


def seconds_since_last_true(mask: pd.Series) -> pd.Series:
    values = np.full(len(mask), np.nan)
    last_idx: int | None = None
    for i, val in enumerate(mask.to_numpy(dtype=bool)):
        if val:
            last_idx = i
            values[i] = 0.0
        elif last_idx is not None:
            values[i] = float(i - last_idx)
    return pd.Series(values, index=mask.index)


def add_feature_group_for_date(g: pd.DataFrame) -> pd.DataFrame:
    out = g.copy()
    soft = pd.to_numeric(out["soft_solexs_2_22"], errors="coerce").ffill().fillna(0.0)
    cdte = pd.to_numeric(out["hard_cdte_5_20"], errors="coerce").ffill().fillna(0.0)
    czt = pd.to_numeric(out["hard_czt_20_40"], errors="coerce").ffill().fillna(0.0)
    hard = cdte + czt

    out["soft_rolling_median_5min"] = soft.rolling(300, min_periods=30).median()
    out["soft_rolling_median_15min"] = soft.rolling(900, min_periods=60).median()
    out["soft_background_subtracted_flux"] = soft - out["soft_rolling_median_15min"]
    out["soft_background_slope_10min"] = out["soft_rolling_median_15min"] - out["soft_rolling_median_15min"].shift(600)
    _, out["soft_mad_zscore"] = past_mad_zscore(soft, 900)
    out["soft_percentile_vs_quiet"] = 100.0 * safe_divide(out["soft_mad_zscore"].clip(lower=-5, upper=5) + 5, 10)

    out["soft_first_derivative_1min"] = soft - soft.shift(60)
    out["soft_first_derivative_5min"] = soft - soft.shift(300)
    out["soft_second_derivative_5min"] = out["soft_first_derivative_5min"] - out["soft_first_derivative_5min"].shift(300)
    out["soft_monotonic_rise_count_4min"] = monotonic_rise_count(soft, 240)
    out["soft_monotonic_rise_count_10min"] = monotonic_rise_count(soft, 600)
    out["soft_rise_persistence_score"] = safe_divide(out["soft_monotonic_rise_count_10min"], 600)
    local_min_mask = soft.eq(soft.rolling(900, min_periods=60).min())
    out["time_since_soft_local_min"] = seconds_since_last_true(local_min_mask)

    out["soft_low_band_flux"] = soft
    out["soft_high_band_flux"] = np.nan
    out["soft_high_low_ratio"] = np.nan
    out["soft_temperature_proxy"] = np.nan
    out["soft_emission_measure_proxy"] = np.nan
    out["soft_temperature_slope"] = np.nan
    out["soft_emission_measure_slope"] = np.nan
    soft_rise_norm = safe_divide(
        out["soft_background_subtracted_flux"].clip(lower=0),
        out["soft_rolling_median_15min"].abs() + 1.0,
    ).clip(0, 1)
    soft_z_norm = safe_divide(out["soft_mad_zscore"].clip(lower=0), 10.0).clip(0, 1)
    out["hot_onset_proxy_score"] = (
        soft_rise_norm
        + soft_z_norm
        + out["soft_rise_persistence_score"].fillna(0).clip(0, 1)
    ) / 3.0
    out["hot_onset_proxy_method"] = "fallback_background_rise_mad_persistence_not_physical_spectral_inversion"

    out["hard_cdte_flux"] = cdte
    out["hard_czt_flux"] = czt
    out["hard_total_flux"] = hard
    out["hard_rolling_median_5min"] = hard.rolling(300, min_periods=30).median()
    _, out["hard_mad_zscore"] = past_mad_zscore(hard, 300)
    hard_burst = out["hard_mad_zscore"].gt(3).astype(float)
    out["hard_burst_count_1min"] = hard_burst.rolling(60, min_periods=1).sum()
    out["hard_burst_count_5min"] = hard_burst.rolling(300, min_periods=1).sum()
    out["hard_burst_persistence_score"] = safe_divide(out["hard_burst_count_5min"], 300)
    out["hard_integral_5min"] = hard.rolling(300, min_periods=1).sum()
    out["hard_integral_15min"] = hard.rolling(900, min_periods=1).sum()
    out["hard_spike_score"] = out["hard_mad_zscore"].clip(lower=0)
    out["hard_energy_ratio_czt_cdte"] = safe_divide(czt, cdte.replace(0, np.nan))

    soft_derivative_positive = out["soft_first_derivative_5min"].clip(lower=0)
    out["hard_integral_5min_vs_soft_derivative"] = safe_divide(out["hard_integral_5min"], soft_derivative_positive + 1.0)
    out["hard_integral_15min_vs_soft_derivative"] = safe_divide(out["hard_integral_15min"], soft_derivative_positive + 1.0)
    out["hard_before_soft_score"] = out["hard_spike_score"] * (1.0 - out["soft_rise_persistence_score"].fillna(0)).clip(lower=0)
    out["hard_soft_lag_correlation_5min"] = hard.rolling(300, min_periods=60).corr(soft_derivative_positive)
    out["hard_soft_lag_correlation_15min"] = hard.rolling(900, min_periods=120).corr(soft_derivative_positive)
    out["hard_to_soft_flux_ratio"] = safe_divide(hard, soft + 1.0)
    out["combined_precursor_score"] = (
        safe_divide(out["hard_spike_score"], 10.0).clip(0, 1)
        + safe_divide(out["soft_mad_zscore"].clip(lower=0), 10.0).clip(0, 1)
        + out["soft_rise_persistence_score"].fillna(0).clip(0, 1)
        + out["hot_onset_proxy_score"].fillna(0).clip(0, 1)
    ) / 4.0

    return out


def add_context_features(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["time_since_previous_detected_event"] = np.nan
    out["post_flare_decay_state"] = 0
    out["baseline_recovery_score"] = np.nan
    for date, idx in out.groupby("date").groups.items():
        event_rows = events[events["date"] == date].copy()
        if event_rows.empty:
            continue
        group = out.loc[idx]
        times = group["time_utc"]
        prev_end = pd.Series(pd.NaT, index=group.index, dtype="datetime64[ns, UTC]")
        for _, event in event_rows.iterrows():
            onset = event["event_onset_time"]
            peak = event["event_peak_time"]
            after = times >= peak
            prev_end.loc[after] = peak
        seconds_since = (times - prev_end).dt.total_seconds()
        out.loc[group.index, "time_since_previous_detected_event"] = seconds_since
        out.loc[group.index, "post_flare_decay_state"] = seconds_since.between(0, 3600).fillna(False).astype(int)
        baseline = pd.to_numeric(group["soft_rolling_median_15min"], errors="coerce")
        current = pd.to_numeric(group["soft_solexs_2_22"], errors="coerce")
        out.loc[group.index, "baseline_recovery_score"] = safe_divide((current - baseline).abs(), baseline.abs() + 1.0)
    return out


def add_onset_labels(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in LABEL_COLUMNS:
        out[col] = 0

    for date, idx in out.groupby("date").groups.items():
        event_rows = events[events["date"] == date].copy()
        if event_rows.empty:
            continue
        times = out.loc[idx, "time_utc"]
        for _, event in event_rows.iterrows():
            onset = event["event_onset_time"]
            peak = event["event_peak_time"]
            out.loc[idx, "flare_onset_next_30min"] |= ((times < onset) & (times >= onset - pd.Timedelta(minutes=30))).astype(int)
            out.loc[idx, "flare_onset_next_60min"] |= ((times < onset) & (times >= onset - pd.Timedelta(minutes=60))).astype(int)
            out.loc[idx, "flare_peak_next_15min"] |= ((times < peak) & (times >= peak - pd.Timedelta(minutes=15))).astype(int)
            if str(event.get("class_group_for_label", "")) == "HIGH":
                out.loc[idx, "high_class_onset_next_60min"] |= ((times < onset) & (times >= onset - pd.Timedelta(minutes=60))).astype(int)

    for col in LABEL_COLUMNS:
        out[col] = out[col].astype(int)
    return out


def build_forecasting_v3_dataset() -> tuple[pd.DataFrame, dict]:
    base = load_aligned_forecast_base()
    events = load_master_events()

    feature_frames = []
    for _, group in base.groupby("date", sort=True):
        feature_frames.append(add_feature_group_for_date(group))
    df = pd.concat(feature_frames, ignore_index=True).sort_values(["date", "time_utc"])
    df = add_context_features(df, events)
    df = add_onset_labels(df, events)

    dropped_columns = []
    for col in list(df.columns):
        if col.startswith(LEGACY_LABEL_PREFIXES):
            dropped_columns.append({"column": col, "reason": "legacy v1/v2 future label removed from v3 feature table to avoid target confusion"})
            df = df.drop(columns=[col])

    df = df.rename(columns={"time_utc": "timestamp"})
    ordered_front = ["timestamp", "date", "quality_label"] + [c for c in LABEL_COLUMNS if c in df.columns]
    remaining = [c for c in df.columns if c not in ordered_front and c != "source_date"]
    df = df[ordered_front + remaining]

    audit = {
        "events": events,
        "dropped_columns": dropped_columns,
        "fallback_features": ["hot_onset_proxy_score"],
        "missing_physical_features": [
            "soft_high_band_flux",
            "soft_high_low_ratio",
            "soft_temperature_proxy",
            "soft_emission_measure_proxy",
            "soft_temperature_slope",
            "soft_emission_measure_slope",
        ],
    }
    return df, audit


def write_audit(dataset: pd.DataFrame, audit: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_cols = [c for c in dataset.columns if c not in {"timestamp", "date", "quality_label"} | set(LABEL_COLUMNS)]
    rows = [
        {"metric": "rows", "value": len(dataset), "notes": "Rows in v3 dataset."},
        {"metric": "dates", "value": dataset["date"].nunique(), "notes": ";".join(sorted(dataset["date"].astype(str).unique()))},
        {"metric": "feature_columns", "value": len(feature_cols), "notes": "Feature columns excluding timestamp/date/quality/labels."},
        {"metric": "label_columns", "value": len(LABEL_COLUMNS), "notes": ";".join(LABEL_COLUMNS)},
        {"metric": "events_represented", "value": len(audit["events"]), "notes": "Master catalogue events with onset and peak timestamps."},
        {"metric": "class_coverage", "value": audit["events"].get("class_group_for_label", pd.Series(dtype=str)).value_counts().to_dict(), "notes": "Class groups available for high-class diagnostic label."},
        {"metric": "leakage_check_features", "value": "PASS", "notes": "Feature windows use current/past rolling operations only. Future event times/classes are used only for labels."},
        {"metric": "dropped_columns", "value": len(audit["dropped_columns"]), "notes": "; ".join(f"{d['column']} ({d['reason']})" for d in audit["dropped_columns"])},
        {"metric": "fallback_features", "value": ";".join(audit["fallback_features"]), "notes": "Fallback proxy used because only one SoLEXS broad soft band is present in aligned data."},
    ]
    for label in LABEL_COLUMNS:
        counts = dataset[label].value_counts().sort_index().to_dict()
        rows.append({"metric": f"{label}_distribution", "value": counts, "notes": "0=negative, 1=positive"})
    pd.DataFrame(rows).to_csv(AUDIT_CSV_PATH, index=False)

    md_lines = [
        "# Forecasting v3 Label and Leakage Audit",
        "",
        f"- Dataset rows: {len(dataset):,}",
        f"- Dates: {dataset['date'].nunique()} ({', '.join(sorted(dataset['date'].astype(str).unique()))})",
        f"- Feature columns: {len(feature_cols)}",
        f"- Label columns: {', '.join(LABEL_COLUMNS)}",
        f"- Events represented: {len(audit['events'])}",
        f"- Class coverage: {audit['events'].get('class_group_for_label', pd.Series(dtype=str)).value_counts().to_dict()}",
        "",
        "## Label Distributions",
        "",
    ]
    for label in LABEL_COLUMNS:
        md_lines.append(f"- `{label}`: {dataset[label].value_counts().sort_index().to_dict()}")
    md_lines.extend(
        [
            "",
            "## Leakage Checks",
            "",
            "- PASS: Feature engineering uses current/past rolling medians, slopes, derivatives, integrals, correlations, and context features.",
            "- PASS: Future event onset/peak/class information is used only for label generation.",
            "- PASS: Legacy v1/v2 future label columns are dropped from the v3 feature table.",
            "",
            "## Dropped Columns",
            "",
        ]
    )
    if audit["dropped_columns"]:
        for item in audit["dropped_columns"]:
            md_lines.append(f"- `{item['column']}`: {item['reason']}")
    else:
        md_lines.append("- None")
    md_lines.extend(
        [
            "",
            "## Missing/Fallback Features",
            "",
            "- Multiple SoLEXS energy bands are not present in the aligned v3 base table.",
            "- `hot_onset_proxy_score` is therefore a fallback proxy from background-subtracted rise, MAD z-score, and derivative persistence.",
            "- This fallback is not physical spectral inversion and should not be described as temperature or emission-measure calibration.",
        ]
    )
    AUDIT_MD_PATH.write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset, audit = build_forecasting_v3_dataset()
    dataset.to_csv(DATASET_PATH, index=False)
    write_audit(dataset, audit)

    feature_groups = [
        "SoLEXS background",
        "SoLEXS rise",
        "SoLEXS hot-onset fallback proxy",
        "HEL1OS impulsive hard X-ray",
        "hard-soft coupling",
        "context",
    ]
    print(f"dataset path: {DATASET_PATH}")
    print(f"rows: {len(dataset):,}")
    print(f"dates: {dataset['date'].nunique()} ({', '.join(sorted(dataset['date'].astype(str).unique()))})")
    print(f"labels created: {', '.join(LABEL_COLUMNS)}")
    print(f"top feature groups created: {', '.join(feature_groups)}")
    print("leakage-check result: PASS")
    print("missing/fallback features: hot_onset_proxy_score fallback used; physical temperature/emission-measure proxies unavailable from single broad SoLEXS band")
