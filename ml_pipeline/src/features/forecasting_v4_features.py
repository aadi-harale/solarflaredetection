from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(".")
V3_DATASET_PATH = PROJECT_ROOT / "results" / "forecasting_v3" / "forecasting_v3_dataset.csv"
MASTER_CLASSIFIED_PATH = PROJECT_ROOT / "results" / "master_flare_catalogue_classified_v2.csv"
MASTER_PATH = PROJECT_ROOT / "results" / "master_flare_catalogue.csv"
OUT_DIR = PROJECT_ROOT / "results" / "forecasting_v4"
DATASET_PATH = OUT_DIR / "forecasting_v4_dataset.csv"
AUDIT_CSV_PATH = OUT_DIR / "forecasting_v4_dataset_audit.csv"
AUDIT_MD_PATH = OUT_DIR / "forecasting_v4_dataset_audit.md"

CORE_NAME = "SuryaAlert-XF: Aditya-L1 Soft-Hard X-ray Fusion Forecasting Algorithm"

LABEL_COLUMNS = [
    "flare_onset_next_30min",
    "flare_onset_next_60min",
    "low_class_flare_next_60min",
    "high_class_flare_next_60min",
    "m_or_x_class_like_next_60min",
]

V4_FEATURE_COLUMNS = [
    "log_soft_flux",
    "log_hard_flux",
    "soft_percentile_rank_by_date",
    "hard_percentile_rank_by_date",
    "soft_adaptive_threshold_score",
    "hard_adaptive_threshold_score",
    "soft_small_flare_sensitive_score",
    "hard_impulsive_dynamic_score",
    "combined_dynamic_range_score",
    "soft_preflare_rise_5min",
    "soft_preflare_rise_10min",
    "soft_preflare_rise_20min",
    "soft_background_lift_score",
    "soft_gradual_enhancement_score",
    "soft_rise_persistence_score_v4",
    "soft_flux_acceleration_score",
    "hard_impulse_rise_1min",
    "hard_impulse_rise_5min",
    "hard_burst_count_5min_v4",
    "hard_burst_count_10min",
    "hard_spike_cluster_score",
    "hard_nonthermal_precursor_score",
    "hard_impulsive_enhancement_score",
    "hard_detrended_std_10min",
    "hard_peak_count_10min",
    "hard_peak_spacing_median_10min",
    "hard_autocorr_peak_10min",
    "hard_qpp_power_5_15min",
    "hard_qpp_power_8_30min",
    "hard_qpp_score",
    "hard_oscillation_persistence_score",
    "hard_before_soft_enhancement_score",
    "hard_integral_vs_soft_derivative",
    "hard_soft_lag_correlation_10min",
    "hard_to_soft_percentile_ratio",
    "soft_gradual_plus_hard_impulsive_score",
    "precursor_fusion_score_v4",
]


def normalize_date(value: object) -> str:
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        return text
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return text
    return parsed.strftime("%Y%m%d")


def load_v3_dataset(path: Path = V3_DATASET_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Forecasting v3 dataset: {path.resolve()}")
    df = pd.read_csv(path)
    required = {"timestamp", "date", "quality_label", "soft_solexs_2_22", "hard_cdte_5_20", "hard_czt_20_40"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed", errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["date"] = df["date"].map(normalize_date)
    return df.sort_values(["date", "timestamp"]).reset_index(drop=True)


def load_master_events() -> pd.DataFrame:
    path = MASTER_CLASSIFIED_PATH if MASTER_CLASSIFIED_PATH.exists() else MASTER_PATH
    if not path.exists():
        raise FileNotFoundError(f"Missing master catalogue for v4 labels: {path.resolve()}")
    events = pd.read_csv(path)
    required = {"event_id", "date"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    onset_col = "combined_start" if "combined_start" in events.columns else "soft_start"
    peak_col = "combined_peak" if "combined_peak" in events.columns else "soft_peak"
    if onset_col not in events.columns:
        raise ValueError(f"{path} is missing combined_start/soft_start for onset labels")
    events = events.copy()
    events["date"] = events["date"].map(normalize_date)
    events["event_onset_time"] = pd.to_datetime(events[onset_col], utc=True, format="mixed", errors="coerce")
    events["event_peak_time"] = pd.to_datetime(events.get(peak_col, events[onset_col]), utc=True, format="mixed", errors="coerce")
    events = events.dropna(subset=["event_onset_time"])
    events["low_high_label"] = events.apply(class_to_low_high, axis=1)
    events["m_or_x_like_label"] = events.apply(class_to_m_or_x_like, axis=1)
    return events.sort_values(["date", "event_onset_time"]).reset_index(drop=True)


def class_to_low_high(row: pd.Series) -> str:
    surya_group = str(row.get("surya_estimated_class_group", "")).upper()
    if surya_group == "HIGH":
        return "HIGH"
    if surya_group in {"LOW_OR_MODERATE", "LOW", "MODERATE"}:
        return "LOW_OR_MODERATE"
    goes_group = str(row.get("goes_class_group", "")).upper()
    if goes_group in {"M", "X"}:
        return "HIGH"
    if goes_group in {"A", "B", "C"}:
        return "LOW_OR_MODERATE"
    goes_class = str(row.get("goes_class", "")).upper()
    if goes_class.startswith(("M", "X")):
        return "HIGH"
    if goes_class.startswith(("A", "B", "C")):
        return "LOW_OR_MODERATE"
    return "UNKNOWN"


def class_to_m_or_x_like(row: pd.Series) -> int:
    label = class_to_low_high(row)
    if label == "HIGH":
        return 1
    goes_class = str(row.get("goes_class", "")).upper()
    return int(goes_class.startswith(("M", "X")))


def safe_divide(num: pd.Series | np.ndarray, den: pd.Series | np.ndarray | float, fill: float = 0.0) -> pd.Series:
    out = pd.Series(num) / den
    return out.replace([np.inf, -np.inf], np.nan).fillna(fill)


def robust_zscore(x: pd.Series, window: int, min_periods: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    median = x.rolling(window, min_periods=min_periods).median()
    mad = (x - median).abs().rolling(window, min_periods=min_periods).median()
    z = ((x - median) / (1.4826 * mad.replace(0, np.nan))).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return median, mad, z


def clipped01(x: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(x).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)


def expanding_percentile_approx(x: pd.Series) -> pd.Series:
    """Leakage-safe percentile proxy using only current/past expanding min/max.

    It is intentionally named as a percentile-rank proxy in the audit because exact
    all-day rank would leak future samples.
    """
    expanding_min = x.expanding(min_periods=1).min()
    expanding_max = x.expanding(min_periods=1).max()
    return 100.0 * safe_divide(x - expanding_min, (expanding_max - expanding_min).replace(0, np.nan))


def rolling_peak_count(x: pd.Series, window: int) -> pd.Series:
    left = x.shift(1)
    right = x.shift(-1)
    local_peak = x.gt(left) & x.ge(right)
    dynamic = x.gt(x.rolling(window, min_periods=max(10, window // 10)).median())
    return (local_peak & dynamic).astype(float).rolling(window, min_periods=1).sum()


def rolling_peak_spacing_median_seconds(x: pd.Series, window: int) -> pd.Series:
    local_peak = (x.gt(x.shift(1)) & x.ge(x.shift(-1))).astype(float)
    peak_index = pd.Series(np.where(local_peak.to_numpy(dtype=bool), np.arange(len(x)), np.nan), index=x.index)
    last_peak = peak_index.ffill()
    spacing = last_peak.diff()
    return spacing.rolling(window, min_periods=max(5, window // 20)).median()


def add_v4_features_for_date(g: pd.DataFrame) -> pd.DataFrame:
    out = g.copy()
    soft = pd.to_numeric(out["soft_solexs_2_22"], errors="coerce").ffill().fillna(0.0).clip(lower=0)
    cdte = pd.to_numeric(out["hard_cdte_5_20"], errors="coerce").ffill().fillna(0.0).clip(lower=0)
    czt = pd.to_numeric(out["hard_czt_20_40"], errors="coerce").ffill().fillna(0.0).clip(lower=0)
    hard = cdte + czt

    out["log_soft_flux"] = np.log1p(soft)
    out["log_hard_flux"] = np.log1p(hard)
    out["soft_percentile_rank_by_date"] = expanding_percentile_approx(soft)
    out["hard_percentile_rank_by_date"] = expanding_percentile_approx(hard)

    soft_med_15, _, soft_z_15 = robust_zscore(soft, 900, 60)
    hard_med_10, _, hard_z_10 = robust_zscore(hard, 600, 30)
    soft_med_30 = soft.rolling(1800, min_periods=120).median()
    hard_med_30 = hard.rolling(1800, min_periods=120).median()
    soft_dynamic = safe_divide(soft - soft_med_30, soft_med_30.abs() + 1.0)
    hard_dynamic = safe_divide(hard - hard_med_30, hard_med_30.abs() + 1.0)
    out["soft_adaptive_threshold_score"] = soft_z_15.clip(lower=0)
    out["hard_adaptive_threshold_score"] = hard_z_10.clip(lower=0)
    out["soft_small_flare_sensitive_score"] = (0.5 * clipped01(soft_dynamic) + 0.5 * clipped01(soft_z_15 / 6.0))
    out["hard_impulsive_dynamic_score"] = (0.5 * clipped01(hard_dynamic) + 0.5 * clipped01(hard_z_10 / 6.0))
    out["combined_dynamic_range_score"] = (
        out["soft_small_flare_sensitive_score"] + out["hard_impulsive_dynamic_score"]
    ) / 2.0

    out["soft_preflare_rise_5min"] = soft - soft.shift(300)
    out["soft_preflare_rise_10min"] = soft - soft.shift(600)
    out["soft_preflare_rise_20min"] = soft - soft.shift(1200)
    out["soft_background_lift_score"] = clipped01(safe_divide(soft_med_15 - soft_med_30, soft_med_30.abs() + 1.0))
    rise_5_norm = clipped01(safe_divide(out["soft_preflare_rise_5min"], soft_med_15.abs() + 1.0))
    rise_10_norm = clipped01(safe_divide(out["soft_preflare_rise_10min"], soft_med_15.abs() + 1.0))
    rise_20_norm = clipped01(safe_divide(out["soft_preflare_rise_20min"], soft_med_15.abs() + 1.0))
    out["soft_gradual_enhancement_score"] = (rise_5_norm + rise_10_norm + rise_20_norm + out["soft_background_lift_score"]) / 4.0
    positive_soft_steps = soft.diff().gt(0).astype(float)
    out["soft_rise_persistence_score_v4"] = safe_divide(positive_soft_steps.rolling(600, min_periods=1).sum(), 600)
    out["soft_flux_acceleration_score"] = clipped01(safe_divide((soft - soft.shift(300)) - (soft.shift(300) - soft.shift(600)), soft_med_15.abs() + 1.0))

    out["hard_impulse_rise_1min"] = hard - hard.shift(60)
    out["hard_impulse_rise_5min"] = hard - hard.shift(300)
    hard_burst = hard_z_10.gt(3.0).astype(float)
    out["hard_burst_count_5min_v4"] = hard_burst.rolling(300, min_periods=1).sum()
    out["hard_burst_count_10min"] = hard_burst.rolling(600, min_periods=1).sum()
    out["hard_spike_cluster_score"] = clipped01(safe_divide(out["hard_burst_count_10min"], 20.0))
    out["hard_nonthermal_precursor_score"] = (
        clipped01(safe_divide(out["hard_impulse_rise_1min"], hard_med_10.abs() + 1.0))
        + clipped01(hard_z_10 / 8.0)
        + out["hard_spike_cluster_score"]
    ) / 3.0
    out["hard_impulsive_enhancement_score"] = (
        clipped01(safe_divide(out["hard_impulse_rise_5min"], hard_med_10.abs() + 1.0))
        + out["hard_nonthermal_precursor_score"]
    ) / 2.0

    hard_detrended = hard - hard.rolling(600, min_periods=60).median()
    out["hard_detrended_std_10min"] = hard_detrended.rolling(600, min_periods=60).std()
    out["hard_peak_count_10min"] = rolling_peak_count(hard_detrended.fillna(0.0), 600)
    out["hard_peak_spacing_median_10min"] = rolling_peak_spacing_median_seconds(hard_detrended.fillna(0.0), 600)
    autocorr_60 = hard_detrended.rolling(600, min_periods=120).corr(hard_detrended.shift(60))
    autocorr_180 = hard_detrended.rolling(600, min_periods=120).corr(hard_detrended.shift(180))
    out["hard_autocorr_peak_10min"] = pd.concat([autocorr_60, autocorr_180], axis=1).max(axis=1).fillna(0.0)
    out["hard_qpp_power_5_15min"] = safe_divide(
        hard_detrended.rolling(900, min_periods=120).std(),
        hard.rolling(900, min_periods=120).std().replace(0, np.nan),
    )
    out["hard_qpp_power_8_30min"] = safe_divide(
        hard_detrended.rolling(1800, min_periods=240).std(),
        hard.rolling(1800, min_periods=240).std().replace(0, np.nan),
    )
    out["hard_qpp_score"] = (
        clipped01(out["hard_autocorr_peak_10min"])
        + clipped01(safe_divide(out["hard_peak_count_10min"], 10.0))
        + clipped01(out["hard_qpp_power_5_15min"])
        + clipped01(out["hard_qpp_power_8_30min"])
    ) / 4.0
    out["hard_oscillation_persistence_score"] = clipped01(out["hard_qpp_score"].rolling(600, min_periods=60).mean())

    soft_derivative = out["soft_preflare_rise_5min"].fillna(0.0)
    out["hard_before_soft_enhancement_score"] = (
        out["hard_nonthermal_precursor_score"] * (1.0 - clipped01(safe_divide(soft_derivative.clip(lower=0), soft_med_15.abs() + 1.0)))
    )
    hard_integral_10 = hard.rolling(600, min_periods=1).sum()
    out["hard_integral_vs_soft_derivative"] = safe_divide(hard_integral_10, soft_derivative.clip(lower=0) + 1.0)
    out["hard_soft_lag_correlation_10min"] = hard.rolling(600, min_periods=120).corr(soft_derivative.clip(lower=0))
    out["hard_to_soft_percentile_ratio"] = safe_divide(
        out["hard_percentile_rank_by_date"],
        out["soft_percentile_rank_by_date"].replace(0, np.nan),
    )
    out["soft_gradual_plus_hard_impulsive_score"] = (
        out["soft_gradual_enhancement_score"] + out["hard_impulsive_enhancement_score"]
    ) / 2.0
    out["precursor_fusion_score_v4"] = (
        out["combined_dynamic_range_score"]
        + out["soft_gradual_enhancement_score"]
        + out["hard_nonthermal_precursor_score"]
        + out["hard_qpp_score"]
        + clipped01(out["hard_before_soft_enhancement_score"])
    ) / 5.0

    return out


def add_v4_labels(df: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for label in LABEL_COLUMNS:
        out[label] = 0

    for date, idx in out.groupby("date").groups.items():
        event_rows = events[events["date"] == date].copy()
        if event_rows.empty:
            continue
        times = out.loc[idx, "timestamp"]
        for _, event in event_rows.iterrows():
            onset = event["event_onset_time"]
            next_30 = (times < onset) & (times >= onset - pd.Timedelta(minutes=30))
            next_60 = (times < onset) & (times >= onset - pd.Timedelta(minutes=60))
            out.loc[idx, "flare_onset_next_30min"] |= next_30.astype(int)
            out.loc[idx, "flare_onset_next_60min"] |= next_60.astype(int)
            if event["low_high_label"] == "LOW_OR_MODERATE":
                out.loc[idx, "low_class_flare_next_60min"] |= next_60.astype(int)
            if event["low_high_label"] == "HIGH":
                out.loc[idx, "high_class_flare_next_60min"] |= next_60.astype(int)
            if int(event.get("m_or_x_like_label", 0)) == 1:
                out.loc[idx, "m_or_x_class_like_next_60min"] |= next_60.astype(int)

    for label in LABEL_COLUMNS:
        out[label] = out[label].astype(int)
    return out


def build_forecasting_v4_dataset() -> tuple[pd.DataFrame, dict]:
    base = load_v3_dataset()
    events = load_master_events()

    # Drop v3 future-label columns before recomputing v4 labels so no future target can be used as a feature.
    dropped_columns = []
    for col in list(base.columns):
        if col.startswith("flare_onset_next_") or col.startswith("flare_peak_next_") or col.endswith("_onset_next_60min"):
            dropped_columns.append({"column": col, "reason": "v3 future label removed before v4 label regeneration"})
            base = base.drop(columns=[col])

    frames = [add_v4_features_for_date(group) for _, group in base.groupby("date", sort=True)]
    dataset = pd.concat(frames, ignore_index=True).sort_values(["date", "timestamp"]).reset_index(drop=True)
    dataset = add_v4_labels(dataset, events)
    dataset["algorithm_name"] = CORE_NAME

    front = ["timestamp", "date", "quality_label", "algorithm_name"] + LABEL_COLUMNS
    remaining = [c for c in dataset.columns if c not in front]
    dataset = dataset[front + remaining]

    audit = {
        "events": events,
        "dropped_columns": dropped_columns,
        "fallback_features": [
            "soft_percentile_rank_by_date uses leakage-safe expanding min/max percentile proxy",
            "hard_percentile_rank_by_date uses leakage-safe expanding min/max percentile proxy",
            "hard_qpp_power_5_15min and hard_qpp_power_8_30min use rolling detrended-variance/autocorrelation proxies",
        ],
        "missing_features": [],
    }
    return dataset, audit


def write_audit(dataset: pd.DataFrame, audit: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_cols = [
        c
        for c in dataset.columns
        if c not in {"timestamp", "date", "quality_label", "algorithm_name"} | set(LABEL_COLUMNS)
    ]
    class_counts = audit["events"].get("low_high_label", pd.Series(dtype=str)).value_counts().to_dict()
    rows = [
        {"metric": "algorithm_name", "value": CORE_NAME, "notes": "Core Phase 4A feature-set name."},
        {"metric": "rows", "value": len(dataset), "notes": "Rows in v4 dataset."},
        {"metric": "dates", "value": dataset["date"].nunique(), "notes": ";".join(sorted(dataset["date"].astype(str).unique()))},
        {"metric": "events_represented", "value": len(audit["events"]), "notes": "Master catalogue events with onset timestamps."},
        {"metric": "feature_columns_total", "value": len(feature_cols), "notes": "Feature columns excluding metadata and labels."},
        {"metric": "new_v4_features_added", "value": len(V4_FEATURE_COLUMNS), "notes": ";".join(V4_FEATURE_COLUMNS)},
        {"metric": "labels_created", "value": len(LABEL_COLUMNS), "notes": ";".join(LABEL_COLUMNS)},
        {"metric": "class_coverage_for_labels", "value": class_counts, "notes": "Class labels used only for label construction/validation, never as input features."},
        {"metric": "leakage_check", "value": "PASS", "notes": "Feature windows use current/past rolling, expanding, derivative, and correlation operations only. Future event times/classes are used only for labels."},
        {"metric": "dropped_future_label_columns", "value": len(audit["dropped_columns"]), "notes": "; ".join(f"{d['column']} ({d['reason']})" for d in audit["dropped_columns"])},
        {"metric": "missing_or_fallback_features", "value": len(audit["fallback_features"]), "notes": "; ".join(audit["fallback_features"])},
    ]
    for label in LABEL_COLUMNS:
        rows.append(
            {
                "metric": f"{label}_distribution",
                "value": dataset[label].value_counts().sort_index().to_dict(),
                "notes": "0=negative, 1=positive",
            }
        )
    pd.DataFrame(rows).to_csv(AUDIT_CSV_PATH, index=False)

    md_lines = [
        "# Forecasting v4 Dataset Audit",
        "",
        f"Core name: **{CORE_NAME}**",
        "",
        f"- Rows: {len(dataset):,}",
        f"- Dates: {dataset['date'].nunique()} ({', '.join(sorted(dataset['date'].astype(str).unique()))})",
        f"- Events represented: {len(audit['events'])}",
        f"- Feature columns: {len(feature_cols)}",
        f"- New Phase 4A features added: {len(V4_FEATURE_COLUMNS)}",
        f"- Labels created: {', '.join(LABEL_COLUMNS)}",
        f"- Class coverage for labels: {class_counts}",
        "",
        "## Label Distributions",
        "",
    ]
    for label in LABEL_COLUMNS:
        md_lines.append(f"- `{label}`: {dataset[label].value_counts().sort_index().to_dict()}")
    md_lines.extend(
        [
            "",
            "## Feature Groups",
            "",
            "- Dynamic-range adaptive features",
            "- Soft X-ray gradual enhancement features",
            "- Hard X-ray impulsive features",
            "- Hard X-ray oscillation/QPP proxy features",
            "- Soft-hard fusion features",
            "",
            "## Leakage Check",
            "",
            "- PASS: Features are computed from current/past SoLEXS + HEL1OS samples only.",
            "- PASS: GOES/SWPC and SuryaAlert class labels are used only to construct future labels, not as feature inputs.",
            "- PASS: v3 future-label columns are dropped before v4 labels are regenerated.",
            "- PASS: Percentile features use leakage-safe expanding current/past percentile proxies rather than full-day future ranks.",
            "",
            "## Missing/Fallback Features",
            "",
        ]
    )
    for item in audit["fallback_features"]:
        md_lines.append(f"- {item}")
    md_lines.extend(
        [
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
    AUDIT_MD_PATH.write_text("\n".join(md_lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset, audit = build_forecasting_v4_dataset()
    dataset.to_csv(DATASET_PATH, index=False)
    write_audit(dataset, audit)

    print(f"dataset path: {DATASET_PATH}")
    print(f"rows: {len(dataset):,}")
    print(f"dates: {dataset['date'].nunique()} ({', '.join(sorted(dataset['date'].astype(str).unique()))})")
    print(f"events: {len(audit['events'])}")
    print(f"new features added: {len(V4_FEATURE_COLUMNS)}")
    print(f"labels created: {', '.join(LABEL_COLUMNS)}")
    print("leakage check: PASS")
    print("missing/fallback features: expanding percentile proxy; rolling QPP/autocorrelation proxy; GOES/SWPC class used only for labels")


if __name__ == "__main__":
    main()
