from __future__ import annotations

import numpy as np
import pandas as pd


SOFT_COL = "soft_solexs_2_22"
HARD_CDTE_COL = "hard_cdte_5_20"
HARD_CZT_COL = "hard_czt_20_40"
EPS = 1.0e-6


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _safe_div(a: pd.Series, b: pd.Series | float) -> pd.Series:
    return a / (b + EPS)


def _past_zscore(x: pd.Series, window: int) -> pd.Series:
    mean = x.rolling(window, min_periods=max(5, window // 5)).mean()
    std = x.rolling(window, min_periods=max(5, window // 5)).std().replace(0, np.nan)
    return ((x - mean) / (std + EPS)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _past_percentile_proxy(x: pd.Series) -> pd.Series:
    expanding_mean = x.expanding(min_periods=30).mean()
    expanding_std = x.expanding(min_periods=30).std().replace(0, np.nan)
    z = ((x - expanding_mean) / (expanding_std + EPS)).clip(-6, 6).fillna(0.0)
    return 1.0 / (1.0 + np.exp(-z))


def _past_autocorr(x: pd.Series, window: int, lag: int) -> pd.Series:
    centered = x - x.rolling(window, min_periods=max(10, window // 4)).mean()
    return centered.rolling(window, min_periods=max(10, window // 4)).corr(centered.shift(lag)).fillna(0.0)


def _hard_peak_count_proxy(score: pd.Series, window: int = 600) -> pd.Series:
    # This is a burst/peak-count proxy, not validated QPP or true inter-peak spacing stability.
    threshold = score.rolling(window, min_periods=30).median() + score.rolling(window, min_periods=30).std().fillna(0.0)
    peaks = score.gt(threshold).astype(int)
    counts = peaks.rolling(window, min_periods=30).sum()
    return (counts / max(1, window / 120)).clip(0, 3).fillna(0.0)


def _safe_integral_ratio(raw: pd.Series) -> pd.Series:
    cleaned = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
    logged = np.log1p(cleaned)
    return logged.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(upper=20.0)


def _add_group_features(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("timestamp").copy()
    soft = _num(g, SOFT_COL)
    hard_cdte = _num(g, HARD_CDTE_COL)
    hard_czt = _num(g, HARD_CZT_COL)
    hard = hard_cdte + hard_czt
    soft_score = _num(g, "soft_score")
    hard_score = _num(g, "hard_score")

    g["soft_flux_rise_5min"] = soft - soft.shift(300)
    g["soft_flux_rise_15min"] = soft - soft.shift(900)
    g["soft_flux_rise_30min"] = soft - soft.shift(1800)
    soft_bg = soft.rolling(1800, min_periods=120).median()
    g["soft_background_lift"] = _safe_div(soft - soft_bg, soft_bg.abs())
    soft_d1 = soft.diff(60)
    g["soft_acceleration"] = soft_d1 - soft_d1.shift(300)
    g["soft_preflare_enhancement_score"] = (
        _past_zscore(g["soft_flux_rise_15min"].fillna(0.0), 1800).clip(lower=0)
        + _past_zscore(g["soft_background_lift"].fillna(0.0), 1800).clip(lower=0)
        + _past_zscore(g["soft_acceleration"].fillna(0.0), 900).clip(lower=0)
    )

    g["hard_flux_rise_1min"] = hard - hard.shift(60)
    g["hard_flux_rise_5min"] = hard - hard.shift(300)
    hard_czt_d1 = hard_czt.diff(60)
    g["hard_czt_acceleration"] = hard_czt_d1 - hard_czt_d1.shift(120)
    g["czt_to_cdte_ratio"] = _safe_div(hard_czt, hard_cdte + EPS)
    g["rolling_hard_to_soft_ratio_1min"] = _safe_div(
        hard.rolling(60, min_periods=5).mean(),
        soft.rolling(60, min_periods=5).mean() + EPS,
    )
    g["rolling_hard_to_soft_ratio_3min"] = _safe_div(
        hard.rolling(180, min_periods=10).mean(),
        soft.rolling(180, min_periods=10).mean() + EPS,
    )
    hard_med = hard.rolling(600, min_periods=60).median()
    hard_std = hard.rolling(600, min_periods=60).std().fillna(0.0)
    spike = hard.gt(hard_med + 3.0 * hard_std).astype(int)
    g["hard_spike_cluster_score"] = spike.rolling(300, min_periods=30).sum().fillna(0.0)
    g["hard_burst_density"] = hard_score.gt(4.0).astype(int).rolling(300, min_periods=30).mean().fillna(0.0)
    g["hard_impulsive_precursor_score"] = (
        _past_zscore(g["hard_flux_rise_1min"].fillna(0.0), 600).clip(lower=0)
        + _past_zscore(g["hard_flux_rise_5min"].fillna(0.0), 900).clip(lower=0)
        + g["hard_spike_cluster_score"].clip(0, 10) / 10.0
        + g["hard_burst_density"].clip(0, 1)
    )

    hard_trend = hard.rolling(600, min_periods=60).median()
    detrended = hard - hard_trend
    g["hard_detrended_variance_10min"] = detrended.rolling(600, min_periods=60).var().fillna(0.0)
    g["hard_autocorrelation_score"] = _past_autocorr(hard, 600, 60).clip(lower=0)
    g["hard_peak_count_proxy"] = _hard_peak_count_proxy(hard_score, 600)
    g["hard_oscillation_proxy_score"] = (
        _past_zscore(g["hard_detrended_variance_10min"], 1800).clip(lower=0)
        + g["hard_autocorrelation_score"]
        + g["hard_peak_count_proxy"]
    )

    g["log_soft_flux"] = np.log1p(soft.clip(lower=0))
    g["log_hard_flux"] = np.log1p(hard.clip(lower=0))
    g["soft_expanding_percentile"] = _past_percentile_proxy(g["log_soft_flux"])
    g["hard_expanding_percentile"] = _past_percentile_proxy(g["log_hard_flux"])
    g["soft_dynamic_range_score"] = g["soft_expanding_percentile"] * _past_zscore(g["log_soft_flux"], 3600).clip(lower=0)
    g["hard_dynamic_range_score"] = g["hard_expanding_percentile"] * _past_zscore(g["log_hard_flux"], 3600).clip(lower=0)

    g["hard_before_soft_score"] = hard_score.rolling(300, min_periods=30).max().shift(60).fillna(0.0) - soft_score.rolling(300, min_periods=30).max().fillna(0.0)
    g["hard_soft_lag_correlation"] = hard.rolling(600, min_periods=60).corr(soft.shift(120)).fillna(0.0)
    hard_integral = hard.rolling(300, min_periods=30).sum().fillna(0.0)
    soft_derivative = soft.diff(60).clip(lower=0).fillna(0.0)
    soft_d1 = soft.diff(60)
    hard_int_3min = hard.rolling(180, min_periods=10).sum()
    g["neupert_derivative_proxy"] = _safe_div(
        hard.rolling(60, min_periods=5).mean(),
        soft_d1.abs().rolling(60, min_periods=5).mean() + EPS,
    )
    g["neupert_coupling_proxy_3min"] = _safe_div(
        hard_int_3min,
        soft.rolling(180, min_periods=10).mean() + EPS,
    )
    g["hard_integral_vs_soft_derivative"] = _safe_div(hard_integral, soft_derivative.rolling(300, min_periods=30).sum().fillna(0.0))
    g["hard_integral_vs_soft_derivative_safe"] = _safe_integral_ratio(g["hard_integral_vs_soft_derivative"])
    if {"hard_trigger", "soft_trigger"}.issubset(g.columns):
        hard_trigger = pd.to_numeric(g["hard_trigger"], errors="coerce").fillna(0).astype(bool)
        soft_trigger = pd.to_numeric(g["soft_trigger"], errors="coerce").fillna(0).astype(bool)
        last_hard_time = g["timestamp"].where(hard_trigger).ffill()
        last_soft_time = g["timestamp"].where(soft_trigger).ffill()
        hard_before_soft = last_hard_time.notna() & (last_soft_time.isna() | (last_hard_time > last_soft_time))
        g["hard_before_soft_active_gap_min"] = (
            (g["timestamp"] - last_hard_time).dt.total_seconds().div(60.0).where(hard_before_soft, 0.0)
        )
    g["soft_hard_precursor_fusion_score"] = (
        g["soft_preflare_enhancement_score"].clip(0, 6)
        + g["hard_impulsive_precursor_score"].clip(0, 6)
        + g["hard_oscillation_proxy_score"].clip(0, 6)
        + g["soft_dynamic_range_score"].clip(0, 6)
        + g["hard_dynamic_range_score"].clip(0, 6)
        + g["hard_before_soft_score"].clip(lower=0, upper=6)
    ) / 6.0

    new_cols = [c for c in g.columns if c not in group.columns]
    g[new_cols] = g[new_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return g


def add_v7lite_xray_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "date", SOFT_COL, HARD_CDTE_COL, HARD_CZT_COL}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"v7-Lite feature build missing required columns: {sorted(missing)}")
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, format="mixed", errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values(["date", "timestamp"]).reset_index(drop=True)
    parts = []
    for _date, group in out.groupby("date", sort=False):
        parts.append(_add_group_features(group))
    return pd.concat(parts, ignore_index=True) if parts else out


V7LITE_FEATURE_COLUMNS = [
    "soft_flux_rise_5min",
    "soft_flux_rise_15min",
    "soft_flux_rise_30min",
    "soft_background_lift",
    "soft_acceleration",
    "soft_preflare_enhancement_score",
    "hard_flux_rise_1min",
    "hard_flux_rise_5min",
    "hard_czt_acceleration",
    "czt_to_cdte_ratio",
    "rolling_hard_to_soft_ratio_1min",
    "rolling_hard_to_soft_ratio_3min",
    "hard_spike_cluster_score",
    "hard_burst_density",
    "hard_impulsive_precursor_score",
    "hard_detrended_variance_10min",
    "hard_autocorrelation_score",
    "hard_peak_count_proxy",
    "hard_oscillation_proxy_score",
    "log_soft_flux",
    "log_hard_flux",
    "soft_expanding_percentile",
    "hard_expanding_percentile",
    "soft_dynamic_range_score",
    "hard_dynamic_range_score",
    "hard_before_soft_score",
    "hard_soft_lag_correlation",
    "neupert_derivative_proxy",
    "neupert_coupling_proxy_3min",
    "hard_integral_vs_soft_derivative",
    "hard_integral_vs_soft_derivative_safe",
    "soft_hard_precursor_fusion_score",
]
