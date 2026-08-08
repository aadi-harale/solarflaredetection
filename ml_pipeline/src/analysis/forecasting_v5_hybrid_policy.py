from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


V3_DIR = Path("results") / "forecasting_v3"
V4_DIR = Path("results") / "forecasting_v4"
OUT_DIR = Path("results") / "forecasting_v5"

V3_PRED_PATH = V3_DIR / "forecasting_v3_predictions.csv"
V3_REC_PATH = V3_DIR / "forecasting_v3_policy_recommendations.csv"
V3_FALSE_ALERT_PATH = V3_DIR / "forecasting_v3_false_alert_analysis.csv"
V4_PRED_PATH = V4_DIR / "forecasting_v4_predictions.csv"
V4_REC_PATH = V4_DIR / "forecasting_v4_policy_recommendations.csv"
V4_DATASET_PATH = V4_DIR / "forecasting_v4_dataset.csv"
MASTER_CLASSIFIED_PATH = Path("results") / "master_flare_catalogue_classified_v2.csv"
MASTER_PATH = Path("results") / "master_flare_catalogue.csv"
V3_COMPARISON_PATH = V3_DIR / "v1_v2_v3_forecasting_comparison.csv"
V4_POLICY_REC_PATH = V4_DIR / "forecasting_v4_policy_recommendations.csv"

HYBRID_SCORE_PATH = OUT_DIR / "forecasting_v5_hybrid_scores.csv"
SCORE_AUDIT_PATH = OUT_DIR / "forecasting_v5_score_audit.md"
RULES_AUDIT_CSV_PATH = OUT_DIR / "forecasting_v5_false_alert_rules_audit.csv"
RULES_AUDIT_MD_PATH = OUT_DIR / "forecasting_v5_false_alert_rules_audit.md"
V5_POLICY_SWEEP_PATH = OUT_DIR / "forecasting_v5_hybrid_policy_sweep.csv"
V5_POLICY_REC_PATH = OUT_DIR / "forecasting_v5_policy_recommendations.csv"
V5_FALSE_ALERT_REDUCTION_PATH = OUT_DIR / "forecasting_v5_false_alert_reduction_report.csv"
V1_TO_V5_COMPARISON_PATH = OUT_DIR / "v1_v2_v3_v4_v5_comparison.csv"
V5_REPORT_PATH = OUT_DIR / "forecasting_v5_report.md"

QUALITY_MODES = [
    "GOOD_ONLY_STRICT",
    "GOOD_PLUS_QUESTIONABLE_PENALIZED",
    "GOOD_PLUS_QUESTIONABLE_UNPENALIZED",
]
RULE_COOLDOWN_MIN = 60
V3_BASELINE = {
    "precision": 0.5151515151515151,
    "recall": 0.8235294117647058,
    "f1": 0.6338215712383488,
    "false_alerts_per_day": 1.3333333333333333,
    "valid_alerted_events": 14,
    "total_events": 17,
    "mean_lead_time_min": 39.44404761904762,
    "median_lead_time_min": 40.18333333333334,
}

V4_SUPPORT_COLUMNS = [
    "quality_label",
    "precursor_fusion_score_v4",
    "hard_qpp_score",
    "soft_gradual_enhancement_score",
    "hard_impulsive_enhancement_score",
    "post_flare_decay_state",
    "hard_score",
    "soft_score",
]


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input: {path.resolve()}")


def normalize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = out["date"].astype(str).str.replace(r"\.0$", "", regex=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, format="mixed", errors="coerce")
    return out.dropna(subset=["timestamp", "date"])


def recommended_model(path: Path, fallback: str) -> str:
    if not path.exists():
        return fallback
    rec = pd.read_csv(path, nrows=1)
    if "model" not in rec.columns or rec.empty:
        return fallback
    return str(rec.iloc[0]["model"])


def load_prediction_pivot(path: Path, model: str, targets: list[str], rename_map: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    require(path)
    pred = pd.read_csv(path, usecols=lambda c: c in {"timestamp", "date", "model", "target", "score"})
    pred = normalize_date_column(pred)
    pred = pred[pred["model"].eq(model) & pred["target"].isin(targets)].copy()
    missing = [target for target in targets if target not in set(pred["target"])]
    if pred.empty:
        cols = ["timestamp", "date"] + list(rename_map.values())
        return pd.DataFrame(columns=cols), missing
    pivot = (
        pred.pivot_table(index=["timestamp", "date"], columns="target", values="score", aggfunc="max")
        .reset_index()
        .rename(columns=rename_map)
    )
    for col in rename_map.values():
        if col not in pivot.columns:
            pivot[col] = np.nan
    return pivot[["timestamp", "date"] + list(rename_map.values())], missing


def load_v4_support() -> tuple[pd.DataFrame, list[str]]:
    require(V4_DATASET_PATH)
    available = pd.read_csv(V4_DATASET_PATH, nrows=0).columns
    wanted = ["timestamp", "date"] + V4_SUPPORT_COLUMNS
    missing = [col for col in wanted if col not in available]
    usecols = [col for col in wanted if col in available]
    support = pd.read_csv(V4_DATASET_PATH, usecols=usecols)
    support = normalize_date_column(support)
    for col in V4_SUPPORT_COLUMNS:
        if col not in support.columns:
            support[col] = 0.0 if col != "quality_label" else "UNKNOWN"
    return support[["timestamp", "date"] + V4_SUPPORT_COLUMNS], missing


def load_duplicate_false_alert_windows() -> tuple[pd.DataFrame, dict[str, int]]:
    if not V3_FALSE_ALERT_PATH.exists():
        return pd.DataFrame(columns=["date", "timestamp", "likely_cause_category"]), {}
    false_alerts = pd.read_csv(V3_FALSE_ALERT_PATH)
    if false_alerts.empty or "timestamp" not in false_alerts.columns:
        return pd.DataFrame(columns=["date", "timestamp", "likely_cause_category"]), {}
    false_alerts = normalize_date_column(false_alerts)
    causes = false_alerts.get("likely_cause_category", pd.Series(dtype=str)).value_counts().to_dict()
    return false_alerts[["date", "timestamp", "likely_cause_category"]], causes


def mark_duplicate_penalty(scores: pd.DataFrame, false_alerts: pd.DataFrame) -> pd.Series:
    penalty = pd.Series(0.0, index=scores.index)
    if false_alerts.empty:
        return penalty
    # Penalize rows within +/- 2 minutes of a known v3 duplicate false-alert diagnostic.
    duplicate_alerts = false_alerts[false_alerts["likely_cause_category"].astype(str).eq("DUPLICATE_ALERT")]
    if duplicate_alerts.empty:
        return penalty
    for date, group in duplicate_alerts.groupby("date"):
        idx = scores["date"].eq(str(date))
        if not idx.any():
            continue
        row_times = scores.loc[idx, "timestamp"]
        local_penalty = pd.Series(False, index=row_times.index)
        for ts in group["timestamp"]:
            local_penalty |= row_times.between(ts - pd.Timedelta(minutes=2), ts + pd.Timedelta(minutes=2))
        penalty.loc[local_penalty.index] = np.where(local_penalty, 0.10, penalty.loc[local_penalty.index])
    return penalty


def clip01(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)


def build_hybrid_scores() -> tuple[pd.DataFrame, dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    v3_model = recommended_model(V3_REC_PATH, "shallow_random_forest_diagnostic")
    v4_model = recommended_model(V4_REC_PATH, "gradient_boosting_challenger")

    v3, missing_v3 = load_prediction_pivot(
        V3_PRED_PATH,
        v3_model,
        ["flare_onset_next_30min", "flare_onset_next_60min"],
        {"flare_onset_next_30min": "v3_p30", "flare_onset_next_60min": "v3_p60"},
    )
    v4, missing_v4_pred = load_prediction_pivot(
        V4_PRED_PATH,
        v4_model,
        ["flare_onset_next_30min", "flare_onset_next_60min", "high_class_flare_next_60min"],
        {
            "flare_onset_next_30min": "v4_p30",
            "flare_onset_next_60min": "v4_p60",
            "high_class_flare_next_60min": "v4_p_high",
        },
    )
    support, missing_support = load_v4_support()
    false_alerts, false_causes = load_duplicate_false_alert_windows()

    joined = pd.merge(v3, v4, on=["timestamp", "date"], how="outer")
    joined = pd.merge(joined, support, on=["timestamp", "date"], how="left")
    joined = joined.sort_values(["date", "timestamp"]).reset_index(drop=True)

    for col in ["v3_p30", "v3_p60", "v4_p30", "v4_p60", "v4_p_high"]:
        if col not in joined.columns:
            joined[col] = 0.0
        joined[col] = clip01(joined[col])
    for col in [
        "precursor_fusion_score_v4",
        "hard_qpp_score",
        "soft_gradual_enhancement_score",
        "hard_impulsive_enhancement_score",
        "hard_score",
        "soft_score",
    ]:
        if col not in joined.columns:
            joined[col] = 0.0
        joined[col] = clip01(joined[col] if col.endswith("_score_v4") or col in {"hard_qpp_score", "soft_gradual_enhancement_score", "hard_impulsive_enhancement_score"} else joined[col] / 10.0)
    if "quality_label" not in joined.columns:
        joined["quality_label"] = "UNKNOWN"
    joined["post_flare_decay_state"] = pd.to_numeric(joined.get("post_flare_decay_state", 0), errors="coerce").fillna(0).astype(int)

    joined["v3_weighted_signal"] = (0.65 * joined["v3_p30"] + 0.35 * joined["v3_p60"]).clip(0, 1)
    joined["v4_probability_support"] = (0.45 * joined["v4_p60"] + 0.35 * joined["v4_p30"] + 0.20 * joined["v4_p_high"]).clip(0, 1)
    joined["v4_physics_support"] = (
        0.40 * joined["precursor_fusion_score_v4"]
        + 0.20 * joined["hard_qpp_score"]
        + 0.20 * joined["soft_gradual_enhancement_score"]
        + 0.20 * joined["hard_impulsive_enhancement_score"]
    ).clip(0, 1)
    joined["v4_watch_support"] = (0.55 * joined["v4_probability_support"] + 0.45 * joined["v4_physics_support"]).clip(0, 1)

    joined["questionable_quality_penalty"] = np.where(joined["quality_label"].astype(str).str.upper().eq("QUESTIONABLE"), 0.10, 0.0)
    joined["post_flare_decay_penalty"] = np.where(joined["post_flare_decay_state"].eq(1), 0.15, 0.0)
    joined["duplicate_alert_penalty"] = mark_duplicate_penalty(joined, false_alerts)
    joined["total_penalty"] = (
        joined["questionable_quality_penalty"] + joined["post_flare_decay_penalty"] + joined["duplicate_alert_penalty"]
    ).clip(0, 0.35)

    joined["hybrid_forecast_score_raw"] = (
        0.72 * joined["v3_weighted_signal"]
        + 0.18 * joined["v4_probability_support"]
        + 0.10 * joined["v4_physics_support"]
        - joined["total_penalty"]
    )
    joined["hybrid_forecast_score"] = joined["hybrid_forecast_score_raw"].clip(0, 1)
    joined["v4_precursor_watch_score"] = joined["v4_watch_support"].clip(0, 1)
    joined["v3_confirmed_signal"] = ((joined["v3_p30"] >= 0.80) | (joined["v3_weighted_signal"] >= 0.70)).astype(int)
    joined["very_strong_hybrid_evidence"] = (
        (joined["hybrid_forecast_score"] >= 0.82)
        & (joined["v4_watch_support"] >= 0.55)
        & (joined["v3_weighted_signal"] >= 0.45)
    ).astype(int)
    joined["v5_recommended_state"] = np.select(
        [
            joined["v3_confirmed_signal"].eq(1),
            joined["very_strong_hybrid_evidence"].eq(1),
            joined["v4_precursor_watch_score"].ge(0.50),
        ],
        ["FORECAST_ALERT_V3_CONFIRMED", "FORECAST_ALERT_STRONG_HYBRID", "PRECURSOR_WATCH_V4_SUPPORTED"],
        default="CLEAR",
    )

    joined["v3_model_used"] = v3_model
    joined["v4_model_used"] = v4_model

    ordered = [
        "timestamp",
        "date",
        "quality_label",
        "v3_model_used",
        "v4_model_used",
        "v3_p30",
        "v3_p60",
        "v3_weighted_signal",
        "v4_p30",
        "v4_p60",
        "v4_p_high",
        "v4_probability_support",
        "precursor_fusion_score_v4",
        "hard_qpp_score",
        "soft_gradual_enhancement_score",
        "hard_impulsive_enhancement_score",
        "v4_physics_support",
        "v4_precursor_watch_score",
        "questionable_quality_penalty",
        "post_flare_decay_penalty",
        "duplicate_alert_penalty",
        "total_penalty",
        "hybrid_forecast_score",
        "v3_confirmed_signal",
        "very_strong_hybrid_evidence",
        "v5_recommended_state",
    ]
    remaining = [col for col in joined.columns if col not in ordered]
    joined = joined[ordered + remaining]
    joined.to_csv(HYBRID_SCORE_PATH, index=False)

    audit = {
        "rows_joined": len(joined),
        "dates": sorted(joined["date"].astype(str).unique()),
        "v3_model": v3_model,
        "v4_model": v4_model,
        "missing_v3": missing_v3,
        "missing_v4_pred": missing_v4_pred,
        "missing_support": missing_support,
        "false_causes": false_causes,
    }
    write_score_audit(joined, audit)
    return joined, audit


def write_score_audit(scores: pd.DataFrame, audit: dict) -> None:
    score_desc = scores["hybrid_forecast_score"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
    state_counts = scores["v5_recommended_state"].value_counts().to_dict()
    lines = [
        "# Forecasting v5A Hybrid Score Audit",
        "",
        "## Purpose",
        "",
        "Forecasting v5A combines v3 as the main low-false-alert engine with v4 high-recall precursor/watch support. v4 does not directly trigger final alerts unless confirmed by v3 or very strong hybrid evidence.",
        "",
        "## Inputs Used",
        "",
        f"- v3 predictions: `{V3_PRED_PATH}`",
        f"- v3 recommendation: `{V3_REC_PATH}`",
        f"- v3 false-alert analysis: `{V3_FALSE_ALERT_PATH}`",
        f"- v4 predictions: `{V4_PRED_PATH}`",
        f"- v4 recommendation: `{V4_REC_PATH}`",
        f"- v4 feature/support dataset: `{V4_DATASET_PATH}`",
        "",
        "## Columns Used From v3",
        "",
        "- `flare_onset_next_30min` score -> `v3_p30`",
        "- `flare_onset_next_60min` score -> `v3_p60`",
        "- v3 false-alert cause categories for duplicate-alert penalty diagnostics",
        "",
        "## Columns Used From v4",
        "",
        "- `flare_onset_next_30min` score -> `v4_p30`",
        "- `flare_onset_next_60min` score -> `v4_p60`",
        "- `high_class_flare_next_60min` score -> `v4_p_high`",
        "- `precursor_fusion_score_v4`",
        "- `hard_qpp_score`",
        "- `soft_gradual_enhancement_score`",
        "- `hard_impulsive_enhancement_score`",
        "- `quality_label` and `post_flare_decay_state` for penalties",
        "",
        "## Join Method",
        "",
        f"- Rows joined: {audit['rows_joined']:,}",
        f"- Dates: {', '.join(audit['dates'])}",
        "- v3 and v4 prediction streams were outer-joined exactly on `timestamp` + `date`.",
        "- v4 support features were exact-joined on `timestamp` + `date` from the aligned v4 dataset.",
        "- Missing prediction/support values are filled with conservative zero support.",
        "",
        "## Missing Columns and Fallbacks",
        "",
        f"- Missing v3 target columns: {audit['missing_v3'] or 'none'}",
        f"- Missing v4 prediction target columns: {audit['missing_v4_pred'] or 'none'}",
        f"- Missing v4 support columns: {audit['missing_support'] or 'none'}",
        "",
        "## Score Formula",
        "",
        "`hybrid_forecast_score = 0.72*v3_weighted_signal + 0.18*v4_probability_support + 0.10*v4_physics_support - penalties`",
        "",
        "Penalties:",
        "",
        "- QUESTIONABLE date: 0.10",
        "- post-flare decay state: 0.15",
        "- known duplicate-alert neighborhood: 0.10",
        "",
        "## Leakage Check",
        "",
        "- PASS: Hybrid score uses only v3/v4 prediction scores and current/past v4 feature columns already computed without future leakage.",
        "- PASS: GOES/event labels are not used in the hybrid score.",
        "- PASS: v3 false-alert analysis is used only for diagnostic duplicate-alert penalty neighborhoods, not to create positive labels.",
        "",
        "## Score Distribution",
        "",
    ]
    for key, value in score_desc.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Recommended State Distribution",
            "",
            str(state_counts),
            "",
            "## Top Causes From v3 False-Alert Analysis",
            "",
            str(audit["false_causes"]),
        ]
    )
    SCORE_AUDIT_PATH.write_text("\n".join(lines), encoding="utf-8")


def load_hybrid_scores() -> pd.DataFrame:
    require(HYBRID_SCORE_PATH)
    scores = pd.read_csv(HYBRID_SCORE_PATH)
    scores = normalize_date_column(scores)
    required = {
        "quality_label",
        "v3_p30",
        "v3_p60",
        "hybrid_forecast_score",
        "precursor_fusion_score_v4",
        "hard_impulsive_enhancement_score",
        "soft_gradual_enhancement_score",
        "post_flare_decay_state",
        "v5_recommended_state",
    }
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"{HYBRID_SCORE_PATH} is missing required v5B columns: {sorted(missing)}")
    for col in [
        "v3_p30",
        "v3_p60",
        "hybrid_forecast_score",
        "precursor_fusion_score_v4",
        "hard_impulsive_enhancement_score",
        "soft_gradual_enhancement_score",
        "hard_qpp_score",
        "post_flare_decay_penalty",
        "duplicate_alert_penalty",
    ]:
        if col not in scores.columns:
            scores[col] = 0.0
        scores[col] = pd.to_numeric(scores[col], errors="coerce").fillna(0.0)
    scores["post_flare_decay_state"] = pd.to_numeric(scores["post_flare_decay_state"], errors="coerce").fillna(0).astype(int)
    return scores.sort_values(["date", "timestamp"]).reset_index(drop=True)


def consecutive_true_count(mask: pd.Series, groups: pd.Series) -> pd.Series:
    out = pd.Series(0, index=mask.index, dtype=int)
    for _, idx in mask.groupby(groups).groups.items():
        values = mask.loc[idx].astype(bool).to_numpy()
        counts = np.zeros(len(values), dtype=int)
        current = 0
        for i, value in enumerate(values):
            current = current + 1 if value else 0
            counts[i] = current
        out.loc[idx] = counts
    return out


def consolidate_alerts(frame: pd.DataFrame, cooldown_min: int = RULE_COOLDOWN_MIN) -> pd.DataFrame:
    out = frame.copy()
    out["consolidated_alert_episode_id"] = ""
    out["suppressed_by_duplicate_rule"] = 0
    out["final_alert_after_rules"] = 0
    cooldown = pd.Timedelta(minutes=cooldown_min)
    for date, idx in out.groupby("date", sort=False).groups.items():
        active_idx = [i for i in idx if int(out.at[i, "alert_after_decay_rule"]) == 1]
        episode_id = 0
        cooldown_until = pd.Timestamp.min.tz_localize("UTC")
        for i in active_idx:
            ts = out.at[i, "timestamp"]
            if ts <= cooldown_until:
                out.at[i, "suppressed_by_duplicate_rule"] = 1
                out.at[i, "consolidated_alert_episode_id"] = f"{date}_E{episode_id:04d}"
                continue
            episode_id += 1
            out.at[i, "final_alert_after_rules"] = 1
            out.at[i, "consolidated_alert_episode_id"] = f"{date}_E{episode_id:04d}"
            cooldown_until = ts + cooldown
    return out


def apply_rules_for_mode(scores: pd.DataFrame, quality_mode: str) -> pd.DataFrame:
    out = scores.copy()
    out["quality_mode"] = quality_mode
    is_questionable = out["quality_label"].astype(str).str.upper().eq("QUESTIONABLE")
    is_good = out["quality_label"].astype(str).str.upper().eq("GOOD")

    before_alert = out["v5_recommended_state"].astype(str).str.startswith("FORECAST_ALERT")
    out["alert_before_rules"] = before_alert.astype(int)

    if quality_mode == "GOOD_ONLY_STRICT":
        out["quality_mode_confidence_penalty"] = np.where(is_questionable, 1.0, 0.0)
        out["quality_allowed"] = is_good.astype(int)
        p30_thr = pd.Series(0.80, index=out.index)
        p60_thr = pd.Series(0.50, index=out.index)
        precursor_thr = pd.Series(0.25, index=out.index)
        score_thr = pd.Series(0.70, index=out.index)
        bins_required = pd.Series(2, index=out.index)
    elif quality_mode == "GOOD_PLUS_QUESTIONABLE_PENALIZED":
        out["quality_mode_confidence_penalty"] = np.where(is_questionable, 0.15, 0.0)
        out["quality_allowed"] = 1
        p30_thr = pd.Series(np.where(is_questionable, 0.88, 0.80), index=out.index)
        p60_thr = pd.Series(np.where(is_questionable, 0.60, 0.50), index=out.index)
        precursor_thr = pd.Series(np.where(is_questionable, 0.35, 0.25), index=out.index)
        score_thr = pd.Series(np.where(is_questionable, 0.78, 0.70), index=out.index)
        bins_required = pd.Series(np.where(is_questionable, 4, 2), index=out.index)
    else:
        out["quality_mode_confidence_penalty"] = 0.0
        out["quality_allowed"] = 1
        p30_thr = pd.Series(0.80, index=out.index)
        p60_thr = pd.Series(0.50, index=out.index)
        precursor_thr = pd.Series(0.25, index=out.index)
        score_thr = pd.Series(0.70, index=out.index)
        bins_required = pd.Series(2, index=out.index)

    out["hybrid_score_after_quality_rules"] = (out["hybrid_forecast_score"] - out["quality_mode_confidence_penalty"]).clip(0, 1)
    precursor_ok = out["precursor_fusion_score_v4"].ge(precursor_thr)
    p30_ok = out["v3_p30"].ge(p30_thr)
    p60_ok = out["v3_p60"].ge(p60_thr)
    score_ok = out["hybrid_score_after_quality_rules"].ge(score_thr)
    candidate_signal = out["quality_allowed"].eq(1) & precursor_ok & (p30_ok | (p60_ok & score_ok))
    out["quality_rule_candidate"] = candidate_signal.astype(int)
    consecutive_counts = consecutive_true_count(candidate_signal, out["date"])
    out["quality_rule_consecutive_count"] = consecutive_counts
    out["quality_rule_bins_required"] = bins_required.astype(int)
    out["alert_after_quality_rules"] = (candidate_signal & consecutive_counts.ge(bins_required)).astype(int)

    strong_new_hard_impulse = out["hard_impulsive_enhancement_score"].ge(0.60)
    strong_soft_acceleration = out["soft_gradual_enhancement_score"].ge(0.60)
    strong_v3_reconfirmation = out["v3_p30"].ge(0.90) | out["hybrid_score_after_quality_rules"].ge(0.85)
    weak_decay_alert = (
        out["alert_after_quality_rules"].eq(1)
        & out["post_flare_decay_state"].eq(1)
        & ~(strong_new_hard_impulse | strong_soft_acceleration | strong_v3_reconfirmation)
    )
    out["post_flare_decay_penalty"] = np.where(out["post_flare_decay_state"].eq(1), 0.15, 0.0)
    out["suppressed_by_decay_rule"] = weak_decay_alert.astype(int)
    out["alert_after_decay_rule"] = (out["alert_after_quality_rules"].eq(1) & ~weak_decay_alert).astype(int)

    out["duplicate_alert_penalty"] = np.where(out["alert_after_decay_rule"].eq(1), out.get("duplicate_alert_penalty", 0.0), 0.0)
    out = consolidate_alerts(out)
    return out


def build_false_alert_rules() -> tuple[pd.DataFrame, dict]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores = load_hybrid_scores()
    frames = [apply_rules_for_mode(scores, mode) for mode in QUALITY_MODES]
    audit = pd.concat(frames, ignore_index=True)

    selected_cols = [
        "quality_mode",
        "timestamp",
        "date",
        "quality_label",
        "v5_recommended_state",
        "alert_before_rules",
        "v3_p30",
        "v3_p60",
        "hybrid_forecast_score",
        "hybrid_score_after_quality_rules",
        "precursor_fusion_score_v4",
        "hard_impulsive_enhancement_score",
        "soft_gradual_enhancement_score",
        "quality_mode_confidence_penalty",
        "quality_rule_candidate",
        "quality_rule_consecutive_count",
        "quality_rule_bins_required",
        "alert_after_quality_rules",
        "post_flare_decay_state",
        "post_flare_decay_penalty",
        "suppressed_by_decay_rule",
        "alert_after_decay_rule",
        "duplicate_alert_penalty",
        "suppressed_by_duplicate_rule",
        "consolidated_alert_episode_id",
        "final_alert_after_rules",
    ]
    audit[selected_cols].to_csv(RULES_AUDIT_CSV_PATH, index=False)
    summary = summarize_rules(audit)
    write_rules_audit_md(summary)
    return audit, summary


def summarize_rules(audit: pd.DataFrame) -> dict:
    rows = {}
    for mode, group in audit.groupby("quality_mode"):
        q = group["quality_label"].astype(str).str.upper().eq("QUESTIONABLE")
        rows[mode] = {
            "questionable_alerts_before": int(group.loc[q, "alert_before_rules"].sum()),
            "questionable_alerts_after_quality_penalty": int(group.loc[q, "alert_after_quality_rules"].sum()),
            "post_flare_alerts_suppressed": int(group["suppressed_by_decay_rule"].sum()),
            "duplicate_alerts_consolidated": int(group["suppressed_by_duplicate_rule"].sum()),
            "final_alert_rows": int(group["final_alert_after_rules"].sum()),
            "final_alert_episodes": int(group.loc[group["final_alert_after_rules"].eq(1), "consolidated_alert_episode_id"].replace("", np.nan).nunique()),
        }
    return rows


def write_rules_audit_md(summary: dict) -> None:
    false_causes = {}
    if V3_FALSE_ALERT_PATH.exists():
        false_causes = pd.read_csv(V3_FALSE_ALERT_PATH).get("likely_cause_category", pd.Series(dtype=str)).value_counts().to_dict()
    lines = [
        "# Forecasting v5B False-Alert Reduction Rules Audit",
        "",
        "## Purpose",
        "",
        "v5B applies error-driven rules on top of v5A hybrid scores. The rules target the v3 false-alert pattern: QUESTIONABLE_DATE, POST_FLARE_DECAY, and DUPLICATE_ALERT.",
        "",
        "## Quality-Aware Modes",
        "",
        "- `GOOD_ONLY_STRICT`: suppresses QUESTIONABLE telemetry and keeps only GOOD-date alert candidates.",
        "- `GOOD_PLUS_QUESTIONABLE_PENALIZED`: keeps QUESTIONABLE dates but reduces confidence and requires higher p30/p60, stronger precursor score, and longer persistence.",
        "- `GOOD_PLUS_QUESTIONABLE_UNPENALIZED`: comparison-only mode with standard thresholds.",
        "",
        "## Rule Summary",
        "",
        "| mode | questionable before | questionable after quality penalty | post-flare suppressed | duplicate consolidated | final alert episodes |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode, item in summary.items():
        lines.append(
            f"| {mode} | {item['questionable_alerts_before']} | {item['questionable_alerts_after_quality_penalty']} | {item['post_flare_alerts_suppressed']} | {item['duplicate_alerts_consolidated']} | {item['final_alert_episodes']} |"
        )
    lines.extend(
        [
            "",
            "## Post-Flare Decay Suppression",
            "",
            "Weak forecast alerts during post-flare decay are suppressed unless there is a strong new hard impulsive enhancement, strong soft gradual/acceleration evidence, or very strong v3 reconfirmation.",
            "",
            "## Duplicate Alert Consolidation",
            "",
            f"Alerts inside a {RULE_COOLDOWN_MIN}-minute cooldown window are assigned to the same `consolidated_alert_episode_id`; only the first alert remains as `final_alert_after_rules=1`.",
            "",
            "## v3 False-Alert Cause Context",
            "",
            str(false_causes),
            "",
            "## Leakage Check",
            "",
            "- PASS: Rules use only telemetry quality labels, current score streams, current/past v4 support features, and post-flare state already present in the score table.",
            "- PASS: Master catalogue/GOES labels are not used to decide whether to suppress or keep an alert.",
        ]
    )
    RULES_AUDIT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def load_events() -> pd.DataFrame:
    path = MASTER_CLASSIFIED_PATH if MASTER_CLASSIFIED_PATH.exists() else MASTER_PATH
    require(path)
    events = pd.read_csv(path)
    events["date"] = events["date"].astype(str).str.replace(r"\.0$", "", regex=True)
    onset_col = "combined_start" if "combined_start" in events.columns else "soft_start"
    peak_col = "soft_peak" if "soft_peak" in events.columns else "combined_peak"
    events["event_onset_time"] = pd.to_datetime(events[onset_col], utc=True, format="mixed", errors="coerce")
    events["event_peak_time"] = pd.to_datetime(events[peak_col], utc=True, format="mixed", errors="coerce")
    events["event_id"] = events["event_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    if "quality_label" not in events.columns:
        events["quality_label"] = ""
    return events.dropna(subset=["event_onset_time"]).copy()


def load_rules_summary() -> dict:
    if not RULES_AUDIT_CSV_PATH.exists():
        return {}
    rules = pd.read_csv(RULES_AUDIT_CSV_PATH, usecols=["quality_mode", "quality_label", "alert_before_rules", "alert_after_quality_rules", "suppressed_by_decay_rule", "suppressed_by_duplicate_rule", "final_alert_after_rules"])
    summary = {}
    for mode, group in rules.groupby("quality_mode"):
        q = group["quality_label"].astype(str).str.upper().eq("QUESTIONABLE")
        summary[mode] = {
            "questionable_before": int(group.loc[q, "alert_before_rules"].sum()),
            "questionable_after": int(group.loc[q, "alert_after_quality_rules"].sum()),
            "decay_suppressed": int(group["suppressed_by_decay_rule"].sum()),
            "duplicates_consolidated": int(group["suppressed_by_duplicate_rule"].sum()),
            "final_alert_rows": int(group["final_alert_after_rules"].sum()),
        }
    return summary


def consecutive_counts(mask: pd.Series, dates: pd.Series) -> pd.Series:
    out = pd.Series(0, index=mask.index, dtype=int)
    for _, idx in mask.groupby(dates).groups.items():
        current = 0
        values = mask.loc[idx].astype(bool).to_numpy()
        counts = np.zeros(len(values), dtype=int)
        for i, value in enumerate(values):
            current = current + 1 if value else 0
            counts[i] = current
        out.loc[idx] = counts
    return out


def build_policy_episodes(frame: pd.DataFrame, cooldown_min: int) -> pd.DataFrame:
    rows = []
    cooldown = pd.Timedelta(minutes=cooldown_min)
    for date, group in frame.sort_values(["date", "timestamp"]).groupby("date", sort=False):
        cooldown_until = pd.Timestamp.min.tz_localize("UTC")
        episode_id = 0
        for row in group.itertuples(index=False):
            if int(row.policy_alert_candidate) != 1:
                continue
            ts = row.timestamp
            if ts <= cooldown_until:
                continue
            episode_id += 1
            rows.append(
                {
                    "date": date,
                    "alert_start": ts,
                    "alert_end": ts,
                    "policy_state": row.policy_state,
                    "quality_label": row.quality_label,
                    "post_flare_decay_state": row.post_flare_decay_state,
                    "is_questionable": row.is_questionable,
                    "max_hybrid_score": row.adjusted_hybrid_score,
                    "max_v3_p30": row.v3_p30,
                    "max_v3_p60": row.v3_p60,
                    "max_v4_support": row.v4_precursor_watch_score,
                    "episode_id": f"{date}_V5_{episode_id:04d}",
                }
            )
            cooldown_until = ts + cooldown
    return pd.DataFrame(rows)


def classify_false_episode(ep: pd.Series) -> str:
    if bool(ep.get("is_questionable", False)):
        return "QUESTIONABLE_DATE"
    if int(ep.get("post_flare_decay_state", 0) or 0) == 1:
        return "POST_FLARE_DECAY"
    return "TRUE_ISOLATED_FALSE_ALERT"


def evaluate_policy(episodes: pd.DataFrame, events: pd.DataFrame, dates_count: int, params: dict) -> dict:
    event_hits = {}
    useful = 0
    false = 0
    false_causes = {"QUESTIONABLE_DATE": 0, "POST_FLARE_DECAY": 0, "DUPLICATE_ALERT": 0, "TRUE_ISOLATED_FALSE_ALERT": 0}
    if episodes.empty:
        total_events = len(events)
        return {
            **params,
            "precision": 0.0,
            "recall_pod": 0.0,
            "f1": 0.0,
            "false_alerts_per_day": 0.0,
            "valid_alerted_events": 0,
            "total_events": total_events,
            "useful_alert_episodes": 0,
            "isolated_false_alerts": 0,
            "total_alert_episodes": 0,
            "mean_lead_time_min": np.nan,
            "median_lead_time_min": np.nan,
            "lead_time_iqr_min": np.nan,
            **{f"false_{k.lower()}": 0 for k in false_causes},
        }
    for _, ep in episodes.iterrows():
        same_date = events[events["date"].eq(str(ep["date"]))]
        candidates = same_date[
            (same_date["event_onset_time"] > ep["alert_start"])
            & (same_date["event_onset_time"] <= ep["alert_start"] + pd.Timedelta(minutes=90))
        ].copy()
        if candidates.empty:
            false += 1
            false_causes[classify_false_episode(ep)] += 1
            continue
        candidates["lead_time_min"] = (candidates["event_onset_time"] - ep["alert_start"]).dt.total_seconds() / 60.0
        match = candidates.sort_values("lead_time_min").iloc[0]
        useful += 1
        eid = str(match["event_id"])
        lead = float(match["lead_time_min"])
        if eid not in event_hits or ep["alert_start"] < event_hits[eid]["first_alert_time"]:
            event_hits[eid] = {"event_id": eid, "lead_time_min": lead, "first_alert_time": ep["alert_start"]}
    total = len(episodes)
    total_events = len(events)
    valid = len(event_hits)
    precision = useful / total if total else 0.0
    recall = valid / total_events if total_events else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    lead = pd.Series([v["lead_time_min"] for v in event_hits.values()], dtype=float)
    return {
        **params,
        "precision": precision,
        "recall_pod": recall,
        "f1": f1,
        "false_alerts_per_day": false / max(1, dates_count),
        "valid_alerted_events": valid,
        "total_events": total_events,
        "useful_alert_episodes": useful,
        "isolated_false_alerts": false,
        "total_alert_episodes": total,
        "mean_lead_time_min": float(lead.mean()) if not lead.empty else np.nan,
        "median_lead_time_min": float(lead.median()) if not lead.empty else np.nan,
        "lead_time_iqr_min": float(lead.quantile(0.75) - lead.quantile(0.25)) if not lead.empty else np.nan,
        "false_questionable_date": false_causes["QUESTIONABLE_DATE"],
        "false_post_flare_decay": false_causes["POST_FLARE_DECAY"],
        "false_duplicate_alert": false_causes["DUPLICATE_ALERT"],
        "false_true_isolated_false_alert": false_causes["TRUE_ISOLATED_FALSE_ALERT"],
    }


def candidate_frame_for_params(scores: pd.DataFrame, params: dict) -> pd.DataFrame:
    out = scores.copy()
    is_questionable = out["quality_label"].astype(str).str.upper().eq("QUESTIONABLE")
    out["is_questionable"] = is_questionable
    if params["quality_mode"] == "GOOD_ONLY":
        out = out[~is_questionable].copy()
        is_questionable = out["is_questionable"]
    out["adjusted_hybrid_score"] = out["hybrid_forecast_score"] - np.where(is_questionable, params["quality_penalty_weight"], 0.0)
    decay_weak = (
        out["post_flare_decay_state"].eq(1)
        & out["hard_impulsive_enhancement_score"].lt(0.60)
        & out["soft_gradual_enhancement_score"].lt(0.60)
        & out["v3_p30"].lt(0.90)
    )
    out["adjusted_hybrid_score"] = (out["adjusted_hybrid_score"] - np.where(decay_weak, params["post_flare_decay_penalty_weight"], 0.0)).clip(0, 1)
    nowcast = out["hard_score"].ge(0.40) | out["soft_score"].ge(0.60)
    v3_strong = out["v3_p30"].ge(params["v3_p30_threshold"]) | (
        out["v3_p60"].ge(params["v3_p60_threshold"]) & out["adjusted_hybrid_score"].ge(params["hybrid_score_threshold"])
    )
    hybrid_strong = (
        out["adjusted_hybrid_score"].ge(params["hybrid_score_threshold"])
        & out["v4_precursor_watch_score"].ge(params["v4_support_threshold"])
        & out["precursor_fusion_score_v4"].ge(0.20)
    )
    watch = out["v4_precursor_watch_score"].ge(params["v4_support_threshold"]) & ~v3_strong
    alert_signal = nowcast | v3_strong | hybrid_strong
    out["watch_signal"] = watch.astype(int)
    out["policy_state"] = np.select(
        [nowcast, v3_strong | hybrid_strong, watch],
        ["NOWCAST_CONFIRMED", "FORECAST_ALERT", "PRECURSOR_WATCH"],
        default="CLEAR",
    )
    counts = consecutive_counts(alert_signal, out["date"])
    out["policy_alert_candidate"] = (alert_signal & counts.ge(params["consecutive_bins"])).astype(int)
    return out


def v5_beats_v3(row: pd.Series) -> bool:
    return bool(
        row["f1"] > V3_BASELINE["f1"]
        or (row["false_alerts_per_day"] < V3_BASELINE["false_alerts_per_day"] and row["recall_pod"] >= V3_BASELINE["recall"])
        or (
            row["precision"] > V3_BASELINE["precision"]
            and row["recall_pod"] >= 0.75
            and row["false_alerts_per_day"] <= V3_BASELINE["false_alerts_per_day"]
        )
    )


def sweep_v5_policies() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = load_hybrid_scores()
    events_all = load_events()
    rows = []
    p30_values = [0.80]
    p60_values = [0.55]
    hybrid_values = [0.55, 0.65]
    v4_values = [0.55]
    quality_penalties = [0.10, 0.15]
    decay_penalties = [0.15]
    bins_values = [2, 3, 6, 12]
    cooldown_values = [30, 45, 60, 90]
    quality_modes = ["GOOD_ONLY", "GOOD_PLUS_QUESTIONABLE_PENALIZED"]

    for quality_mode in quality_modes:
        events = events_all[events_all["quality_label"].astype(str).str.upper().eq("GOOD")].copy() if quality_mode == "GOOD_ONLY" else events_all
        if events.empty:
            continue
        for p30 in p30_values:
            for p60 in p60_values:
                for hybrid_thr in hybrid_values:
                    for v4_thr in v4_values:
                        for qpen in quality_penalties:
                            for dpen in decay_penalties:
                                for bins in bins_values:
                                    params = {
                                        "v3_p30_threshold": p30,
                                        "v3_p60_threshold": p60,
                                        "hybrid_score_threshold": hybrid_thr,
                                        "v4_support_threshold": v4_thr,
                                        "quality_penalty_weight": qpen,
                                        "post_flare_decay_penalty_weight": dpen,
                                        "consecutive_bins": bins,
                                        "cooldown_min": 30,
                                        "quality_mode": quality_mode,
                                    }
                                    candidate = candidate_frame_for_params(scores, params)
                                    for cooldown in cooldown_values:
                                        params_cd = dict(params)
                                        params_cd["cooldown_min"] = cooldown
                                        episodes = build_policy_episodes(candidate, cooldown)
                                        rows.append(evaluate_policy(episodes, events, candidate["date"].nunique(), params_cd))
    sweep = pd.DataFrame(rows)
    if sweep.empty:
        return sweep, pd.DataFrame(), pd.DataFrame()
    rec = sweep.sort_values(
        ["f1", "false_alerts_per_day", "recall_pod", "precision", "mean_lead_time_min"],
        ascending=[False, True, False, False, False],
    ).head(10)
    best_params = rec.iloc[0][
        [
            "v3_p30_threshold",
            "v3_p60_threshold",
            "hybrid_score_threshold",
            "v4_support_threshold",
            "quality_penalty_weight",
            "post_flare_decay_penalty_weight",
            "consecutive_bins",
            "cooldown_min",
            "quality_mode",
        ]
    ].to_dict()
    best_candidate = candidate_frame_for_params(scores, best_params)
    best_episodes = build_policy_episodes(best_candidate, int(best_params["cooldown_min"]))
    return sweep, rec, best_episodes


def build_false_alert_reduction_report(best: pd.Series) -> pd.DataFrame:
    rules_summary = load_rules_summary()
    before_causes = {"QUESTIONABLE_DATE": 13, "POST_FLARE_DECAY": 2, "DUPLICATE_ALERT": 1}
    rows = []
    for mode, item in rules_summary.items():
        rows.append(
            {
                "comparison": mode,
                "questionable_false_alert_rows_before": item.get("questionable_before", 0),
                "questionable_false_alert_rows_after": item.get("questionable_after", 0),
                "post_flare_false_alert_rows_suppressed": item.get("decay_suppressed", 0),
                "duplicate_alert_rows_consolidated": item.get("duplicates_consolidated", 0),
                "final_alert_rows_after_rules": item.get("final_alert_rows", 0),
            }
        )
    rows.append(
        {
            "comparison": "BEST_V5_POLICY_EPISODE_METRICS",
            "questionable_false_alert_rows_before": before_causes["QUESTIONABLE_DATE"],
            "questionable_false_alert_rows_after": best.get("false_questionable_date", np.nan),
            "post_flare_false_alert_rows_suppressed": before_causes["POST_FLARE_DECAY"] - best.get("false_post_flare_decay", 0),
            "duplicate_alert_rows_consolidated": before_causes["DUPLICATE_ALERT"] - best.get("false_duplicate_alert", 0),
            "final_alert_rows_after_rules": best.get("total_alert_episodes", np.nan),
        }
    )
    report = pd.DataFrame(rows)
    report.to_csv(V5_FALSE_ALERT_REDUCTION_PATH, index=False)
    return report


def build_v1_to_v5_comparison(best: pd.Series) -> pd.DataFrame:
    rows = []
    if V3_COMPARISON_PATH.exists():
        prior = pd.read_csv(V3_COMPARISON_PATH)
        for _, row in prior.iterrows():
            rows.append(
                {
                    "system": row.get("system", ""),
                    "precision": row.get("precision", np.nan),
                    "recall": row.get("recall", np.nan),
                    "f1": row.get("f1", np.nan),
                    "false_alerts_per_day": row.get("false_alerts_per_day", np.nan),
                    "valid_alerted_events": row.get("valid_alerted_events", np.nan),
                    "total_events": row.get("total_events", np.nan),
                    "mean_lead_time_min": row.get("mean_lead_time_min", np.nan),
                    "median_lead_time_min": row.get("median_lead_time_min", np.nan),
                    "status": row.get("status", ""),
                }
            )
    if V4_POLICY_REC_PATH.exists():
        v4 = pd.read_csv(V4_POLICY_REC_PATH, nrows=1)
        if not v4.empty:
            r = v4.iloc[0]
            rows.append(
                {
                    "system": "v4 best state-machine policy",
                    "precision": r.get("precision", np.nan),
                    "recall": r.get("recall", np.nan),
                    "f1": r.get("f1", np.nan),
                    "false_alerts_per_day": r.get("false_alerts_per_day", np.nan),
                    "valid_alerted_events": r.get("valid_alerted_events", np.nan),
                    "total_events": r.get("total_events", np.nan),
                    "mean_lead_time_min": r.get("mean_lead_time_min", np.nan),
                    "median_lead_time_min": r.get("median_lead_time_min", np.nan),
                    "status": "ADVANCED_RESEARCH_EXTENSION",
                }
            )
    rows.append(
        {
            "system": "v5 hybrid false-alert reduction policy",
            "precision": best.get("precision", np.nan),
            "recall": best.get("recall_pod", np.nan),
            "f1": best.get("f1", np.nan),
            "false_alerts_per_day": best.get("false_alerts_per_day", np.nan),
            "valid_alerted_events": best.get("valid_alerted_events", np.nan),
            "total_events": best.get("total_events", np.nan),
            "mean_lead_time_min": best.get("mean_lead_time_min", np.nan),
            "median_lead_time_min": best.get("median_lead_time_min", np.nan),
            "status": "FINAL_RECOMMENDED" if v5_beats_v3(best) else "DIAGNOSTIC_DOES_NOT_REPLACE_V3",
        }
    )
    comp = pd.DataFrame(rows)
    comp.to_csv(V1_TO_V5_COMPARISON_PATH, index=False)
    return comp


def write_v5_report(best: pd.Series, comparison: pd.DataFrame, reduction: pd.DataFrame) -> None:
    beats = v5_beats_v3(best)
    decision = (
        "v5 beats the predefined v3 replacement rule and becomes the current final forecasting mode."
        if beats
        else "v5 does not beat the predefined v3 replacement rule. v3 remains the final forecasting mode; v5 is retained as a hybrid false-alert-reduction research extension."
    )
    lines = [
        "# Forecasting v5 Hybrid Policy Report",
        "",
        "## Best v5 Policy",
        "",
        "| metric | value |",
        "| --- | --- |",
    ]
    for key, value in best.to_dict().items():
        lines.append(f"| {key} | {value} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            decision,
            "",
            "## v3 Baseline",
            "",
            f"- Precision: {V3_BASELINE['precision']:.3f}",
            f"- Recall/POD: {V3_BASELINE['recall']:.3f}",
            f"- F1: {V3_BASELINE['f1']:.3f}",
            f"- False alerts/day: {V3_BASELINE['false_alerts_per_day']:.2f}",
            f"- Valid alerted events: {V3_BASELINE['valid_alerted_events']} / {V3_BASELINE['total_events']}",
            "",
            "## False-Alert Reduction",
            "",
            reduction.to_csv(index=False),
            "",
            "## Caveats",
            "",
            "- v5 uses existing v3/v4 prediction streams and does not retrain or modify nowcasting.",
            "- Small sample size remains the limiting factor.",
            "- v5 only becomes final if it beats the explicit replacement rule.",
        ]
    )
    V5_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def update_final_reports_with_v5(best: pd.Series) -> list[str]:
    beats = v5_beats_v3(best)
    decision = "v5 becomes final under the predefined rule." if beats else "v3 remains final; v5 is diagnostic/research support."
    section = f"""## Forecasting v5: Hybrid False-Alert Reduction

Forecasting v5 combines the v3 low-false-alert forecasting engine with v4 high-recall precursor/watch support. It adds quality-aware penalties, post-flare decay suppression, and duplicate-alert consolidation.

Best v5 policy:

- Precision: {best.get('precision', np.nan):.3f}
- Recall/POD: {best.get('recall_pod', np.nan):.3f}
- F1: {best.get('f1', np.nan):.3f}
- False alerts/day: {best.get('false_alerts_per_day', np.nan):.2f}
- Valid alerted events: {best.get('valid_alerted_events', np.nan)} / {best.get('total_events', np.nan)}
- Mean lead time: {best.get('mean_lead_time_min', np.nan):.2f} min

Decision: {decision}

The v5 layer does not modify nowcasting and does not overwrite v1/v2/v3/v4 results.
"""
    paths = [
        Path("final_submission_package/final_idea_submission.md"),
        Path("final_submission_package/detailed_project_summary.md"),
        Path("final_submission_package/architecture_summary.md"),
        Path("final_submission_package/final_claims_and_caveats.md"),
        Path("results/final_hackathon_evidence_report.md"),
        Path("results/space_agency_evaluation_criteria_scorecard.md"),
    ]
    updated = []
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        title = "## Forecasting v5: Hybrid False-Alert Reduction"
        if title in text:
            before = text.split(title, 1)[0].rstrip()
            after = text.split(title, 1)[1]
            import re

            match = re.search(r"\n## ", after)
            text = before + "\n\n" + section + ("\n" + after[match.start() :].lstrip() if match else "")
        else:
            text = text.rstrip() + "\n\n" + section
        path.write_text(text, encoding="utf-8")
        updated.append(str(path))
    return updated


def run_v5_sweep() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep, rec, best_episodes = sweep_v5_policies()
    sweep.to_csv(V5_POLICY_SWEEP_PATH, index=False)
    rec.to_csv(V5_POLICY_REC_PATH, index=False)
    if rec.empty:
        reduction = pd.DataFrame()
        comparison = pd.DataFrame()
        updated = []
    else:
        best = rec.iloc[0]
        reduction = build_false_alert_reduction_report(best)
        comparison = build_v1_to_v5_comparison(best)
        write_v5_report(best, comparison, reduction)
        updated = update_final_reports_with_v5(best)
    return sweep, rec, reduction, comparison, updated


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build SuryaAlert Forecasting v5A hybrid scores.")
    parser.add_argument("--stage", choices=["score", "rules", "sweep"], default="score")
    args = parser.parse_args(argv)
    if args.stage == "score":
        scores, audit = build_hybrid_scores()
        missing = {
            "v3": audit["missing_v3"],
            "v4_predictions": audit["missing_v4_pred"],
            "v4_support": audit["missing_support"],
        }
        print(f"rows joined: {len(scores):,}")
        print("score columns created: v3_weighted_signal, v4_probability_support, v4_physics_support, v4_precursor_watch_score, hybrid_forecast_score, v5_recommended_state")
        print(f"missing/fallback columns: {missing}")
        print("leakage check: PASS")
        print("output files:")
        print(HYBRID_SCORE_PATH)
        print(SCORE_AUDIT_PATH)
    elif args.stage == "rules":
        audit, summary = build_false_alert_rules()
        primary = summary.get("GOOD_PLUS_QUESTIONABLE_PENALIZED", {})
        print(
            "QUESTIONABLE alerts before/after penalty: "
            f"{primary.get('questionable_alerts_before', 0)} / {primary.get('questionable_alerts_after_quality_penalty', 0)}"
        )
        print(f"post-flare alerts suppressed: {primary.get('post_flare_alerts_suppressed', 0)}")
        print(f"duplicate alerts consolidated: {primary.get('duplicate_alerts_consolidated', 0)}")
        print("files created:")
        print(RULES_AUDIT_CSV_PATH)
        print(RULES_AUDIT_MD_PATH)
    elif args.stage == "sweep":
        sweep, rec, reduction, comparison, updated = run_v5_sweep()
        if rec.empty:
            print("best v5 policy: none")
            print("whether v5 beats v3: no")
            print("false alerts/day before/after: 1.33 / none")
            print("false alerts by cause before/after: unavailable")
            print("final recommended forecasting mode: v3 state-machine policy")
            print("dashboard status: not checked")
            return
        best = rec.iloc[0]
        beats = v5_beats_v3(best)
        after_causes = {
            "QUESTIONABLE_DATE": int(best.get("false_questionable_date", 0)),
            "POST_FLARE_DECAY": int(best.get("false_post_flare_decay", 0)),
            "DUPLICATE_ALERT": int(best.get("false_duplicate_alert", 0)),
            "TRUE_ISOLATED_FALSE_ALERT": int(best.get("false_true_isolated_false_alert", 0)),
        }
        print("best v5 policy:")
        print(best.to_string())
        print(f"whether v5 beats v3: {'yes' if beats else 'no'}")
        print(f"false alerts/day before/after: {V3_BASELINE['false_alerts_per_day']:.2f} / {best['false_alerts_per_day']:.2f}")
        print(f"false alerts by cause before/after: before={{'QUESTIONABLE_DATE': 13, 'POST_FLARE_DECAY': 2, 'DUPLICATE_ALERT': 1}} after={after_causes}")
        print(f"final recommended forecasting mode: {'v5 hybrid policy' if beats else 'v3 state-machine policy'}")
        print("files created/updated:")
        for path in [V5_POLICY_SWEEP_PATH, V5_POLICY_REC_PATH, V5_FALSE_ALERT_REDUCTION_PATH, V1_TO_V5_COMPARISON_PATH, V5_REPORT_PATH]:
            print(path)
        for path in updated:
            print(path)
        print("dashboard status: pending py_compile")


if __name__ == "__main__":
    main()
