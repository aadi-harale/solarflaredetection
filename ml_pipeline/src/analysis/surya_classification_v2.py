from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(".")
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

MASTER_PATH = RESULTS_DIR / "master_flare_catalogue.csv"
CLASS_AWARE_PATH = RESULTS_DIR / "class_aware_event_summary.csv"
GOES_MATCHING_PATH = RESULTS_DIR / "goes_matching_report.csv"

CLASSIFIED_PATH = RESULTS_DIR / "master_flare_catalogue_classified_v2.csv"
SUMMARY_CSV_PATH = RESULTS_DIR / "surya_class_estimation_summary.csv"
REPORT_PATH = RESULTS_DIR / "surya_class_estimation_report.md"
CONFUSION_PATH = RESULTS_DIR / "surya_class_confusion_matrix.csv"
PLOT_SURYA_CLASS_PATH = RESULTS_DIR / "plot_surya_estimated_class_distribution.csv"
PLOT_SURYA_GOES_AGREEMENT_PATH = RESULTS_DIR / "plot_surya_vs_goes_class_agreement.csv"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in [MASTER_PATH, CLASS_AWARE_PATH, GOES_MATCHING_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    master = pd.read_csv(MASTER_PATH)
    class_aware = pd.read_csv(CLASS_AWARE_PATH)
    goes = pd.read_csv(GOES_MATCHING_PATH)
    master["event_id"] = master["event_id"].astype(str)
    master["date"] = master["date"].astype(str)
    class_aware["surya_event_id"] = class_aware["surya_event_id"].astype(str)
    goes["surya_event_id"] = goes["surya_event_id"].astype(str)
    for col in ["soft_start", "soft_peak", "soft_end", "hard_start", "hard_peak", "combined_start", "combined_end"]:
        if col in master.columns:
            master[col] = pd.to_datetime(master[col], utc=True, format="mixed", errors="coerce")
    return master, class_aware, goes


def load_lightcurve(date: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{date}_combined_lightcurves_long.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"time_utc", "instrument", "detector", "band", "count_rate"}
    if required - set(df.columns):
        return pd.DataFrame()
    df = df.copy()
    df["time_utc"] = pd.to_datetime(df["time_utc"], utc=True, format="mixed", errors="coerce")
    df["count_rate"] = pd.to_numeric(df["count_rate"], errors="coerce")
    df = df.dropna(subset=["time_utc", "count_rate"])
    for col in ["instrument", "detector", "band"]:
        df[col] = df[col].astype(str)
    # Duplicate files can yield duplicate timestamps; average them for event feature extraction.
    return (
        df.groupby(["time_utc", "instrument", "detector", "band"], as_index=False)["count_rate"]
        .mean()
        .sort_values("time_utc")
    )


def series_for(df: pd.DataFrame, mask: pd.Series) -> pd.Series:
    subset = df.loc[mask, ["time_utc", "count_rate"]]
    if subset.empty:
        return pd.Series(dtype=float)
    return subset.groupby("time_utc")["count_rate"].mean().sort_index()


def finite_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.median()) if not values.empty else np.nan


def finite_max(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(values.max()) if not values.empty else np.nan


def event_features(row: pd.Series, lightcurve: pd.DataFrame) -> dict:
    if lightcurve.empty:
        return {
            "surya_soft_peak_value": np.nan,
            "surya_soft_background_value": np.nan,
            "surya_soft_net_peak": np.nan,
            "surya_soft_peak_to_background_ratio": np.nan,
            "surya_hard_peak_value": np.nan,
            "surya_hard_to_soft_ratio": np.nan,
        }

    soft = series_for(
        lightcurve,
        (lightcurve["instrument"] == "SoLEXS") & lightcurve["band"].str.contains("2-22", regex=False),
    )
    hard = series_for(
        lightcurve,
        (lightcurve["instrument"] == "HEL1OS")
        & (
            lightcurve["band"].str.contains("5-20", regex=False)
            | lightcurve["band"].str.contains("20-40", regex=False)
        ),
    )

    soft_peak_time = row["soft_peak"]
    soft_start = row["soft_start"]
    hard_peak_time = row["hard_peak"] if pd.notna(row.get("hard_peak")) else row["hard_start"]

    peak_window = soft.loc[(soft.index >= soft_peak_time - pd.Timedelta(seconds=60)) & (soft.index <= soft_peak_time + pd.Timedelta(seconds=60))]
    background_window = soft.loc[(soft.index >= soft_start - pd.Timedelta(minutes=30)) & (soft.index < soft_start)]
    if background_window.empty:
        background_window = soft.loc[(soft.index >= soft_peak_time - pd.Timedelta(minutes=60)) & (soft.index < soft_peak_time - pd.Timedelta(minutes=5))]

    hard_window = hard.loc[(hard.index >= hard_peak_time - pd.Timedelta(seconds=60)) & (hard.index <= hard_peak_time + pd.Timedelta(seconds=60))]
    if hard_window.empty:
        hard_window = hard.loc[(hard.index >= row["combined_start"]) & (hard.index <= row["combined_end"])]

    soft_peak = finite_max(peak_window)
    soft_bg = finite_median(background_window)
    soft_net = soft_peak - soft_bg if pd.notna(soft_peak) and pd.notna(soft_bg) else np.nan
    soft_ratio = soft_peak / soft_bg if pd.notna(soft_peak) and pd.notna(soft_bg) and soft_bg > 0 else np.nan
    hard_peak = finite_max(hard_window)
    hard_soft_ratio = hard_peak / soft_peak if pd.notna(hard_peak) and pd.notna(soft_peak) and soft_peak > 0 else np.nan

    return {
        "surya_soft_peak_value": soft_peak,
        "surya_soft_background_value": soft_bg,
        "surya_soft_net_peak": soft_net,
        "surya_soft_peak_to_background_ratio": soft_ratio,
        "surya_hard_peak_value": hard_peak,
        "surya_hard_to_soft_ratio": hard_soft_ratio,
    }


def goes_multiclass(goes_class: object) -> str:
    text = "" if pd.isna(goes_class) else str(goes_class).strip().upper()
    if not text:
        return ""
    prefix = text[0]
    if prefix == "C":
        return "C-like"
    if prefix == "M":
        return "M-like"
    if prefix == "X":
        return "X-like"
    return "sub-C-like"


def low_high(label: str) -> str:
    return "HIGH" if label in {"M-like", "X-like"} else "LOW_OR_MODERATE"


def calibrated_boundaries(train: pd.DataFrame) -> tuple[float, float, float]:
    matched = train[
        train["goes_match_status"].isin(["EXACT_PEAK_MATCH", "WINDOW_OVERLAP_MATCH"])
        & train["goes_class_group"].isin(["C", "M", "X"])
        & train["surya_soft_net_peak"].notna()
    ]
    if matched.empty:
        return np.nan, np.nan, np.nan

    c_vals = matched.loc[matched["goes_class_group"] == "C", "surya_soft_net_peak"]
    m_vals = matched.loc[matched["goes_class_group"] == "M", "surya_soft_net_peak"]
    x_vals = matched.loc[matched["goes_class_group"] == "X", "surya_soft_net_peak"]

    c_m = np.nan
    if not c_vals.empty and not m_vals.empty:
        c_m = float(np.sqrt(max(c_vals.max(), 1e-9) * max(m_vals.min(), 1e-9)))
    elif not m_vals.empty:
        c_m = float(m_vals.min())

    m_x = np.nan
    if not m_vals.empty and not x_vals.empty:
        m_x = float(np.sqrt(max(m_vals.max(), 1e-9) * max(x_vals.min(), 1e-9)))
    elif not x_vals.empty:
        m_x = float(x_vals.min())

    sub_c = np.nan
    if not c_vals.empty:
        sub_c = float(c_vals.min() * 0.5)
    elif pd.notna(c_m):
        sub_c = float(c_m * 0.5)

    return sub_c, c_m, m_x


def estimate_label(net_peak: float, boundaries: tuple[float, float, float]) -> str:
    sub_c, c_m, m_x = boundaries
    if pd.isna(net_peak):
        return "UNKNOWN"
    if pd.notna(sub_c) and net_peak < sub_c:
        return "sub-C-like"
    if pd.notna(c_m) and net_peak < c_m:
        return "C-like"
    if pd.notna(m_x) and net_peak >= m_x:
        return "X-like"
    return "M-like"


def confidence(net_peak: float, boundaries: tuple[float, float, float], quality_label: str) -> float:
    vals = np.array([b for b in boundaries if pd.notna(b) and b > 0], dtype=float)
    if pd.isna(net_peak) or vals.size == 0:
        return 0.0
    nearest = float(np.min(np.abs(np.log(max(net_peak, 1e-9)) - np.log(vals))))
    conf = min(0.95, max(0.35, nearest / 2.0))
    if quality_label == "QUESTIONABLE":
        conf *= 0.75
    return round(float(conf), 3)


def add_classification(master: pd.DataFrame, goes: pd.DataFrame) -> pd.DataFrame:
    lightcurves = {date: load_lightcurve(date) for date in sorted(master["date"].unique())}
    feature_rows = []
    for _, row in master.iterrows():
        feature_rows.append(event_features(row, lightcurves.get(str(row["date"]), pd.DataFrame())))
    out = pd.concat([master.reset_index(drop=True), pd.DataFrame(feature_rows)], axis=1)

    # Leave-one-event-out calibration where possible; fallback uses all matched events.
    all_boundaries = calibrated_boundaries(out)
    labels = []
    for idx, row in out.iterrows():
        train = out.drop(index=idx)
        boundaries = calibrated_boundaries(train)
        if any(pd.isna(b) for b in boundaries):
            boundaries = all_boundaries
        labels.append((estimate_label(row["surya_soft_net_peak"], boundaries), boundaries))

    out["surya_estimated_class_label"] = [x[0] for x in labels]
    out["surya_estimated_class_group"] = out["surya_estimated_class_label"].map(low_high)
    out["surya_class_confidence"] = [
        confidence(row["surya_soft_net_peak"], labels[i][1], row["quality_label"]) for i, row in out.iterrows()
    ]
    out["surya_class_method"] = "empirical GOES-equivalent proxy calibrated on matched events"
    out["surya_class_reliability_flag"] = np.where(
        out["quality_label"].eq("QUESTIONABLE"), "QUESTIONABLE_TELEMETRY_SMALL_SAMPLE_DIAGNOSTIC_ONLY", "SMALL_SAMPLE_DIAGNOSTIC_ONLY"
    )

    out["goes_multiclass_label"] = out["goes_class"].map(goes_multiclass)
    out["goes_low_high_group"] = out["goes_multiclass_label"].map(lambda x: low_high(x) if x else "")
    out["class_agreement_with_goes"] = np.where(
        ~out["goes_match_status"].isin(["EXACT_PEAK_MATCH", "WINDOW_OVERLAP_MATCH"]),
        "NO_TIMING_SUPPORTED_GOES_CLASS",
        np.where(
            out["surya_estimated_class_group"].eq(out["goes_low_high_group"]),
            "LOW_HIGH_AGREE",
            "LOW_HIGH_DISAGREE",
        ),
    )
    out["class_notes"] = (
        "SuryaAlert GOES-equivalent proxy class estimate from SoLEXS peak/background features; "
        "HEL1OS hard/soft ratio is supporting diagnostic evidence. GOES/SWPC class is external validation, "
        "not the source of the output label. Multiclass labels are diagnostic because the dataset is small."
    )
    return out


def write_summary(classified: pd.DataFrame) -> None:
    timing = classified[classified["goes_match_status"].isin(["EXACT_PEAK_MATCH", "WINDOW_OVERLAP_MATCH"])]
    low_high_counts = classified["surya_estimated_class_group"].value_counts().rename_axis("surya_estimated_class_group").reset_index(name="count")
    multi_counts = classified["surya_estimated_class_label"].value_counts().rename_axis("surya_estimated_class_label").reset_index(name="count")
    agree = int((timing["class_agreement_with_goes"] == "LOW_HIGH_AGREE").sum())
    timing_total = int(len(timing))
    multi_agree = int((timing["surya_estimated_class_label"] == timing["goes_multiclass_label"]).sum())

    summary_rows = [
        {"metric": "events_classified", "value": len(classified), "notes": "All v1 master catalogue events received a v2 SuryaAlert-estimated class."},
        {"metric": "low_high_agreement_with_goes", "value": f"{agree} / {timing_total}", "notes": "Only timing-supported GOES matches counted."},
        {"metric": "multiclass_agreement_with_goes", "value": f"{multi_agree} / {timing_total}", "notes": "Diagnostic only; small sample and empirical proxy thresholds."},
        {"metric": "method", "value": "empirical GOES-equivalent proxy calibrated on matched events", "notes": "Not a direct GOES flux measurement."},
    ]
    for _, row in low_high_counts.iterrows():
        summary_rows.append(
            {
                "metric": f"estimated_group_count_{row['surya_estimated_class_group']}",
                "value": int(row["count"]),
                "notes": "Stable low/high SuryaAlert proxy class group.",
            }
        )
    for _, row in multi_counts.iterrows():
        summary_rows.append(
            {
                "metric": f"estimated_multiclass_count_{row['surya_estimated_class_label']}",
                "value": int(row["count"]),
                "notes": "Diagnostic multiclass SuryaAlert proxy class label.",
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_CSV_PATH, index=False)

    confusion = pd.crosstab(
        timing["goes_multiclass_label"],
        timing["surya_estimated_class_label"],
        dropna=False,
    ).reset_index()
    confusion.to_csv(CONFUSION_PATH, index=False)

    low_high_counts.to_csv(PLOT_SURYA_CLASS_PATH, index=False)
    agreement_counts = classified["class_agreement_with_goes"].value_counts().rename_axis("class_agreement_with_goes").reset_index(name="count")
    agreement_counts.to_csv(PLOT_SURYA_GOES_AGREEMENT_PATH, index=False)

    report = f"""# SuryaAlert-Estimated Flare Classification Report

## Method

SuryaAlert now produces a v2 GOES-equivalent proxy class estimate generated from SoLEXS/HEL1OS event features. The main features are SoLEXS soft peak value, soft background value, background-subtracted net peak, peak/background ratio, HEL1OS hard peak value, and hard-to-soft ratio.

The method is labelled: **empirical GOES-equivalent proxy calibrated on matched events**. It is not a direct GOES flux measurement and should not be interpreted as physical GOES flux calibration.

GOES/SWPC class remains external validation, not the only source of class labels.

## Summary

- Events classified: {len(classified)}
- Estimated LOW/HIGH counts: {low_high_counts.to_dict(orient="records")}
- Estimated sub-C/C/M/X-like counts: {multi_counts.to_dict(orient="records")}
- LOW/HIGH agreement with GOES: {agree} / {timing_total}
- Multiclass agreement with GOES: {multi_agree} / {timing_total}

## Reliability Caveats

- Multiclass labels are diagnostic because the dataset is small.
- Leave-one-event-out threshold calibration is used where possible, but the sample is still too small for robust calibration.
- Reliability flags include `SMALL_SAMPLE_DIAGNOSTIC_ONLY` and `QUESTIONABLE_TELEMETRY_SMALL_SAMPLE_DIAGNOSTIC_ONLY`.
- HEL1OS hard/soft ratio is supporting evidence, not the only class signal.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def update_reports() -> None:
    section = """## SuryaAlert-estimated flare classification

SuryaAlert now includes a v2 diagnostic flare classification catalogue. The class estimate is generated by SuryaAlert from SoLEXS/HEL1OS event features, including soft X-ray peak/background behavior and hard-to-soft support.

The class method is a GOES-equivalent proxy class estimate: **empirical GOES-equivalent proxy calibrated on matched events**. It is not a direct GOES flux measurement.

GOES/SWPC class is used for external validation. Multiclass labels are diagnostic because the dataset is small.

Files:

- `results/master_flare_catalogue_classified_v2.csv`
- `results/surya_class_estimation_summary.csv`
- `results/surya_class_estimation_report.md`
- `results/surya_class_confusion_matrix.csv`
"""
    for name in ["final_hackathon_evidence_report.md", "space_agency_evaluation_criteria_scorecard.md", "hackathon_diagnostic_report.md"]:
        path = RESULTS_DIR / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        title = "## SuryaAlert-estimated flare classification"
        if title in text:
            before = text.split(title, 1)[0].rstrip()
            after = text.split(title, 1)[1]
            import re

            match = re.search(r"\n## ", after)
            text = before + "\n\n" + section + ("\n" + after[match.start() :].lstrip() if match else "")
        else:
            text = text.rstrip() + "\n\n" + section
        path.write_text(text, encoding="utf-8")


def main() -> None:
    master, _, goes = load_inputs()
    classified = add_classification(master, goes)
    classified.to_csv(CLASSIFIED_PATH, index=False)
    write_summary(classified)
    update_reports()

    print("SuryaAlert classification v2 complete.")
    print(f"Saved: {CLASSIFIED_PATH}")
    print(f"Saved: {SUMMARY_CSV_PATH}")
    print(f"Saved: {REPORT_PATH}")
    print(f"Saved: {CONFUSION_PATH}")
    print("Estimated class counts:")
    print(classified["surya_estimated_class_group"].value_counts().to_string())
    timing = classified[classified["goes_match_status"].isin(["EXACT_PEAK_MATCH", "WINDOW_OVERLAP_MATCH"])]
    print("LOW/HIGH agreement with GOES:")
    print((timing["class_agreement_with_goes"] == "LOW_HIGH_AGREE").sum(), "/", len(timing))
    print("Multiclass agreement with GOES:")
    print((timing["surya_estimated_class_label"] == timing["goes_multiclass_label"]).sum(), "/", len(timing))
    print("Caveat: SMALL_SAMPLE_DIAGNOSTIC_ONLY; proxy class is not direct GOES flux calibration.")


if __name__ == "__main__":
    main()
