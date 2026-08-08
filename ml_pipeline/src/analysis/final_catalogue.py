from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = [
        RESULTS_DIR / "combined_nowcast_catalogue_clean.csv",
        RESULTS_DIR / "goes_matching_report.csv",
        RESULTS_DIR / "class_aware_event_summary.csv",
        RESULTS_DIR / "quality_group_metric_comparison.csv",
        RESULTS_DIR / "false_early_goes_followup_analysis.csv",
        RESULTS_DIR / "valid_alerts_by_quality_group.csv",
        RESULTS_DIR / "goes_cross_validation_summary.md",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(str(path) for path in missing))

    catalogue = pd.read_csv(required[0])
    goes_matching = pd.read_csv(required[1])
    class_summary = pd.read_csv(required[2])
    quality_groups = pd.read_csv(required[3])
    false_goes = pd.read_csv(required[4])
    valid_alerts = pd.read_csv(required[5])
    return catalogue, goes_matching, class_summary, quality_groups, false_goes, valid_alerts


def normalize_date(value: object) -> str:
    return str(value).replace("-", "")[:8]


def write_candidate_catalogues(catalogue: pd.DataFrame, goes_matching: pd.DataFrame) -> None:
    enriched = catalogue.copy()
    enriched["date"] = enriched["source_date"].map(normalize_date)
    goes = goes_matching.rename(columns={"surya_event_id": "global_event_id"}).copy()
    goes["global_event_id"] = goes["global_event_id"].astype(str)
    enriched["global_event_id"] = enriched["global_event_id"].astype(str)
    enriched = enriched.merge(
        goes[
            [
                "global_event_id",
                "goes_match_status",
                "goes_event_id",
                "goes_class",
                "goes_class_group",
                "goes_active_region",
            ]
        ],
        on="global_event_id",
        how="left",
    )

    soft = pd.DataFrame(
        {
            "event_id": enriched["global_event_id"],
            "date": enriched["date"],
            "quality_label": enriched["quality_label"],
            "soft_start": enriched["event_start"],
            "soft_peak": enriched["soft_peak_time"],
            "soft_end": enriched["event_end"],
            "soft_peak_counts": enriched["soft_peak_counts"],
            "max_soft_score": enriched["max_soft_score"],
            "goes_match_status": enriched["goes_match_status"],
            "goes_event_id": enriched["goes_event_id"],
            "goes_class": enriched["goes_class"],
            "goes_class_group": enriched["goes_class_group"],
            "active_region": enriched["goes_active_region"],
            "notes": "Soft X-ray candidate from cleaned SuryaAlert nowcast catalogue.",
        }
    )
    soft.to_csv(RESULTS_DIR / "soft_xray_candidate_catalogue.csv", index=False)

    hard = pd.DataFrame(
        {
            "event_id": enriched["global_event_id"],
            "date": enriched["date"],
            "quality_label": enriched["quality_label"],
            "hard_start": enriched["hard_trigger_time"],
            "hard_peak": enriched["hard_peak_time"],
            "hard_end": "",
            "max_hard_score": enriched["max_hard_score"],
            "hard_to_soft_lead_time_min": enriched["lead_time_min"],
            "alert_type": enriched["alert_type"],
            "goes_match_status": enriched["goes_match_status"],
            "goes_event_id": enriched["goes_event_id"],
            "goes_class": enriched["goes_class"],
            "goes_class_group": enriched["goes_class_group"],
            "active_region": enriched["goes_active_region"],
            "notes": "Hard X-ray candidate from cleaned SuryaAlert nowcast catalogue; hard_end not available in source outputs.",
        }
    )
    hard.to_csv(RESULTS_DIR / "hard_xray_candidate_catalogue.csv", index=False)


def write_master_catalogue(
    catalogue: pd.DataFrame,
    goes_matching: pd.DataFrame,
    class_summary: pd.DataFrame,
    valid_alerts: pd.DataFrame,
) -> pd.DataFrame:
    cat = catalogue.copy()
    cat["date"] = cat["source_date"].map(normalize_date)
    cat["global_event_id"] = cat["global_event_id"].astype(str)
    goes = goes_matching.rename(columns={"surya_event_id": "global_event_id"}).copy()
    goes["global_event_id"] = goes["global_event_id"].astype(str)
    classes = class_summary.rename(columns={"surya_event_id": "global_event_id"}).copy()
    classes["global_event_id"] = classes["global_event_id"].astype(str)

    valid = valid_alerts.copy()
    valid["date"] = valid["date"].map(normalize_date)
    valid["event_id"] = valid["event_id"].astype(str)
    valid_lookup = valid.set_index(["date", "event_id"])

    merged = cat.merge(
        goes[
            [
                "global_event_id",
                "goes_match_status",
                "goes_event_id",
                "goes_start_time_utc",
                "goes_peak_time_utc",
                "goes_end_time_utc",
                "goes_class",
                "goes_class_group",
                "goes_active_region",
                "notes",
            ]
        ],
        on="global_event_id",
        how="left",
    ).merge(
        classes[["global_event_id", "class_source"]],
        on="global_event_id",
        how="left",
    )

    rows = []
    for _, row in merged.iterrows():
        key = (str(row["date"]), str(row["event_id"]))
        valid_row = valid_lookup.loc[key] if key in valid_lookup.index else None
        valid_alerted = valid_row is not None
        first_alert_time = "" if valid_row is None else valid_row["first_valid_alert_time"]
        alert_lead = np.nan if valid_row is None else valid_row["lead_time_min"]
        match_status = row.get("goes_match_status", "")
        quality = row.get("quality_label", "")
        origin = "SURYA_GOES_MATCHED" if match_status in ["EXACT_PEAK_MATCH", "WINDOW_OVERLAP_MATCH"] else "SURYA_CANDIDATE_GOES_NEAREST_ONLY"
        data_quality_notes = "GOOD quality date." if quality == "GOOD" else "QUESTIONABLE quality date; interpret metrics with caution."
        if match_status == "NEAREST_ONLY":
            interpretation = "SuryaAlert cleaned event has nearest GOES event on same date, but no peak/window match."
        elif match_status == "NO_MATCH":
            interpretation = "SuryaAlert cleaned event has no GOES event on same date."
        else:
            interpretation = "SuryaAlert cleaned event externally supported by NOAA/SWPC GOES timing match."

        combined_candidates = pd.to_datetime(
            [row["event_start"], row["event_end"], row["soft_peak_time"], row["hard_trigger_time"], row["hard_peak_time"]],
            utc=True,
            errors="coerce",
        )
        combined_start = combined_candidates.min()
        combined_end = combined_candidates.max()

        rows.append(
            {
                "event_id": row["global_event_id"],
                "date": row["date"],
                "quality_label": quality,
                "origin_flag": origin,
                "soft_start": row["event_start"],
                "soft_peak": row["soft_peak_time"],
                "soft_end": row["event_end"],
                "hard_start": row["hard_trigger_time"],
                "hard_peak": row["hard_peak_time"],
                "hard_end": "",
                "combined_start": combined_start,
                "combined_peak": row["soft_peak_time"],
                "combined_end": combined_end,
                "hard_to_soft_lead_time_min": row["lead_time_min"],
                "valid_forecast_alerted": valid_alerted,
                "first_alert_time": first_alert_time,
                "alert_lead_time_min": alert_lead,
                "goes_match_status": match_status,
                "goes_event_id": row.get("goes_event_id", ""),
                "goes_class": row.get("goes_class", ""),
                "goes_class_group": row.get("goes_class_group", ""),
                "active_region": row.get("goes_active_region", ""),
                "class_source": row.get("class_source", ""),
                "data_quality_notes": data_quality_notes,
                "interpretation_notes": interpretation,
            }
        )

    master = pd.DataFrame(rows)
    master.to_csv(RESULTS_DIR / "master_flare_catalogue.csv", index=False)
    return master


def write_scorecard(
    master: pd.DataFrame,
    quality_groups: pd.DataFrame,
    false_goes: pd.DataFrame,
) -> pd.DataFrame:
    overall = quality_groups[quality_groups["group_name"] == "GOOD + QUESTIONABLE"].iloc[0]
    exact = int((master["goes_match_status"] == "EXACT_PEAK_MATCH").sum())
    window = int((master["goes_match_status"] == "WINDOW_OVERLAP_MATCH").sum())
    nearest = int((master["goes_match_status"] == "NEAREST_ONLY").sum())
    no_match = int((master["goes_match_status"] == "NO_MATCH").sum())
    class_counts = master[master["goes_match_status"].isin(["EXACT_PEAK_MATCH", "WINDOW_OVERLAP_MATCH"])]["goes_class_group"].value_counts()
    false_60 = int(false_goes["followed_by_goes_within_60min"].sum())
    false_90 = int(false_goes["followed_by_goes_within_90min"].sum())
    isolated = int((false_goes["reclassified_as"] == "TRUE_ISOLATED_FALSE_ALARM").sum())

    rows = [
        {
            "criterion": "low/high flare detection",
            "metric_name": "GOES external class coverage",
            "value": f"16 GOES-matched SuryaAlert events: C={int(class_counts.get('C', 0))}, M={int(class_counts.get('M', 0))}, X={int(class_counts.get('X', 0))}",
            "evidence_file": "results/master_flare_catalogue.csv",
            "interpretation": "Prototype detects externally matched C/M/X events, with most matched events in M/X classes for this dataset.",
            "limitations": "Class coverage is from matched cleaned candidates only; this is not a complete all-GOES-event detection-rate study.",
        },
        {
            "criterion": "TPR/FAR",
            "metric_name": "External matching of cleaned events",
            "value": f"EXACT={exact}, WINDOW={window}, NEAREST_ONLY={nearest}, NO_MATCH={no_match}",
            "evidence_file": "results/goes_matching_report.csv",
            "interpretation": "16 of 17 cleaned SuryaAlert events have GOES peak/window support; one has only a nearest same-date GOES event.",
            "limitations": "This validates cleaned event timing against GOES, but does not measure recall against all 108 GOES events.",
        },
        {
            "criterion": "TPR/FAR",
            "metric_name": "Operational forecast precision/recall/F1",
            "value": f"precision={float(overall['event_level_precision']):.3f}, recall={float(overall['event_level_recall']):.3f}, F1={float(overall['event_level_f1']):.3f}",
            "evidence_file": "results/quality_group_metric_comparison.csv",
            "interpretation": "Event-level operational warning performance remains weak but measurable.",
            "limitations": "QUESTIONABLE dates are included and the dataset is still small.",
        },
        {
            "criterion": "TPR/FAR",
            "metric_name": "False-early GOES follow-up",
            "value": f"GOES within 60 min={false_60}, GOES within 90 min={false_90}, true isolated false alarms={isolated}",
            "evidence_file": "results/false_early_goes_followup_analysis.csv",
            "interpretation": "Many false-early alerts precede GOES events, but degraded data still makes most cases ambiguous.",
            "limitations": "This does not retroactively change alert labels or thresholds.",
        },
        {
            "criterion": "lead time",
            "metric_name": "Forecast-alerted GOES-matched events and mean alert lead",
            "value": f"valid alerted GOES-matched events={int(master['valid_forecast_alerted'].sum())}, mean alert lead={master.loc[master['valid_forecast_alerted'], 'alert_lead_time_min'].astype(float).mean():.2f} min",
            "evidence_file": "results/master_flare_catalogue.csv",
            "interpretation": "The prototype produced useful forecast alerts for 8 GOES-matched events with positive lead time.",
            "limitations": "Lead time is calculated from the existing SuryaAlert alert episode evaluation, not from a new model.",
        },
    ]
    scorecard = pd.DataFrame(rows)
    scorecard.to_csv(RESULTS_DIR / "space_agency_evaluation_criteria_scorecard.csv", index=False)

    md = "# Space Agency Evaluation Criteria Scorecard\n\n"
    md += "This scorecard maps existing SuryaAlert outputs to hackathon evaluation criteria without changing model logic, thresholds, labels, or training.\n\n"
    md += dataframe_to_markdown(scorecard)
    md += "\n\n## Headline Numbers\n\n"
    md += f"- SuryaAlert cleaned events: {len(master)}\n"
    md += f"- GOES timing-supported events: {exact + window} (`EXACT_PEAK_MATCH={exact}`, `WINDOW_OVERLAP_MATCH={window}`)\n"
    md += f"- Matched GOES classes: C={int(class_counts.get('C', 0))}, M={int(class_counts.get('M', 0))}, X={int(class_counts.get('X', 0))}\n"
    md += f"- Valid alerted GOES-matched events: {int(master['valid_forecast_alerted'].sum())}\n"
    md += f"- False-early alerts followed by GOES within 60/90 min: {false_60}/{false_90}\n"
    md += f"- True isolated false alarms after GOES check: {isolated}\n"
    md += "\n## Caveat\n\nThese are prototype diagnostics on the current labelled dataset. They should not be presented as final scientific reliability.\n"
    (RESULTS_DIR / "space_agency_evaluation_criteria_scorecard.md").write_text(md, encoding="utf-8")
    return scorecard


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    text = df.copy()
    for col in text.columns:
        text[col] = text[col].map(lambda value: "" if pd.isna(value) else str(value))
    headers = list(text.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in text.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in headers) + " |")
    return "\n".join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    catalogue, goes_matching, class_summary, quality_groups, false_goes, valid_alerts = read_inputs()
    write_candidate_catalogues(catalogue, goes_matching)
    master = write_master_catalogue(catalogue, goes_matching, class_summary, valid_alerts)
    write_scorecard(master, quality_groups, false_goes)

    matched = master["goes_match_status"].isin(["EXACT_PEAK_MATCH", "WINDOW_OVERLAP_MATCH"])
    class_counts = master[matched]["goes_class_group"].value_counts()
    print("Final hackathon catalogue and Space Agency scorecard created.")
    print(f"Master catalogue events: {len(master)}")
    print(f"GOES timing-supported events: {int(matched.sum())}")
    print(f"Matched classes: C={int(class_counts.get('C', 0))}, M={int(class_counts.get('M', 0))}, X={int(class_counts.get('X', 0))}")
    print(f"Valid forecast-alerted events: {int(master['valid_forecast_alerted'].sum())}")
    print("Outputs written to results/")


if __name__ == "__main__":
    main()
