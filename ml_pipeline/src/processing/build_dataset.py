from __future__ import annotations

from pathlib import Path

from src.io.read_lightcurves import (
    build_lightcurves_for_date,
    date_to_display,
    discover_available_dates,
    select_dates_for_mode,
)
from src.utils.config import is_dev_mode
from src.utils.plotting import plot_lightcurves


PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
REPORTS_DIR = Path("reports")


def process_dates(dates: list[str]) -> list[Path]:
    PROCESSED_DIR.mkdir(exist_ok=True, parents=True)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    outputs = []
    for date in dates:
        print("\n" + "=" * 90)
        print(f"Building combined lightcurves for {date_to_display(date)}")
        all_df = build_lightcurves_for_date(date)

        out_csv = PROCESSED_DIR / f"{date}_combined_lightcurves_long.csv"
        all_df.to_csv(out_csv, index=False)
        outputs.append(out_csv)
        print(f"\nSaved CSV: {out_csv}")

        plot_lightcurves(all_df, RESULTS_DIR / f"{date}_solexs_hel1os_quicklook.png", date_to_display(date))

    return outputs


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True, parents=True)

    matched_dates, availability = discover_available_dates()
    availability_path = REPORTS_DIR / "date_availability.csv"
    availability.to_csv(availability_path, index=False)

    print("Date availability:")
    print(availability.to_string(index=False))
    print(f"\nSaved: {availability_path}")

    dates = select_dates_for_mode(matched_dates, is_dev_mode())
    if not dates:
        raise RuntimeError("No dates have both SoLEXS and HEL1OS Level-1 lightcurve data.")

    process_dates(dates)
