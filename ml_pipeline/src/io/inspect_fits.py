from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd
from astropy.io import fits

from src.io.extract import RAW_DIR, ZIPS_DIR, extract_all_zips
from src.utils.config import is_dev_mode


REPORTS_DIR = Path("reports")


def find_candidate_files(raw_dir: Path = RAW_DIR, june03_only: bool = False) -> list[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Missing raw data directory: {raw_dir.resolve()}")

    files = []
    for path in raw_dir.rglob("*"):
        if not path.is_file():
            continue

        name = path.name.lower()
        is_solexs_lc = name.endswith(".lc.gz") or name.endswith(".lc")
        is_hel1os_lc = name.startswith("lightcurve_") and name.endswith((".fits", ".fit", ".fits.gz"))
        is_spectrum = "spectra" in name and name.endswith((".fits", ".fit", ".fits.gz"))
        is_event = name == "evt.fits"
        is_gti = "gti" in name and name.endswith((".fits", ".fit", ".gz"))

        if june03_only and "20260603" not in str(path) and "2026\\06\\03" not in str(path):
            continue

        if is_solexs_lc or is_hel1os_lc or is_spectrum or is_event or is_gti:
            files.append(path)

    return sorted(files)


def classify_file(path: Path) -> str:
    path_str = str(path).lower()

    if "solexs" in path_str and (path_str.endswith(".lc.gz") or path_str.endswith(".lc")):
        return "SoLEXS lightcurve"
    if "lightcurve_cdte" in path_str:
        return "HEL1OS CdTe lightcurve"
    if "lightcurve_czt" in path_str:
        return "HEL1OS CZT lightcurve"
    if "spectra" in path_str:
        return "spectral file"
    if "gti" in path_str:
        return "GTI file"
    if path.name.lower() == "evt.fits":
        return "event file"
    return "unknown"


def inspect_fits(path: Path) -> list[dict]:
    rows = []

    print("\n" + "=" * 100)
    print(f"FILE: {path}")
    print(f"TYPE: {classify_file(path)}")

    try:
        with fits.open(path) as hdul:
            print("\nHDU SUMMARY:")
            hdul.info()

            for i, hdu in enumerate(hdul):
                header = hdu.header
                columns = []

                if hasattr(hdu, "columns") and hdu.columns is not None:
                    columns = list(hdu.columns.names)

                print(f"\n--- HDU {i} ---")
                print(f"EXTNAME : {header.get('EXTNAME', hdu.name)}")
                print(f"NAXIS   : {header.get('NAXIS')}")
                print(f"NAXIS1  : {header.get('NAXIS1', '')}")
                print(f"NAXIS2  : {header.get('NAXIS2', '')}")

                useful_keys = [
                    "TELESCOP",
                    "INSTRUME",
                    "DETNAM",
                    "DATE-OBS",
                    "DATE-END",
                    "TSTART",
                    "TSTOP",
                    "TIMEUNIT",
                    "TIMESYS",
                    "MJDREF",
                    "EXTNAME",
                    "ELOW",
                    "EHIGH",
                ]

                print("HEADER KEYS:")
                for key in useful_keys:
                    if key in header:
                        print(f"  {key}: {header[key]}")

                if columns:
                    print("COLUMNS:")
                    for col in columns:
                        print(f"  - {col}")

                rows.append(
                    {
                        "file": str(path),
                        "file_type": classify_file(path),
                        "hdu_index": i,
                        "extname": header.get("EXTNAME", hdu.name),
                        "naxis": header.get("NAXIS"),
                        "naxis1": header.get("NAXIS1", ""),
                        "naxis2": header.get("NAXIS2", ""),
                        "tstart": header.get("TSTART", ""),
                        "tstop": header.get("TSTOP", ""),
                        "date_obs": header.get("DATE-OBS", ""),
                        "date_end": header.get("DATE-END", ""),
                        "detnam": header.get("DETNAM", ""),
                        "elow": header.get("ELOW", ""),
                        "ehigh": header.get("EHIGH", ""),
                        "columns": ", ".join(columns),
                    }
                )

    except Exception as exc:
        print(f"[ERROR] Could not read FITS file: {path}")
        print(f"Reason: {exc}")
        rows.append(
            {
                "file": str(path),
                "file_type": classify_file(path),
                "hdu_index": "",
                "extname": "",
                "naxis": "",
                "naxis1": "",
                "naxis2": "",
                "tstart": "",
                "tstop": "",
                "date_obs": "",
                "date_end": "",
                "detnam": "",
                "elow": "",
                "ehigh": "",
                "columns": f"ERROR: {exc}",
            }
        )

    return rows


def list_zip_contents(zips_dir: Path = ZIPS_DIR) -> pd.DataFrame:
    if not zips_dir.exists():
        raise FileNotFoundError(f"Missing ZIP directory: {zips_dir.resolve()}")

    rows = []
    for zip_path in sorted(zips_dir.glob("*.zip")):
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                rows.append(
                    {
                        "zip_file": zip_path.name,
                        "member": member.filename,
                        "size_bytes": member.file_size,
                        "compressed_bytes": member.compress_size,
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("STEP 1: Listing ZIP contents...")
    zip_df = list_zip_contents()
    zip_report = REPORTS_DIR / "zip_contents.csv"
    zip_df.to_csv(zip_report, index=False)
    print(f"[SAVED] {zip_report}")

    print("\nSTEP 2: Extracting ZIPs...")
    if is_dev_mode() and RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        print(f"[DEV_MODE] Using existing raw data in {RAW_DIR}; skipping ZIP extraction.")
    else:
        extract_all_zips()

    print("\nSTEP 3: Finding useful FITS files...")
    files = find_candidate_files(june03_only=is_dev_mode())
    if is_dev_mode():
        print("[DEV_MODE] Inspecting only 2026-06-03 files.")

    if not files:
        raise FileNotFoundError("No candidate FITS/lightcurve files found. Check data/zips and data/raw.")

    print(f"[OK] Found {len(files)} candidate files:")
    for path in files:
        print(f"  - {classify_file(path)} :: {path}")

    print("\nSTEP 4: Inspecting FITS structures...")
    all_rows = []
    for path in files:
        all_rows.extend(inspect_fits(path))

    report_path = REPORTS_DIR / "fits_inspection_report.csv"
    pd.DataFrame(all_rows).to_csv(report_path, index=False)

    print("\n" + "=" * 100)
    print("[DONE]")
    print(f"ZIP report saved to:  {zip_report}")
    print(f"FITS report saved to: {report_path}")
