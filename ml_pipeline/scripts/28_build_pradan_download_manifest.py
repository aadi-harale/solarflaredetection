from __future__ import annotations

from pathlib import Path

import pandas as pd


OUT_DIR = Path("results") / "data_expansion"
TARGET_EVENT_PATH = OUT_DIR / "target_event_dates_for_download.csv"
TARGET_QUIET_PATH = OUT_DIR / "target_quiet_dates_for_download.csv"
MANIFEST_PATH = OUT_DIR / "pradan_download_manifest.csv"
HIGH_PRIORITY_PATH = OUT_DIR / "pradan_download_manifest_high_priority.csv"
MANIFEST_MD_PATH = OUT_DIR / "pradan_download_manifest.md"

PAYLOADS = ["SoLEXS", "HEL1OS"]
MANIFEST_COLUMNS = [
    "date",
    "payload",
    "level",
    "start_time_utc",
    "end_time_utc",
    "target_type",
    "priority",
    "reason",
    "expected_benefit",
    "download_status",
    "notes",
]


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required target file: {path.resolve()}")


def format_date(value: object) -> str:
    text = str(value).strip().replace(".0", "")
    if len(text) != 8 or not text.isdigit():
        raise ValueError(f"Expected YYYYMMDD date, got: {value!r}")
    return f"{text[:4]}-{text[4:6]}-{text[6:]}"


def load_targets() -> pd.DataFrame:
    require(TARGET_EVENT_PATH)
    require(TARGET_QUIET_PATH)
    event = pd.read_csv(TARGET_EVENT_PATH)
    quiet = pd.read_csv(TARGET_QUIET_PATH)
    targets = pd.concat([event, quiet], ignore_index=True)
    required = {"date", "target_type", "priority", "reason", "expected_benefit", "notes"}
    missing = required - set(targets.columns)
    if missing:
        raise ValueError(f"Target files are missing required columns: {sorted(missing)}")
    targets["date"] = targets["date"].astype(str).str.replace(r"\.0$", "", regex=True)
    targets = targets.drop_duplicates(subset=["date", "target_type", "priority", "reason"])
    priority_rank = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    merged_rows = []
    for date, group in targets.groupby("date", sort=True):
        priorities = group["priority"].astype(str).str.upper().tolist()
        best_priority = sorted(priorities, key=lambda value: priority_rank.get(value, 99))[0]
        target_type = "FLARE" if group["target_type"].astype(str).str.upper().eq("FLARE").any() else "QUIET"
        merged_rows.append(
            {
                "date": date,
                "target_type": target_type,
                "priority": best_priority,
                "reason": " | ".join(dict.fromkeys(group["reason"].astype(str))),
                "expected_benefit": " | ".join(dict.fromkeys(group["expected_benefit"].astype(str))),
                "notes": " | ".join(dict.fromkeys(group["notes"].astype(str))),
            }
        )
    return pd.DataFrame(merged_rows).sort_values(["priority", "target_type", "date"]).reset_index(drop=True)


def build_manifest() -> pd.DataFrame:
    targets = load_targets()
    rows = []
    for _, target in targets.iterrows():
        day = format_date(target["date"])
        for payload in PAYLOADS:
            rows.append(
                {
                    "date": target["date"],
                    "payload": payload,
                    "level": "Level-1",
                    "start_time_utc": f"{day} 00:00:00",
                    "end_time_utc": f"{day} 23:59:59",
                    "target_type": target["target_type"],
                    "priority": target["priority"],
                    "reason": target["reason"],
                    "expected_benefit": target["expected_benefit"],
                    "download_status": "PENDING",
                    "notes": target["notes"],
                }
            )
    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    return manifest.sort_values(["priority", "target_type", "date", "payload"]).reset_index(drop=True)


def write_markdown(manifest: pd.DataFrame, high: pd.DataFrame) -> None:
    target_dates = manifest["date"].nunique()
    lines = [
        "# PRADAN Download Manifest",
        "",
        "This manifest expands each Phase 6 target date into full-day Aditya-L1 Level-1 download windows for both SoLEXS and HEL1OS.",
        "",
        "Full-day windows are used because forecasting needs pre-flare background, preflare precursor activity, flare peak, and post-flare decay context.",
        "",
        f"- Total target dates: {target_dates}",
        f"- Total download rows: {len(manifest)}",
        f"- High-priority target dates: {high['date'].nunique()}",
        f"- High-priority download rows: {len(high)}",
        "",
        "## Required Payloads",
        "",
        "- SoLEXS Level-1",
        "- HEL1OS Level-1",
        "",
        "## Files",
        "",
        f"- `{MANIFEST_PATH}`",
        f"- `{HIGH_PRIORITY_PATH}`",
        "",
        "## High-Priority Preview",
        "",
        high.head(20).to_csv(index=False),
    ]
    MANIFEST_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    high = manifest[manifest["priority"].astype(str).str.upper().eq("HIGH")].copy()
    manifest.to_csv(MANIFEST_PATH, index=False)
    high.to_csv(HIGH_PRIORITY_PATH, index=False)
    write_markdown(manifest, high)
    print(f"total dates: {manifest['date'].nunique()}")
    print(f"total download rows: {len(manifest)}")
    print(f"high-priority dates: {high['date'].nunique()}")
    print(f"high-priority download rows: {len(high)}")
    print("output file paths:")
    print(MANIFEST_PATH)
    print(HIGH_PRIORITY_PATH)
    print(MANIFEST_MD_PATH)


if __name__ == "__main__":
    main()
