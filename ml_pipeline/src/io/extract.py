from __future__ import annotations

from pathlib import Path
import zipfile

from src.utils.config import is_dev_mode


ZIPS_DIR = Path("data/zips")
RAW_DIR = Path("data/raw")


def safe_extract_zip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            target = (out_dir / member.filename).resolve()
            if not str(target).startswith(str(out_dir.resolve())):
                raise RuntimeError(f"Unsafe ZIP path detected: {member.filename}")
        zf.extractall(out_dir)


def extract_all_zips(zips_dir: Path = ZIPS_DIR, raw_dir: Path = RAW_DIR) -> None:
    if not zips_dir.exists():
        raise FileNotFoundError(f"Missing ZIP directory: {zips_dir.resolve()}")

    zip_files = sorted(zips_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError(f"No ZIP files found in {zips_dir.resolve()}")

    raw_dir.mkdir(parents=True, exist_ok=True)

    for zip_path in zip_files:
        out_dir = raw_dir / zip_path.stem
        if out_dir.exists():
            print(f"[SKIP] Already extracted: {zip_path.name}")
            continue

        print(f"[EXTRACT] {zip_path.name} -> {out_dir}")
        safe_extract_zip(zip_path, out_dir)


def extract_nested_zips(root: Path = RAW_DIR) -> None:
    if not root.exists():
        raise FileNotFoundError(f"Missing raw data directory: {root.resolve()}")

    while True:
        zip_files = sorted(root.rglob("*.zip"))
        pending = []

        for zip_path in zip_files:
            out_dir = zip_path.with_suffix("")
            if not out_dir.exists():
                pending.append((zip_path, out_dir))

        if not pending:
            print("[DONE] No more nested ZIPs to extract.")
            break

        for zip_path, out_dir in pending:
            print(f"[EXTRACT NESTED] {zip_path} -> {out_dir}")
            safe_extract_zip(zip_path, out_dir)


def main() -> None:
    if is_dev_mode() and RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        print(f"[DEV_MODE] Using existing raw data in {RAW_DIR}; skipping ZIP extraction.")
        return

    extract_all_zips()
    extract_nested_zips()
