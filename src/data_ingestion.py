"""
src/data_ingestion.py

Handles downloading the Olist Brazilian E-Commerce dataset via KaggleHub
and copying the raw CSVs into the project's data/raw/ folder for consistency.
"""

import os
import shutil
from pathlib import Path

import kagglehub
from dotenv import load_dotenv

# Load KAGGLE_USERNAME / KAGGLE_KEY from .env
load_dotenv()

DATASET_SLUG = "olistbr/brazilian-ecommerce"

# Project root is two levels up from this file (src/data_ingestion.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def download_olist_dataset() -> Path:
    """
    Downloads the Olist e-commerce dataset via KaggleHub (cached locally by
    kagglehub after the first call) and returns the path to the cached folder.
    """
    cache_path = kagglehub.dataset_download(DATASET_SLUG)
    print(f"Dataset downloaded/cached at: {cache_path}")
    return Path(cache_path)


def copy_to_raw(cache_path: Path, raw_dir: Path = RAW_DATA_DIR) -> None:
    """
    Copies all CSV files from the KaggleHub cache directory into data/raw/,
    so the rest of the project always reads from a consistent local path
    regardless of where kagglehub decides to cache things.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(cache_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {cache_path}")

    for csv_file in csv_files:
        destination = raw_dir / csv_file.name
        shutil.copy2(csv_file, destination)
        print(f"Copied {csv_file.name} -> {destination}")

    print(f"\n{len(csv_files)} files copied to {raw_dir}")


def ensure_olist_data() -> Path:
    """
    Convenience function: downloads (if needed) and copies the Olist dataset
    into data/raw/. Safe to call every time — kagglehub caches downloads,
    and copy_to_raw will just overwrite with the same files.
    """
    cache_path = download_olist_dataset()
    copy_to_raw(cache_path)
    return RAW_DATA_DIR


if __name__ == "__main__":
    ensure_olist_data()