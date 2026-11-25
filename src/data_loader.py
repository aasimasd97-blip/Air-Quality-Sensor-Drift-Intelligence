"""
Data loading and LIBSVM parsing module for the UCI Gas Sensor Array Drift Dataset.
"""
import os
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    DATASET_RAW_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    PROCESSED_DATASET_PATH,
    PROCESSED_CSV_PATH,
    SAMPLE_DATASET_PATH,
    REFERENCE_DATASET_PATH,
    GAS_CLASSES,
    FEATURE_NAMES,
    TOTAL_FEATURES,
    RANDOM_STATE
)

DATASET_DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip"


def download_dataset_if_needed(target_dir: Optional[Path] = None) -> Path:
    """
    Checks if raw batch files exist; if not, downloads and extracts the UCI dataset.
    """
    raw_dir = target_dir or RAW_DATA_DIR
    dataset_dir = raw_dir / "Dataset"
    
    # Check if all batch files 1-10 are already present
    all_present = dataset_dir.exists() and all((dataset_dir / f"batch{b}.dat").exists() for b in range(1, 11))
    if all_present:
        return dataset_dir

    print(f"[DataLoader] Downloading UCI Gas Sensor Drift Dataset from {DATASET_DOWNLOAD_URL}...")
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "gas_drift_dataset.zip"

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    req = urllib.request.Request(DATASET_DOWNLOAD_URL, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as response, open(zip_path, "wb") as out_file:
        out_file.write(response.read())

    print(f"[DataLoader] Download complete ({zip_path.stat().st_size / (1024*1024):.2f} MB). Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(raw_dir)

    print("[DataLoader] Dataset successfully extracted.")
    return dataset_dir


def parse_libsvm_file(file_path: Path, batch_id: int) -> pd.DataFrame:
    """
    Parses a single LIBSVM-formatted batch file into a pandas DataFrame.
    Format per line: <gas_class> 1:<val1> 2:<val2> ... 128:<val128>
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Batch file not found: {file_path}")

    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            gas_class = int(tokens[0])
            record = {
                "batch_id": int(batch_id),
                "gas_class": gas_class,
                "gas_name": GAS_CLASSES.get(gas_class, f"Class_{gas_class}")
            }
            
            # Pre-fill features with 0.0 or parse explicit key:value pairs
            for item in tokens[1:]:
                if ":" in item:
                    feat_idx, val = item.split(":", 1)
                    record[f"f{feat_idx}"] = float(val)
            
            records.append(record)

    df = pd.DataFrame(records)
    
    # Ensure all 128 feature columns exist and are ordered
    for feat in FEATURE_NAMES:
        if feat not in df.columns:
            df[feat] = 0.0

    ordered_cols = ["batch_id", "gas_class", "gas_name"] + FEATURE_NAMES
    return df[ordered_cols]


def load_all_batches(raw_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads and aggregates all 10 batches into a single consolidated DataFrame.
    Preserves original batch IDs and gas class labels.
    """
    dataset_dir = download_dataset_if_needed(raw_dir)
    batch_dfs = []
    print("[DataLoader] Loading and parsing 10 chronological batches...")
    
    for b in range(1, 11):
        batch_file = dataset_dir / f"batch{b}.dat"
        b_df = parse_libsvm_file(batch_file, batch_id=b)
        print(f"  - Loaded Batch {b}: {len(b_df)} observations")
        batch_dfs.append(b_df)
        
    full_df = pd.concat(batch_dfs, ignore_index=True)
    print(f"[DataLoader] Total observations aggregated: {len(full_df)} across {full_df['batch_id'].nunique()} batches.")
    return full_df


def save_processed_data(df: pd.DataFrame, output_dir: Optional[Path] = None) -> Tuple[Path, Path]:
    """
    Saves processed dataset as Parquet and CSV, along with a reference baseline (Batch 1)
    and a lightweight sample dataset.
    """
    out_dir = output_dir or PROCESSED_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    
    parquet_path = out_dir / "gas_drift_processed.parquet"
    csv_path = out_dir / "gas_drift_processed.csv"
    
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    print(f"[DataLoader] Processed dataset saved to:\n  - {parquet_path}\n  - {csv_path}")

    # Save Batch 1 as dedicated reference baseline
    batch1_df = df[df["batch_id"] == 1].copy()
    batch1_df.to_csv(REFERENCE_DATASET_PATH, index=False)
    print(f"[DataLoader] Reference baseline (Batch 1) saved to: {REFERENCE_DATASET_PATH}")

    # Save a representative stratified sample for quick testing / upload demo
    sample_df = df.groupby(["batch_id", "gas_class"], group_keys=False).apply(
        lambda g: g.sample(min(len(g), 5), random_state=RANDOM_STATE)
    ).reset_index(drop=True)
    sample_df.to_csv(SAMPLE_DATASET_PATH, index=False)
    print(f"[DataLoader] Sample dataset ({len(sample_df)} records) saved to: {SAMPLE_DATASET_PATH}")

    return parquet_path, csv_path


def load_processed_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Loads the processed dataset from Parquet if available, falling back to CSV
    or regenerating from raw if neither exists.
    """
    target_parquet = file_path or PROCESSED_DATASET_PATH
    if target_parquet.exists():
        return pd.read_parquet(target_parquet)
    
    if PROCESSED_CSV_PATH.exists():
        return pd.read_csv(PROCESSED_CSV_PATH)

    print("[DataLoader] Processed dataset not found. Generating from raw batch files...")
    df = load_all_batches()
    save_processed_data(df)
    return df


if __name__ == "__main__":
    df = load_all_batches()
    save_processed_data(df)
    print("DataLoader pipeline executed successfully!")
