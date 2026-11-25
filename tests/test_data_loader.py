"""
Unit tests for data loader and LIBSVM parser.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import FEATURE_NAMES, GAS_CLASSES, DATASET_RAW_DIR
from src.data_loader import parse_libsvm_file, load_processed_data


def test_parse_libsvm_single_batch():
    """Verify single batch file parsing integrity."""
    batch1_path = DATASET_RAW_DIR / "batch1.dat"
    if not batch1_path.exists():
        pytest.skip("batch1.dat not found in raw data directory.")
        
    df = parse_libsvm_file(batch1_path, batch_id=1)
    
    assert len(df) == 445, f"Expected 445 records in batch 1, found {len(df)}"
    assert "batch_id" in df.columns
    assert "gas_class" in df.columns
    assert "gas_name" in df.columns
    assert df["batch_id"].iloc[0] == 1
    
    # Check that all 128 features are present
    for f in FEATURE_NAMES:
        assert f in df.columns, f"Feature {f} missing from parsed DataFrame"


def test_load_processed_data():
    """Verify full processed dataset load."""
    df = load_processed_data()
    assert len(df) == 13910, f"Expected 13,910 records, found {len(df)}"
    assert df["batch_id"].nunique() == 10, "Expected 10 unique batches"
    assert df["gas_class"].nunique() == 6, "Expected 6 gas classes"
    assert df[FEATURE_NAMES].isnull().sum().sum() == 0, "Processed dataset contains nulls in features"
