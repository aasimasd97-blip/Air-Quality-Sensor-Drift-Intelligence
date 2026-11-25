"""
Unit tests for data quality diagnostics and chronological scaling.
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import FEATURE_NAMES
from src.preprocessing import DataQualityChecker, ChronologicalPreprocessor


def test_data_quality_checker():
    """Verify missing, inf, and duplicate detection."""
    # Create test dataframe
    data = {f: np.random.randn(50) for f in FEATURE_NAMES}
    df = pd.DataFrame(data)
    df["batch_id"] = 1
    df["gas_class"] = 1
    
    # Inject missing and infinite values
    df.loc[0, "f1"] = np.nan
    df.loc[1, "f2"] = np.inf
    
    report = DataQualityChecker.inspect_dataframe(df)
    assert report["total_nulls"] == 1
    assert report["total_infinite_values"] == 1
    assert not report["is_clean"]
    
    cleaned = DataQualityChecker.clean_dataframe(df)
    clean_report = DataQualityChecker.inspect_dataframe(cleaned)
    assert clean_report["is_clean"]


def test_chronological_scaling_no_leakage():
    """Verify that scaler fits strictly on train batches and normalizes properly."""
    np.random.seed(42)
    rows = []
    for b in [1, 2]:
        for _ in range(100):
            r = {f"f{i}": (np.random.randn() + (b * 5.0)) for i in range(1, 129)}
            r["batch_id"] = b
            r["gas_class"] = 1
            rows.append(r)
    df = pd.DataFrame(rows)

    prep = ChronologicalPreprocessor(scaler_type="standard")
    train_df, test_df = prep.fit_transform_chronological(df, train_batch_ids=[1])
    
    # Train mean should be approximately 0.0 and std approximately 1.0
    train_f1_mean = train_df["f1"].mean()
    train_f1_std = train_df["f1"].std()
    assert abs(train_f1_mean) < 0.05, f"Train mean not centered: {train_f1_mean}"
    assert abs(train_f1_std - 1.0) < 0.1, f"Train std not scaled: {train_f1_std}"

    # Test set (Batch 2 was shifted by +5) should have positive non-zero mean
    assert test_df["f1"].mean() > 2.0, "Test set was inappropriately scaled to zero (data leakage)"
