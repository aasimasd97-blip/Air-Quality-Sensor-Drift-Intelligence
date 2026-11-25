"""
Exploratory Data Analysis (EDA) metrics and summary generators.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    FEATURE_NAMES,
    GAS_CLASSES,
    NUM_SENSORS,
    FEATURES_PER_SENSOR
)
from src.feature_engineering import SensorFeatureExtractor


def get_dataset_overview_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Returns high-level metadata and counts for the overview dashboard.
    """
    total_records = len(df)
    total_sensors = NUM_SENSORS
    total_features = len([c for c in FEATURE_NAMES if c in df.columns])
    batches = sorted(df["batch_id"].unique().tolist()) if "batch_id" in df.columns else []
    gas_counts = df["gas_name"].value_counts().to_dict() if "gas_name" in df.columns else {}
    
    batch_counts = df["batch_id"].value_counts().sort_index().to_dict() if "batch_id" in df.columns else {}

    return {
        "total_records": total_records,
        "total_sensors": total_sensors,
        "total_features": total_features,
        "batches": batches,
        "total_batches": len(batches),
        "gas_classes": list(gas_counts.keys()),
        "gas_distribution": gas_counts,
        "batch_distribution": batch_counts
    }


def get_batch_class_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a pivot table of measurement counts per batch and gas class.
    """
    pivot = pd.crosstab(
        df["batch_id"],
        df["gas_name"],
        margins=True,
        margins_name="Total"
    )
    return pivot


def get_sensor_response_profiles(df: pd.DataFrame, batch_id: Optional[int] = None) -> pd.DataFrame:
    """
    Computes average steady-state and dynamic response profiles for each of the 16 sensors.
    """
    subset = df[df["batch_id"] == batch_id] if batch_id is not None else df
    profiles = []
    
    for s in range(1, NUM_SENSORS + 1):
        cols = SensorFeatureExtractor.get_sensor_columns(s)
        s_data = subset[cols]
        profiles.append({
            "Sensor_ID": f"Sensor {s}",
            "Sensor_Num": s,
            "Steady_State_Peak": s_data[cols[0]].mean(),
            "Steady_State_Norm": s_data[cols[1]].mean(),
            "Transient_Rising_Avg": s_data[cols[2:5]].mean().mean(),
            "Transient_Decaying_Avg": s_data[cols[5:8]].mean().mean(),
            "Mean_Response": s_data.mean().mean(),
            "Std_Response": s_data.std().mean()
        })
        
    return pd.DataFrame(profiles)


def get_batch_variability_summary(df: pd.DataFrame, feature_name: str = "f1") -> pd.DataFrame:
    """
    Tracks how a specific feature's mean, std, and coefficient of variation (CV) change across all 10 batches.
    """
    summary = []
    for b in sorted(df["batch_id"].unique()):
        b_data = df[df["batch_id"] == b][feature_name]
        mean_val = b_data.mean()
        std_val = b_data.std()
        cv = (std_val / abs(mean_val)) if mean_val != 0 else 0
        summary.append({
            "batch_id": b,
            "mean": mean_val,
            "std": std_val,
            "median": b_data.median(),
            "cv": cv,
            "count": len(b_data)
        })
    return pd.DataFrame(summary)
