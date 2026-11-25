"""
Model Evaluation and Chronological Degradation Tracking Module.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import GAS_CLASSES


def format_classification_metrics(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Formats the comparison results of multiple models into a clean display DataFrame.
    """
    rows = []
    for model_name, res in results.items():
        rows.append({
            "Model": model_name,
            "Accuracy": round(res["accuracy"], 4),
            "Weighted F1": round(res["f1_weighted"], 4),
            "Weighted Precision": round(res["precision"], 4),
            "Weighted Recall": round(res["recall"], 4),
            "Training Time (s)": round(res["fit_time_seconds"], 3),
            "Train Samples": res.get("train_samples", "-"),
            "Test Samples": res.get("test_samples", "-")
        })
    return pd.DataFrame(rows).sort_values("Weighted F1", ascending=False).reset_index(drop=True)


def get_per_class_metrics(classification_report_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Extracts class-level Precision, Recall, and F1-score from sklearn classification_report dictionary.
    """
    rows = []
    for gas_id, gas_name in GAS_CLASSES.items():
        if gas_name in classification_report_dict:
            metrics = classification_report_dict[gas_name]
            rows.append({
                "Gas Class": gas_name,
                "Precision": round(metrics.get("precision", 0.0), 4),
                "Recall": round(metrics.get("recall", 0.0), 4),
                "F1-Score": round(metrics.get("f1-score", 0.0), 4),
                "Support": int(metrics.get("support", 0))
            })
    return pd.DataFrame(rows)


def calculate_degradation_statistics(progressive_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes drift-induced degradation statistics:
    - Initial vs Final batch accuracy
    - Absolute degradation percentage
    - Batch with maximum drop
    """
    if progressive_df.empty or len(progressive_df) < 2:
        return {}

    b_init = progressive_df.iloc[0]
    b_final = progressive_df.iloc[-1]
    
    init_acc = b_init["accuracy"]
    final_acc = b_final["accuracy"]
    abs_drop = (init_acc - final_acc) * 100.0
    rel_drop = ((init_acc - final_acc) / init_acc) * 100.0 if init_acc > 0 else 0.0
    
    # Calculate step-by-step drops
    diffs = progressive_df["accuracy"].diff().fillna(0)
    worst_drop_idx = diffs.idxmin()
    worst_batch = progressive_df.loc[worst_drop_idx, "batch_id"]
    worst_drop_val = abs(diffs.loc[worst_drop_idx]) * 100.0

    return {
        "initial_batch_id": int(b_init["batch_id"]),
        "initial_accuracy": round(init_acc * 100.0, 2),
        "final_batch_id": int(b_final["batch_id"]),
        "final_accuracy": round(final_acc * 100.0, 2),
        "absolute_accuracy_drop_pct": round(abs_drop, 2),
        "relative_accuracy_drop_pct": round(rel_drop, 2),
        "sharpest_drop_batch": int(worst_batch),
        "sharpest_drop_pct": round(worst_drop_val, 2)
    }
