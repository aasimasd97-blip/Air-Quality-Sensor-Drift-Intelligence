"""
Data Preprocessing, Quality Validation, and Leakage-Free Scaling Module.
"""
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, RobustScaler

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    FEATURE_NAMES,
    SCALER_PATH,
    RANDOM_STATE
)


class DataQualityChecker:
    """
    Performs data integrity and quality diagnostics:
    - Missing values detection
    - Infinite values detection
    - Duplicate records verification
    - Data type consistency
    """
    
    @staticmethod
    def inspect_dataframe(df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Runs comprehensive quality checks on the dataset.
        """
        feats = feature_cols or [c for c in FEATURE_NAMES if c in df.columns]
        
        # Missing values
        null_counts = df[feats].isnull().sum()
        total_nulls = int(null_counts.sum())
        cols_with_nulls = {k: int(v) for k, v in null_counts[null_counts > 0].items()}
        
        # Infinite values
        num_matrix = df[feats].to_numpy()
        inf_mask = np.isinf(num_matrix)
        total_infs = int(np.sum(inf_mask))
        
        # Duplicate records (based on features)
        dup_count = int(df.duplicated(subset=feats).sum())
        
        # Batch and class consistency
        batches = sorted(df["batch_id"].unique().tolist()) if "batch_id" in df.columns else []
        classes = sorted(df["gas_class"].unique().tolist()) if "gas_class" in df.columns else []
        
        report = {
            "total_rows": len(df),
            "total_features": len(feats),
            "total_nulls": total_nulls,
            "columns_with_nulls": cols_with_nulls,
            "total_infinite_values": total_infs,
            "duplicate_feature_rows": dup_count,
            "batches_present": batches,
            "gas_classes_present": classes,
            "is_clean": (total_nulls == 0 and total_infs == 0)
        }
        return report

    @staticmethod
    def clean_dataframe(df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Replaces infinite values with NaN and applies median imputation if necessary.
        """
        cleaned = df.copy()
        feats = feature_cols or [c for c in FEATURE_NAMES if c in cleaned.columns]
        cleaned[feats] = cleaned[feats].replace([np.inf, -np.inf], np.nan)
        if cleaned[feats].isnull().sum().sum() > 0:
            cleaned[feats] = cleaned[feats].fillna(cleaned[feats].median())
        return cleaned


class ChronologicalPreprocessor:
    """
    Ensures strict chronological data partitioning to prevent data leakage.
    Scalers are fitted EXCLUSIVELY on training batch(es) and applied to future batches.
    """
    
    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type
        if scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = FEATURE_NAMES

    def fit_on_batches(self, df: pd.DataFrame, train_batch_ids: List[int]) -> "ChronologicalPreprocessor":
        """
        Fits scaler strictly on designated training batch(es).
        """
        train_mask = df["batch_id"].isin(train_batch_ids)
        train_data = df.loc[train_mask, self.feature_names]
        if len(train_data) == 0:
            raise ValueError(f"No samples found for training batches: {train_batch_ids}")
            
        self.scaler.fit(train_data)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms feature columns using fitted scaler, keeping metadata columns intact.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transforming data.")
        
        transformed_df = df.copy()
        transformed_matrix = self.scaler.transform(transformed_df[self.feature_names])
        transformed_df[self.feature_names] = transformed_matrix
        return transformed_df

    def fit_transform_chronological(
        self,
        df: pd.DataFrame,
        train_batch_ids: List[int] = [1]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convenience method to fit on train batches and return scaled train & test sets.
        """
        self.fit_on_batches(df, train_batch_ids)
        
        train_mask = df["batch_id"].isin(train_batch_ids)
        test_mask = ~train_mask
        
        train_df = self.transform(df[train_mask].copy())
        test_df = self.transform(df[test_mask].copy()) if test_mask.any() else pd.DataFrame()
        
        return train_df, test_df

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Saves fitted scaler to disk."""
        target_path = filepath or SCALER_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, target_path)
        return target_path

    def load(self, filepath: Optional[Path] = None) -> "ChronologicalPreprocessor":
        """Loads fitted scaler from disk."""
        target_path = filepath or SCALER_PATH
        if not target_path.exists():
            raise FileNotFoundError(f"Scaler file not found at {target_path}")
        self.scaler = joblib.load(target_path)
        self.is_fitted = True
        return self


def compute_descriptive_statistics(df: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Computes statistical moments: count, mean, std, median, IQR, min, max, skewness, kurtosis.
    """
    feats = feature_cols or [c for c in FEATURE_NAMES if c in df.columns]
    data = df[feats]
    
    desc = pd.DataFrame({
        "Mean": data.mean(),
        "Std": data.std(),
        "Median": data.median(),
        "Min": data.min(),
        "Max": data.max(),
        "IQR": data.quantile(0.75) - data.quantile(0.25),
        "Skewness": data.skew(),
        "Kurtosis": data.kurtosis()
    })
    return desc.round(4)


if __name__ == "__main__":
    from src.data_loader import load_processed_data
    df = load_processed_data()
    
    # 1. Quality Check
    report = DataQualityChecker.inspect_dataframe(df)
    print("[DataQualityChecker Report]:", report)
    
    # 2. Chronological Scaling (Fit strictly on Batch 1)
    preprocessor = ChronologicalPreprocessor(scaler_type="standard")
    train_df, test_df = preprocessor.fit_transform_chronological(df, train_batch_ids=[1])
    preprocessor.save()
    print(f"[Preprocessing] Scaler fitted on Batch 1 and saved to {SCALER_PATH}")
    print(f"  Train samples (Batch 1): {len(train_df)}, Test samples (Batches 2-10): {len(test_df)}")
