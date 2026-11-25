"""
Gas Classification models, chronological evaluation, and progressive degradation tracking.
"""
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import time
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    FEATURE_NAMES,
    GAS_CLASSES,
    GAS_CLASSIFIER_PATH,
    MODEL_METADATA_PATH,
    RANDOM_STATE
)


def get_model_instances() -> Dict[str, Any]:
    """Returns freshly configured model instances for comparison."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=4,
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=RANDOM_STATE
        ),
        "Support Vector Machine (Linear)": CalibratedClassifierCV(
            LinearSVC(dual=False, C=1.0, random_state=RANDOM_STATE)
        )
    }


def train_and_evaluate_all_models(
    df_scaled: pd.DataFrame,
    train_batches: List[int] = [1],
    test_batches: Optional[List[int]] = None
) -> Tuple[Dict[str, Dict[str, Any]], Any]:
    """
    Trains Random Forest, SVM, and Logistic Regression on chronological train batches,
    and evaluates on test batches.
    """
    train_mask = df_scaled["batch_id"].isin(train_batches)
    if test_batches is not None:
        test_mask = df_scaled["batch_id"].isin(test_batches)
    else:
        test_mask = ~train_mask

    X_train = df_scaled.loc[train_mask, FEATURE_NAMES]
    y_train = df_scaled.loc[train_mask, "gas_class"]
    X_test = df_scaled.loc[test_mask, FEATURE_NAMES]
    y_test = df_scaled.loc[test_mask, "gas_class"]

    models = get_model_instances()
    results = {}
    best_model = None
    best_f1 = -1.0

    target_names = [GAS_CLASSES[i] for i in sorted(GAS_CLASSES.keys())]

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - t0
        
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred, labels=sorted(GAS_CLASSES.keys()))
        report = classification_report(
            y_test,
            y_pred,
            labels=sorted(GAS_CLASSES.keys()),
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
        
        results[name] = {
            "model_obj": model,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_weighted": float(f1),
            "fit_time_seconds": float(fit_time),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = (name, model)

    return results, best_model


def progressive_batch_evaluation(
    df_scaled: pd.DataFrame,
    model: Any,
    batch_col: str = "batch_id",
    target_col: str = "gas_class"
) -> pd.DataFrame:
    """
    Evaluates a trained model across each chronological batch individually (1 to 10),
    tracking accuracy, precision, recall, and F1 degradation over time.
    """
    records = []
    unique_batches = sorted(df_scaled[batch_col].unique())
    
    for b in unique_batches:
        b_df = df_scaled[df_scaled[batch_col] == b]
        X_b = b_df[FEATURE_NAMES]
        y_b = b_df[target_col]
        
        y_pred = model.predict(X_b)
        acc = accuracy_score(y_b, y_pred)
        prec = precision_score(y_b, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_b, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_b, y_pred, average="weighted", zero_division=0)
        
        records.append({
            "batch_id": int(b),
            "sample_count": len(b_df),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1)
        })
        
    return pd.DataFrame(records)


def save_classifier_and_metadata(
    model: Any,
    model_name: str,
    metrics: Dict[str, Any],
    metadata_path: Optional[Path] = None
) -> Tuple[Path, Path]:
    """
    Saves trained classifier to models/gas_classifier.pkl and metadata to models/model_metadata.json.
    """
    model_path = GAS_CLASSIFIER_PATH
    meta_path = metadata_path or MODEL_METADATA_PATH
    
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    
    meta_content = {
        "model_name": model_name,
        "trained_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "feature_count": len(FEATURE_NAMES),
        "target_classes": GAS_CLASSES
    }
    
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_content, f, indent=2)
        
    print(f"[GasClassifier] Model saved to {model_path}")
    print(f"[GasClassifier] Metadata saved to {meta_path}")
    return model_path, meta_path


def load_gas_classifier() -> Tuple[Any, Dict[str, Any]]:
    """Loads gas classifier model and metadata."""
    if not GAS_CLASSIFIER_PATH.exists():
        raise FileNotFoundError(f"Trained model not found at {GAS_CLASSIFIER_PATH}")
    model = joblib.load(GAS_CLASSIFIER_PATH)
    meta = {}
    if MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
    return model, meta


if __name__ == "__main__":
    from src.data_loader import load_processed_data
    from src.preprocessing import ChronologicalPreprocessor

    df = load_processed_data()
    prep = ChronologicalPreprocessor().load()
    scaled_df = prep.transform(df)

    results, best = train_and_evaluate_all_models(scaled_df, train_batches=[1])
    best_name, best_model_obj = best
    
    print(f"\n--- Model Comparison (Trained on Batch 1, Tested on Batches 2-10) ---")
    for name, res in results.items():
        print(f"[{name}] Acc: {res['accuracy']:.4f} | Prec: {res['precision']:.4f} | Rec: {res['recall']:.4f} | F1: {res['f1_weighted']:.4f}")

    # Progressive degradation
    prog_df = progressive_batch_evaluation(scaled_df, best_model_obj)
    print("\n--- Progressive Batch Accuracy Degradation ---")
    print(prog_df[["batch_id", "sample_count", "accuracy", "f1_score"]].to_string(index=False))

    # Save best model
    save_classifier_and_metadata(best_model_obj, best_name, {
        "accuracy": results[best_name]["accuracy"],
        "f1_weighted": results[best_name]["f1_weighted"],
        "progressive_degradation": prog_df.to_dict(orient="records")
    })
