# gspo_readmit/metrics.py
from typing import Dict, List
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def compute_classification_metrics(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    """
    Compute classification metrics: accuracy, precision, recall, F1, and confusion matrix.
    
    Args:
        predictions: List of predicted labels (0 or 1)
        labels: List of true labels (0 or 1)
    
    Returns:
        Dictionary with metrics
    """
    if len(predictions) == 0 or len(labels) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "precision_class_0": 0.0,
            "recall_class_0": 0.0,
            "f1_class_0": 0.0,
            "precision_class_1": 0.0,
            "recall_class_1": 0.0,
            "f1_class_1": 0.0,
            "tn": 0, "fp": 0, "fn": 0, "tp": 0,
        }
    
    # Convert to numpy arrays
    y_pred = np.array(predictions)
    y_true = np.array(labels)
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics (binary classification)
    precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
    
    # Per-class metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Confusion matrix: [TN, FP], [FN, TP]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    elif cm.size == 1:
        # Only one class present
        if y_true[0] == 0:
            tn, fp, fn, tp = int(cm[0, 0]), 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, int(cm[0, 0])
    else:
        # Fallback
        tn, fp, fn, tp = 0, 0, 0, 0
    
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "precision_class_0": float(precision_per_class[0]) if len(precision_per_class) > 0 else 0.0,
        "recall_class_0": float(recall_per_class[0]) if len(recall_per_class) > 0 else 0.0,
        "f1_class_0": float(f1_per_class[0]) if len(f1_per_class) > 0 else 0.0,
        "precision_class_1": float(precision_per_class[1]) if len(precision_per_class) > 1 else 0.0,
        "recall_class_1": float(recall_per_class[1]) if len(recall_per_class) > 1 else 0.0,
        "f1_class_1": float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    
    return metrics

