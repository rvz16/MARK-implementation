import json
import csv
from typing import Dict, Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, f1_score


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy using Hungarian algorithm for optimal matching.
    
    Returns:
        accuracy: float
        mapping: dict mapping predicted labels to true labels
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_pred.size == y_true.size
    
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    acc = float(w[row_ind, col_ind].sum()) / y_pred.size
    
    # Create mapping from predicted to true labels
    mapping = {row: col for row, col in zip(row_ind, col_ind)}
    
    return acc, mapping


def clustering_acc(y_true, y_pred) -> float:
    """Calculate clustering accuracy (for backward compatibility)."""
    acc, _ = cluster_acc(y_true, y_pred)
    return acc


def compute_all(y_true, y_pred) -> Dict[str, float]:
    """
    Compute all clustering evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted cluster assignments
    
    Returns:
        Dict with ACC, NMI, ARI, F1 scores
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    
    # Clustering accuracy with optimal mapping
    acc, mapping = cluster_acc(y_true, y_pred)
    
    # NMI - doesn't need label matching
    nmi = float(normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic'))
    
    # ARI - doesn't need label matching
    ari = float(adjusted_rand_score(y_true, y_pred))
    
    # F1 with mapped labels
    # Map predicted labels to true label space for proper F1 calculation
    y_pred_mapped = np.array([mapping.get(p, p) for p in y_pred])
    
    # Use macro F1 across all classes
    try:
        f1 = float(f1_score(y_true, y_pred_mapped, average="macro", zero_division=0))
    except Exception:
        # Fallback if labels don't match
        f1 = 0.0
    
    return {"ACC": acc, "NMI": nmi, "ARI": ari, "F1": f1}


def save_metrics(metrics: Dict[str, Any], json_path: str, csv_path: str) -> None:
    """Save metrics to JSON and CSV files."""
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in metrics.items():
            writer.writerow([k, v])
