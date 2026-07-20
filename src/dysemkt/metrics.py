from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64).clip(1e-7, 1 - 1e-7)
    result = {
        "accuracy": float(accuracy_score(labels, probabilities >= 0.5)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
    }
    result["roc_auc"] = float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else float("nan")
    return result

