from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score


def classification_report(y_true, y_pred, classes: list[str]) -> dict:
    labels = list(range(len(classes)))
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    present = sorted(set(np.asarray(y_true).tolist()))
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        # macro-F1 over the classes that actually occur in this test set
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "per_class_recall": {classes[i]: float(rec[i]) for i in present},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classes": classes,
    }
