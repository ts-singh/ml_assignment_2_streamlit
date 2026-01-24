"""Evaluation metrics used in the assignment."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef, confusion_matrix, classification_report
)


def _get_scores(model, X):
    """Return probability score for positive class.

    For models without predict_proba, fall back to decision_function.
    """
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
        # positive class assumed label 1
        return proba[:, 1]
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(X)
        # normalize scores to [0,1] for AUC stability
        smin, smax = scores.min(), scores.max()
        if smax - smin < 1e-12:
            return np.zeros_like(scores)
        return (scores - smin) / (smax - smin)
    # last resort
    return model.predict(X)


def evaluate_binary(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_score = _get_scores(model, X_test)

    metrics = {
        'Accuracy': float(accuracy_score(y_test, y_pred)),
        'AUC': float(roc_auc_score(y_test, y_score)),
        'Precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'Recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'F1': float(f1_score(y_test, y_pred, zero_division=0)),
        'MCC': float(matthews_corrcoef(y_test, y_pred)),
    }
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['benign', 'malignant'], zero_division=0)
    return metrics, cm, report
