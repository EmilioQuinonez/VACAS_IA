"""Selección con validación y evaluación de probabilidades y decisiones binarias."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)

def choose_threshold(actual, probability, target_recall=0.80):
    """Máxima precisión con recall suficiente; desempata por mayor umbral."""
    precision, recall, thresholds = precision_recall_curve(actual, probability)

    # Encontrar umbrales que cumplen el objetivo de target_recall
    eligible = np.flatnonzero(recall[:-1] >= target_recall)
    
    if not len(eligible):
        raise ValueError("No hay umbral que cumpla la sensibilidad objetivo.")
    best = max(eligible, key=lambda i: (precision[i], thresholds[i]))
    
    return float(thresholds[best])

def evaluate(actual, probability, threshold):
    """Evalúa probabilidades y decisiones con average precision y otras métricas."""
    actual, probability = np.asarray(actual), np.asarray(probability)
    result = {
        "n": len(actual), "prevalence": float(actual.mean()),
        "accuracy_no_event": float(1 - actual.mean()),
        "roc_auc": float(roc_auc_score(actual, probability)),
        "average_precision": float(average_precision_score(actual, probability)),
        "brier": float(brier_score_loss(actual, probability)),
    }
    
    for name, cutoff in (("threshold_0.5", 0.5), ("threshold_recall", threshold)):
        prediction = probability >= cutoff
        result[name] = {
            "threshold": float(cutoff), "accuracy": float(accuracy_score(actual, prediction)),
            "precision": float(precision_score(actual, prediction, zero_division=0)),
            "recall": float(recall_score(actual, prediction)),
            "f1": float(f1_score(actual, prediction, zero_division=0)),
            "confusion_matrix": confusion_matrix(actual, prediction, labels=[0, 1]).tolist(),
        }
    
    return result

def explanation_coverage(explanations, probability, threshold):
    alerts = np.asarray(probability) >= threshold
    complete = explanations[[f"factor_{i}" for i in range(1, 4)]].ne("").all(axis=1).to_numpy()
    
    return {
        "alerts": int(alerts.sum()), "alerts_with_three_factors": int((alerts & complete).sum()),
        "coverage": float(complete[alerts].mean()) if alerts.any() else None,
    }
