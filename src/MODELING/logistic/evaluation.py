"""Selección con validación y evaluación de probabilidades y decisiones binarias."""

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, precision_recall_curve, precision_recall_fscore_support, roc_auc_score,
)

def choose_threshold(actual, probability, target_recall=0.80):
    """Mayor umbral que alcanza el recall objetivo en validación."""
    actual, probability = np.asarray(actual), np.asarray(probability, dtype=float)
    if not actual.sum():
        raise ValueError("Validación requiere positivos para elegir el umbral.")
    _, recall, thresholds = precision_recall_curve(actual, probability)

    # Encontrar umbrales que cumplen el objetivo de target_recall
    eligible = np.flatnonzero(recall[:-1] >= target_recall)
    
    best = eligible[np.argmax(thresholds[eligible])]
    
    return float(thresholds[best])

def evaluate(actual, probability, threshold):
    """Evalúa probabilidades y decisiones con average precision y otras métricas."""
    actual, probability = np.asarray(actual), np.asarray(probability, dtype=float)
    n = len(actual)
    positives = int(actual.sum())
    prevalence = float(actual.mean()) if n else None
    both_classes = 0 < positives < n
    result = {
        "n": n, "positives": positives, "prevalence": prevalence,
        "accuracy_no_event": 1 - prevalence if n else None,
        "accuracy_majority": max(prevalence, 1 - prevalence) if n else None,
        "f1_always_alert": float(f1_score(actual, np.ones(n), zero_division=0)) if n else None,
        "roc_auc": float(roc_auc_score(actual, probability)) if both_classes else None,
        "average_precision": float(average_precision_score(actual, probability)) if positives else None,
        "brier": float(brier_score_loss(actual, probability)) if n else None,
        "log_loss": float(log_loss(actual, probability, labels=[0, 1])) if n else None,
        "undefined_metrics_note": None if both_classes else "Sin ambas clases: ROC-AUC no calculable; sin positivos: AP y recall no calculables. Sin predicciones positivas: precisión no calculable. Vacío: métricas no calculables.",
    }
    
    for name, cutoff in (("threshold_0.5", 0.5), ("threshold_recall", threshold)):
        prediction = probability >= cutoff
        matrix = confusion_matrix(actual, prediction, labels=[0, 1]) if n else np.zeros((2, 2), dtype=int)
        scores = precision_recall_fscore_support(
            actual, prediction, average="binary", zero_division=np.nan,
        )[:3] if n else (np.nan, np.nan, np.nan)
        result[name] = {
            "threshold": float(cutoff), "accuracy": float(accuracy_score(actual, prediction)) if n else None,
            **{key: None if np.isnan(value) else float(value)
               for key, value in zip(("precision", "recall", "f1"), scores)},
            "confusion_matrix": matrix.tolist(),
        }
    
    return result

def explanation_coverage(explanations, probability, threshold):
    alerts = np.asarray(probability) >= threshold
    complete = explanations[[f"factor_{i}" for i in range(1, 4)]].ne("").all(axis=1).to_numpy()
    
    return {
        "alerts": int(alerts.sum()), "alerts_with_three_factors": int((alerts & complete).sum()),
        "coverage": float(complete[alerts].mean()) if alerts.any() else None,
    }
