"""Umbral de recall y métricas por conjunto."""

from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)


def choose_threshold(model, X, y, val_rows, target_recall):
    """Mayor umbral que en val alcanza target_recall."""
    probability = model.predict_proba(X.iloc[val_rows])[:, 1]
    # El recall baja al subir el umbral; recall[:-1] corresponde a cada umbral.
    _, recall, thresholds = precision_recall_curve(y.iloc[val_rows], probability)

    return float(thresholds[recall[:-1] >= target_recall].max())


def evaluate(model, X, y, sets, recall_threshold):
    """
    Por conjunto: prevalencia, accuracy de responder siempre "sin evento", ROC-AUC, PR-AUC,
    Brier (calibración) y accuracy, precisión, recall y F1 con umbral 0.5 y con recall_threshold.
    """
    metrics = {}
    for name, rows in sets.items():
        actual = y.iloc[rows].to_numpy()
        probability = model.predict_proba(X.iloc[rows])[:, 1]
        metrics[name] = {
            "prevalence": float(actual.mean()),
            "accuracy_no_event": float(1 - actual.mean()),
            "roc_auc": float(roc_auc_score(actual, probability)),
            "pr_auc": float(average_precision_score(actual, probability)),
            "brier": float(brier_score_loss(actual, probability)),
        }
        for label, threshold in [("threshold_0.5", 0.5), ("threshold_recall", recall_threshold)]:
            prediction = (probability >= threshold).astype(int)
            metrics[name][label] = {
                "accuracy": float(accuracy_score(actual, prediction)),
                "precision": float(precision_score(actual, prediction, zero_division=0)),
                "recall": float(recall_score(actual, prediction)),
                "f1": float(f1_score(actual, prediction)),
            }

    return metrics

