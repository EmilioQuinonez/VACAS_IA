"""Umbral de decisión, métricas por conjunto y tablas CSV de la LSTM (mismo criterio que la regresión logística)."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, f1_score, log_loss,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)

from dataset import EVENTS

SPLIT_LABELS = {
    "train": "Entrenamiento", "val": "Validación", "test": "Prueba completa",
    "test_vacas_conocidas": "Prueba vacas conocidas", "test_vacas_nuevas": "Prueba vacas nuevas",
}


def choose_threshold(actual, probability, target_recall):
    """Umbral con mayor F1 entre los que alcanzan target_recall; recall[:-1] corresponde a cada umbral."""
    precision, recall, thresholds = precision_recall_curve(actual, probability)
    precision, recall = precision[:-1], recall[:-1]
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator != 0)
    eligible = np.flatnonzero(recall >= target_recall)
    best = eligible[f1[eligible] == f1[eligible].max()][-1]

    return float(thresholds[best])


def f1_at_threshold(actual, probability, threshold):
    return float(f1_score(actual, (probability >= threshold).astype(int)))


def evaluate(actual, probability, threshold):
    """
    Prevalencia, referencias (F1 de alertar siempre y exactitud de la clase mayoritaria), ROC-AUC,
    PR-AUC, Brier y log loss, y F1, recall, precisión y exactitud con el umbral elegido en val.
    """
    prevalence = float(actual.mean())
    prediction = (probability >= threshold).astype(int)
    return {
        "n": int(len(actual)),
        "prevalence": prevalence,
        "f1_always_alert": 2 * prevalence / (1 + prevalence),
        "accuracy_majority": max(prevalence, 1 - prevalence),
        "roc_auc": float(roc_auc_score(actual, probability)),
        "pr_auc": float(average_precision_score(actual, probability)),
        "brier": float(brier_score_loss(actual, probability)),
        "log_loss": float(log_loss(actual, probability, labels=[0, 1])),
        "threshold_recall": {
            "f1": float(f1_score(actual, prediction)),
            "recall": float(recall_score(actual, prediction)),
            "precision": float(precision_score(actual, prediction, zero_division=0)),
            "accuracy": float(accuracy_score(actual, prediction)),
        },
    }


def save_metric_tables(metrics, directory):
    """Un CSV por evento: filas = métricas (F1 primero), columnas = conjuntos, con cuatro decimales."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = (
        ("Observaciones", "n", False), ("F1", "f1", True), ("Sensibilidad", "recall", True),
        ("Precisión", "precision", True), ("F1 (alertar siempre)", "f1_always_alert", False),
        ("ROC-AUC", "roc_auc", False), ("PR-AUC", "pr_auc", False), ("Brier", "brier", False),
        ("Log loss", "log_loss", False), ("Exactitud", "accuracy", True),
        ("Exactitud (mayoritaria)", "accuracy_majority", False),
    )
    for event in EVENTS:
        table = {"Métrica": [label for label, _, _ in rows] + ["Umbral (val)"]}
        for split, label in SPLIT_LABELS.items():
            scores = metrics[event][split]
            values = [(scores["threshold_recall"] if decision else scores)[key] for _, key, decision in rows]
            table[label] = ([str(values[0])] + [f"{value:.4f}" for value in values[1:]]
                            + [f"{metrics[event]['threshold_recall_val']:.4f}"])
        pd.DataFrame(table).to_csv(directory / f"{event}.csv", index=False, encoding="utf-8-sig")
