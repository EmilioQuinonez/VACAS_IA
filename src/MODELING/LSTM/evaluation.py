"""Umbral, métricas, calibración y tablas CSV."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
    """Umbral con mayor F1 entre los que alcanzan target_recall (misma regla que la regresión logística)."""
    precision, recall, thresholds = precision_recall_curve(actual, probability)
    # El último punto de la curva no tiene umbral.
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    eligible = np.flatnonzero(recall >= target_recall)
    best = eligible[f1[eligible] == f1[eligible].max()][-1]  # Si hay empate, el umbral más alto

    return float(thresholds[best])


def mean_f1(actual, probability, target_recall):
    """F1 promedio de los eventos, cada uno con su umbral."""
    scores = []
    for n in range(actual.shape[1]):
        threshold = choose_threshold(actual[:, n], probability[:, n], target_recall)
        scores.append(f1_score(actual[:, n], probability[:, n] >= threshold))

    return float(np.mean(scores))


def evaluate(actual, probability, threshold):
    """Métricas de un conjunto; las de threshold_recall usan el umbral elegido en val."""
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


def logit(probability):
    probability = np.clip(probability, 1e-7, 1 - 1e-7)
    return np.log(probability / (1 - probability))


def fit_calibration(actual, probability):
    """Calibración de Platt: a y b por evento, ajustados con val."""
    calibration = {}
    for n, event in enumerate(EVENTS):
        # C grande: sin regularización.
        model = LogisticRegression(C=1e6).fit(logit(probability[:, [n]]), actual[:, n])
        calibration[event] = {"a": float(model.coef_[0, 0]), "b": float(model.intercept_[0])}

    return calibration


def calibrate(probability, calibration):
    """p = sigmoide(a · log-odds + b) por evento."""
    calibrated = np.zeros(probability.shape)
    for n, event in enumerate(EVENTS):
        a, b = calibration[event]["a"], calibration[event]["b"]
        log_odds = logit(probability[:, n]).astype(float)  # A float64 antes de multiplicar por a
        calibrated[:, n] = 1 / (1 + np.exp(-(a * log_odds + b)))

    return calibrated


def save_metric_tables(metrics, directory):
    """Un CSV por evento: filas = métricas, columnas = conjuntos."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rows = [
        ("F1", "f1"), ("Sensibilidad", "recall"), ("Precisión", "precision"),
        ("F1 (alertar siempre)", "f1_always_alert"), ("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"),
        ("Brier", "brier"), ("Log loss", "log_loss"), ("Exactitud", "accuracy"),
        ("Exactitud (mayoritaria)", "accuracy_majority"),
    ]
    for event in EVENTS:
        table = {"Métrica": ["Observaciones"] + [label for label, _ in rows] + ["Umbral (val)"]}
        for split, label in SPLIT_LABELS.items():
            scores = {**metrics[event][split], **metrics[event][split]["threshold_recall"]}
            table[label] = ([str(scores["n"])] + [f"{scores[key]:.4f}" for _, key in rows]
                            + [f"{metrics[event]['threshold_recall_val']:.4f}"])
        pd.DataFrame(table).to_csv(directory / f"{event}.csv", index=False, encoding="utf-8-sig")
