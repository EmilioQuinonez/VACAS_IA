"""Umbral de recall, métricas por conjunto y factores SHAP de cada alerta."""

import numpy as np
import pandas as pd
import xgboost as xgb
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


def shap_contributions(model, X, slope):
    """
    Aporte SHAP de cada atributo a cada sesión, en log-odds calibrados: TreeSHAP de XGBoost
    multiplicado por la pendiente de Platt, así que el valor base calibrado más la suma de los
    aportes da el logit de la probabilidad calibrada. Las dummies maquina_* se suman en "maquina".
    """
    raw = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
    # La última columna es el valor base, igual en todas las sesiones.
    contributions = pd.DataFrame(slope * raw[:, :-1], index=X.index, columns=X.columns)
    group = lambda column: "maquina" if column.startswith("maquina_") else column

    return contributions.T.groupby(group, sort=False).sum().T


def top_factors(contributions, X, k=3):
    """
    Los k aportes positivos más grandes de cada sesión: atributo, su valor en la sesión y su
    aporte. El valor de "maquina" es la máquina (referencia si no tiene dummy); si hay menos de
    k aportes positivos, los que faltan quedan vacíos.
    """
    machines = [column for column in X.columns if column.startswith("maquina_")]
    has_dummy = X[machines].eq(1).any(axis=1)
    machine = X[machines].idxmax(axis=1).str.removeprefix("maquina_").where(has_dummy, "referencia")
    values = X.drop(columns=machines).assign(maquina=machine)[contributions.columns].to_numpy()

    names = contributions.columns.to_numpy()
    order = np.argsort(-contributions.to_numpy(), axis=1)[:, :k]
    rows = np.arange(len(contributions))[:, None]
    top = contributions.to_numpy()[rows, order]
    factors = {}
    for i in range(k):
        positive = top[:, i] > 0
        factors[f"factor_{i + 1}"] = np.where(positive, names[order[:, i]], None)
        factors[f"valor_{i + 1}"] = np.where(positive, values[rows[:, 0], order[:, i]], None)
        factors[f"aporte_{i + 1}"] = np.where(positive, top[:, i], np.nan)

    return pd.DataFrame(factors, index=contributions.index)


def check_additivity(model, X, contributions, platt):
    """
    Error máximo entre el logit calibrado y el valor base calibrado más la suma de los aportes
    ya agrupados (comprueba la escala de Platt y que el agrupamiento de maquina_* no pierda aportes).
    """
    base = model.get_booster().predict(xgb.DMatrix(X.iloc[:1]), pred_contribs=True)[0, -1]
    logit = platt["slope"] * model.predict(X, output_margin=True) + platt["intercept"]
    calibrated_base = platt["slope"] * base + platt["intercept"]

    return float(np.abs(contributions.sum(axis=1).to_numpy() + calibrated_base - logit).max())
