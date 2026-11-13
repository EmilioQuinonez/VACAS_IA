"""
Búsqueda al azar de hiperparámetros del XGBoost; la mejor combinación es v5 en versions.py.

Cada combinación se aplica a las 4 salidas y sigue los mismos pasos que main.py (entrenar con
train, calibrar con Platt y fijar el umbral de recall en val), pero solo se mide en train y val:
test no se calcula. Se elige con el F1 de alguna anomalía en val (redondeado a 3 decimales); si
hay empate, decide el PR-AUC de alguna anomalía en val y luego la menor brecha de ROC-AUC entre
train y val. Todas las combinaciones van a OUTPUT_PATH.

Para cambiar la búsqueda, edita N_COMBINATIONS, SEED o SPACE. La ganadora se copia a mano en
VERSIONS["v5"] de versions.py y se entrena con main.py.

Uso, desde la raíz del repo y después de build_history_features.py:
    python src/MODELING/XGBoost/random_search.py
"""

from functools import partial

import numpy as np
import pandas as pd

from dataset import EVENTS, build_features, load_sessions, split_by_time
from evaluation import choose_threshold, evaluate
from main import INPUT_PATH, RANDOM_STATE, TARGET_RECALL, TRAIN_FRACTION, VAL_FRACTION
from versions import BASE_DIR
from xgboost_model import calibrate, train_event

N_COMBINATIONS = 40
SEED = 42
SPACE = {
    "learning_rate": [0.03, 0.05, 0.1],
    "n_estimators": [100, 200, 300, 400, 600],
    "max_depth": [3, 4, 5, 6, 7, 8],
    "min_child_weight": [1, 5, 10, 20, 50],
    "gamma": [0, 0.5, 1, 2, 5],
    "reg_lambda": [1, 5, 10, 20],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
}
OUTPUT_PATH = BASE_DIR / "models" / "v5" / "busqueda.csv"


def sample_params(rng):
    """Una combinación al azar de SPACE: un valor de cada hiperparámetro."""
    return {name: values[rng.integers(len(values))] for name, values in SPACE.items()}


def score(params, X, labels, sets):
    """F1 y PR-AUC de val y brecha de ROC-AUC train − val de cada salida con params."""
    fit = partial(train_event, params=params, random_state=RANDOM_STATE)
    row = {}
    for event in EVENTS:
        y = labels[event]
        calibrated = calibrate(fit(X, y, sets["train"]), X, y, sets["val"])
        threshold = choose_threshold(calibrated, X, y, sets["val"], TARGET_RECALL)
        m = evaluate(calibrated, X, y, sets, threshold)
        row[f"f1_val_{event}"] = m["val"]["threshold_recall"]["f1"]
        row[f"pr_auc_val_{event}"] = m["val"]["pr_auc"]
        row[f"brecha_roc_{event}"] = m["train"]["roc_auc"] - m["val"]["roc_auc"]

    return row


def main():
    sessions = load_sessions(INPUT_PATH)
    X, labels = build_features(sessions)
    idx = split_by_time(sessions["inicio"], TRAIN_FRACTION, VAL_FRACTION)
    sets = {"train": idx["train"], "val": idx["val"]}
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    rows = []
    for number in range(1, N_COMBINATIONS + 1):
        params = sample_params(rng)
        rows.append({"combinacion": number, **params, **score(params, X, labels, sets)})
        print(f"[search] {number}/{N_COMBINATIONS} F1 val alguna_anomalia "
              f"{rows[-1]['f1_val_alguna_anomalia']:.4f} | {params}", flush=True)

    results = pd.DataFrame(rows).assign(f1_redondeado=lambda d: d["f1_val_alguna_anomalia"].round(3))
    results = results.sort_values(["f1_redondeado", "pr_auc_val_alguna_anomalia", "brecha_roc_alguna_anomalia"],
                                  ascending=[False, False, True]).drop(columns="f1_redondeado")
    results.to_csv(OUTPUT_PATH, index=False)

    best = {name: results[name].iloc[0].item() for name in ["combinacion", *SPACE]}
    print(f"\nBest combination: {best}")
    print(f"Saved to {OUTPUT_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
