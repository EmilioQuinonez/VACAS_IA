"""
XGBoost base: un clasificador binario por evento de ordeño y uno para "alguna anomalía" (el
ordeño no fue exitoso), con los valores por defecto de la librería y sin regularización (se
agrega solo si aparece sobreajuste).

Por cada una de las 4 salidas, con un corte cronológico 70/15/15 (las sesiones más antiguas a
train, las más recientes a test): el modelo se entrena una sola vez con train, se calibra en val
(Platt, porque scale_pos_weight infla las probabilidades), elige en val el umbral de recall con
las probabilidades calibradas y se evalúa en train, val y test.
Todo va a MODELS_DIR, que se versiona: el XGBoost (.json), metrics.json (con los 2 coeficientes
de Platt de cada salida) y las curvas de aprendizaje (log loss de train y val por árbol, sin
calibrar). Las gráficas se generan aparte con plot_results.py, a partir de lo que guarda este script.

Uso, desde la raíz del repo:
    python src/MODELING/XGBoost/main.py
"""

import json
from functools import partial
from pathlib import Path

from dataset import EVENTS, build_features, load_sessions, split_by_time
from evaluation import choose_threshold, evaluate
from xgboost_model import calibrate, train_event

ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = ROOT / "Data" / "dataset_vacas_integrado.csv"
MODELS_DIR = Path(__file__).resolve().parent / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"
CURVES_PATH = MODELS_DIR / "curvas_aprendizaje.json"

TRAIN_FRACTION = 0.70  # train: el 70 % más antiguo de las sesiones
VAL_FRACTION = 0.15  # val: el 15 % siguiente; test: el 15 % más reciente
TARGET_RECALL = 0.80  # Criterio de negocio: detectar al menos 80 % de las anomalías.
N_TREES = 100  # Valor por defecto de XGBoost.
RANDOM_STATE = 42


def print_summary(event, results, threshold):
    """Métricas de un evento en cada conjunto."""
    print(f"\n{event} (threshold for recall >= {TARGET_RECALL:.0%} on val: {threshold:.3f})")
    for name in ("train", "val", "test"):
        m = results[name]
        print(f"  {name:9}: prevalence {m['prevalence']:.1%} | ROC-AUC {m['roc_auc']:.3f} | "
              f"PR-AUC {m['pr_auc']:.3f} | Brier {m['brier']:.3f} | accuracy no event {m['accuracy_no_event']:.1%}")
        for label in ("threshold_0.5", "threshold_recall"):
            t = m[label]
            print(f"             {label:16}: accuracy {t['accuracy']:.1%} | precision {t['precision']:.1%} | "
                  f"recall {t['recall']:.1%} | F1 {t['f1']:.3f}")


def main():
    sessions = load_sessions(INPUT_PATH)
    X, labels = build_features(sessions)
    idx = split_by_time(sessions["inicio"], TRAIN_FRACTION, VAL_FRACTION)
    MODELS_DIR.mkdir(exist_ok=True)

    fit = partial(train_event, n_trees=N_TREES, random_state=RANDOM_STATE)

    results, curves = {}, {}
    for event in EVENTS:
        y = labels[event]
        print(f"\n[training] {event}: train, validating on val")
        model = fit(X, y, idx["train"], idx["val"])
        log = model.evals_result()
        curves[event] = {"train": log["validation_0"]["logloss"], "val": log["validation_1"]["logloss"]}
        calibrated = calibrate(model, X, y, idx["val"])
        threshold = choose_threshold(calibrated, X, y, idx["val"], TARGET_RECALL)
        results[event] = {"threshold_recall_val": threshold,
                          "platt": {"slope": calibrated.slope, "intercept": calibrated.intercept}}
        results[event].update(evaluate(calibrated, X, y, idx, threshold))
        model.save_model(MODELS_DIR / f"{event}.json")
        print_summary(event, results[event], threshold)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    with open(CURVES_PATH, "w", encoding="utf-8") as file:
        json.dump(curves, file)
    print(f"\nModels, metrics and learning curves in {MODELS_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
