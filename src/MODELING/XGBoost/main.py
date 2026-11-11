"""
XGBoost base: un clasificador binario por evento de ordeño y uno para "alguna anomalía" (el
ordeño no fue exitoso), con los valores por defecto de la librería y sin regularización (se
agrega solo si aparece sobreajuste).

Por cada una de las 4 salidas, con un corte cronológico 70/15/15 (las sesiones más antiguas a
train, las más recientes a test): el modelo se entrena una sola vez con train, elige en val el
umbral de recall y se evalúa en train, val y test.
Los modelos van a MODELS_DIR (se versionan); metrics.json y el log, a OUTPUT_DIR (fuera de Git).
Las gráficas se generan aparte con plot_results.py, a partir de lo que guarda este script.

Uso, desde la raíz del repo (con -u el log se escribe al momento, para el panel /entrenamiento):
    python -u src/MODELING/XGBoost/main.py >| Data/modelos/xgboost_base/xgboost_baseline.log 2>&1
"""

import json
from functools import partial
from pathlib import Path

from dataset import EVENTS, build_features, load_sessions, split_by_time
from evaluation import choose_threshold, evaluate
from xgboost_model import train_event

ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = ROOT / "Data" / "dataset_vacas_integrado.csv"
OUTPUT_DIR = ROOT / "Data" / "modelos" / "xgboost_base"
LOG_PATH = OUTPUT_DIR / "xgboost_baseline.log"
MODELS_DIR = Path(__file__).resolve().parent / "models"

TRAIN_FRACTION = 0.70  # train: el 70 % más antiguo de las sesiones
VAL_FRACTION = 0.15  # val: el 15 % siguiente; test: el 15 % más reciente
TARGET_RECALL = 0.80  # Criterio de negocio: detectar al menos 80 % de las anomalías.
N_TREES = 100  # Valor por defecto de XGBoost; fijo para numerar las épocas del log.
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    fit = partial(train_event, n_trees=N_TREES, total_epochs=N_TREES * len(EVENTS), random_state=RANDOM_STATE)

    results = {}
    for number, event in enumerate(EVENTS):
        y = labels[event]
        print(f"\n[training] {event}: train, validating on val")
        model = fit(X, y, idx["train"], idx["val"], number * N_TREES)
        threshold = choose_threshold(model, X, y, idx["val"], TARGET_RECALL)
        results[event] = {"threshold_recall_val": threshold}
        results[event].update(evaluate(model, X, y, idx, threshold))
        model.save_model(MODELS_DIR / f"{event}.json")
        print_summary(event, results[event], threshold)

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"\nModels in {MODELS_DIR.relative_to(ROOT)}; metrics in {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
