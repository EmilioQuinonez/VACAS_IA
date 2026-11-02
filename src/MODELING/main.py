"""
XGBoost base: un clasificador binario por evento de ordeño, con los valores por defecto de la
librería y sin regularización (se agrega solo si aparece sobreajuste).

Por evento:
- Corte cronológico (fechas del Doc): el modelo de train elige en val el umbral de recall y el
  modelo final, reentrenado con train + val, se evalúa en test.
- Corte por vaca (si RUN_BY_COW): GroupKFold sobre todas las sesiones con el umbral anterior.
Modelos finales y metrics.json van a OUTPUT_DIR (fuera de Git). Las gráficas se generan aparte
con plot_results.py, a partir de lo que guarda este script.

Uso, desde la raíz del repo (con -u el log se escribe al momento, para el panel /entrenamiento):
    python -u src/MODELING/main.py >| Data/modelos/xgboost_base/xgboost_baseline.log 2>&1
"""

import json
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import EVENTS, build_features, load_sessions, split_by_time
from evaluation import choose_threshold, cross_validate_by_cow, evaluate
from xgboost_model import train_event

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "Data" / "dataset_vacas_integrado.csv"
OUTPUT_DIR = ROOT / "Data" / "modelos" / "xgboost_base"
LOG_PATH = OUTPUT_DIR / "xgboost_baseline.log"

TRAIN_END = pd.Timestamp("2025-07-01")  # train: antes de julio de 2025
VAL_END = pd.Timestamp("2025-10-01")  # val: julio a septiembre; test: desde octubre de 2025
TARGET_RECALL = 0.80  # Criterio de negocio: detectar al menos 80 % de las anomalías.
N_TREES = 100  # Valor por defecto de XGBoost; fijo para numerar las épocas del log.
N_FOLDS = 5
RANDOM_STATE = 42
RUN_BY_COW = True  # False corre solo el corte cronológico


def print_summary(event, results, threshold):
    """Métricas de un evento en cada conjunto y, si hay, media ± desviación del corte por vaca."""
    print(f"\n{event} (threshold for recall >= {TARGET_RECALL:.0%} on val: {threshold:.3f}; "
          f"train and val with the train model, train_val and test with the final one)")
    for name in ("train", "val", "train_val", "test"):
        m = results[name]
        print(f"  {name:9}: prevalence {m['prevalence']:.1%} | ROC-AUC {m['roc_auc']:.3f} | "
              f"PR-AUC {m['pr_auc']:.3f} | Brier {m['brier']:.3f} | accuracy no event {m['accuracy_no_event']:.1%}")
        for label in ("threshold_0.5", "threshold_recall"):
            t = m[label]
            print(f"             {label:16}: accuracy {t['accuracy']:.1%} | precision {t['precision']:.1%} | "
                  f"recall {t['recall']:.1%} | F1 {t['f1']:.3f}")
    if "by_cow" in results:
        mean, std = results["by_cow"]["mean"], results["by_cow"]["std"]
        print(f"  by cow   : " + " | ".join(f"{k} {mean[k]:.3f} ± {std[k]:.3f}" for k in mean))


def main():
    sessions = load_sessions(INPUT_PATH)
    X, labels = build_features(sessions)
    idx = split_by_time(sessions["inicio"], TRAIN_END, VAL_END)
    final_rows = np.concatenate([idx["train"], idx["val"]])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Épocas del log: modelo de train, modelo final y, si aplica, un entrenamiento por fold.
    fits_per_event = 2 + (N_FOLDS if RUN_BY_COW else 0)
    fit = partial(train_event, n_trees=N_TREES, total_epochs=fits_per_event * N_TREES * len(EVENTS),
                  random_state=RANDOM_STATE)

    results = {}
    for number, event in enumerate(EVENTS):
        y = labels[event]
        first_epoch = number * fits_per_event * N_TREES
        print(f"\n[training] {event}: train, validating on val")
        model = fit(X, y, idx["train"], idx["val"], first_epoch)
        threshold = choose_threshold(model, X, y, idx["val"], TARGET_RECALL)
        results[event] = {"threshold_recall_val": threshold}
        results[event].update(evaluate(model, X, y, {"train": idx["train"], "val": idx["val"]}, threshold))

        print(f"\n[training] {event}: final model on train + val")
        final = fit(X, y, final_rows, first_epoch=first_epoch + N_TREES)
        final.save_model(OUTPUT_DIR / f"{event}.json")
        results[event].update(evaluate(final, X, y, {"train_val": final_rows, "test": idx["test"]}, threshold))

        if RUN_BY_COW:
            print(f"\n[training] {event}: {N_FOLDS} folds by cow")
            results[event]["by_cow"] = cross_validate_by_cow(
                fit, X, y, sessions["id_vaca"], threshold, N_FOLDS, first_epoch + 2 * N_TREES, N_TREES
            )
        print_summary(event, results[event], threshold)

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"\nModels and metrics in {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
