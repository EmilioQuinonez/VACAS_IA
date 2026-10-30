"""
XGBoost base: un clasificador binario por evento de ordeño, con los valores por defecto
de la librería y sin regularización (se agrega solo si aparece sobreajuste).

Las variables siguen la tabla de selección de datos del Doc (pestaña DP): solo lo conocido
al entrar al robot. Dos cortes:
- Cronológico (fechas del Doc): el modelo de train elige en val el umbral de recall y el
  modelo final, reentrenado con train + val, se evalúa en test.
- Por vaca: GroupKFold de 5 sobre todas las sesiones, con el umbral del corte cronológico.
Modelos y métricas van a Data/modelos/xgboost_base/ (fuera de Git).

El log imprime cada árbol como una época de Keras (verbose=2) para el panel /entrenamiento.

Uso:
    python -u src/MODELING/xgboost_baseline.py >| Data/modelos/xgboost_base/xgboost_baseline.log 2>&1
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
from xgboost.callback import TrainingCallback

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "Data" / "dataset_vacas_integrado.csv"
OUTPUT_DIR = ROOT / "Data" / "modelos" / "xgboost_base"

EVENTS = [
    "mechanical_connection_failure_event", "health_anomaly_event",
    "behavior_interruption_event",
]
QUARTERS = ("di", "dd", "ti", "td")
EVENT_KEYS = ("mecanico", "salud", "conducta")
# Tabla de selección de datos del Doc, columna XGBoost; las máquinas (maquina_*) se agregan aparte.
FEATURES = [
    "intervalo_horas", "intervalo_horas_sin_dato", "regreso_tras_periodo_seco", "estado_numero_de_ordeno",
    "t_sesion", "thi_sesion", "horas_thi_alto_24h", "sangre_tasa_maquina_30d",
    "fme_connection", "fme_milk_production", "fme_incomplete", "kicks_ratio_7d",
    *[f"blood_diff_{quarter}" for quarter in QUARTERS],
    "box_time_to_yield_ratio", "flies_proxy",
    # Creadas por src/DP/build_history_features.py.
    "dias_en_leche", "dias_en_hato", "mes", "hora", "dia_semana",
    *[f"{key}_vaca_30d" for key in EVENT_KEYS], *[f"{key}_hato_7d" for key in EVENT_KEYS],
    "vacas_activas_7d", *[f"{key}_previo" for key in EVENT_KEYS],
    "mdi_previo", "conductividad_max_previa", "produccion_previa", "cuarto_inactivo_previo",
]
TRAIN_END = pd.Timestamp("2025-07-01")  # train: antes de julio de 2025
VAL_END = pd.Timestamp("2025-10-01")  # val: julio a septiembre; test: desde octubre de 2025
N_FOLDS = 5
TARGET_RECALL = 0.80  # Criterio de negocio: detectar al menos 80 % de las anomalías.
N_TREES = 100  # Valor por defecto de XGBoost; fijo para numerar las épocas del log.
TOTAL_EPOCHS = (2 + N_FOLDS) * N_TREES * len(EVENTS)
RANDOM_STATE = 42


class KerasStyleLog(TrainingCallback):
    """Imprime cada árbol como época de Keras; sin val, solo accuracy y loss."""

    def __init__(self, first_epoch, total_epochs):
        super().__init__()
        self.first_epoch = first_epoch
        self.total_epochs = total_epochs
        self.best = float("inf")
        self.elapsed = 0.0
        self.printed = 0

    def before_iteration(self, model, epoch, evals_log):
        self.start = time.perf_counter()
        return False

    def after_iteration(self, model, epoch, evals_log):
        duration = time.perf_counter() - self.start
        # Keras imprime segundos enteros: se arrastra el resto para que la suma sea el tiempo real.
        self.elapsed += duration
        seconds = int(self.elapsed) - self.printed
        self.printed += seconds

        train, val = evals_log["validation_0"], evals_log.get("validation_1")
        number = self.first_epoch + epoch + 1
        summary = (f"1/1 - {seconds}s - {int(duration * 1000)}ms/step - "
                   f"accuracy: {1 - train['error'][-1]:.4f} - loss: {train['logloss'][-1]:.4f}")
        if val:
            summary += f" - val_accuracy: {1 - val['error'][-1]:.4f} - val_loss: {val['logloss'][-1]:.4f}"
        print(f"Epoch {number}/{self.total_epochs}")
        print(summary)
        if val and val["logloss"][-1] < self.best:
            print(f"Epoch {number}: val_loss improved from {self.best:.5f} to {val['logloss'][-1]:.5f}")
            self.best = val["logloss"][-1]

        return False


def load_sessions(path):
    """
    Lee el dataset integrado y deja solo los ordeños: integrate_derivates.py pone las
    etiquetas en 0 a rechazos y echadas, que entrarían como ordeños sin evento.
    """
    sessions = pd.read_csv(path, dtype={"id_vaca": str}, parse_dates=["inicio"], low_memory=False)
    not_milking = sessions[["accion_rechazada", "accion_echada_sin_ordenar"]].eq(1).any(axis=1)
    print(f"[data] {len(sessions):,} visits; {not_milking.sum():,} rejections and lie-downs removed")

    return sessions[~not_milking].reset_index(drop=True)


def build_features(sessions):
    """X (FEATURES y dummies maquina_*; los NaN se conservan) e y (una columna 0/1 por evento)."""
    machines = [column for column in sessions.columns if column.startswith("maquina_")]
    X = sessions[FEATURES + machines]
    print(f"[features] {X.shape[1]} columns ({len(machines)} machines)")

    return X, sessions[EVENTS]


def split_by_time(start):
    """Posiciones de train, val y test según TRAIN_END y VAL_END."""
    idx = {
        "train": np.flatnonzero(start < TRAIN_END),
        "val": np.flatnonzero((start >= TRAIN_END) & (start < VAL_END)),
        "test": np.flatnonzero(start >= VAL_END),
    }
    for name, rows in idx.items():
        dates = start.iloc[rows]
        print(f"[split] {name}: {len(rows):,} sessions, {dates.min():%d/%m/%Y} to {dates.max():%d/%m/%Y}")

    return idx


def train_event(X, y, rows, val_rows=None, first_epoch=0):
    """
    Entrena un XGBoost con valores por defecto; scale_pos_weight compensa el desbalance.
    val_rows solo se evalúa para el log: no detiene ni cambia el entrenamiento.
    """
    X_fit, y_fit = X.iloc[rows], y.iloc[rows]
    eval_set = [(X_fit, y_fit)]
    if val_rows is not None:
        eval_set.append((X.iloc[val_rows], y.iloc[val_rows]))
    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=N_TREES,
        scale_pos_weight=(y_fit == 0).sum() / (y_fit == 1).sum(),
        eval_metric=["error", "logloss"],
        callbacks=[KerasStyleLog(first_epoch, TOTAL_EPOCHS)],
        random_state=RANDOM_STATE,
    )

    return model.fit(X_fit, y_fit, eval_set=eval_set, verbose=False)


def choose_threshold(model, X, y, val_rows):
    """Mayor umbral que en val alcanza TARGET_RECALL."""
    probability = model.predict_proba(X.iloc[val_rows])[:, 1]
    # El recall baja al subir el umbral; recall[:-1] corresponde a cada umbral.
    _, recall, thresholds = precision_recall_curve(y.iloc[val_rows], probability)

    return float(thresholds[recall[:-1] >= TARGET_RECALL].max())


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


def cross_validate_by_cow(X, y, cows, recall_threshold, first_epoch):
    """
    GroupKFold por vaca: cada fold entrena sin un 20 % de las vacas y se evalúa en ellas.
    Devuelve media, desviación y valores por fold de las métricas sin umbral y con recall_threshold.
    """
    folds = []
    for number, (train_rows, test_rows) in enumerate(GroupKFold(n_splits=N_FOLDS).split(X, y, cows)):
        model = train_event(X, y, train_rows, test_rows, first_epoch + number * N_TREES)
        m = evaluate(model, X, y, {"fold": test_rows}, recall_threshold)["fold"]
        folds.append({
            "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"], "brier": m["brier"],
            **m["threshold_recall"],
        })
    folds = pd.DataFrame(folds)

    return {"mean": folds.mean().to_dict(), "std": folds.std().to_dict(), "folds": folds.to_dict("records")}


def main():
    """Por evento: corte cronológico (train, val y modelo final en test) y corte por vaca."""
    sessions = load_sessions(INPUT_PATH)
    X, labels = build_features(sessions)
    idx = split_by_time(sessions["inicio"])
    final_rows = np.concatenate([idx["train"], idx["val"]])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    for number, event in enumerate(EVENTS):
        y = labels[event]
        first_epoch = number * (2 + N_FOLDS) * N_TREES
        print(f"\n[training] {event}: train, validating on val")
        model = train_event(X, y, idx["train"], idx["val"], first_epoch)
        threshold = choose_threshold(model, X, y, idx["val"])
        results[event] = {"threshold_recall_val": threshold}
        results[event].update(evaluate(model, X, y, {"train": idx["train"], "val": idx["val"]}, threshold))

        print(f"\n[training] {event}: final model on train + val")
        final = train_event(X, y, final_rows, first_epoch=first_epoch + N_TREES)
        final.save_model(OUTPUT_DIR / f"{event}.json")
        results[event].update(evaluate(final, X, y, {"train_val": final_rows, "test": idx["test"]}, threshold))

        print(f"\n[training] {event}: {N_FOLDS} folds by cow")
        results[event]["by_cow"] = cross_validate_by_cow(
            X, y, sessions["id_vaca"], threshold, first_epoch + 2 * N_TREES
        )

        print(f"\n{event} (threshold for recall >= {TARGET_RECALL:.0%} on val: {threshold:.3f}; "
              f"train and val with the train model, train_val and test with the final one)")
        for name in ("train", "val", "train_val", "test"):
            m = results[event][name]
            print(f"  {name:9}: prevalence {m['prevalence']:.1%} | ROC-AUC {m['roc_auc']:.3f} | "
                  f"PR-AUC {m['pr_auc']:.3f} | Brier {m['brier']:.3f} | accuracy no event {m['accuracy_no_event']:.1%}")
            for label in ("threshold_0.5", "threshold_recall"):
                t = m[label]
                print(f"             {label:16}: accuracy {t['accuracy']:.1%} | precision {t['precision']:.1%} | "
                      f"recall {t['recall']:.1%} | F1 {t['f1']:.3f}")
        mean, std = results[event]["by_cow"]["mean"], results[event]["by_cow"]["std"]
        print(f"  by cow   : " + " | ".join(f"{k} {mean[k]:.3f} ± {std[k]:.3f}" for k in mean))

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(f"\nModels and metrics in {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
