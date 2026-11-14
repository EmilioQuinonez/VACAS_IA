"""
LSTM: una red multitarea con una salida sigmoide por evento de ordeño y una más para
"alguna anomalía" (lo contrario de successful_milking_event).

La red lee la secuencia de las K sesiones previas de la vaca (cada paso con los valores de esa
sesión) y, después de la capa recurrente, el contexto de la sesión a predecir (clima, máquina,
calendario y carga del hato).
- Corte cronológico 70/15/15 sin espacio entre periodos: train entrena, val detiene el
  entrenamiento (early stopping cuando el F1 promedio de val no mejora en patience épocas) y elige
  el umbral (mayor F1 entre los que alcanzan recall de 80 %), y test evalúa.
- Test completo y separado en vacas conocidas (en train o val) y nuevas (primer ordeño en test).
- Tres factores por alerta de test con Integrated Gradients (explain.py).

Cada iteración se guarda por versión (VERSION), como en la regresión logística:
- modelos/<VERSION>/: lstm.keras, config.json, scaler.json, metrics.json, history.json y lstm.log.
- metrics/<VERSION>/: un CSV de métricas por evento, alertas_test.csv.gz (los 3 factores de cada
  alerta) y aportes_promedio.csv.
- img/<VERSION>/: las mismas gráficas que la regresión logística (plot_results.py).

Requiere correr antes src/MODELING/LSTM/build_history_features.py.

Uso, desde la raíz del repo (con -u el log se escribe al momento):
    python -u src/MODELING/LSTM/main.py >| src/MODELING/LSTM/modelos/v1/lstm.log 2>&1
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from tensorflow import keras

from dataset import EVENTS, build_inputs, load_sessions, split_by_time, split_test_by_cow
from evaluation import SPLIT_LABELS, choose_threshold, evaluate, f1_at_threshold, save_metric_tables
from explain import explain
from plot_results import save_plots

ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "Data" / "dataset_vacas_integrado.csv"

VERSION = "v1"  # Cambia esta versión para guardar una nueva iteración.
MODEL_DIR = MODULE_DIR / "modelos" / VERSION
METRICS_DIR = MODULE_DIR / "metrics" / VERSION
FIGURES_DIR = MODULE_DIR / "img" / VERSION

TRAIN_FRACTION = 0.70  # train: el 70 % más antiguo de las sesiones
VAL_FRACTION = 0.15  # val: el 15 % siguiente; test: el 15 % más reciente
TARGET_RECALL = 0.80  # Criterio de negocio: detectar al menos 80 % de las anomalías.
TARGET_F1 = 0.70
RANDOM_STATE = 42
CONFIG = {
    "k": 20,  # Sesiones previas que lee la red (el reporte permite de 10 a 30)
    "units": 64,
    "dense_units": 32,
    "dropout": 0.2,
    "learning_rate": 0.001,
    "epochs": 50,
    "batch_size": 512,
    "patience": 5,
}


def build_model(config, n_sequence, n_context):
    """Masking y LSTM sobre la secuencia; su salida se une al contexto antes de las capas densas."""
    sequence = keras.Input(shape=(config["k"], n_sequence), name="secuencia")
    context = keras.Input(shape=(n_context,), name="contexto")
    x = keras.layers.Masking(mask_value=0.0)(sequence)
    x = keras.layers.LSTM(config["units"], dropout=config["dropout"])(x)
    x = keras.layers.Concatenate()([x, context])
    x = keras.layers.Dense(config["dense_units"], activation="relu")(x)
    x = keras.layers.Dropout(config["dropout"])(x)
    output = keras.layers.Dense(len(EVENTS), activation="sigmoid", name="eventos")(x)

    model = keras.Model([sequence, context], output)
    model.compile(optimizer=keras.optimizers.Adam(config["learning_rate"]), loss="binary_crossentropy")

    return model


class EpochMetrics(keras.callbacks.Callback):
    """
    Al final de cada época agrega a logs val_f1, el F1 promedio de los 4 eventos en val con el umbral
    con que se evalúa (lo vigila el early stopping), y log loss y accuracy (umbral 0.5) de cada evento
    en train, val y test para las curvas de aprendizaje, como en la regresión logística (test solo
    como diagnóstico; no decide nada). Va antes de EarlyStopping para que este y el historial ya vean
    estos valores. Predecir los tres conjuntos agrega unos segundos por época.
    """

    def __init__(self, X_seq, X_ctx, y, idx, batch_size):
        super().__init__()
        self.splits = {name: ([X_seq[idx[name]], X_ctx[idx[name]]], y[idx[name]]) for name in ("train", "val", "test")}
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        for split, (inputs, actual) in self.splits.items():
            probability = self.model.predict(inputs, batch_size=self.batch_size, verbose=0)
            for n, event in enumerate(EVENTS):
                logs[f"{split}_log_loss_{event}"] = float(log_loss(actual[:, n], probability[:, n], labels=[0, 1]))
                logs[f"{split}_accuracy_{event}"] = float(accuracy_score(actual[:, n], probability[:, n] >= 0.5))
            if split == "val":
                f1 = [f1_at_threshold(actual[:, n], probability[:, n],
                                      choose_threshold(actual[:, n], probability[:, n], TARGET_RECALL))
                      for n in range(len(EVENTS))]
                logs["val_f1"] = float(np.mean(f1))
        print(f"  val_f1: {logs['val_f1']:.4f}", flush=True)


def train(config, X_seq, X_ctx, y, idx):
    """Entrena con train y detiene con val; devuelve el modelo con los pesos de la mejor época y su historial."""
    print(f"\n[training] LSTM {VERSION} {config}", flush=True)
    keras.utils.set_random_seed(RANDOM_STATE)
    model = build_model(config, X_seq.shape[2], X_ctx.shape[1])
    callbacks = [EpochMetrics(X_seq, X_ctx, y, idx, config["batch_size"]),
                 keras.callbacks.EarlyStopping(monitor="val_f1", mode="max", patience=config["patience"],
                                               restore_best_weights=True, verbose=1)]
    train_rows = idx["train"]
    history = model.fit([X_seq[train_rows], X_ctx[train_rows]], y[train_rows], epochs=config["epochs"],
                        batch_size=config["batch_size"], verbose=2, callbacks=callbacks)

    return model, history.history


def save_json(data, name):
    with open(MODEL_DIR / name, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=float)


def print_summary(event, scores):
    """F1, recall y precisión con el umbral de val por conjunto, y los criterios de negocio en test."""
    threshold = scores["threshold_recall_val"]
    print(f"\n{event} (best F1 threshold with recall >= {TARGET_RECALL:.0%} on val: {threshold:.3f})")
    for name in SPLIT_LABELS:
        m, t = scores[name], scores[name]["threshold_recall"]
        print(f"  {name:20}: F1 {t['f1']:.3f} (always alert {m['f1_always_alert']:.3f}) | "
              f"recall {t['recall']:.1%} | precision {t['precision']:.1%} | PR-AUC {m['pr_auc']:.3f} | "
              f"Brier {m['brier']:.3f} | log loss {m['log_loss']:.3f}")
    for criterion, passed in scores["business_criteria"].items():
        print(f"    {criterion}: {'cumple' if passed else 'no cumple'}")


def main():
    sessions = load_sessions(INPUT_PATH)
    idx = split_by_time(sessions["inicio"], TRAIN_FRACTION, VAL_FRACTION)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X_seq, X_ctx, y, scaler = build_inputs(sessions, idx["train"], CONFIG["k"])
    model, history = train(CONFIG, X_seq, X_ctx, y, idx)
    model.save(MODEL_DIR / "lstm.keras")
    save_json(CONFIG, "config.json")
    save_json(history, "history.json")
    save_json(scaler, "scaler.json")

    groups = split_test_by_cow(sessions["id_vaca"], idx)
    idx.update(groups)
    probabilities = {name: model.predict([X_seq[rows], X_ctx[rows]], batch_size=CONFIG["batch_size"], verbose=0)
                     for name, rows in idx.items()}

    results = {name: {"n_cows": int(sessions["id_vaca"].iloc[rows].nunique()), "n_sessions": len(rows)}
               for name, rows in groups.items()}
    alerts, contributions = [], []
    for number, event in enumerate(EVENTS):
        threshold = choose_threshold(y[idx["val"], number], probabilities["val"][:, number], TARGET_RECALL)
        results[event] = {"threshold_recall_val": threshold}
        for name, probability in probabilities.items():
            results[event][name] = evaluate(y[idx[name], number], probability[:, number], threshold)

        print(f"[explain] {event}", flush=True)
        is_alert = probabilities["test"][:, number] >= threshold
        alert_rows = idx["test"][is_alert]
        factors, contribution = explain(model, X_seq, X_ctx, alert_rows, sessions, number, CONFIG["batch_size"])
        alert = pd.DataFrame({
            "id_vaca": sessions["id_vaca"].iloc[alert_rows].to_numpy(),
            "inicio": sessions["inicio"].iloc[alert_rows].to_numpy(),
            "evento": event,
            "real": y[alert_rows, number].astype(int),
            "probabilidad": probabilities["test"][is_alert, number],
        })
        alerts.append(pd.concat([alert, factors], axis=1))
        contributions.append(pd.DataFrame({"event": event, "variable": contribution.columns,
                                           "aporte_medio": contribution.mean().to_numpy()}))
        complete = factors[[f"factor_{i}" for i in range(1, 4)]].ne("").all(axis=1)
        results[event]["test"]["explanations"] = {"alerts": len(factors), "coverage": float(complete.mean())}

        test = results[event]["test"]
        results[event]["business_criteria"] = {
            f"Recall test >= {TARGET_RECALL:.0%}": test["threshold_recall"]["recall"] >= TARGET_RECALL,
            f"F1 test >= {TARGET_F1}": test["threshold_recall"]["f1"] >= TARGET_F1,
            "F1 test > alertar siempre": test["threshold_recall"]["f1"] > test["f1_always_alert"],
            "3 factores en >= 80 % de las alertas": test["explanations"]["coverage"] >= 0.80,
        }
        print_summary(event, results[event])

    save_json(results, "metrics.json")
    save_metric_tables(results, METRICS_DIR)
    pd.concat(alerts, ignore_index=True).to_csv(METRICS_DIR / "alertas_test.csv.gz", index=False)
    contributions = pd.concat(contributions, ignore_index=True)
    contributions.round(6).to_csv(METRICS_DIR / "aportes_promedio.csv", index=False, encoding="utf-8-sig")
    save_plots(results, history, y[idx["test"]], probabilities["test"], contributions, FIGURES_DIR)
    print(f"\nModel in {MODEL_DIR.relative_to(ROOT)}; tables in {METRICS_DIR.relative_to(ROOT)}; "
          f"figures in {FIGURES_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
