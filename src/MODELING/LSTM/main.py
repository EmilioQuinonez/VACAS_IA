"""
LSTM multitarea: una salida sigmoide por evento de ordeño y una para "alguna anomalía".

La red lee las K sesiones previas de la vaca y, después de la LSTM, el contexto de la sesión a
predecir. Corte cronológico 70/15/15: train entrena; val detiene el entrenamiento por F1, calibra
y elige el umbral; test evalúa. Da 3 factores por alerta (explain.py).

Versión final (v5); las anteriores están en su commit. Salidas en modelos/, metrics/ e img/<VERSION>.

Requiere correr antes src/MODELING/LSTM/build_history_features.py.

Uso, desde la raíz del repo (con -u el log se escribe al momento):
    python -u src/MODELING/LSTM/main.py >| src/MODELING/LSTM/modelos/v5/lstm.log 2>&1
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss
from tensorflow import keras

from dataset import EVENTS, build_inputs, load_sessions, split_by_time, split_test_by_cow
from evaluation import (
    SPLIT_LABELS, calibrate, choose_threshold, evaluate, fit_calibration, mean_f1, save_metric_tables,
)
from explain import explain
from plot_results import save_plots

ROOT = Path(__file__).resolve().parents[3]
MODULE_DIR = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "Data" / "dataset_vacas_integrado.csv"

VERSION = "v5"  # Cambia esta versión para guardar una nueva iteración.
MODEL_DIR = MODULE_DIR / "modelos" / VERSION
METRICS_DIR = MODULE_DIR / "metrics" / VERSION
FIGURES_DIR = MODULE_DIR / "img" / VERSION

TRAIN_FRACTION = 0.70  # train: el 70 % más antiguo de las sesiones
VAL_FRACTION = 0.15  # val: el 15 % siguiente; test: el 15 % más reciente
TARGET_RECALL = 0.80  # Criterio de negocio: detectar al menos 80 % de las anomalías.
TARGET_F1 = 0.70
RANDOM_STATE = 42
CONFIG = {
    "k": 30,  # Sesiones previas que lee la red (el reporte permite de 10 a 30); v3 eligió 30
    "units": 64,
    "dense_units": 32,
    "dropout": 0.2,
    "learning_rate": 0.001,  # Inicial; ReduceLROnPlateau lo baja (v4)
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
    Al final de cada época calcula val_f1 (lo vigilan EarlyStopping y ReduceLROnPlateau) y el log
    loss y accuracy de cada evento en train, val y test para las curvas de aprendizaje.
    """

    def __init__(self, X_seq, X_ctx, y, idx, batch_size):
        super().__init__()
        self.splits = {}
        for name in ("train", "val", "test"):
            rows = idx[name]
            self.splits[name] = ([X_seq[rows], X_ctx[rows]], y[rows])
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        for split, (inputs, actual) in self.splits.items():
            probability = self.model.predict(inputs, batch_size=self.batch_size, verbose=0)
            for n, event in enumerate(EVENTS):
                logs[f"{split}_log_loss_{event}"] = float(log_loss(actual[:, n], probability[:, n], labels=[0, 1]))
                logs[f"{split}_accuracy_{event}"] = float(accuracy_score(actual[:, n], probability[:, n] >= 0.5))
            if split == "val":
                logs["val_f1"] = mean_f1(actual, probability, TARGET_RECALL)
        print(f"  val_f1: {logs['val_f1']:.4f}", flush=True)


def train(config, X_seq, X_ctx, y, idx):
    """Entrena con train y se detiene con val; devuelve el modelo de la mejor época y su historial."""
    print(f"\n[training] LSTM {VERSION} {config}", flush=True)
    keras.utils.set_random_seed(RANDOM_STATE)
    model = build_model(config, X_seq.shape[2], X_ctx.shape[1])
    callbacks = [
        EpochMetrics(X_seq, X_ctx, y, idx, config["batch_size"]),
        keras.callbacks.EarlyStopping(monitor="val_f1", mode="max", patience=config["patience"],
                                      restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_f1", mode="max", factor=0.5, patience=2,
                                          min_lr=1e-5, verbose=1),
    ]
    rows = idx["train"]
    history = model.fit([X_seq[rows], X_ctx[rows]], y[rows], epochs=config["epochs"],
                        batch_size=config["batch_size"], verbose=2, callbacks=callbacks)

    return model, history.history


def save_json(data, name):
    with open(MODEL_DIR / name, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=float)


def print_summary(event, scores):
    """Métricas de cada conjunto y criterios de negocio."""
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
    # 1. Datos
    sessions = load_sessions(INPUT_PATH)
    idx = split_by_time(sessions["inicio"], TRAIN_FRACTION, VAL_FRACTION)
    X_seq, X_ctx, y, scaler = build_inputs(sessions, idx["train"], CONFIG["k"])

    # 2. Entrenamiento
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model, history = train(CONFIG, X_seq, X_ctx, y, idx)
    model.save(MODEL_DIR / "lstm.keras")
    save_json(CONFIG, "config.json")
    save_json(history, "history.json")
    save_json(scaler, "scaler.json")

    # 3. Probabilidades calibradas con val
    groups = split_test_by_cow(sessions["id_vaca"], idx)
    idx.update(groups)
    probabilities = {}
    for name, rows in idx.items():
        probabilities[name] = model.predict([X_seq[rows], X_ctx[rows]], batch_size=CONFIG["batch_size"], verbose=0)
    calibration = fit_calibration(y[idx["val"]], probabilities["val"])
    for name in probabilities:
        probabilities[name] = calibrate(probabilities[name], calibration)
    save_json(calibration, "calibracion.json")

    # 4. Umbral, métricas y 3 factores por evento
    results = {}
    for name, rows in groups.items():
        results[name] = {"n_cows": int(sessions["id_vaca"].iloc[rows].nunique()), "n_sessions": len(rows)}
    alerts, contributions = [], []
    for number, event in enumerate(EVENTS):
        threshold = choose_threshold(y[idx["val"], number], probabilities["val"][:, number], TARGET_RECALL)
        results[event] = {"threshold_recall_val": threshold}
        for name, probability in probabilities.items():
            results[event][name] = evaluate(y[idx[name], number], probability[:, number], threshold)

        print(f"[explain] {event}", flush=True)
        is_alert = probabilities["test"][:, number] >= threshold
        alert_rows = idx["test"][is_alert]
        factors, contribution = explain(model, X_seq, X_ctx, alert_rows, sessions, number,
                                        CONFIG["batch_size"], calibration[event]["a"])
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
        complete = factors[["factor_1", "factor_2", "factor_3"]].ne("").all(axis=1)
        results[event]["test"]["explanations"] = {"alerts": len(factors), "coverage": float(complete.mean())}

        test = results[event]["test"]
        results[event]["business_criteria"] = {
            f"Recall test >= {TARGET_RECALL:.0%}": test["threshold_recall"]["recall"] >= TARGET_RECALL,
            f"F1 test >= {TARGET_F1}": test["threshold_recall"]["f1"] >= TARGET_F1,
            "F1 test > alertar siempre": test["threshold_recall"]["f1"] > test["f1_always_alert"],
            "3 factores en >= 80 % de las alertas": test["explanations"]["coverage"] >= 0.80,
        }
        print_summary(event, results[event])

    # 5. Tablas y gráficas
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
