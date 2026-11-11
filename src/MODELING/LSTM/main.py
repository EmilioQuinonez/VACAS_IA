"""
LSTM base: una red multitarea con una salida sigmoide por evento de ordeño y una más para
"alguna anomalía" (lo contrario de successful_milking_event).

La red lee la secuencia de las K sesiones previas de la vaca (cada paso con los valores de esa
sesión) y, después de la capa LSTM, el contexto de la sesión a predecir (clima, máquina,
calendario y carga del hato).
- Corte cronológico 70/15/15 sin espacio entre periodos: train entrena, val detiene el
  entrenamiento (early stopping cuando el PR-AUC promedio de val no mejora en PATIENCE épocas)
  y elige el umbral de recall, y test evalúa.
- Vacas en test: las mismas métricas para vacas conocidas (en train), solo en val y nuevas
  (primer ordeño en test), con los mismos grupos que XGBoost y la regresión logística.
Modelo, metrics.json e history.json van a OUTPUT_DIR (src/MODELING/LSTM/modelos/lstm_base).

Requiere correr antes src/MODELING/LSTM/build_history_features.py.

Uso, desde la raíz del repo (con -u el log se escribe al momento):
    python -u src/MODELING/LSTM/main.py >| src/MODELING/LSTM/modelos/lstm_base/lstm_base.log 2>&1
"""

import json
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from tensorflow import keras

from dataset import EVENTS, build_inputs, load_sessions, split_by_time, split_test_by_cow

ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = ROOT / "Data" / "dataset_vacas_integrado.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "modelos" / "lstm_base"

TRAIN_FRACTION = 0.70  # train: el 70 % más antiguo de las sesiones
VAL_FRACTION = 0.15  # val: el 15 % siguiente; test: el 15 % más reciente
K = 20  # Sesiones previas que lee la red
UNITS = 64
DROPOUT = 0.2
EPOCHS = 50
BATCH_SIZE = 512
PATIENCE = 5
TARGET_RECALL = 0.80  # Criterio de negocio: detectar al menos 80 % de las anomalías.
RANDOM_STATE = 42


def build_model(k, n_sequence, n_context):
    """Masking y LSTM sobre la secuencia; su salida se une al contexto antes de las capas densas."""
    sequence = keras.Input(shape=(k, n_sequence), name="secuencia")
    context = keras.Input(shape=(n_context,), name="contexto")
    x = keras.layers.Masking(mask_value=0.0)(sequence)
    x = keras.layers.LSTM(UNITS, dropout=DROPOUT)(x)
    x = keras.layers.Concatenate()([x, context])
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dropout(DROPOUT)(x)
    output = keras.layers.Dense(len(EVENTS), activation="sigmoid", name="eventos")(x)

    model = keras.Model([sequence, context], output)
    # Con varias salidas sigmoide, "accuracy" se toma como categórica; se pide la binaria.
    # pr_auc: PR-AUC de cada salida promediado (multi_label), el que vigila el early stopping.
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=[keras.metrics.BinaryAccuracy(name="accuracy"),
                           keras.metrics.AUC(curve="PR", multi_label=True, num_labels=len(EVENTS), name="pr_auc")])

    return model


def evaluate(actual, probability, threshold):
    """
    Prevalencia, accuracy de responder siempre "sin evento", ROC-AUC, PR-AUC, Brier (calibración)
    y accuracy, precisión, recall y F1 con umbral 0.5 y con el umbral de recall.
    """
    metrics = {
        "prevalence": float(actual.mean()),
        "accuracy_no_event": float(1 - actual.mean()),
        "roc_auc": float(roc_auc_score(actual, probability)),
        "pr_auc": float(average_precision_score(actual, probability)),
        "brier": float(brier_score_loss(actual, probability)),
    }
    for label, value in [("threshold_0.5", 0.5), ("threshold_recall", threshold)]:
        prediction = (probability >= value).astype(int)
        metrics[label] = {
            "accuracy": float(accuracy_score(actual, prediction)),
            "precision": float(precision_score(actual, prediction, zero_division=0)),
            "recall": float(recall_score(actual, prediction)),
            "f1": float(f1_score(actual, prediction)),
        }

    return metrics


def main():
    keras.utils.set_random_seed(RANDOM_STATE)
    sessions = load_sessions(INPUT_PATH)
    idx = split_by_time(sessions["inicio"], TRAIN_FRACTION, VAL_FRACTION)
    X_seq, X_ctx, y = build_inputs(sessions, idx["train"], K)
    idx.update(split_test_by_cow(sessions["id_vaca"], idx))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def inputs(rows):
        return [X_seq[rows], X_ctx[rows]]

    print(f"\n[training] LSTM: train, early stopping when mean PR-AUC on val does not improve in {PATIENCE} epochs")
    model = build_model(K, X_seq.shape[2], X_ctx.shape[1])
    history = model.fit(
        inputs(idx["train"]), y[idx["train"]],
        validation_data=(inputs(idx["val"]), y[idx["val"]]),
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=PATIENCE,
                                                 restore_best_weights=True, verbose=1)],
    )
    model.save(OUTPUT_DIR / "lstm.keras")
    probabilities = {name: model.predict(inputs(rows), batch_size=BATCH_SIZE, verbose=0)
                     for name, rows in idx.items() if len(rows)}

    results = {}
    for number, event in enumerate(EVENTS):
        # Mayor umbral que en val alcanza TARGET_RECALL; recall[:-1] corresponde a cada umbral.
        _, recall, thresholds = precision_recall_curve(y[idx["val"], number], probabilities["val"][:, number])
        threshold = float(thresholds[recall[:-1] >= TARGET_RECALL].max())
        results[event] = {"threshold_recall_val": threshold}
        for name, probability in probabilities.items():
            results[event][name] = evaluate(y[idx[name], number], probability[:, number], threshold)

        print(f"\n{event} (threshold for recall >= {TARGET_RECALL:.0%} on val: {threshold:.3f})")
        for name in probabilities:
            m = results[event][name]
            print(f"  {name:20}: prevalence {m['prevalence']:.1%} | ROC-AUC {m['roc_auc']:.3f} | "
                  f"PR-AUC {m['pr_auc']:.3f} | Brier {m['brier']:.3f} | accuracy no event {m['accuracy_no_event']:.1%}")
            for label in ("threshold_0.5", "threshold_recall"):
                t = m[label]
                print(f"                        {label:16}: accuracy {t['accuracy']:.1%} | precision "
                      f"{t['precision']:.1%} | recall {t['recall']:.1%} | F1 {t['f1']:.3f}")

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    with open(OUTPUT_DIR / "history.json", "w", encoding="utf-8") as file:
        json.dump({key: [float(v) for v in values] for key, values in history.history.items()}, file, indent=2)
    print(f"\nModel and metrics in {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
