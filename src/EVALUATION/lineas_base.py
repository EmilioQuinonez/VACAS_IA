"""
Líneas base sin modelo para comparar contra la regresión logística, XGBoost y la LSTM (fase de Evaluation).

Usa los mismos ordeños, etiquetas y corte cronológico 70/15/15 que los modelos, y mide en test tres reglas
para cada salida (los tres eventos y alguna anomalía):

- Alertar siempre: alerta en todos los ordeños. Su F1 es 2p / (1 + p), con p la prevalencia.
- Persistencia: alerta si el ordeño anterior de la misma vaca, en el mismo bloque de historial, tuvo el
  evento. El primer ordeño de cada bloque no tiene anterior y no alerta.
- Tasa de la vaca en 30 días: proporción de ordeños de la vaca con el evento en los 30 días previos. El umbral
  se fija en val como el más alto que detecta al menos el 80 % de los eventos, la misma regla que XGBoost, y
  se aplica sin cambios en test.

Lee Data/dataset_vacas_integrado.csv sin modificarlo y guarda los resultados en
src/EVALUATION/metrics/lineas_base.json.

Uso, desde la raíz del repo:
    python src/EVALUATION/lineas_base.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "Data" / "dataset_vacas_integrado.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "metrics" / "lineas_base.json"

# Mismo corte que los modelos: se reutiliza su función para que train, val y test sean idénticos.
sys.path.insert(0, str(ROOT / "src" / "MODELING" / "XGBoost"))
from dataset import split_by_time

EVENTS = [
    "mechanical_connection_failure_event", "health_anomaly_event",
    "behavior_interruption_event", "alguna_anomalia",
]
TRAIN_FRACTION, VAL_FRACTION = 0.70, 0.15
TARGET_RECALL = 0.80
COW_WINDOW = "30D"


def load_milkings(path):
    """Ordeños (sin rechazos ni echadas) ordenados por vaca e inicio, con alguna_anomalia."""
    columns = ["id_vaca", "inicio", "bloque_historial", "accion_rechazada", "accion_echada_sin_ordenar",
               *EVENTS[:3], "successful_milking_event"]
    sessions = pd.read_csv(path, usecols=columns, dtype={"id_vaca": str}, parse_dates=["inicio"])
    milking = sessions[["accion_rechazada", "accion_echada_sin_ordenar"]].eq(0).all(axis=1)
    sessions = sessions[milking].sort_values(["id_vaca", "inicio"], kind="stable").reset_index(drop=True)
    sessions["alguna_anomalia"] = 1 - sessions["successful_milking_event"]
    print(f"[data] {len(sessions):,} ordeños de {sessions['id_vaca'].nunique()} vacas")
    return sessions


def persistence(sessions, event):
    """1 si el ordeño anterior de la vaca, en el mismo bloque, tuvo el evento; 0 si no o si no hay anterior."""
    previous = sessions.groupby(["id_vaca", "bloque_historial"])[event].shift(1)
    return previous.fillna(0).astype(int).to_numpy()


def cow_rate(sessions, event):
    """Proporción de ordeños de la vaca con el evento en [inicio - 30 días, inicio); NaN sin historial."""
    rates = sessions.set_index("inicio").groupby("id_vaca")[event].rolling(COW_WINDOW, closed="left").mean()
    return rates.to_numpy()


def highest_threshold(actual, score, target_recall):
    """Umbral más alto que en val alcanza target_recall (misma regla que XGBoost)."""
    _, recall, thresholds = precision_recall_curve(actual, score)
    return float(thresholds[recall[:-1] >= target_recall].max())


def scores(actual, prediction):
    return {
        "f1": float(f1_score(actual, prediction)),
        "recall": float(recall_score(actual, prediction)),
        "precision": float(precision_score(actual, prediction, zero_division=0)),
        "alert_rate": float(np.mean(prediction)),
    }


def main():
    sessions = load_milkings(INPUT_PATH)
    idx = split_by_time(sessions["inicio"], TRAIN_FRACTION, VAL_FRACTION)
    train, val, test = idx["train"], idx["val"], idx["test"]

    results = {"split": {name: len(rows) for name, rows in idx.items()}}
    for event in EVENTS:
        actual = sessions[event].to_numpy()
        prevalence = float(actual[test].mean())

        rate = cow_rate(sessions, event)
        rate = np.where(np.isnan(rate), np.nanmedian(rate[train]), rate)  # sin historial: mediana de train
        threshold = highest_threshold(actual[val], rate[val], TARGET_RECALL)

        results[event] = {
            "prevalence_test": prevalence,
            "alertar_siempre": {"f1": 2 * prevalence / (1 + prevalence), "recall": 1.0,
                                "precision": prevalence, "alert_rate": 1.0},
            "persistencia": scores(actual[test], persistence(sessions, event)[test]),
            "tasa_vaca_30d": {"threshold_val": threshold,
                              **scores(actual[test], (rate[test] >= threshold).astype(int))},
        }

        print(f"\n{event} (prevalencia en test {prevalence:.1%})")
        for name in ("alertar_siempre", "persistencia", "tasa_vaca_30d"):
            r = results[event][name]
            print(f"  {name:16}: F1 {r['f1']:.3f} | recall {r['recall']:.1%} | precisión {r['precision']:.1%} | "
                  f"alerta en {r['alert_rate']:.1%} de los ordeños")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
    print(f"\nResultados en {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
