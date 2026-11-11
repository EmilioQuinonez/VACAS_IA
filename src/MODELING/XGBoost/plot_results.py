"""
Genera las gráficas del XGBoost base a partir de lo que guardó main.py (modelos,
metrics.json y las curvas de aprendizaje), sin reentrenar. Solo cifras agregadas, sin datos por vaca.

Las figuras van a FIGURES_DIR (src/MODELING/XGBoost/img), que sí se versiona.

Uso, desde la raíz del repo y después de main.py:
    python src/MODELING/XGBoost/plot_results.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import plots
from dataset import EVENT_KEYS, EVENTS, build_features, load_sessions, split_by_time
from main import CURVES_PATH, INPUT_PATH, MODELS_DIR, N_TREES, OUTPUT_DIR, TRAIN_FRACTION, VAL_FRACTION

FIGURES_DIR = Path(__file__).resolve().parent / "img"


def read_curves(path):
    """Log loss de train y val por árbol de cada evento (en el orden de EVENTS), como lo guarda main.py."""
    curves = json.loads(path.read_text(encoding="utf-8"))
    frames = [pd.DataFrame({"event": number, "tree": np.arange(1, len(curves[event]["train"]) + 1),
                            "loss": curves[event]["train"], "val_loss": curves[event]["val"]})
              for number, event in enumerate(EVENTS)]

    return pd.concat(frames, ignore_index=True)


def load_test():
    """
    Atributos, etiquetas, inicio y probabilidades de los modelos guardados en test; fechas de val y
    grupo de la vaca de cada sesión de test (Conocida si estuvo en train, Solo en val o Nueva) y
    tasa de cada salida de la vaca a 30 días.
    """
    sessions = load_sessions(INPUT_PATH)
    X, labels = build_features(sessions)
    idx = split_by_time(sessions["inicio"], TRAIN_FRACTION, VAL_FRACTION)
    test, val_dates = idx["test"], sessions["inicio"].iloc[idx["val"]]
    models, probabilities = [], []
    for event in EVENTS:
        model = XGBClassifier()
        model.load_model(MODELS_DIR / f"{event}.json")
        models.append(model)
        probabilities.append(model.predict_proba(X.iloc[test])[:, 1])

    val_label = f"Val ({val_dates.min():%d/%m/%Y}–{val_dates.max():%d/%m/%Y})"
    cows = sessions["id_vaca"]
    test_cows = cows.iloc[test]
    groups = np.select([test_cows.isin(cows.iloc[idx["train"]]), test_cows.isin(cows.iloc[idx["val"]])],
                       ["Conocida", "Solo en val"], default="Nueva")

    return (X.iloc[test], labels.iloc[test], sessions["inicio"].iloc[test], models, probabilities, val_label, groups,
            cow_rates(sessions, X, labels).iloc[test])


def cow_rates(sessions, X, labels):
    """
    Tasa de cada salida de la vaca en los 30 días previos, en el orden de EVENTS: la de los eventos
    es su *_vaca_30d; la de alguna anomalía se calcula igual que en build_history_features.py
    (ventana [inicio - 30 d, inicio) por vaca) y solo sirve para la gráfica 6, no entra al modelo.
    """
    m = pd.DataFrame({"id_vaca": sessions["id_vaca"], "inicio": sessions["inicio"],
                      "anomaly": labels["alguna_anomalia"]}).sort_values(["id_vaca", "inicio"], kind="stable")
    rate = m.set_index("inicio").groupby("id_vaca")["anomaly"].rolling("30D", closed="left").mean()
    rates = X[[f"{key}_vaca_30d" for key in EVENT_KEYS]].copy()
    rates["alguna_anomalia_vaca_30d"] = pd.Series(rate.to_numpy(), index=m.index)

    return rates


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=150)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((OUTPUT_DIR / "metrics.json").read_text(encoding="utf-8"))

    X, y, start, models, probabilities, val_label, groups, rates = load_test()
    log = read_curves(CURVES_PATH)
    save(plots.learning_curves(log, N_TREES, val_label), "curvas_aprendizaje.png")
    test_period = f"{start.min():%d/%m/%Y}–{start.max():%d/%m/%Y}"
    save(plots.roc_pr(y, probabilities, metrics, test_period), "roc_pr_test.png")
    save(plots.calibration(y, probabilities, metrics), "calibracion_test.png")
    save(plots.probability_distributions(y, probabilities, metrics), "distribucion_probabilidades.png")
    save(plots.gain_importance(models), "importancia_gain.png")
    save(plots.model_vs_feature(rates, y, probabilities), "modelo_vs_una_variable.png")
    save(plots.weekly_performance(y, start, probabilities, metrics), "desempeno_semanal_test.png")
    save(plots.metrics_by_cow_group(y, probabilities, groups, metrics), "vacas_nuevas_metricas.png")
    save(plots.new_cow_ramp(y, probabilities, X["dias_en_hato"].to_numpy(), groups, metrics),
         "vacas_nuevas_arranque.png")
    save(plots.probabilities_by_cow_group(y, probabilities, groups, metrics), "vacas_nuevas_probabilidades.png")
    save(plots.confusion_matrices(y, probabilities, metrics), "matrices_confusion_test.png")
    print(f"Figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
