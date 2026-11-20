"""
Genera en img/<VERSION>/ las gráficas de la versión VERSION de versions.py, sin reentrenar.
ALERTS_PATH (fuera de Git) tiene una fila por alerta de test con su probabilidad calibrada y sus
3 factores SHAP, sin el id de la vaca.

Uso, desde la raíz del repo y después de main.py:
    python src/MODELING/XGBoost/plot_results.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import plots
from dataset import EVENTS, build_features, load_sessions, split_by_time
from evaluation import check_additivity, shap_contributions, top_factors
from main import CURVES_PATH, INPUT_PATH, METRICS_PATH, TRAIN_FRACTION, VAL_FRACTION
from versions import FIGURES_DIR, MODELS_DIR, VERSION, VERSIONS
from xgboost_model import PlattXGB

ALERTS_PATH = INPUT_PATH.parent / "modelos" / "xgboost" / VERSION / "alertas_test.csv"
ADDITIVITY_TOLERANCE = 1e-3


def read_curves(path):
    """Log loss de train y val por árbol de cada evento (en el orden de EVENTS), como lo guarda main.py."""
    curves = json.loads(path.read_text(encoding="utf-8"))
    frames = [pd.DataFrame({"event": number, "tree": np.arange(1, len(curves[event]["train"]) + 1),
                            "loss": curves[event]["train"], "val_loss": curves[event]["val"]})
              for number, event in enumerate(EVENTS)]

    return pd.concat(frames, ignore_index=True)


def load_test(metrics):
    """
    Atributos, etiquetas, inicio y probabilidades calibradas de los modelos guardados en test; fechas de val y
    grupo de la vaca de cada sesión de test (Conocida si estuvo en train, Solo en val o Nueva).
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
        calibrated = PlattXGB(model, **metrics[event]["platt"])
        probabilities.append(calibrated.predict_proba(X.iloc[test])[:, 1])

    val_label = f"Val ({val_dates.min():%d/%m/%Y}–{val_dates.max():%d/%m/%Y})"
    cows = sessions["id_vaca"]
    test_cows = cows.iloc[test]
    groups = np.select([test_cows.isin(cows.iloc[idx["train"]]), test_cows.isin(cows.iloc[idx["val"]])],
                       ["Conocida", "Solo en val"], default="Nueva")

    return X.iloc[test], labels.iloc[test], sessions["inicio"].iloc[test], models, probabilities, val_label, groups


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=150)
    plt.close(fig)


def explain_alerts(X, start, models, probabilities, metrics):
    """
    Aportes SHAP de cada modelo en test y 3 factores de cada alerta (probabilidad calibrada mayor o
    igual al umbral de recall de val); guarda las alertas en ALERTS_PATH e imprime en qué proporción
    los 3 factores suben el riesgo y el error de aditividad.
    """
    contributions, factors, alerts = [], [], []
    for event, model, probability in zip(EVENTS, models, probabilities):
        platt, threshold = metrics[event]["platt"], metrics[event]["threshold_recall_val"]
        alert = probability >= threshold
        contribution = shap_contributions(model, X, platt["slope"])
        error = check_additivity(model, X, contribution, platt)
        assert error < ADDITIVITY_TOLERANCE, f"{event}: SHAP additivity error {error:.2e}"
        top = top_factors(contribution[alert], X[alert])
        all_up = (top.filter(like="aporte_") > 0).all(axis=1).mean()
        print(f"[shap] {event}: {alert.sum():,} alerts, {all_up:.1%} with the 3 factors raising the risk, "
              f"additivity error {error:.1e}")
        contributions.append(contribution)
        factors.append(top)
        alerts.append(pd.DataFrame({"inicio": start[alert], "evento": event, "probabilidad": probability[alert],
                                    "umbral": threshold}).join(top))
    ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(alerts).to_csv(ALERTS_PATH, index=False)

    return contributions, factors


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    X, y, start, models, probabilities, val_label, groups = load_test(metrics)
    log = read_curves(CURVES_PATH)
    save(plots.learning_curves(log, VERSIONS[VERSION]["n_estimators"], val_label), "curvas_aprendizaje.png")
    save(plots.calibration(y, probabilities, metrics), "calibracion_test.png")
    save(plots.probability_distributions(y, probabilities, metrics), "distribucion_probabilidades.png")
    save(plots.gain_importance(models), "importancia_gain.png")
    contributions, factors = explain_alerts(X, start, models, probabilities, metrics)
    save(plots.shap_importance(contributions), "importancia_shap.png")
    save(plots.top_factor_share(factors), "factores_alertas.png")
    save(plots.weekly_performance(y, start, probabilities, metrics), "desempeno_semanal_test.png")
    save(plots.metrics_by_cow_group(y, probabilities, groups, metrics), "vacas_nuevas_metricas.png")
    save(plots.new_cow_ramp(y, probabilities, X["dias_en_hato"].to_numpy(), groups, metrics),
         "vacas_nuevas_arranque.png")
    save(plots.probabilities_by_cow_group(y, probabilities, groups, metrics), "vacas_nuevas_probabilidades.png")
    save(plots.confusion_matrices(y, probabilities, metrics), "matrices_confusion_test.png")
    print(f"Figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
