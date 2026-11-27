"""
Carga el modelo final, la versión del XGBoost de versions.py, y para la última sesión de un historial
da la probabilidad calibrada de cada evento, si pasa el umbral elegido en val y la variable que más
aumenta el riesgo, con la misma función que usa el XGBoost para sus alertas de test (TreeSHAP; si
ninguna variable sube el riesgo, no se da ninguna). Es una asociación, no una causa.
"""

import json

import pandas as pd
from xgboost import XGBClassifier

from nueva_sesion import ROOT, load_module

XGB_DIR = ROOT / "src" / "MODELING" / "XGBoost"
BASELINES_PATH = ROOT / "src" / "EVALUATION" / "metrics" / "lineas_base.json"

xgb_dataset = load_module("xgb_dataset", XGB_DIR / "dataset.py")
xgb_evaluation = load_module("xgb_evaluation", XGB_DIR / "evaluation.py")
xgb_model = load_module("xgboost_model", XGB_DIR / "xgboost_model.py")
xgb_versions = load_module("xgb_versions", XGB_DIR / "versions.py")

EVENTS = xgb_dataset.EVENTS
EVENT_NAMES = dict(zip(EVENTS, ("Falla mecánica", "Salud", "Conducta", "Alguna anomalía")))
TOP_N = 1  # Solo la variable más relevante (objetivo 2 de minería de datos)


def read_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def text(value):
    """Valor de un factor como en las alertas de test: 4 cifras significativas, «faltante» o el texto."""
    if isinstance(value, str):
        return value
    return "faltante" if pd.isna(value) else f"{value:.4g}"


def top_factors(top):
    """
    Factores de una fila de xgb_evaluation.top_factors (columnas factor_N, valor_N y aporte_N); los
    vacíos son los que no aumentan el riesgo y se omiten.
    """
    found = []
    for n in range(1, TOP_N + 1):
        name = top[f"factor_{n}"]
        if isinstance(name, str) and name:
            found.append({"variable": name, "valor": text(top[f"valor_{n}"]), "aporte": float(top[f"aporte_{n}"])})
    return found


def performance(f1, recall, precision, brier):
    return {"F1": f1, "Recall": recall, "Precisión": precision, "Brier": brier}


class XGBoostModel:
    name = f"XGBoost {xgb_versions.VERSION}"

    def __init__(self):
        metrics = read_json(xgb_versions.MODELS_DIR / "metrics.json")
        self.models, self.platt, self.thresholds, self.performance = {}, {}, {}, {}
        for event in EVENTS:
            model = XGBClassifier()
            model.load_model(xgb_versions.MODELS_DIR / f"{event}.json")
            self.models[event] = model
            self.platt[event] = metrics[event]["platt"]
            self.thresholds[event] = metrics[event]["threshold_recall_val"]
            test = metrics[event]["test"]  # Métricas de test con el umbral elegido en val
            t = test["threshold_recall"]
            self.performance[event] = performance(t["f1"], t["recall"], t["precision"], test["brier"])

    def predict(self, sessions):
        """sessions: historial de una vaca con la sesión a predecir al final."""
        X = sessions.iloc[[-1]][xgb_dataset.FEATURES].astype(float)

        results = {}
        for event in EVENTS:
            model, platt = self.models[event], self.platt[event]
            probability = xgb_model.PlattXGB(model, platt["slope"], platt["intercept"]).predict_proba(X)[0, 1]
            # TreeSHAP en log-odds calibrados, con las dummies de máquina sumadas en "maquina".
            contributions = xgb_evaluation.shap_contributions(model, X, platt["slope"])
            top = xgb_evaluation.top_factors(contributions, X, TOP_N).iloc[0]
            threshold = self.thresholds[event]
            results[event] = {"probabilidad": float(probability), "umbral": float(threshold),
                              "alerta": bool(probability >= threshold), "factores": top_factors(top)}
        return results


def baseline():
    """Persistencia (alertar si el ordeño anterior tuvo el evento), de src/EVALUATION/lineas_base.py."""
    if not BASELINES_PATH.exists():
        return None
    baselines = read_json(BASELINES_PATH)
    return {event: performance(*(baselines[event]["persistencia"][k] for k in ("f1", "recall", "precision")), None)
            for event in EVENTS}


def load_model():
    return XGBoostModel()
