"""
Carga los modelos finales (LSTM v5, regresión logística v4 y la versión del XGBoost de
versions.py) y, para la última sesión de un historial, da la probabilidad calibrada de cada
evento, si pasa el umbral elegido en val y la variable que más aumenta el riesgo, con la misma
función que usa cada modelo para sus alertas de test (si ninguna variable sube el riesgo, no se da
ninguna). Es una asociación, no una causa.
"""

import json
import sys

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from xgboost import XGBClassifier

from nueva_sesion import ROOT, is_milking, load_module

MODELING = ROOT / "src" / "MODELING"
LSTM_DIR = MODELING / "LSTM"
XGB_DIR = MODELING / "XGBoost"
LOGISTIC_DIR = MODELING / "logistic"
BASELINES_PATH = ROOT / "src" / "EVALUATION" / "metrics" / "lineas_base.json"

# La LSTM importa sus módulos por nombre (dataset, evaluation); la logística es un paquete de src.
sys.path[:0] = [str(LSTM_DIR), str(ROOT)]
import dataset as lstm_dataset  # noqa: E402
from evaluation import calibrate as lstm_calibrate  # noqa: E402
from explain import explain as lstm_explain  # noqa: E402
from src.MODELING.logistic.dataset import build_features as logistic_features  # noqa: E402
from src.MODELING.logistic.logistic_model import explain as logistic_explain  # noqa: E402

xgb_dataset = load_module("xgb_dataset", XGB_DIR / "dataset.py")
xgb_evaluation = load_module("xgb_evaluation", XGB_DIR / "evaluation.py")
xgb_model = load_module("xgboost_model", XGB_DIR / "xgboost_model.py")
xgb_versions = load_module("xgb_versions", XGB_DIR / "versions.py")

EVENTS = lstm_dataset.EVENTS
EVENT_NAMES = dict(zip(EVENTS, ("Falla mecánica", "Salud", "Conducta", "Alguna anomalía")))
LSTM_VERSION, LOGISTIC_VERSION = "v5", "v4"
TOP_N = 1  # Solo la variable más relevante (objetivo 2 de minería de datos)
HERD_DAYS = 10  # Las tasas del hato de la logística usan los 7 días anteriores; se pasa un margen


def read_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def result(probability, threshold, factors):
    return {"probabilidad": float(probability), "umbral": float(threshold),
            "alerta": bool(probability >= threshold), "factores": factors}


def top_factors(top, amount):
    """
    Factores de una fila con las columnas factor_N, valor_N y <amount>_N (y sesion_N en la LSTM)
    de los explain de cada modelo; los vacíos son los que no aumentan el riesgo y se omiten.
    """
    found = []
    for n in range(1, TOP_N + 1):
        name = top[f"factor_{n}"]
        if isinstance(name, str) and name:
            found.append({"variable": name, "valor": text(top[f"valor_{n}"]),
                          "aporte": float(top[f"{amount}_{n}"]), "sesion": top.get(f"sesion_{n}", "")})
    return found


def text(value):
    """Valor de un factor como en las alertas de test: 4 cifras significativas, «faltante» o el texto."""
    if isinstance(value, str):
        return value
    return "faltante" if pd.isna(value) else f"{value:.4g}"


def performance(f1, recall, precision, brier):
    return {"F1": f1, "Recall": recall, "Precisión": precision, "Brier": brier}


def from_threshold(test):
    """Métricas de test de metrics.json (LSTM y XGBoost) con el umbral elegido en val."""
    t = test["threshold_recall"]
    return performance(t["f1"], t["recall"], t["precision"], test["brier"])


class LSTMModel:
    name = f"LSTM {LSTM_VERSION}"

    def __init__(self):
        folder = LSTM_DIR / "modelos" / LSTM_VERSION
        self.model = keras.models.load_model(folder / "lstm.keras")
        # Integrated Gradients llama 32 veces al modelo por evento: compilado tarda una fracción.
        self.compiled = tf.function(self.model, reduce_retracing=True)
        self.config = read_json(folder / "config.json")
        self.scaler = read_json(folder / "scaler.json")
        self.calibration = read_json(folder / "calibracion.json")
        metrics = read_json(folder / "metrics.json")
        self.thresholds = {event: metrics[event]["threshold_recall_val"] for event in EVENTS}
        self.performance = {event: from_threshold(metrics[event]["test"]) for event in EVENTS}

    def predict(self, sessions, herd):
        """sessions: historial de una vaca con la sesión a predecir al final."""
        sessions = lstm_dataset.prepare_sessions(sessions)
        X_seq, X_ctx, _, _ = lstm_dataset.build_inputs(sessions, None, self.config["k"], self.scaler)
        rows = np.array([len(sessions) - 1])
        raw = self.model.predict([X_seq[rows], X_ctx[rows]], verbose=0)
        probabilities = lstm_calibrate(raw, self.calibration)[0]

        results = {}
        for number, event in enumerate(EVENTS):
            factors, _ = lstm_explain(self.compiled, X_seq, X_ctx, rows, sessions, number,
                                      self.config["batch_size"], self.calibration[event]["a"])
            results[event] = result(probabilities[number], self.thresholds[event],
                                    top_factors(factors.iloc[0], "aporte"))
        return results


class LogisticModel:
    name = f"Regresión logística {LOGISTIC_VERSION}"

    def __init__(self):
        self.bundle = joblib.load(LOGISTIC_DIR / "models" / f"{LOGISTIC_VERSION}.joblib")
        self.performance = {}
        for event in EVENTS:
            table = pd.read_csv(LOGISTIC_DIR / "metrics" / LOGISTIC_VERSION / f"{event}.csv",
                                index_col=0, encoding="utf-8-sig")["Prueba completa"].astype(float)
            self.performance[event] = performance(table["F1"], table["Sensibilidad"], table["Precisión"], table["Brier"])

    def predict(self, sessions, herd):
        """
        sessions: historial de una vaca con la sesión a predecir al final. herd: ordeños del
        resto del hato, para las tasas del hato.
        """
        milkings = sessions[is_milking(sessions)]
        start = sessions["inicio"].iloc[-1]
        recent = (herd["inicio"] < start) & (herd["inicio"] >= start.normalize() - pd.Timedelta(days=HERD_DAYS))
        others = herd[recent & (herd["id_vaca"] != sessions["id_vaca"].iloc[-1])]
        history = pd.concat([others, milkings], ignore_index=True)
        X = logistic_features(sessions.iloc[[-1]], history=history,
                              machine_columns=self.bundle["machine_columns"])[self.bundle["input_columns"]]
        transformed = self.bundle["preprocessor"].transform(X)

        results = {}
        for event in EVENTS:
            model = self.bundle["models"][event]
            probability = model.predict_proba(transformed)[0, 1]
            top = logistic_explain(model, self.bundle["preprocessor"], transformed, X, TOP_N).iloc[0]
            results[event] = result(probability, self.bundle["thresholds"][event], top_factors(top, "aporte_log_odds"))
        return results


class XGBoostModel:
    name = f"XGBoost {xgb_versions.VERSION}"

    def __init__(self):
        metrics = read_json(xgb_versions.MODELS_DIR / "metrics.json")
        self.models, self.platt, self.thresholds = {}, {}, {}
        for event in EVENTS:
            model = XGBClassifier()
            model.load_model(xgb_versions.MODELS_DIR / f"{event}.json")
            self.models[event] = model
            self.platt[event] = metrics[event]["platt"]
            self.thresholds[event] = metrics[event]["threshold_recall_val"]
        self.performance = {event: from_threshold(metrics[event]["test"]) for event in EVENTS}

    def predict(self, sessions, herd):
        """sessions: historial de una vaca con la sesión a predecir al final."""
        X = sessions.iloc[[-1]][xgb_dataset.FEATURES].astype(float)

        results = {}
        for event in EVENTS:
            model, platt = self.models[event], self.platt[event]
            probability = xgb_model.PlattXGB(model, platt["slope"], platt["intercept"]).predict_proba(X)[0, 1]
            # TreeSHAP en log-odds calibrados, con las dummies de máquina sumadas en "maquina".
            contributions = xgb_evaluation.shap_contributions(model, X, platt["slope"])
            top = xgb_evaluation.top_factors(contributions, X, TOP_N).iloc[0]
            results[event] = result(probability, self.thresholds[event], top_factors(top, "aporte"))
        return results


def baseline():
    """Persistencia (alertar si el ordeño anterior tuvo el evento), de src/EVALUATION/lineas_base.py."""
    if not BASELINES_PATH.exists():
        return None
    baselines = read_json(BASELINES_PATH)
    return {event: performance(*(baselines[event]["persistencia"][k] for k in ("f1", "recall", "precision")), None)
            for event in EVENTS}


def load_models():
    return [LSTMModel(), LogisticModel(), XGBoostModel()]
