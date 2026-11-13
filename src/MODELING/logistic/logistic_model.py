"""Relleno con cero y bandera, escala de train y regresión logística explicable."""

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.preprocessing import StandardScaler

from .dataset import binary_features, feature_group
from .evaluation import choose_threshold

LOGISTIC_CONFIG = {
    "loss": "log_loss", 
    "penalty": "elasticnet", 
    "alpha": 0.0001,    
    "l1_ratio": 0.5, 
    "learning_rate": "optimal", 
    "average": True,
    "class_weight": None, 
    "random_state": 42,
    "max_iter": 2000, 
    "tol": 1e-4, 
    "n_iter_no_change": 5,
}


class CowPreprocessor:
    """Completa faltantes con cero y bandera; ajusta escala y referencia solo en train."""

    def prepare(self, X):
        self.columns = list(X.columns)
        self.binary = binary_features(self.columns)
        self.continuous = [c for c in self.columns if c not in self.binary]
        expanded = self._fill_missing(X)
        self.scaler = StandardScaler().fit(expanded[self.continuous])
        self.transformed_names = list(expanded.columns)
        expanded[self.continuous] = self.scaler.transform(expanded[self.continuous])
        self.reference = np.ascontiguousarray(expanded.to_numpy(dtype=np.float64)).mean(axis=0)
        
        return self

    def _fill_missing(self, X):
        values = X[self.columns]
        # Todas las columnas reciben bandera para mantener el esquema entre particiones.
        flags = values.isna().astype(float).add_suffix("__faltante")
        return pd.concat([values.fillna(0), flags], axis=1)

    def transform(self, X):
        expanded = self._fill_missing(X)
        expanded[self.continuous] = self.scaler.transform(expanded[self.continuous])
        return np.ascontiguousarray(expanded.to_numpy(dtype=np.float64))


def train_event(transformed, actual, config=None, target_recall=0.80, show_params_settings=True):
    """Entrena solo con train y registra loss y accuracy a umbral 0.5 por época."""
    params = dict(LOGISTIC_CONFIG if config is None else config)
    if not len(actual["val"]) or not np.asarray(actual["val"]).sum():
        raise ValueError("Validación requiere positivos para elegir el umbral y calcular F1.")
    # La tolerancia y la paciencia se aplican al F1 de validación.
    model = SGDClassifier(**{**params, "tol": None})
    history = []

    def record(epoch):
        losses = {}
        for split, data in transformed.items():
            probability = model.predict_proba(data)[:, 1] if epoch else np.full(len(data), 0.5)
            losses[split] = float(log_loss(actual[split], probability, labels=[0, 1]))
            history.append({
                "iteration": epoch, "split": split, "log_loss": losses[split],
                "accuracy": float(accuracy_score(actual[split], probability >= 0.5)),
            })
        return losses

    record(0)
    best_f1, reference_f1, best_epoch, best_model, stale_epochs = -np.inf, -np.inf, 0, None, 0
    for epoch in range(1, params["max_iter"] + 1):
        model.partial_fit(transformed["train"], actual["train"], classes=[0, 1] if epoch == 1 else None)
        losses = record(epoch)
        val_probability = model.predict_proba(transformed["val"])[:, 1]
        threshold = choose_threshold(actual["val"], val_probability, target_recall)
        val_f1 = float(f1_score(actual["val"], val_probability >= threshold, zero_division=0))
        if val_f1 > best_f1:
            best_f1, best_epoch, best_model = val_f1, epoch, deepcopy(model)
        if reference_f1 == -np.inf or (val_f1 > reference_f1 and val_f1 - reference_f1 >= params["tol"]):
            reference_f1, stale_epochs = val_f1, 0
        else:
            stale_epochs += 1
        stopped = stale_epochs >= params["n_iter_no_change"]
        if show_params_settings and (epoch == 1 or epoch % 10 == 0 or stopped or epoch == params["max_iter"]):
            print(f"  [learning] época {epoch}/{params['max_iter']}; Log Loss train={losses['train']:.4f}; F1 val={val_f1:.4f}; mejor F1 val={best_f1:.4f}", flush=True)
        if stopped:
            break
    details = {
        **params, "iterations": epoch, "converged": stopped,
        "stop_reason": "validation_f1_plateau" if stopped else "max_iter",
        "best_epoch": best_epoch, "best_validation_f1": best_f1,
    }
    return best_model, details, history


def explain(model, preprocessor, transformed, X, top_n=3):
    """Tres asociaciones de mayor contribución absoluta, agrupadas por variable."""
    weights = model.coef_[0]
    contribution = (transformed - preprocessor.reference) * weights
    groups = [feature_group(c.removesuffix("__faltante")) for c in preprocessor.transformed_names]
    group_names = list(dict.fromkeys(groups))
    grouped = np.column_stack([
        contribution[:, np.array(groups) == name].sum(axis=1)
        for name in group_names
    ])
    
    ranking = np.argsort(-np.abs(grouped), axis=1, kind="stable")[:, :top_n]
    output = pd.DataFrame(index=X.index)
    
    for rank in range(top_n):
        if rank >= ranking.shape[1]:
            output[f"factor_{rank + 1}"] = ""
            output[f"aporte_log_odds_{rank + 1}"] = 0.0
            output[f"direccion_{rank + 1}"] = "sin aporte"
            output[f"valor_{rank + 1}"] = ""
            continue
        indices = ranking[:, rank]
        amounts = grouped[np.arange(len(X)), indices]
        names = np.asarray(group_names, dtype=object)[indices]
        nonzero = np.abs(amounts) > 1e-12
        output[f"factor_{rank + 1}"] = np.where(nonzero, names, "")
        output[f"aporte_log_odds_{rank + 1}"] = amounts
        output[f"direccion_{rank + 1}"] = np.where(~nonzero, "sin aporte", np.where(amounts > 0, "aumenta riesgo", "reduce riesgo"))
        values = np.full(len(X), "", dtype=object)
        
        for name in np.unique(names[nonzero]):
            rows = np.flatnonzero((names == name) & nonzero)
            columns = [c for c in X if feature_group(c) == name]
            raw = X.iloc[rows][columns].to_numpy()
            if name == "produccion_previa":
                raw = np.expm1(raw)
            unit = " s/kg" if name == "box_time_to_yield_ratio" else ""
            values[rows] = [
                "; ".join(f"{c}={v:.6g}{unit}" if pd.notna(v) else f"{c}=faltante" for c, v in zip(columns, row))
                for row in raw
            ]
        output[f"valor_{rank + 1}"] = values
    output["base_log_odds"] = float(model.intercept_[0] + preprocessor.reference @ weights)
    output["suma_aportes_log_odds"] = grouped.sum(axis=1)
    output["asociaciones_no_causales"] = True

    return output


def coefficient_table(models, preprocessor):
    rows = []
    groups = [feature_group(c.removesuffix("__faltante")) for c in preprocessor.transformed_names]
    
    for event, model in models.items():
        for column, group, weight in zip(preprocessor.transformed_names, groups, model.coef_[0]):
            scale = "por 1 desviación estándar en la escala del predictor" if column in preprocessor.continuous else "cambio de 0 a 1"
            if column == "produccion_previa":
                scale += " log1p"
            if column == "box_time_to_yield_ratio":
                scale += " (predictor en s/kg)"
            if column == "mdi_previo" or column in ("blood_diff_di", "blood_diff_dd", "blood_diff_ti"):
                scale += " log1p original"
            rows.append({
                "event": event, "variable": column, "grupo": group, "coeficiente": float(weight),
                "odds_ratio": float(np.exp(weight)), "interpretacion": scale + "; multiplica odds, no probabilidad",
                "direccion": "aumenta riesgo" if weight > 0 else "reduce riesgo" if weight < 0 else "sin asociación",
            })
    
    return pd.DataFrame(rows)
