"""Imputación ajustada en train, regresión logística y explicaciones locales."""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import average_precision_score, log_loss
from sklearn.preprocessing import StandardScaler

from .dataset import binary_features, feature_group


class CowPreprocessor:
    """Medianas de entrenamiento por vaca, respaldo global, indicadores y escala fija."""

    def fit(self, X, cows):
        self.columns = list(X.columns)
        self.binary = binary_features(self.columns)
        self.continuous = [c for c in self.columns if c not in self.binary]
        self.global_medians = X.median().fillna(0)
        self.empty_columns = X.columns[X.isna().all()].tolist()
        self.cow_medians = X.assign(_cow=np.asarray(cows)).groupby("_cow").median()
        expanded = self._impute(X, cows)
        self.scaler = StandardScaler().fit(expanded[self.continuous])
        self.transformed_names = list(expanded.columns)
        expanded[self.continuous] = self.scaler.transform(expanded[self.continuous])
        self.reference = np.ascontiguousarray(expanded.to_numpy(dtype=np.float64)).mean(axis=0)
        
        return self

    def _impute(self, X, cows):
        cow_values = self.cow_medians.reindex(np.asarray(cows)).set_axis(X.index)
        filled = X.fillna(cow_values).fillna(self.global_medians)
        flags = X.isna().astype(float).add_suffix("__faltante")
        return pd.concat([filled, flags], axis=1)

    def transform(self, X, cows):
        expanded = self._impute(X, cows)
        expanded[self.continuous] = self.scaler.transform(expanded[self.continuous])
        return np.ascontiguousarray(expanded.to_numpy(dtype=np.float64))

def train_event(X, y, C, l1_ratio, random_state=42, max_iter=500, tol=1e-4):
    """Entrena desde cero y devuelve parámetros e iteraciones ejecutadas."""
    model = LogisticRegression(
        solver="saga", C=C, l1_ratio=l1_ratio, class_weight=None,
        random_state=random_state, max_iter=max_iter, tol=tol,
    )
    model.fit(X, y)
    
    details = {
        "C": C, "l1_ratio": l1_ratio,
        "iterations": int(model.n_iter_.max()), "max_iter": max_iter, "tol": tol,
    }
    
    return model, details


def training_curve(model, transformed, actual):
    """Reconstruye puntos de un ajuste SAGA sin modificar el modelo.

    Cada punto se ajusta desde cero con la misma semilla que el modelo original.
    El último punto reutiliza el modelo evaluado.
    La pérdida es log_loss sin la penalización de regularización.
    """
    final_iteration = int(model.n_iter_.max())
    checkpoints = [n for n in (1, 5, 10, 25, 50, 100, 200) if n < final_iteration]
    params = model.get_params()
    rows = []

    for iteration in [*checkpoints, final_iteration]:
        if iteration == final_iteration:
            checkpoint_model = model
        else:
            # Estos ajustes se cortan deliberadamente antes de converger.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                checkpoint_model, _ = train_event(
                    transformed["train"], actual["train"], params["C"], params["l1_ratio"],
                    params["random_state"], iteration, params["tol"],
                )

        for name in ("train", "val", "test"):
            probability = checkpoint_model.predict_proba(transformed[name])[:, 1]
            rows.append({
                "iteration": int(checkpoint_model.n_iter_.max()), "split": name,
                "log_loss": float(log_loss(actual[name], probability, labels=[0, 1])),
                "average_precision": float(average_precision_score(actual[name], probability)),
            })

    return pd.DataFrame(rows)


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
            if name in ("produccion_previa", "box_time_to_yield_ratio"):
                raw = np.expm1(raw)
            values[rows] = [
                "; ".join(f"{c}={v:.6g}" if pd.notna(v) else f"{c}=faltante" for c, v in zip(columns, row))
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
            if column in ("produccion_previa", "box_time_to_yield_ratio"):
                scale += " log1p"
            if column == "mdi_previo":
                scale += " log1p original"
            rows.append({
                "event": event, "variable": column, "grupo": group, "coeficiente": float(weight),
                "odds_ratio": float(np.exp(weight)), "interpretacion": scale,
            })
    
    return pd.DataFrame(rows)
