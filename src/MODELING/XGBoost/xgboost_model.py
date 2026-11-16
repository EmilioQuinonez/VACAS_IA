"""Clasificador XGBoost por evento y su calibración."""

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def train_event(X, y, rows, val_rows=None, *, params, random_state):
    """
    Entrena un XGBoost con los hiperparámetros de params (los que faltan quedan con el valor
    por defecto de XGBoost, incluida la regularización L2 reg_lambda = 1); scale_pos_weight
    compensa el desbalance. val_rows solo se evalúa para las curvas de aprendizaje: no
    detiene ni cambia el entrenamiento.
    """
    X_fit, y_fit = X.iloc[rows], y.iloc[rows]
    eval_set = [(X_fit, y_fit)]
    if val_rows is not None:
        eval_set.append((X.iloc[val_rows], y.iloc[val_rows]))
    model = XGBClassifier(
        objective="binary:logistic",
        scale_pos_weight=(y_fit == 0).sum() / (y_fit == 1).sum(),
        eval_metric="logloss",
        random_state=random_state,
        **params,
    )

    return model.fit(X_fit, y_fit, eval_set=eval_set, verbose=False)


class PlattXGB:
    """XGBoost con probabilidad calibrada: sigmoide(slope · log-odds del XGBoost + intercept)."""

    def __init__(self, model, slope, intercept):
        self.model, self.slope, self.intercept = model, slope, intercept

    def predict_proba(self, X):
        p = expit(self.slope * self.model.predict(X, output_margin=True) + self.intercept)
        return np.column_stack([1 - p, p])


def calibrate(model, X, y, rows):
    """
    Calibración de Platt ajustada en rows sobre los log-odds del XGBoost, sin reentrenarlo:
    corrige la inflación de scale_pos_weight sin cambiar el orden de las probabilidades.
    """
    margin = model.predict(X.iloc[rows], output_margin=True).reshape(-1, 1)
    platt = LogisticRegression(penalty=None).fit(margin, y.iloc[rows])
    return PlattXGB(model, float(platt.coef_[0, 0]), float(platt.intercept_[0]))
