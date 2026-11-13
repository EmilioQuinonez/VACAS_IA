"""
Versiones del XGBoost: hiperparámetros de cada una y la que se corre.

Para entrenar o graficar otra versión, cambia solo VERSION y corre main.py y luego plot_results.py.
Lo que no aparece en una versión queda con el valor por defecto de XGBoost (learning_rate 0.3,
max_depth 6, min_child_weight 1, gamma 0, reg_lambda 1, reg_alpha 0, subsample y colsample_bytree 1).
En todas, scale_pos_weight, la calibración de Platt y el umbral los pone main.py; no hay early stopping.
"""

from pathlib import Path

VERSIONS = {
    # Base: valores por defecto de XGBoost.
    "v1": {"n_estimators": 100},
    # Pasos más chicos, para que val no se estanque desde los primeros árboles.
    "v2": {"n_estimators": 200, "learning_rate": 0.05},
    # Árboles más simples y regularizados.
    "v3": {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 4, "min_child_weight": 10, "gamma": 1,
           "reg_lambda": 5},
    # Árboles profundos frenados con regularización y muestreo de filas y atributos.
    "v4": {"n_estimators": 200, "learning_rate": 0.05, "max_depth": 6, "min_child_weight": 20, "gamma": 2,
           "reg_lambda": 10, "subsample": 0.8, "colsample_bytree": 0.7},
}
VERSION = "v4"

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models" / VERSION
FIGURES_DIR = BASE_DIR / "img" / VERSION
