"""
Hiperparámetros de cada versión del XGBoost; VERSION elige cuál corren main.py y plot_results.py.
Lo que no aparece en una versión queda con el valor por defecto de XGBoost.
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
    # Mejor de las 40 combinaciones al azar de random_search.py (combinación 5 de models/v5/busqueda.csv).
    "v5": {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 8, "min_child_weight": 1, "gamma": 5,
           "reg_lambda": 20, "subsample": 0.7, "colsample_bytree": 0.9},
}
VERSION = "v5"

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models" / VERSION
FIGURES_DIR = BASE_DIR / "img" / VERSION
