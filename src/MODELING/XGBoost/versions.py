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
}
VERSION = "v1"

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models" / VERSION
FIGURES_DIR = BASE_DIR / "img" / VERSION
