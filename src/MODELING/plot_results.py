"""
Genera las gráficas del XGBoost base a partir de lo que guardó main.py (modelos finales,
metrics.json y el log), sin reentrenar. Solo cifras agregadas, sin datos por vaca.

Las figuras van a FIGURES_DIR, fuera de Git por ahora; para versionarlas basta con cambiar esa ruta.

Uso, desde la raíz del repo y después de main.py:
    python src/MODELING/plot_results.py
"""

import json
import re

import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBClassifier

import plots
from dataset import EVENTS, build_features, load_sessions, split_by_time
from main import INPUT_PATH, LOG_PATH, N_TREES, OUTPUT_DIR, TRAIN_FRACTION, VAL_FRACTION

FIGURES_DIR = OUTPUT_DIR / "graficas"


def read_log(path, fits_per_event, chrono_fits):
    """
    Log loss de train y val por época; block 0 = modelo de train, 1 = final (si se reentrenó con
    train + val) y los siguientes (by_cow) = folds por vaca.
    """
    rows, epoch = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        header = re.match(r"^Epoch (\d+)/\d+$", line)
        if header:
            epoch = int(header.group(1))
        elif line.startswith("1/1 - "):
            metrics = dict(re.findall(r"(\w+): (-?[\d.]+)", line))
            rows.append({"epoch": epoch, "loss": float(metrics["loss"]),
                         "val_loss": float(metrics.get("val_loss", "nan"))})
    log = pd.DataFrame(rows)
    fit_number = (log["epoch"] - 1) // N_TREES
    log["event"] = fit_number // fits_per_event
    log["block"] = fit_number % fits_per_event
    log["tree"] = (log["epoch"] - 1) % N_TREES + 1
    log["by_cow"] = log["block"] >= chrono_fits

    return log


def load_test():
    """Atributos, etiquetas, inicio y probabilidades de los modelos guardados en test; fechas de val."""
    sessions = load_sessions(INPUT_PATH)
    X, labels = build_features(sessions)
    idx = split_by_time(sessions["inicio"], TRAIN_FRACTION, VAL_FRACTION)
    test, val_dates = idx["test"], sessions["inicio"].iloc[idx["val"]]
    models, probabilities = [], []
    for event in EVENTS:
        model = XGBClassifier()
        model.load_model(OUTPUT_DIR / f"{event}.json")
        models.append(model)
        probabilities.append(model.predict_proba(X.iloc[test])[:, 1])

    val_label = f"Val ({val_dates.min():%d/%m/%Y}–{val_dates.max():%d/%m/%Y})"

    return X.iloc[test], labels.iloc[test], sessions["inicio"].iloc[test], models, probabilities, val_label


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=150)
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((OUTPUT_DIR / "metrics.json").read_text(encoding="utf-8"))
    has_by_cow = all("by_cow" in metrics[event] for event in EVENTS)
    chrono_fits = 1 + ("train_val" in metrics[EVENTS[0]])
    fits_per_event = chrono_fits + (len(metrics[EVENTS[0]]["by_cow"]["folds"]) if has_by_cow else 0)

    X, y, start, models, probabilities, val_label = load_test()
    log = read_log(LOG_PATH, fits_per_event, chrono_fits)
    save(plots.learning_curves(log, N_TREES, val_label), "curvas_aprendizaje.png")
    test_period = f"{start.min():%d/%m/%Y}–{start.max():%d/%m/%Y}"
    save(plots.roc_pr(y, probabilities, metrics, test_period), "roc_pr_test.png")
    save(plots.calibration(y, probabilities, metrics), "calibracion_test.png")
    save(plots.probability_distributions(y, probabilities, metrics), "distribucion_probabilidades.png")
    save(plots.gain_importance(models), "importancia_gain.png")
    save(plots.model_vs_feature(X, y, probabilities), "modelo_vs_una_variable.png")
    if has_by_cow:
        save(plots.by_cow(metrics), "corte_por_vaca.png")
    save(plots.weekly_performance(y, start, probabilities, metrics), "desempeno_semanal_test.png")
    print(f"Figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
