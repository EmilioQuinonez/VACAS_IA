"""Genera las seis gráficas con los resultados en memoria que entrega main.py."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from .dataset import EVENTS, EVENT_NAMES

BLUE, ORANGE, REFERENCE = "#2a78d6", "#eb6834", "#9a9890"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
})


def roc_pr(predictions, metrics):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for i, (event, name) in enumerate(zip(EVENTS, EVENT_NAMES)):
        data = predictions[predictions["event"].eq(event)]
        actual, probability = data["actual"], data["probability"]
        result = metrics[event]["test"]
        decision = result["threshold_recall"]
        tn, fp, fn, tp = np.asarray(decision["confusion_matrix"]).ravel()
        ax = axes[0, i]
        fpr, tpr, _ = roc_curve(actual, probability)
        ax.plot(fpr, tpr, color=BLUE, label="Modelo")
        ax.plot([0, 1], [0, 1], "--", color=REFERENCE)
        ax.plot(fp / (fp + tn), tp / (tp + fn), "o", color=ORANGE, label="Umbral de validación")
        ax.set(title=f"{name}\nROC-AUC {result['roc_auc']:.3f}", xlabel="Tasa de falsos positivos", ylabel="Sensibilidad")
        ax = axes[1, i]
        precision, recall, _ = precision_recall_curve(actual, probability)
        ax.plot(recall, precision, color=BLUE)
        ax.axhline(result["prevalence"], color=REFERENCE, linestyle="--")
        ax.plot(decision["recall"], decision["precision"], "o", color=ORANGE)
        ax.set(title=f"AP {result['average_precision']:.3f} · prevalencia {result['prevalence']:.1%}", xlabel="Sensibilidad", ylabel="Precisión", ylim=(0, 1))
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("Curvas ROC y precisión–sensibilidad en prueba")
    return fig


def calibration(predictions, metrics):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), constrained_layout=True)
    for event, name, ax in zip(EVENTS, EVENT_NAMES, axes):
        data = predictions[predictions["event"].eq(event)]
        observed, predicted = calibration_curve(data["actual"], data["probability"], n_bins=10)
        ax.plot([0, 1], [0, 1], "--", color=REFERENCE, label="Calibración perfecta")
        ax.plot(predicted, observed, "o-", color=BLUE, label="Modelo")
        ax.set(title=f"{name}\nBrier {metrics[event]['test']['brier']:.3f}", xlabel="Probabilidad predicha", ylabel="Frecuencia observada", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(loc="upper left")
    fig.suptitle("Calibración de probabilidades en prueba")
    return fig


def confusion(metrics):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
    for event, name, ax in zip(EVENTS, EVENT_NAMES, axes.ravel()):
        matrix = np.asarray(metrics[event]["test"]["threshold_recall"]["confusion_matrix"])
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:,}", ha="center", va="center", color="white" if value > matrix.max() / 2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Sin evento", "Evento"], yticklabels=["Sin evento", "Evento"], xlabel="Predicción", ylabel="Real", title=f"{name} · umbral {metrics[event]['threshold_recall_val']:.3f}")
        ax.grid(False)
    fig.suptitle("Matrices de confusión en prueba con el umbral de validación")
    return fig


def probability_distributions(predictions, metrics):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for event, name, ax in zip(EVENTS, EVENT_NAMES, axes.ravel()):
        data = predictions[predictions["event"].eq(event)]
        for label, title, color in ((0, "Sin evento", BLUE), (1, "Con evento", ORANGE)):
            ax.hist(data.loc[data["actual"].eq(label), "probability"], bins=np.linspace(0, 1, 41), density=True, histtype="step", linewidth=2, label=title, color=color)
        ax.axvline(metrics[event]["threshold_recall_val"], color=REFERENCE, linestyle="--", label="Umbral de validación")
        ax.set(title=name, xlabel="Probabilidad predicha", ylabel="Densidad")
    axes[0, 0].legend()
    fig.suptitle("Probabilidades en prueba según la clase real")
    return fig


def coefficients(frame):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    for event, name, ax in zip(EVENTS, EVENT_NAMES, axes.ravel()):
        data = frame[frame["event"].eq(event)].copy()
        data = data.iloc[np.argsort(np.abs(data["coeficiente"].to_numpy()))[-10:]]
        weights = data["coeficiente"].to_numpy()
        ax.barh(data["variable"], weights, color=np.where(weights >= 0, ORANGE, BLUE))
        ax.axvline(0, color=REFERENCE, linewidth=1)
        ax.set(title=name, xlabel="Coeficiente: numéricas por 1 DE; binarias de 0 a 1")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Coeficientes de mayor magnitud de los modelos de evaluación")
    return fig


def learning_curves(metrics):
    """Pérdida y desempeño por iteración para cada conjunto y evento."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    for i, (event, name) in enumerate(zip(EVENTS, EVENT_NAMES)):
        history = pd.DataFrame(metrics[event]["learning_curve"])
        max_iter = int(metrics[event]["parameters"]["max_iter"])
        for split, label, color in (("train", "Train", BLUE), ("val", "Validación", ORANGE), ("test", "Test", REFERENCE)):
            data = history.loc[history["split"].eq(split)].sort_values("iteration")
            for row, metric in enumerate(("log_loss", "average_precision")):
                axes[row, i].plot(data["iteration"], data[metric], marker="o", color=color, label=label)
        xlabel = f"Iteraciones de entrenamiento (max_iter={max_iter})"
        axes[0, i].set(title=name, xlabel=xlabel, ylabel="Log loss (menor es mejor)", ylim=(0, None))
        axes[1, i].set(xlabel=xlabel, ylabel="Average precision (mayor es mejor)", ylim=(0, 1))
        for ax in axes[:, i]:
            ax.set_xlim(0, max_iter)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    axes[0, 0].legend()
    axes[1, 0].legend()
    fig.suptitle("Curvas de aprendizaje por iteraciones del modelo con parámetros fijos\nPuntos reconstruidos desde la misma inicialización; test solo se evalúa")
    return fig


def save_plots(metrics, predictions, coefficient_data, output_dir=None):
    """Guarda los archivos png de las graficas."""
    figures = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent / "results" / "graficas"
    builders = (
        ("roc_pr_test", roc_pr, (predictions, metrics)),
        ("calibracion_test", calibration, (predictions, metrics)),
        ("matrices_confusion_test", confusion, (metrics,)),
        ("distribucion_probabilidades", probability_distributions, (predictions, metrics)),
        ("coeficientes_evaluacion", coefficients, (coefficient_data,)),
        ("curvas_aprendizaje", learning_curves, (metrics,)),
    )
    figures.mkdir(parents=True, exist_ok=True)
    for name, build, args in builders:
        figure = build(*args)
        figure.savefig(figures / f"{name}.png", dpi=150)
        plt.close(figure)
        print(f"[figure] {name}.png", flush=True)
    print(f"Gráficas en {figures}")
