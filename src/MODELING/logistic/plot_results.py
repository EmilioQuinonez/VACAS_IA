"""Gráficas de evaluación y avance por iteraciones de los cuatro eventos."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from .dataset import EVENTS, EVENT_NAMES
from matplotlib.ticker import MaxNLocator

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
        ax = axes[0, i]
        
        if actual.nunique() == 2:
            fpr, tpr, _ = roc_curve(actual, probability)
            ax.plot(fpr, tpr, color=BLUE, label="Modelo")
            rates = confusion_matrix(actual, probability >= decision["threshold"], labels=[0, 1], normalize="true")
            ax.plot(rates[0, 1], rates[1, 1], "o", color=ORANGE, label="Umbral de validación")
        
        ax.plot([0, 1], [0, 1], "--", color=REFERENCE)
        auc = "no calculable" if result["roc_auc"] is None else f"{result['roc_auc']:.3f}"
        ax.set(title=f"{name}\nROC-AUC {auc}", xlabel="Tasa de falsos positivos", ylabel="Sensibilidad")
        ax = axes[1, i]
        
        if actual.sum():
            precision, recall, _ = precision_recall_curve(actual, probability)
            ax.plot(recall, precision, color=BLUE)
        
        ax.axhline(result["prevalence"], color=REFERENCE, linestyle="--")
        if decision["recall"] is not None and decision["precision"] is not None:
            ax.plot(decision["recall"], decision["precision"], "o", color=ORANGE)
        ap = "no calculable" if result["average_precision"] is None else f"{result['average_precision']:.3f}"
        ax.set(title=f"AP {ap} · prevalencia {result['prevalence']:.1%}", xlabel="Sensibilidad", ylabel="Precisión", ylim=(0, 1))
    
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


def test_group_comparison(metrics, measures, title):
    """Compara conocidas y nuevas en test con el mismo modelo y umbral de val."""
    columns = min(3, len(measures))
    rows = (len(measures) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4.5 * rows), squeeze=False)
    positions = np.arange(len(EVENTS))
    groups = (("known_cows", "Conocidas", BLUE), ("new_cows", "Nuevas", ORANGE))
    for ax, (label, key) in zip(axes.flat, measures):
        for offset, (group, name, color) in zip((-0.18, 0.18), groups):
            values = []
            for event in EVENTS:
                scores = metrics[event][group]
                value = scores[key] if key in scores else scores["threshold_recall"][key]
                values.append(np.nan if value is None else value)
            counts = metrics[group]
            legend = f"{name} (vacas: {counts['n_cows']:,}; sesiones: {counts['n_sessions']:,})"
            bars = ax.bar(positions + offset, values, width=0.36, color=color, label=legend)
            for bar, value in zip(bars, values):
                text = "N/D" if np.isnan(value) else f"{value:.3f}"
                ax.annotate(text, (bar.get_x() + bar.get_width() / 2, 0 if np.isnan(value) else value),
                            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8)
        ax.set(title=label, xticks=positions, xticklabels=EVENT_NAMES)
        ax.tick_params(axis="x", labelrotation=15, labelsize=8)
        ax.margins(y=0.20)
        ax.set_ylim(bottom=0)
        ax.grid(axis="x", visible=False)
        if key not in ("brier", "log_loss"):
            ax.set_ylim(0, 1.12)
    for ax in axes.flat[len(measures):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10)
    fig.suptitle(title + "\nConocidas: aparecen antes de test; nuevas: primer ordeño registrado en test", fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 0.91))
    return fig


def learning_curves(metrics, metric="log_loss"):
    """Historial de train/validación/test registrado durante las épocas SGD."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    colors = {"train": BLUE, "val": ORANGE, "test": "#169b78"}
    labels = {"train": "Train", "val": "Validación", "test": "Test (solo diagnóstico)"}
    ylabel = "Accuracy (umbral 0.5)" if metric == "accuracy" else "Log loss"
    for event, name, ax in zip(EVENTS, EVENT_NAMES, axes):
        history = metrics[event]["learning_curve"]
        for split in ("train", "val", "test"):
            points = sorted((row for row in history if row["split"] == split), key=lambda row: row["iteration"])
            iterations = [row["iteration"] for row in points]
            values = [row[metric] for row in points]
            ax.plot(iterations, values, color=colors[split], label=f"{labels[split]}: {values[-1]:.4f}", linewidth=2, marker=".", markersize=3)
            ax.plot(iterations[-1], values[-1], "o", color=colors[split], markersize=4)
        final_iteration = metrics[event]["parameters"]["iterations"]
        ax.set(title=f"{name}\n{final_iteration} épocas ejecutadas", xlabel="Épocas SGD", ylabel=ylabel, xlim=(0, max(1, final_iteration)))
        if metric == "accuracy":
            ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.legend(loc="upper right", fontsize=8)
    direction = "mayor es mejor; umbral fijo 0.5" if metric == "accuracy" else "menor es mejor"
    fig.suptitle(f"Avance del ajuste: {ylabel} en entrenamiento, validación y test ({direction})\nHistorial registrado durante un único entrenamiento incremental por evento", fontsize=12)
    return fig


def save_learning_curves(metrics, output_dir):
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for metric, filename in (("log_loss", "curvas_aprendizaje.png"), ("accuracy", "curvas_accuracy.png")):
        figure = learning_curves(metrics, metric)
        path = directory / filename
        figure.savefig(path, dpi=150)
        plt.close(figure)
        print(f"[figure] {path}", flush=True)


def save_plots(metrics, predictions, coefficient_data, output_dir=None):
    """Guarda los archivos png de las graficas."""
    figures = Path(output_dir) if output_dir is not None else Path(__file__).resolve().parent / "img"
    builders = (
        ("roc_pr_test", roc_pr, (predictions, metrics)),
        ("calibracion_test", calibration, (predictions, metrics)),
        ("matrices_confusion_test", confusion, (metrics,)),
        ("coeficientes_evaluacion", coefficients, (coefficient_data,)),
        ("comparacion_test_metricas", test_group_comparison, (metrics, (
            ("ROC-AUC (mayor es mejor)", "roc_auc"), ("AP (mayor es mejor)", "average_precision"),
            ("Precisión (mayor es mejor)", "precision"), ("Sensibilidad (mayor es mejor)", "recall"),
            ("F1 (mayor es mejor)", "f1"), ("Exactitud (mayor es mejor)", "accuracy"),
        ), "Comparación de métricas en test: mismo modelo y umbral elegido en validación")),
        ("comparacion_test_perdidas", test_group_comparison, (metrics, (
            ("Brier (menor es mejor)", "brier"), ("Log loss (menor es mejor)", "log_loss"),
        ), "Comparación de pérdidas en test: menor es mejor")),
    )
    figures.mkdir(parents=True, exist_ok=True)
    for name, build, args in builders:
        figure = build(*args)
        figure.savefig(figures / f"{name}.png", dpi=150)
        plt.close(figure)
        print(f"[figure] {name}.png", flush=True)
    if all("learning_curve" in metrics[event] for event in EVENTS):
        save_learning_curves(metrics, figures)
    print(f"Gráficas en {figures}")
