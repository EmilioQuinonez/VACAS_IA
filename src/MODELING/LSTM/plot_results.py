"""
Gráficas de la LSTM, las mismas que la regresión logística. main.py las guarda en img/<VERSION>:
- roc_pr_test.png, calibracion_test.png y matrices_confusion_test.png: test con el umbral elegido
  en val (mayor F1 con recall >= 80 %).
- aportes_evaluacion.png: equivalente a coeficientes_evaluacion de la regresión logística; aporte
  promedio de cada variable al log-odds en las alertas de test (Integrated Gradients).
- comparacion_test_metricas.png y comparacion_test_perdidas.png: vacas conocidas contra nuevas.
- curvas_aprendizaje.png y curvas_accuracy.png: log loss y accuracy (umbral 0.5) por época de cada
  evento en train, val y test.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from dataset import EVENTS

EVENT_NAMES = ["Mecánico", "Salud", "Conducta", "Alguna anomalía"]
TOP_FEATURES = 10

BLUE, ORANGE, AQUA, REFERENCE = "#2a78d6", "#eb6834", "#169b78", "#9a9890"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9a9890",
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "lines.linewidth": 2,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
})


def roc_pr(y, probability, metrics):
    """ROC (arriba) y precisión-recall (abajo) en test; el punto es el umbral elegido en val."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for number, (event, name) in enumerate(zip(EVENTS, EVENT_NAMES)):
        actual, score = y[:, number], probability[:, number]
        result, decision = metrics[event]["test"], metrics[event]["test"]["threshold_recall"]
        rates = confusion_matrix(actual, score >= metrics[event]["threshold_recall_val"], normalize="true")

        ax = axes[0, number]
        fpr, tpr, _ = roc_curve(actual, score)
        ax.plot(fpr, tpr, color=BLUE, label="LSTM")
        ax.plot(rates[0, 1], rates[1, 1], "o", color=ORANGE, markersize=8, label="Umbral de val")
        ax.plot([0, 1], [0, 1], "--", color=REFERENCE, linewidth=1, label="Azar")
        ax.set(title=f"{name}\nROC-AUC {result['roc_auc']:.3f}", xlabel="Tasa de falsos positivos",
               ylabel="Sensibilidad")

        ax = axes[1, number]
        precision, recall, _ = precision_recall_curve(actual, score)
        ax.plot(recall, precision, color=BLUE)
        ax.plot(decision["recall"], decision["precision"], "o", color=ORANGE, markersize=8)
        ax.axhline(result["prevalence"], color=REFERENCE, linestyle="--", linewidth=1)
        ax.set(title=f"PR-AUC {result['pr_auc']:.3f} · prevalencia {result['prevalence']:.1%}",
               xlabel="Sensibilidad", ylabel="Precisión", ylim=(0, 1.02))
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("Curvas ROC y precisión-sensibilidad en test (la línea punteada es el azar)")
    return fig


def calibration(y, probability, metrics):
    """Frecuencia observada contra probabilidad predicha en 10 intervalos."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), constrained_layout=True)
    for number, (event, name, ax) in enumerate(zip(EVENTS, EVENT_NAMES, axes)):
        observed, predicted = calibration_curve(y[:, number], probability[:, number], n_bins=10)
        ax.plot([0, 1], [0, 1], "--", color=REFERENCE, linewidth=1, label="Calibración perfecta")
        ax.plot(predicted, observed, "o-", color=BLUE, markersize=6, label="LSTM")
        ax.set(title=f"{name}\nBrier {metrics[event]['test']['brier']:.3f}", xlabel="Probabilidad predicha",
               ylabel="Frecuencia observada", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(loc="upper left")
    fig.suptitle("Calibración en test: bajo la diagonal, el modelo sobrestima la probabilidad")
    return fig


def confusion(y, probability, metrics):
    """Conteos de test con el umbral de val; entre paréntesis, el porcentaje de cada fila real."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), constrained_layout=True)
    for number, (event, name, ax) in enumerate(zip(EVENTS, EVENT_NAMES, axes)):
        threshold = metrics[event]["threshold_recall_val"]
        matrix = confusion_matrix(y[:, number], probability[:, number] >= threshold)
        shares = matrix / matrix.sum(axis=1, keepdims=True)
        ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:,}\n({shares[i, j]:.1%})", ha="center", va="center",
                    color="white" if shares[i, j] > 0.5 else "#0b0b0b")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Sin evento", "Evento"], yticklabels=["Sin evento", "Evento"],
               xlabel="Predicción", ylabel="Real", title=f"{name} · umbral {threshold:.3f}")
        ax.grid(False)
    fig.suptitle("Matrices de confusión en test con el umbral de val")
    return fig


def contributions(frame):
    """Las variables de mayor aporte promedio al log-odds en las alertas de test, por evento."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    for event, name, ax in zip(EVENTS, EVENT_NAMES, axes.ravel()):
        data = frame[frame["event"].eq(event)]
        data = data.iloc[np.argsort(np.abs(data["aporte_medio"].to_numpy()))[-TOP_FEATURES:]]
        weights = data["aporte_medio"].to_numpy()
        ax.barh(data["variable"], weights, color=np.where(weights >= 0, ORANGE, BLUE))
        ax.axvline(0, color=REFERENCE, linewidth=1)
        ax.set(title=name, xlabel="Aporte promedio al log-odds (naranja: aumenta el riesgo; azul: lo reduce)")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Aportes de mayor magnitud en las alertas de test (Integrated Gradients; asociaciones, no causas)")
    return fig


def test_group_comparison(metrics, measures, title):
    """Compara vacas conocidas y nuevas en test con el mismo modelo y el umbral de val."""
    columns = min(3, len(measures))
    rows = (len(measures) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4.5 * rows), squeeze=False)
    positions = np.arange(len(EVENTS))
    groups = (("test_vacas_conocidas", "Conocidas", BLUE), ("test_vacas_nuevas", "Nuevas", ORANGE))
    for ax, (label, key) in zip(axes.flat, measures):
        for offset, (group, name, color) in zip((-0.18, 0.18), groups):
            scores = [metrics[event][group] for event in EVENTS]
            values = [s[key] if key in s else s["threshold_recall"][key] for s in scores]
            counts = metrics[group]
            bars = ax.bar(positions + offset, values, width=0.36, color=color,
                          label=f"{name} (vacas: {counts['n_cows']:,}; sesiones: {counts['n_sessions']:,})")
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
        ax.set(title=label, xticks=positions, xticklabels=EVENT_NAMES)
        ax.tick_params(axis="x", labelrotation=15, labelsize=8)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.20)
        if key not in ("brier", "log_loss"):
            ax.set_ylim(0, 1.12)
    for ax in axes.flat[len(measures):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=10)
    fig.suptitle(title + "\nConocidas: aparecen antes de test; nuevas: primer ordeño registrado en test", fontsize=13)
    fig.tight_layout(rect=(0, 0.08, 1, 0.91))
    return fig


def learning_curves(history, metric):
    """Log loss o accuracy (umbral 0.5) de cada evento por época; la línea punteada es la época restaurada."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
    colors = {"train": BLUE, "val": ORANGE, "test": AQUA}
    labels = {"train": "Train", "val": "Validación", "test": "Test (solo diagnóstico)"}
    ylabel = "Accuracy (umbral 0.5)" if metric == "accuracy" else "Log loss"
    best = int(np.argmax(history["val_f1"])) + 1
    for event, name, ax in zip(EVENTS, EVENT_NAMES, axes):
        for split in ("train", "val", "test"):
            values = history[f"{split}_{metric}_{event}"]
            epochs = np.arange(1, len(values) + 1)
            ax.plot(epochs, values, color=colors[split], marker=".", markersize=3,
                    label=f"{labels[split]}: {values[best - 1]:.4f}")
        ax.axvline(best, color=REFERENCE, linestyle="--", linewidth=1)
        ax.set(title=f"{name}\n{len(epochs)} épocas ejecutadas; restaurada: {best}", xlabel="Época", ylabel=ylabel)
        if metric == "accuracy":
            ax.set_ylim(0, 1)
        ax.legend(loc="best", fontsize=8)
    direction = "mayor es mejor; umbral fijo 0.5" if metric == "accuracy" else "menor es mejor"
    fig.suptitle(f"Avance del ajuste: {ylabel} en entrenamiento, validación y test ({direction})\n"
                 "Early stopping por F1 promedio de validación", fontsize=12)
    return fig


def save_plots(metrics, history, y_test, probability_test, contribution_data, output_dir):
    """Guarda las gráficas en output_dir con los mismos nombres que la regresión logística."""
    figures = Path(output_dir)
    figures.mkdir(parents=True, exist_ok=True)
    builders = (
        ("roc_pr_test", roc_pr, (y_test, probability_test, metrics)),
        ("calibracion_test", calibration, (y_test, probability_test, metrics)),
        ("matrices_confusion_test", confusion, (y_test, probability_test, metrics)),
        ("aportes_evaluacion", contributions, (contribution_data,)),
        ("comparacion_test_metricas", test_group_comparison, (metrics, (
            ("ROC-AUC (mayor es mejor)", "roc_auc"), ("PR-AUC (mayor es mejor)", "pr_auc"),
            ("Precisión (mayor es mejor)", "precision"), ("Sensibilidad (mayor es mejor)", "recall"),
            ("F1 (mayor es mejor)", "f1"), ("Exactitud (mayor es mejor)", "accuracy"),
        ), "Comparación de métricas en test: mismo modelo y umbral elegido en validación")),
        ("comparacion_test_perdidas", test_group_comparison, (metrics, (
            ("Brier (menor es mejor)", "brier"), ("Log loss (menor es mejor)", "log_loss"),
        ), "Comparación de pérdidas en test: menor es mejor")),
        ("curvas_aprendizaje", learning_curves, (history, "log_loss")),
        ("curvas_accuracy", learning_curves, (history, "accuracy")),
    )
    for name, build, args in builders:
        figure = build(*args)
        figure.savefig(figures / f"{name}.png", dpi=150, bbox_inches="tight")
        plt.close(figure)
        print(f"[figure] {name}.png", flush=True)
