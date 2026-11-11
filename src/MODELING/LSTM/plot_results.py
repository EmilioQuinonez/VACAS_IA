"""
Gráficas de la LSTM base. Carga el modelo guardado por main.py, vuelve a armar las mismas
entradas y predice val y test; no reentrena. Las gráficas van a OUTPUT_DIR/graficas:
- curvas_aprendizaje.png: pérdida y accuracy por época en train y val.
- roc_pr_test.png, calibracion_test.png, matrices_confusion_test.png y
  distribucion_probabilidades.png: test con el umbral de recall elegido en val.
- desempeno_semanal_test.png: precisión y recall de cada semana de test.
- vacas_nuevas_metricas.png: test separado en vacas conocidas, solo en val y nuevas.
- importancia_permutacion.png: cuánto baja el PR-AUC de val al revolver cada variable.

Requiere correr antes src/MODELING/LSTM/main.py.

Uso, desde la raíz del repo:
    python -u src/MODELING/LSTM/plot_results.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_curve
from tensorflow import keras

from dataset import (
    CONTEXT_FEATURES, EVENTS, SEQUENCE_FEATURES, build_inputs, load_sessions, split_by_time, split_test_by_cow,
)
from main import BATCH_SIZE, INPUT_PATH, K, OUTPUT_DIR, RANDOM_STATE, ROOT, TRAIN_FRACTION, VAL_FRACTION

FIGURES_DIR = OUTPUT_DIR / "graficas"
EVENT_NAMES = ["Mecánico", "Salud", "Conducta", "Alguna anomalía"]
MIN_WEEK_SESSIONS = 1000  # Semanas incompletas en los bordes de test no se grafican.
TOP_FEATURES = 10

BLUE, ORANGE, AQUA, REFERENCE = "#2a78d6", "#eb6834", "#1baf7a", "#9a9890"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9a9890",
    "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8, "lines.linewidth": 2,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
})


def learning_curves(history):
    """Pérdida y accuracy por época; la línea punteada marca la época que se restauró."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    epochs = np.arange(1, len(history["loss"]) + 1)
    best = int(np.argmin(history["val_loss"])) + 1
    for ax, key, label in [(axes[0], "loss", "Pérdida (binary cross-entropy)"),
                           (axes[1], "accuracy", "Accuracy binaria (umbral 0.5)")]:
        ax.plot(epochs, history[key], color=BLUE, label=f"Train: {history[key][best - 1]:.4f}")
        ax.plot(epochs, history[f"val_{key}"], color=ORANGE, label=f"Val: {history[f'val_{key}'][best - 1]:.4f}")
        ax.axvline(best, color=REFERENCE, linestyle="--", linewidth=1)
        ax.set(title=label, xlabel="Época", ylabel=label)
        ax.legend(title=f"Época restaurada: {best}", loc="best")
    fig.suptitle("Curvas de aprendizaje de la LSTM (las cuatro salidas juntas)")
    return fig


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
        ax.set(title=f"{name}\nROC-AUC {result['roc_auc']:.3f}", xlabel="Tasa de falsos positivos", ylabel="Recall")

        ax = axes[1, number]
        precision, recall, _ = precision_recall_curve(actual, score)
        ax.plot(recall, precision, color=BLUE)
        ax.plot(decision["recall"], decision["precision"], "o", color=ORANGE, markersize=8)
        ax.axhline(result["prevalence"], color=REFERENCE, linestyle="--", linewidth=1)
        ax.set(title=f"PR-AUC {result['pr_auc']:.3f} · prevalencia {result['prevalence']:.1%}",
               xlabel="Recall", ylabel="Precisión", ylim=(0, 1.02))
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("Curvas ROC y precisión-recall en test (la línea punteada es el azar)")
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


def probability_distributions(y, probability, metrics):
    """Histograma de la probabilidad por clase real; a la derecha del umbral se emite alerta."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), constrained_layout=True)
    bins = np.linspace(0, 1, 41)
    for number, (event, name, ax) in enumerate(zip(EVENTS, EVENT_NAMES, axes)):
        actual, score = y[:, number], probability[:, number]
        ax.hist(score[actual == 0], bins=bins, density=True, color=BLUE, alpha=0.6, label="Sin evento")
        ax.hist(score[actual == 1], bins=bins, density=True, color=ORANGE, alpha=0.6, label="Con evento")
        ax.axvline(metrics[event]["threshold_recall_val"], color="#0b0b0b", linestyle="--", linewidth=1, label="Umbral de val")
        ax.set(title=name, xlabel="Probabilidad predicha", ylabel="Densidad", xlim=(0, 1))
    axes[0].legend(loc="upper right")
    fig.suptitle("Probabilidades en test por clase real")
    return fig


def weekly_performance(y, probability, start, metrics):
    """Precisión y recall de cada semana de test con el umbral fijo de val."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), constrained_layout=True, sharey=True)
    week = start.dt.to_period("W").dt.start_time.to_numpy()
    for number, (event, name, ax) in enumerate(zip(EVENTS, EVENT_NAMES, axes)):
        alert = probability[:, number] >= metrics[event]["threshold_recall_val"]
        actual = y[:, number] == 1
        counts = pd.DataFrame({"aciertos": actual & alert, "alertas": alert, "eventos": actual, "ordenos": 1}).groupby(week).sum()
        counts = counts[counts["ordenos"] >= MIN_WEEK_SESSIONS]
        ax.plot(counts.index, counts["aciertos"] / counts["eventos"], "o-", color=BLUE, markersize=5, label="Recall")
        ax.plot(counts.index, counts["aciertos"] / counts["alertas"], "o-", color=ORANGE, markersize=5, label="Precisión")
        ax.axhline(0.80, color=REFERENCE, linestyle="--", linewidth=1)
        ax.set(title=name, xlabel="Semana", ylim=(0, 1))
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    axes[0].set_ylabel("Valor")
    axes[0].legend(loc="lower left")
    fig.suptitle(f"Desempeño semanal en test con el umbral de val (semanas con ≥ {MIN_WEEK_SESSIONS:,} ordeños; "
                 "línea punteada: recall objetivo 80 %)")
    return fig


def metrics_by_cow_group(metrics, groups, cows):
    """Mismas métricas en test para vacas conocidas, solo en val y nuevas (umbral de val)."""
    measures = [("ROC-AUC", "roc_auc"), ("PR-AUC", "pr_auc"), ("Precisión", "precision"),
                ("Recall", "recall"), ("F1", "f1")]
    labels = [("test_vacas_conocidas", "Conocidas (en train)", BLUE),
              ("test_vacas_solo_val", "Solo en val", AQUA),
              ("test_vacas_nuevas", "Nuevas (primer ordeño en test)", ORANGE)]
    fig, axes = plt.subplots(1, len(measures), figsize=(24, 4.8), constrained_layout=True)
    positions = np.arange(len(EVENTS))
    for ax, (label, key) in zip(axes, measures):
        for offset, (group, name, color) in zip((-0.27, 0, 0.27), labels):
            scores = [metrics[e][group] for e in EVENTS]
            values = [s[key] if key in s else s["threshold_recall"][key] for s in scores]
            rows = groups[group]
            bars = ax.bar(positions + offset, values, width=0.27, color=color, edgecolor="#fcfcfb", linewidth=1.5,
                          label=f"{name}: {cows.iloc[rows].nunique()} vacas, {len(rows):,} ordeños")
            ax.bar_label(bars, fmt="%.2f", fontsize=6.5, padding=2)
        ax.set(title=label, xticks=positions, ylim=(0, 1.1))
        ax.set_xticklabels(EVENT_NAMES, rotation=15, fontsize=8)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Desempeño en test por grupo de vaca, con el umbral de val")
    return fig


def permutation_importance(model, seq, ctx, y):
    """
    Caída del PR-AUC de val al revolver una variable entre sesiones (en todos los pasos si es de la
    secuencia). Los pasos de relleno siguen en 0 para que Masking los siga ignorando.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    order = rng.permutation(len(y))
    real_step = (seq != 0).any(axis=2)

    def pr_auc(inputs):
        probability = model.predict(inputs, batch_size=BATCH_SIZE, verbose=0)
        return np.array([average_precision_score(y[:, n], probability[:, n]) for n in range(len(EVENTS))])

    baseline = pr_auc([seq, ctx])
    drops = {}
    for j, feature in enumerate(SEQUENCE_FEATURES):
        shuffled = seq.copy()
        shuffled[:, :, j] = seq[order, :, j] * real_step
        drops[(feature, "secuencia")] = baseline - pr_auc([shuffled, ctx])
        print(f"  [importance] {feature}: {np.round(drops[(feature, 'secuencia')], 4)}", flush=True)
    for j, feature in enumerate(CONTEXT_FEATURES):
        shuffled = ctx.copy()
        shuffled[:, j] = ctx[order, j]
        drops[(feature, "contexto")] = baseline - pr_auc([seq, shuffled])
        print(f"  [importance] {feature}: {np.round(drops[(feature, 'contexto')], 4)}", flush=True)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), constrained_layout=True)
    for number, (name, ax) in enumerate(zip(EVENT_NAMES, axes)):
        top = sorted(drops, key=lambda key: drops[key][number])[-TOP_FEATURES:]
        ax.barh([feature for feature, _ in top], [drops[key][number] for key in top],
                color=[BLUE if part == "secuencia" else ORANGE for _, part in top])
        ax.axvline(0, color=REFERENCE, linewidth=1)
        ax.set(title=f"{name}\nPR-AUC de val sin revolver: {baseline[number]:.3f}", xlabel="Caída del PR-AUC")
        ax.tick_params(axis="y", labelsize=8)
        ax.grid(axis="y", visible=False)
    axes[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
                   labels=["Paso de la secuencia", "Contexto de la sesión"], loc="lower right")
    fig.suptitle(f"Importancia por permutación en val: las {TOP_FEATURES} variables que más bajan el PR-AUC al revolverlas")
    return fig


def save(fig, name):
    fig.savefig(FIGURES_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {name}", flush=True)


def main():
    metrics = json.loads((OUTPUT_DIR / "metrics.json").read_text(encoding="utf-8"))
    history = json.loads((OUTPUT_DIR / "history.json").read_text(encoding="utf-8"))
    model = keras.models.load_model(OUTPUT_DIR / "lstm.keras")

    sessions = load_sessions(INPUT_PATH)
    idx = split_by_time(sessions["inicio"], TRAIN_FRACTION, VAL_FRACTION)
    X_seq, X_ctx, y = build_inputs(sessions, idx["train"], K)
    test = idx["test"]
    probability = model.predict([X_seq[test], X_ctx[test]], batch_size=BATCH_SIZE, verbose=0)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    groups = split_test_by_cow(sessions["id_vaca"], idx)

    save(learning_curves(history), "curvas_aprendizaje.png")
    save(roc_pr(y[test], probability, metrics), "roc_pr_test.png")
    save(calibration(y[test], probability, metrics), "calibracion_test.png")
    save(confusion(y[test], probability, metrics), "matrices_confusion_test.png")
    save(probability_distributions(y[test], probability, metrics), "distribucion_probabilidades.png")
    save(weekly_performance(y[test], probability, sessions["inicio"].iloc[test], metrics), "desempeno_semanal_test.png")
    save(metrics_by_cow_group(metrics, groups, sessions["id_vaca"]), "vacas_nuevas_metricas.png")
    print("\n[importance] permutation on val")
    val = idx["val"]
    save(permutation_importance(model, X_seq[val], X_ctx[val], y[val]), "importancia_permutacion.png")
    print(f"\nFigures in {FIGURES_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
