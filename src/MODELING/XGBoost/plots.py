"""
Gráficas del XGBoost base; cada función devuelve la figura y plot_results.py la guarda.

Editar: títulos y etiquetas dentro de cada función; nombres de los eventos en EVENT_NAMES;
colores en SERIES (papel de cada serie, igual en todas las gráficas), INK, MUTED, GRID,
SURFACE y REFERENCE.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, brier_score_loss, confusion_matrix, precision_recall_curve,
    precision_score, recall_score, roc_auc_score, roc_curve,
)

from dataset import EVENTS

EVENT_NAMES = ["Falla mecánica", "Salud", "Conducta", "Alguna anomalía"]
TOP_FEATURES = 10
MIN_WEEK_SESSIONS = 1000  # semanas incompletas (inicio y fin del test) se omiten

# Paleta validada (skill dataviz, modo claro): el color marca el papel de la serie, no el evento.
SERIES = {"train": "#2a78d6", "val": "#eb6834", "cows": "#1baf7a"}
# Grupo de la vaca en test según dónde apareció antes (ver plot_results.load_test).
COW_GROUPS = {"Conocida": SERIES["train"], "Solo en val": SERIES["cows"], "Nueva": SERIES["val"]}
DAY_BINS = ["0–30", "31–90", "> 90"]  # días en el hato; 0–7 tiene muy pocas sesiones y va en 0–30
INK, MUTED, GRID, SURFACE, REFERENCE = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb", "#9a9890"
SEQUENTIAL = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # rampa azul de 100 a 700 (matriz de confusión)

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "lines.linewidth": 2,
    "legend.frameon": False,
})


def label_line_end(ax, x, y, text, color):
    """Etiqueta directa al final de la línea (el aqua no llega a 3:1 de contraste)."""
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=9, color=MUTED)
    ax.plot([x], [y], "o", color=color, markersize=4)


def learning_curves(log, n_trees, val_label):
    """1. Log loss por árbol: train y val del corte cronológico y, si hay, vacas fuera (media de los folds)."""
    fig, axes = plt.subplots(1, len(EVENTS), figsize=(17.5, 4), sharey=True)
    for event, ax in enumerate(axes):
        chrono = log[(log["event"] == event) & (log["block"] == 0)]
        by_cow = log[(log["event"] == event) & log["by_cow"]]
        series = [
            ("Train", chrono["tree"], chrono["loss"], SERIES["train"]),
            (val_label, chrono["tree"], chrono["val_loss"], SERIES["val"]),
        ]
        if len(by_cow):
            folds = by_cow.groupby("tree")["val_loss"].mean()
            n_folds = by_cow["block"].nunique()
            series.append((f"Vacas fuera (media {n_folds} folds)", folds.index, folds, SERIES["cows"]))
        for name, x, y, color in series:
            x, y = x.to_numpy(), y.to_numpy()
            ax.plot(x, y, color=color, label=name)
            label_line_end(ax, x[-1], y[-1], f"{y[-1]:.3f}", color)
        ax.set_title(EVENT_NAMES[event], loc="left")
        ax.set_xlabel("Árbol")
        ax.set_xlim(0, n_trees + 14)
    axes[0].set_ylabel("Log loss")
    axes[0].legend(loc="upper right")
    fig.suptitle("Curvas de aprendizaje: log loss por árbol (menor es mejor)",
                 x=0.01, ha="left", fontsize=12)
    return fig


def roc_pr(y, probabilities, metrics, test_period):
    """2. Curvas ROC y PR en test con el punto del umbral elegido en val."""
    fig, axes = plt.subplots(2, len(EVENTS), figsize=(17.5, 8))
    for i, event in enumerate(EVENTS):
        actual, p, m = y[event].to_numpy(), probabilities[i], metrics[event]
        threshold = m["threshold_recall_val"]
        predicted = p >= threshold
        fpr_t = (predicted & (actual == 0)).sum() / (actual == 0).sum()
        tpr_t = (predicted & (actual == 1)).sum() / (actual == 1).sum()
        precision_t = (predicted & (actual == 1)).sum() / predicted.sum()

        ax = axes[0, i]
        fpr, tpr, _ = roc_curve(actual, p)
        ax.plot([0, 1], [0, 1], color=REFERENCE, linestyle="--", linewidth=1, label="Azar / prevalencia")
        ax.plot(fpr, tpr, color=SERIES["train"], label="Modelo")
        ax.plot(fpr_t, tpr_t, "o", color=SERIES["val"], markersize=8, label="Umbral elegido en val")
        ax.set_title(f"{EVENT_NAMES[i]}\nROC-AUC {m['test']['roc_auc']:.3f} · umbral {threshold:.2f}", loc="left")
        ax.set_xlabel("Falsos positivos (1 − especificidad)")

        ax = axes[1, i]
        precision, recall, _ = precision_recall_curve(actual, p)
        ax.axhline(actual.mean(), color=REFERENCE, linestyle="--", linewidth=1)
        ax.plot(recall, precision, color=SERIES["train"])
        ax.plot(tpr_t, precision_t, "o", color=SERIES["val"], markersize=8)
        ax.set_title(f"PR-AUC {m['test']['pr_auc']:.3f} · prevalencia {actual.mean():.1%}\n"
                     f"en el umbral: recall {tpr_t:.0%}, precisión {precision_t:.0%}", loc="left")
        ax.set_xlabel("Recall (sensibilidad)")
        ax.set_ylim(0, 1)
    axes[0, 0].set_ylabel("Verdaderos positivos (recall)")
    axes[1, 0].set_ylabel("Precisión")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper right", ncol=3)
    fig.suptitle(f"Curvas ROC (arriba) y PR (abajo) en test, {test_period}", x=0.01, ha="left", fontsize=12)
    return fig


def calibration(y, probabilities, metrics):
    """3. Probabilidad predicha contra frecuencia real en test."""
    fig, axes = plt.subplots(1, len(EVENTS), figsize=(17.5, 4.5), sharey=True)
    for i, event in enumerate(EVENTS):
        actual, p, ax = y[event].to_numpy(), probabilities[i], axes[i]
        observed, predicted = calibration_curve(actual, p, n_bins=10)
        constant = actual.mean() * (1 - actual.mean())
        ax.plot([0, 1], [0, 1], color=REFERENCE, linestyle="--", linewidth=1, label="Calibración perfecta")
        ax.plot(predicted, observed, "o-", color=SERIES["train"], markersize=5, label="Modelo")
        ax.set_title(f"{EVENT_NAMES[i]}\nBrier {metrics[event]['test']['brier']:.3f} (constante {constant:.3f})", loc="left")
        ax.set_xlabel("Probabilidad predicha")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Frecuencia real del evento")
    axes[0].legend(loc="upper left")
    fig.suptitle("Calibración en test: bajo la diagonal, el modelo sobrestima la probabilidad",
                 x=0.01, ha="left", fontsize=12)
    return fig


def probability_distributions(y, probabilities, metrics):
    """4. Distribución de la probabilidad en test por clase real."""
    fig, axes = plt.subplots(1, len(EVENTS), figsize=(17.5, 4))
    bins = np.linspace(0, 1, 41)
    for i, event in enumerate(EVENTS):
        actual, p, ax = y[event].to_numpy(), probabilities[i], axes[i]
        ax.hist(p[actual == 0], bins=bins, density=True, histtype="step", linewidth=2,
                color=SERIES["train"], label="Sin evento")
        ax.hist(p[actual == 1], bins=bins, density=True, histtype="step", linewidth=2,
                color=SERIES["val"], label="Con evento")
        threshold = metrics[event]["threshold_recall_val"]
        ax.axvline(threshold, color=REFERENCE, linestyle="--", linewidth=1)
        ax.annotate(f"umbral {threshold:.2f}", (threshold, ax.get_ylim()[1] * 0.95), xytext=(4, 0),
                    textcoords="offset points", fontsize=9, color=MUTED, va="top")
        ax.set_title(EVENT_NAMES[i], loc="left")
        ax.set_xlabel("Probabilidad predicha")
    axes[0].set_ylabel("Densidad")
    axes[0].legend(loc="upper right")
    fig.suptitle("Probabilidades en test: a la derecha del umbral se emite alerta", x=0.01, ha="left", fontsize=12)
    return fig


def gain_importance(models):
    """5. Los atributos con más gain de cada modelo final."""
    fig, axes = plt.subplots(1, len(EVENTS), figsize=(18.5, 4.8))
    for i, model in enumerate(models):
        gain = pd.Series(model.get_booster().get_score(importance_type="gain"))
        share = (gain / gain.sum()).sort_values(ascending=False).head(TOP_FEATURES)[::-1]
        ax = axes[i]
        ax.barh(share.index, share.to_numpy(), color=SERIES["train"], height=0.6)
        for y_pos, value in enumerate(share.to_numpy()):
            ax.annotate(f"{value:.0%}", (value, y_pos), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=9, color=MUTED)
        ax.set_title(EVENT_NAMES[i], loc="left")
        ax.set_xlabel("Proporción del gain total")
        ax.set_xlim(0, share.max() * 1.2)
        ax.grid(axis="y", visible=False)
    fig.suptitle(f"Importancia por gain: los {TOP_FEATURES} atributos que más usa cada modelo final",
                 x=0.01, ha="left", fontsize=12)
    return fig


def model_vs_feature(rates, y, probabilities):
    """
    6. ROC del modelo contra la tasa del evento de la vaca en 30 días usada sola; rates trae una
    columna por salida, en el orden de EVENTS.
    """
    fig, axes = plt.subplots(1, len(EVENTS), figsize=(17.5, 4.5), sharey=True)
    for i, event in enumerate(EVENTS):
        feature = rates.columns[i]
        known = rates[feature].notna().to_numpy()
        actual, p, f = y[event].to_numpy()[known], probabilities[i][known], rates[feature].to_numpy()[known]
        ax = axes[i]
        ax.plot([0, 1], [0, 1], color=REFERENCE, linestyle="--", linewidth=1, label="Azar")
        for name, score, color in [("Modelo", p, SERIES["train"]), (feature, f, SERIES["val"])]:
            fpr, tpr, _ = roc_curve(actual, score)
            ax.plot(fpr, tpr, color=color, label=f"{name} (AUC {roc_auc_score(actual, score):.3f})")
        ax.set_title(EVENT_NAMES[i], loc="left")
        ax.set_xlabel("Falsos positivos")
        ax.legend(loc="lower right")
    axes[0].set_ylabel("Verdaderos positivos")
    fig.suptitle("Modelo contra una sola variable en test (sesiones con tasa a 30 días): "
                 "en mecánica y conducta la iguala; en salud y alguna anomalía el modelo sí aporta",
                 x=0.01, ha="left", fontsize=12)
    return fig


def by_cow(metrics):
    """7. Métricas de cada fold por vaca, su media y el test cronológico como referencia."""
    names = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "recall": "Recall"}
    fig, axes = plt.subplots(1, len(EVENTS), figsize=(17.5, 4.2), sharey=True)
    for i, event in enumerate(EVENTS):
        folds, ax = pd.DataFrame(metrics[event]["by_cow"]["folds"]), axes[i]
        test = metrics[event]["test"]
        chrono = {"roc_auc": test["roc_auc"], "pr_auc": test["pr_auc"],
                  "recall": test["threshold_recall"]["recall"]}
        for x, key in enumerate(names):
            jitter = np.linspace(-0.12, 0.12, len(folds))
            ax.plot(x + jitter, folds[key], "o", color=SERIES["cows"], markersize=6, alpha=0.8,
                    label="Fold por vaca" if x == 0 else None)
            ax.plot([x - 0.22, x + 0.22], [folds[key].mean()] * 2, color=SERIES["cows"], linewidth=2.5,
                    label="Media de los folds" if x == 0 else None)
            ax.plot(x + 0.32, chrono[key], "D", color=SERIES["val"], markersize=7,
                    label="Test cronológico" if x == 0 else None)
            ax.annotate(f"{folds[key].mean():.2f}", (x - 0.22, folds[key].mean()), xytext=(-4, 0),
                        textcoords="offset points", ha="right", va="center", fontsize=9, color=MUTED)
        ax.set_xticks(range(len(names)), names.values())
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_ylim(0.3, 1)
        ax.set_title(EVENT_NAMES[i], loc="left")
        ax.grid(axis="x", visible=False)
    axes[1].legend(loc="lower left")
    fig.suptitle(f"Corte por vaca ({len(folds)} folds): el modelo rinde parecido con vacas que no vio",
                 x=0.01, ha="left", fontsize=12)
    return fig


def weekly_performance(y, start, probabilities, metrics):
    """8. Recall, precisión y prevalencia por semana en test."""
    week = start.dt.to_period("W-SUN").dt.start_time.to_numpy()
    fig, axes = plt.subplots(1, len(EVENTS), figsize=(18.5, 4.2), sharey=True)
    for i, event in enumerate(EVENTS):
        frame = pd.DataFrame({
            "week": week, "actual": y[event].to_numpy(),
            "alert": probabilities[i] >= metrics[event]["threshold_recall_val"],
        })
        frame["hit"] = frame["actual"].eq(1) & frame["alert"]
        weekly = frame.groupby("week").agg(n=("actual", "size"), events=("actual", "sum"),
                                           alerts=("alert", "sum"), hits=("hit", "sum"))
        weekly = weekly[weekly["n"] >= MIN_WEEK_SESSIONS]
        series = [
            ("Recall", weekly["hits"] / weekly["events"], SERIES["train"], "-"),
            ("Precisión", weekly["hits"] / weekly["alerts"], SERIES["val"], "-"),
            ("Prevalencia", weekly["events"] / weekly["n"], REFERENCE, "--"),
        ]
        ax = axes[i]
        for name, values, color, style in series:
            ax.plot(values.index, values * 100, style, color=color, marker="o", markersize=4, label=name)
            ax.annotate(f"{name} {values.mean():.0%}", (values.index[-1], values.iloc[-1] * 100), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=MUTED)
        ax.set_title(EVENT_NAMES[i], loc="left")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", labelrotation=45)
        ax.margins(x=0.25)
    axes[0].set_ylabel("%")
    axes[0].legend(loc="center left")
    fig.suptitle(f"Desempeño semanal en test con el umbral de val (semanas con ≥ {MIN_WEEK_SESSIONS:,} ordeños; "
                 "promedio en la etiqueta)", x=0.01, ha="left", fontsize=12)
    return fig


def metrics_by_cow_group(y, probabilities, groups, metrics):
    """9. ROC-AUC, PR-AUC y, con el umbral de val, recall y precisión en test por grupo de vaca."""
    names = ["ROC-AUC", "PR-AUC", "Recall", "Precisión"]
    width = 0.26
    fig, axes = plt.subplots(1, len(EVENTS), figsize=(18.5, 4.4), sharey=True)
    for i, event in enumerate(EVENTS):
        ax, threshold = axes[i], metrics[event]["threshold_recall_val"]
        for j, (group, color) in enumerate(COW_GROUPS.items()):
            rows = groups == group
            actual, p = y[event].to_numpy()[rows], probabilities[i][rows]
            alert = p >= threshold
            values = [roc_auc_score(actual, p), average_precision_score(actual, p),
                      recall_score(actual, alert), precision_score(actual, alert)]
            x = np.arange(len(names)) + (j - 1) * width
            ax.bar(x, values, width, color=color, label=f"{group} ({rows.sum():,} ordeños)")
            for xi, value in zip(x, values):
                ax.annotate(f"{value:.2f}", (xi, value), xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=7, color=MUTED)
        ax.set_xticks(range(len(names)), names)
        ax.set_ylim(0, 1.05)
        ax.set_title(EVENT_NAMES[i], loc="left")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Valor en test")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=3)
    fig.suptitle("Desempeño en test por grupo de vaca", x=0.01, ha="left", fontsize=12)
    return fig


def new_cow_ramp(y, probabilities, days_in_herd, groups, metrics):
    """
    10. Recall, precisión y prevalencia de las vacas que no estuvieron en train según sus días en el
    hato; la línea punteada es el valor de las vacas conocidas.
    """
    new = groups != "Conocida"
    day_bin = pd.cut(days_in_herd[new], [-np.inf, 30, 90, np.inf], labels=DAY_BINS)
    fig, axes = plt.subplots(1, len(EVENTS), figsize=(18.5, 4.4), sharey=True)
    for i, event in enumerate(EVENTS):
        actual, alert = y[event].to_numpy(), probabilities[i] >= metrics[event]["threshold_recall_val"]
        frame = pd.DataFrame({"bin": day_bin, "actual": actual[new], "alert": alert[new]})
        frame["hit"] = frame["actual"].eq(1) & frame["alert"]
        by_bin = frame.groupby("bin", observed=False).agg(n=("actual", "size"), events=("actual", "sum"),
                                                          alerts=("alert", "sum"), hits=("hit", "sum"))
        known_actual, known_alert = actual[~new] == 1, alert[~new]
        series = [
            ("Recall", by_bin["hits"] / by_bin["events"], SERIES["train"], "-",
             (known_actual & known_alert).sum() / known_actual.sum()),
            ("Precisión", by_bin["hits"] / by_bin["alerts"], SERIES["val"], "-",
             (known_actual & known_alert).sum() / known_alert.sum()),
            ("Prevalencia", by_bin["events"] / by_bin["n"], REFERENCE, "--", known_actual.mean()),
        ]
        ax, x = axes[i], np.arange(len(DAY_BINS))
        for name, values, color, style, known in series:
            ax.plot(x, values * 100, style, color=color, marker="o", markersize=5, label=name)
            ax.axhline(known * 100, color=color, linestyle=":", linewidth=1.2)
            for xi, value in zip(x, values):
                ax.annotate(f"{value:.0%}", (xi, value * 100), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=8, color=MUTED)
        ax.set_xticks(x, [f"{label}\n{n:,} ordeños" for label, n in zip(DAY_BINS, by_bin["n"])])
        ax.set_xlim(-0.3, len(DAY_BINS) - 0.5)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Días en el hato")
        ax.set_title(EVENT_NAMES[i], loc="left")
    axes[0].plot([], [], ":", color=MUTED, label="Vacas conocidas")
    axes[0].set_ylabel("%")
    axes[0].legend(loc="center left")
    fig.suptitle(f"Vacas que no estuvieron en train ({new.sum():,} ordeños de test) según sus días en el hato, "
                 "con el umbral de val", x=0.01, ha="left", fontsize=12)
    return fig


def probabilities_by_cow_group(y, probabilities, groups, metrics):
    """
    11. Calibración en test (arriba) y probabilidad de los ordeños con evento (abajo) de las vacas
    conocidas y de las que no estuvieron en train.
    """
    bins = np.linspace(0, 1, 26)
    sets = [("Conocidas", groups == "Conocida", SERIES["train"]),
            ("No vistas en train", groups != "Conocida", SERIES["val"])]
    fig, axes = plt.subplots(2, len(EVENTS), figsize=(17.5, 8))
    for i, event in enumerate(EVENTS):
        actual, p = y[event].to_numpy(), probabilities[i]
        threshold = metrics[event]["threshold_recall_val"]
        top, bottom = axes[0, i], axes[1, i]
        top.plot([0, 1], [0, 1], color=REFERENCE, linestyle="--", linewidth=1)
        for name, rows, color in sets:
            observed, predicted = calibration_curve(actual[rows], p[rows], n_bins=10)
            top.plot(predicted, observed, "o-", color=color, markersize=4,
                     label=f"{name} (Brier {brier_score_loss(actual[rows], p[rows]):.3f})")
            events = p[rows & (actual == 1)]
            bottom.hist(events, bins=bins, density=True, histtype="step", linewidth=2, color=color,
                        label=f"{name}: {(events < threshold).mean():.0%} bajo el umbral")
        top.set_title(EVENT_NAMES[i], loc="left")
        top.set_xlabel("Probabilidad predicha")
        top.set_xlim(0, 1)
        top.set_ylim(0, 1)
        top.legend(loc="upper left", fontsize=8)
        bottom.axvline(threshold, color=REFERENCE, linestyle="--", linewidth=1, label=f"Umbral {threshold:.2f}")
        bottom.set_xlabel("Probabilidad predicha (solo ordeños con evento)")
        bottom.legend(loc="upper left", fontsize=8)
    axes[0, 0].set_ylabel("Frecuencia real del evento")
    axes[1, 0].set_ylabel("Densidad")
    fig.suptitle("Vacas conocidas frente a no vistas en train: calibración (arriba) y probabilidad de los "
                 "ordeños con evento (abajo)", x=0.01, ha="left", fontsize=12)
    return fig


def confusion_matrices(y, probabilities, metrics):
    """12. Matriz de confusión en test con el umbral elegido en val; el color es el % de cada fila."""
    cmap = LinearSegmentedColormap.from_list("azul", [SURFACE, *SEQUENTIAL])
    rows, columns = ["Sin evento", "Con evento"], ["Sin alerta", "Alerta"]
    fig, axes = plt.subplots(1, len(EVENTS), figsize=(17.5, 4.2))
    for i, event in enumerate(EVENTS):
        actual, ax = y[event].to_numpy(), axes[i]
        threshold = metrics[event]["threshold_recall_val"]
        matrix = confusion_matrix(actual, probabilities[i] >= threshold, labels=[0, 1])
        share = matrix / matrix.sum(axis=1, keepdims=True)
        ax.pcolormesh(share, cmap=cmap, vmin=0, vmax=1, edgecolors=SURFACE, linewidth=2)
        for row in range(2):
            for col in range(2):
                color = SURFACE if share[row, col] > 0.55 else INK
                ax.text(col + 0.5, row + 0.5, f"{matrix[row, col]:,}\n{share[row, col]:.0%} de la fila",
                        ha="center", va="center", fontsize=10, color=color)
        ax.set_xticks([0.5, 1.5], columns)
        ax.set_yticks([0.5, 1.5], rows)
        ax.invert_yaxis()
        ax.grid(False)
        ax.tick_params(length=0)
        ax.set_title(f"{EVENT_NAMES[i]} · umbral {threshold:.2f}", loc="left")
        ax.set_xlabel("Predicción")
    axes[0].set_ylabel("Clase real")
    fig.suptitle("Matrices de confusión en test con el umbral elegido en val "
                 "(la diagonal son los aciertos)", x=0.01, ha="left", fontsize=12)
    return fig
