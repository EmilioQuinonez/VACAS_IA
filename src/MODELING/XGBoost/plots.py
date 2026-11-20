"""Gráficas del XGBoost: cada función arma una figura y plot_results.py la guarda."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score, precision_score, recall_score

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
    """1. Log loss por árbol: train y val del corte cronológico."""
    fig, axes = plt.subplots(1, len(EVENTS), figsize=(17.5, 4), sharey=True)
    for event, ax in enumerate(axes):
        chrono = log[log["event"] == event]
        series = [
            ("Train", chrono["tree"], chrono["loss"], SERIES["train"]),
            (val_label, chrono["tree"], chrono["val_loss"], SERIES["val"]),
        ]
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
    fig.suptitle("Calibración en test (Platt ajustado en val): bajo la diagonal, el modelo sobrestima",
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
    """5. Los atributos con más gain de cada modelo."""
    gains = [pd.Series(model.get_booster().get_score(importance_type="gain")) for model in models]
    return bars_by_event([gain / gain.sum() for gain in gains], "Proporción del gain total",
                         lambda value: f"{value:.0%}",
                         f"Importancia por gain: los {TOP_FEATURES} atributos que más usa cada modelo")


def bars_by_event(series, xlabel, label, title):
    """Barras horizontales de los TOP_FEATURES valores más altos de cada salida, con etiqueta al final."""
    fig, axes = plt.subplots(1, len(EVENTS), figsize=(18.5, 4.8))
    for i, values in enumerate(series):
        top = values.sort_values(ascending=False).head(TOP_FEATURES)[::-1]
        ax = axes[i]
        ax.barh(top.index, top.to_numpy(), color=SERIES["train"], height=0.6)
        for y_pos, value in enumerate(top.to_numpy()):
            ax.annotate(label(value), (value, y_pos), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=9, color=MUTED)
        ax.set_title(EVENT_NAMES[i], loc="left")
        ax.set_xlabel(xlabel)
        ax.set_xlim(0, top.max() * 1.2)
        ax.grid(axis="y", visible=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12)
    return fig


def shap_importance(contributions):
    """12. Aporte SHAP medio (en valor absoluto) de cada atributo en test."""
    return bars_by_event([c.abs().mean() for c in contributions], "Aporte medio |SHAP| (log-odds)",
                         lambda value: f"{value:.2f}",
                         f"Importancia SHAP en test: los {TOP_FEATURES} atributos que más mueven el riesgo")


def top_factor_share(factors):
    """
    13. Proporción de alertas de test en que cada atributo es uno de sus 3 factores (mayor aporte en
    valor absoluto), separada según si el aporte sube (naranja) o baja (azul) el riesgo.
    """
    fig, axes = plt.subplots(1, len(EVENTS), figsize=(18.5, 4.8))
    for i, f in enumerate(factors):
        names = f.filter(like="factor_").to_numpy().ravel()
        up = f.filter(like="aporte_").to_numpy().ravel() > 0
        share = pd.DataFrame({"sube": pd.Series(names[up]).value_counts(),
                              "baja": pd.Series(names[~up]).value_counts()}).fillna(0) / len(f)
        top = share.assign(total=share.sum(axis=1)).sort_values("total").tail(TOP_FEATURES)
        ax = axes[i]
        ax.barh(top.index, top["sube"], color=SERIES["val"], height=0.6, label="Sube el riesgo")
        ax.barh(top.index, top["baja"], left=top["sube"], color=SERIES["train"], height=0.6, label="Baja el riesgo")
        for y_pos, value in enumerate(top["total"]):
            ax.annotate(f"{value:.0%}", (value, y_pos), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=9, color=MUTED)
        ax.set_title(EVENT_NAMES[i], loc="left")
        ax.set_xlabel("Proporción de alertas")
        ax.set_xlim(0, top["total"].max() * 1.2)
        ax.grid(axis="y", visible=False)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", ncol=2)
    fig.suptitle(f"Factores de las alertas en test: los {TOP_FEATURES} atributos que más aparecen entre "
                 "los 3 de mayor aporte", x=0.01, ha="left", fontsize=12)
    return fig


def weekly_performance(y, start, probabilities, metrics):
    """7. Sensibilidad, precisión y prevalencia por semana en test."""
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
            ("Sensibilidad", weekly["hits"] / weekly["events"], SERIES["train"], "-"),
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
    """8. Con el umbral de val, F1, sensibilidad y precisión en test por grupo de vaca, y su Brier."""
    names = ["F1", "Sensibilidad", "Precisión", "Brier"]
    width = 0.26
    fig, axes = plt.subplots(1, len(EVENTS), figsize=(18.5, 4.4), sharey=True)
    for i, event in enumerate(EVENTS):
        ax, threshold = axes[i], metrics[event]["threshold_recall_val"]
        for j, (group, color) in enumerate(COW_GROUPS.items()):
            rows = groups == group
            actual, p = y[event].to_numpy()[rows], probabilities[i][rows]
            alert = p >= threshold
            values = [f1_score(actual, alert), recall_score(actual, alert),
                      precision_score(actual, alert, zero_division=0), brier_score_loss(actual, p)]
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
    fig.suptitle("Desempeño en test por grupo de vaca (Brier: más bajo es mejor)", x=0.01, ha="left", fontsize=12)
    return fig


def new_cow_ramp(y, probabilities, days_in_herd, groups, metrics):
    """
    9. Sensibilidad, precisión y prevalencia de las vacas que no estuvieron en train según sus días en el
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
            ("Sensibilidad", by_bin["hits"] / by_bin["events"], SERIES["train"], "-",
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
    10. Calibración en test (arriba) y probabilidad de los ordeños con evento (abajo) de las vacas
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
    """11. Matriz de confusión en test con el umbral elegido en val; el color es el % de cada fila."""
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
