"""Entrena cuatro regresiones con train y las evalúa con validación y test.

Desde la raíz: py src/MODELING/logistic/main.py
Desde esta carpeta: py main.py
"""

from pathlib import Path

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "src.MODELING.logistic"

import joblib
import numpy as np
import pandas as pd

from .dataset import EVENTS, build_features, get_labels, load_sessions, split_by_time
from .evaluation import choose_threshold, evaluate, explanation_coverage
from .logistic_model import LOGISTIC_CONFIG, CowPreprocessor, coefficient_table, explain, train_event
from .plot_results import save_plots

MODULE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = MODULE_DIR
INPUT_PATH = MODULE_DIR.parents[2] / "Data" / "dataset_vacas_integrado.csv"
TRAIN_FRACTION, VAL_FRACTION, TARGET_RECALL = 0.70, 0.15, 0.80


def fit_model(event, transformed, labels, parts, config):
    """Un único entrenamiento incremental del evento con historial por época."""
    print(f"[fit] {event}; configuración global={config}", flush=True)
    actual = {name: labels[event].iloc[rows].to_numpy() for name, rows in parts.items()}
    model, details, history = train_event(transformed, actual, config)
    print(f"  épocas={details['iterations']}; motivo de parada={details['stop_reason']}", flush=True)
    return model, details, history


def print_summary(event, results, threshold):
    def number(value):
        return "no calculable" if value is None else f"{value:.4f}"

    columns = [(
        "Métrica", "Observaciones", "ROC-AUC", "AP", "Brier", "Log loss", "Precisión",
        "Sensibilidad", "F1", "Exactitud", "Exactitud (mayoritaria)",
    )]
    for name, label in (
        ("train", "Entrenamiento"), ("val", "Validación"), ("test", "Prueba completa"),
    ):
        scores = results[name]
        decision = scores["threshold_recall"]
        columns.append((
            label, f"{scores['n']:,}", number(scores["roc_auc"]), number(scores["average_precision"]),
            number(scores["brier"]), number(scores["log_loss"]),
            number(decision["precision"]), number(decision["recall"]),
            number(decision["f1"]), number(decision["accuracy"]),
            number(scores["accuracy_majority"]),
        ))

    widths = [max(len(cell) for cell in column) for column in columns]
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    lines = [f"\n[result] {event}", f"Umbral elegido en validación: {threshold:.6f}", border]
    for index, row in enumerate(zip(*columns)):
        cells = [cell.ljust(widths[0]) if col == 0 else cell.rjust(widths[col]) for col, cell in enumerate(row)]
        lines.append("| " + " | ".join(cells) + " |")
        if index == 0:
            lines.append(border)
    lines.extend((border, ""))
    print("\n".join(lines), flush=True)


def _difference(first, second):
    return None if first is None or second is None else float(first - second)


def run(input_path=INPUT_PATH, output_dir=OUTPUT_DIR, config=None):
    config = dict(LOGISTIC_CONFIG if config is None else config)
    output_dir = Path(output_dir)
    images_dir, models_dir = output_dir / "img", output_dir / "models"
    images_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    sessions = load_sessions(input_path)
    labels = get_labels(sessions)
    parts, split = split_by_time(sessions["inicio"], TRAIN_FRACTION, VAL_FRACTION)
    # Un resultado que todavía no terminó al corte no puede entrenar ni elegir umbrales.
    completion = sessions["inicio"] + pd.to_timedelta(sessions["duracion_seg"], unit="s")
    split["unavailable_labels_purged"] = {}
    for name, boundary in (("train", split["train_end"]), ("val", split["val_end"])):
        rows = parts[name]
        available = completion.iloc[rows].lt(pd.Timestamp(boundary)).to_numpy()
        split["unavailable_labels_purged"][name] = int((~available).sum())
        parts[name] = rows[available]
        print(f"[split] {name}: {len(parts[name]):,} etiquetas disponibles; {int((~available).sum())} aún sin terminar al corte", flush=True)
    split["counts"] = {name: len(rows) for name, rows in parts.items()}
    print("[features] Reutilizando atributos de DP y construyendo historial adicional", flush=True)
    X = build_features(sessions)
    cows = sessions["id_vaca"]
    train = parts["train"]
    preprocessor = CowPreprocessor().fit(X.iloc[train])
    matrix = preprocessor.transform(X)
    transformed = {name: matrix[rows] for name, rows in parts.items()}
    print(f"[features] {X.shape[1]} predictores; {matrix.shape[1]} columnas con indicadores", flush=True)
    test_rows = parts["test"]
    first_seen = sessions.groupby("id_vaca")["inicio"].min()
    new_cows = first_seen.index[first_seen >= pd.Timestamp(split["test_start"])].tolist()
    new_mask = cows.iloc[test_rows].isin(new_cows).to_numpy()
    known_mask = ~new_mask
    known_count = cows.iloc[test_rows].iloc[known_mask].nunique()
    metrics = {
        "status": "fitting", "split": split, "logistic_config": config,
        "new_cows": {"n_cows": len(new_cows), "n_sessions": int(new_mask.sum())},
        "known_cows": {"n_cows": int(known_count), "n_sessions": int(known_mask.sum())},
    }
    models, thresholds, predictions = {}, {}, []
    for event in EVENTS:
        model, parameters, history = fit_model(event, transformed, labels, parts, config)
        if not parameters["converged"]:
            metrics["status"] = "incomplete_convergence"
            metrics[event] = {"parameters": parameters}
            raise RuntimeError(f"{event} agotó max_iter sin cumplir el criterio de parada de train. Ajusta LOGISTIC_CONFIG; no se publica el modelo.")
        models[event] = model
        probabilities = {name: model.predict_proba(data)[:, 1] for name, data in transformed.items()}
        threshold = choose_threshold(labels[event].iloc[parts["val"]], probabilities["val"], TARGET_RECALL)
        thresholds[event] = threshold
        scores = {"threshold_recall_val": threshold, "parameters": parameters, "learning_curve": history}
        for name, rows in parts.items():
            scores[name] = evaluate(labels[event].iloc[rows], probabilities[name], threshold)
        scores["new_cows"] = evaluate(labels[event].iloc[test_rows].to_numpy()[new_mask], probabilities["test"][new_mask], threshold)
        scores["known_cows"] = evaluate(labels[event].iloc[test_rows].to_numpy()[known_mask], probabilities["test"][known_mask], threshold)
        scores["new_cows_difference_from_test"] = {
            key: _difference(scores["new_cows"][key], scores["test"][key])
            for key in ("prevalence", "roc_auc", "average_precision", "brier", "log_loss")
        }
        scores["new_cows_difference_from_test"].update({
            key: _difference(scores["new_cows"]["threshold_recall"][key], scores["test"]["threshold_recall"][key])
            for key in ("accuracy", "precision", "recall", "f1")
        })
        scores["generalization_difference"] = {
            f"{later}_minus_{earlier}": {
                key: _difference(scores[later][key], scores[earlier][key])
                for key in ("average_precision", "roc_auc", "brier", "log_loss")
            }
            for earlier, later in (("train", "val"), ("train", "test"), ("val", "test"))
        }
        frame = sessions.iloc[test_rows][["id_vaca", "inicio"]].reset_index(drop=True)
        frame["event"] = event
        frame["actual"] = labels[event].iloc[test_rows].to_numpy()
        frame["probability"] = probabilities["test"]
        frame["threshold"] = threshold
        frame["prediction"] = (frame["probability"] >= threshold).astype(int)
        alert_positions = np.flatnonzero(frame["prediction"])
        explanation = explain(model, preprocessor, transformed["test"][alert_positions], X.iloc[test_rows[alert_positions]].reset_index(drop=True))
        # Solo las alertas reciben factores; los datos se conservan en memoria.
        explanation.index = alert_positions
        frame = frame.join(explanation)
        for column in [c for c in explanation if c.startswith(("factor_", "valor_", "direccion_"))]:
            frame[column] = frame[column].fillna("")
        scores["test"]["explanations"] = explanation_coverage(frame, frame["probability"], threshold)
        predictions.append(frame)
        metrics[event] = scores
        print_summary(event, scores, threshold)
        auc_gap = _difference(scores["train"]["roc_auc"], scores["test"]["roc_auc"])
        print(f"  Diferencia ROC-AUC (train - test): {'no calculable' if auc_gap is None else f'{auc_gap:.4f}'}", flush=True)

    anomaly = metrics["alguna_anomalia"]["test"]
    metrics["business_criteria"] = {
        "anomaly_recall_at_least_80_percent": (anomaly["threshold_recall"]["recall"] or 0) >= TARGET_RECALL,
        "anomaly_accuracy_at_least_80_percent": anomaly["threshold_recall"]["accuracy"] >= 0.80,
        "three_factors_coverage_at_least_80_percent": (anomaly["explanations"]["coverage"] or 0) >= 0.80,
        "note": "Factores predictivos asociados; no son causas demostradas ni validación del socio formador.",
    }
    coefficient_data = coefficient_table(models, preprocessor)
    prediction_data = pd.concat(predictions, ignore_index=True)
    prediction_data["probability_alguna_anomalia"] = np.tile(predictions[-1]["probability"].to_numpy(), len(EVENTS))
    bundle = {
        "preprocessor": preprocessor,
        "models": models, "thresholds": thresholds, "logistic_config": config,
        "input_columns": list(X.columns), "transformed_columns": preprocessor.transformed_names,
        "machine_columns": sorted(c for c in X if c.startswith("maquina_")),
        "preparation": "dp_features_zero_missing_flags_train_scaler_v1",
        "split": split,
    }
    joblib.dump(bundle, models_dir / "model.joblib", compress=3)
    save_plots(metrics, prediction_data, coefficient_data, images_dir)
    metrics["status"] = "evaluation_complete"
    print(f"[done] Gráficas en {images_dir}; modelos en {models_dir}", flush=True)
    return metrics


if __name__ == "__main__":
    run()
