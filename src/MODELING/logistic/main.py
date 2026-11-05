"""Entrena, evalúa y genera seis gráficas con particiones cronológicas por cuantiles.

Desde la raíz: py src/MODELING/logistic/main.py
Desde esta carpeta: py main.py
"""

import argparse
from pathlib import Path

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "src.MODELING.logistic"

import joblib
import pandas as pd

from .dataset import EVENTS, build_features, get_labels, load_sessions, split_by_time
from .evaluation import choose_threshold, evaluate, explanation_coverage
from .logistic_model import CowPreprocessor, coefficient_table, explain, train_event, training_curve
from .plot_results import save_plots

MODULE_DIR = Path(__file__).resolve().parent
INPUT_PATH = MODULE_DIR.parents[2] / "Data" / "dataset_vacas_integrado.csv"
TRAIN_FRACTION, VAL_FRACTION = 0.70, 0.15
# Elección manual a partir de results/graficas/hiperparametros_validacion.png.
FIXED_HYPERPARAMETERS = {
    "mechanical_connection_failure_event": {"C": 0.01, "l1_ratio": 0.1},
    "health_anomaly_event": {"C": 0.1, "l1_ratio": 0.9},
    "behavior_interruption_event": {"C": 0.01, "l1_ratio": 0.9},
    "alguna_anomalia": {"C": 0.1, "l1_ratio": 0.1},
}
TARGET_RECALL, RANDOM_STATE = 0.80, 42
# Evaluar el ajuste obtenido con hasta 500 iteraciones, aunque no haya convergido.
MAX_ITER, TOL = 500, 1e-4

def fit_model(event, transformed, labels, parts):
    """Ajusta parámetros fijos y permite evaluar el modelo aunque no converja."""
    train_labels = labels[event].iloc[parts["train"]].to_numpy()
    params = FIXED_HYPERPARAMETERS[event]
    print(f"[fit] {event}; C={params['C']:g}; l1_ratio={params['l1_ratio']:g}; max_iter={MAX_ITER}; parámetros fijos", flush=True)
    model, details = train_event(
        transformed["train"], train_labels, params["C"], params["l1_ratio"],
        RANDOM_STATE, MAX_ITER, TOL,
    )
    print(f"  iteraciones ejecutadas={details['iterations']}", flush=True)
    return model, details


def print_summary(event, results, threshold):
    print(f"\n[result] {event}; umbral={threshold:.6f}", flush=True)

    parameters = results["parameters"]
    print(
        f"  Hiperparámetros fijos: "
        f"C={parameters['C']:g}; l1_ratio={parameters['l1_ratio']:g}",
        flush=True,
    )
    
    for name in ("train", "val", "test"):
        scores = results[name]
        decision = scores["threshold_recall"]
        print(
            f"  {name}: AP={scores['average_precision']:.4f}; Brier={scores['brier']:.4f}; "
            f"precisión={decision['precision']:.3%}; sensibilidad={decision['recall']:.3%}; "
            f"F1={decision['f1']:.4f}; accuracy={decision['accuracy']:.3%}", flush=True,
        )


def run(input_path=INPUT_PATH):
    # Cargar y dividir por cuantiles sin excluir intervalos entre conjuntos.
    sessions = load_sessions(input_path)
    labels = get_labels(sessions)
    parts, split = split_by_time(sessions["inicio"], TRAIN_FRACTION, VAL_FRACTION)
    for event in EVENTS:
        for name, rows in parts.items():
            if labels[event].iloc[rows].nunique() != 2:
                raise ValueError(f"{event}: {name} no contiene ambas clases.")

    # Preparar: historial anterior a cada sesión e imputación/escala solo de train.
    print("[features] Construyendo historial en memoria", flush=True)
    X = build_features(sessions)
    
    cows = sessions["id_vaca"]
    preprocessor = CowPreprocessor().fit(X.iloc[parts["train"]], cows.iloc[parts["train"]])
    transformed = {name: preprocessor.transform(X.iloc[rows], cows.iloc[rows]) for name, rows in parts.items()}
    print(f"[features] {X.shape[1]} predictores; {transformed['train'].shape[1]} columnas con indicadores", flush=True)
    
    if preprocessor.empty_columns:
        print(f"[imputation] Sin datos en train; respaldo 0: {preprocessor.empty_columns}", flush=True)

    artifacts = MODULE_DIR / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    models, thresholds, predictions = {}, {}, []
    metrics = {"split": split}

    # Entrenar con parámetros fijos; validación ajusta el umbral y test solo evalúa.
    for event in EVENTS:
        model, parameters = fit_model(event, transformed, labels, parts)
        
        models[event] = model
        probabilities = {name: model.predict_proba(data)[:, 1] for name, data in transformed.items()}
        threshold = choose_threshold(labels[event].iloc[parts["val"]], probabilities["val"], TARGET_RECALL)
        thresholds[event] = threshold
        scores = {"threshold_recall_val": threshold, "parameters": parameters}
        
        for name, rows in parts.items():
            probability = probabilities[name]
            scores[name] = evaluate(labels[event].iloc[rows], probability, threshold)
            
            if name == "test":
                frame = pd.DataFrame({
                    "event": event, "actual": labels[event].iloc[rows].to_numpy(),
                    "probability": probability,
                })
                
                explanations = explain(model, preprocessor, transformed[name], X.iloc[rows].reset_index(drop=True))
                scores[name]["explanations"] = explanation_coverage(explanations, probability, threshold)
                predictions.append(frame)
        
        # Diagnóstico del ajuste fijo; test nunca elige el modelo ni la iteración.
        print(f"[learning] {event}; reconstruyendo curvas por iteraciones del modelo fijo", flush=True)
        actual = {name: labels[event].iloc[rows].to_numpy() for name, rows in parts.items()}
        scores["learning_curve"] = training_curve(model, transformed, actual).to_dict("records")
        metrics[event] = scores
        print_summary(event, scores, threshold)

    anomaly_test = metrics["alguna_anomalia"]["test"]
    metrics["business_criteria"] = {
        "anomaly_recall_at_least_80_percent": anomaly_test["threshold_recall"]["recall"] >= TARGET_RECALL,
        "anomaly_accuracy_at_least_80_percent": anomaly_test["threshold_recall"]["accuracy"] >= 0.80,
        "three_factors_coverage_at_least_80_percent": (anomaly_test["explanations"]["coverage"] or 0) >= 0.80,
        "note": "Los factores son asociaciones del modelo, no causas demostradas.",
    }
    print("\n[criterios]", metrics["business_criteria"], flush=True)

    # Guardar un único paquete: contiene los mismos modelos que se evaluaron.
    bundle = {
        "preprocessor": preprocessor, "models": models, "thresholds": thresholds,
        "machine_columns": sorted(c for c in X if c.startswith("maquina_")),
        "hyperparameters": {event: metrics[event]["parameters"] for event in EVENTS},
    }
    joblib.dump(bundle, artifacts / "model.joblib", compress=3)
    save_plots(
        metrics, pd.concat(predictions, ignore_index=True),
        coefficient_table(models, preprocessor),
    )
    print(f"\n[done] Modelo en {artifacts / 'model.joblib'}", flush=True)
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH, help="Dataset integrado; no se modifica.")
    args = parser.parse_args()
    
    run(args.input)
