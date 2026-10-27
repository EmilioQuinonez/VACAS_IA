"""
Primera iteración de modelado: un XGBoost binario por cada evento de ordeño.

Lee Data/dataset_vacas_integrado.csv (limpieza, clima, derivados y concentrado) y entrena
un clasificador por evento con los valores por defecto de XGBoost, sin regularización:
solo se agrega cuando la brecha entre train y val muestre sobreajuste. Los atributos son
únicamente los que se conocen cuando la vaca entra al robot; las mediciones de la sesión
actual son consecuencia del evento y se excluyen.

El split es cronológico (70 % más antiguo train, 10 % val, 20 % más reciente test). Los
modelos y las métricas se guardan en Data/modelos/xgboost_base/, fuera de Git.

Cada árbol se imprime como una época de Keras (verbose=2), con loss/accuracy de train y
val, para seguir el entrenamiento con el mismo formato que las redes del equipo. Las
épocas se numeran seguidas entre los tres eventos (1 a 300).

Uso (con -u la salida llega al log al momento):
    python -u src/MODELING/xgboost_baseline.py >| Data/modelos/xgboost_base/xgboost_baseline.log 2>&1
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from xgboost import XGBClassifier
from xgboost.callback import TrainingCallback

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "Data" / "dataset_vacas_integrado.csv"
OUTPUT_DIR = ROOT / "Data" / "modelos" / "xgboost_base"

EVENTOS = [
    "mechanical_connection_failure_event", "health_anomaly_event",
    "behavior_interruption_event",
]
CUARTOS = ("di", "dd", "ti", "td")
# Atributos disponibles al inicio de la sesión: historial de la vaca, clima y concentrado.
ATRIBUTOS = [
    "fme_connection", "fme_milk_production", "fme_incomplete", "kicks_ratio_7d",
    *[f"blood_diff_{cuarto}" for cuarto in CUARTOS],
    "box_time_to_yield_ratio", "flies_proxy",
    "t_sesion", "thi_sesion", "horas_thi_alto_24h",
    "conc_sesion_previa", "tope_sesion_previa", "conc_d1", "conc_media_7d",
    "intervalo_horas", "regreso_tras_periodo_seco", "estado_numero_de_ordeno",
]
VAL_SIZE = 0.1
TEST_SIZE = 0.2
RECALL_OBJETIVO = 0.80  # Criterio de negocio: detectar al menos 80 % de las anomalías.
N_ARBOLES = 100  # Valor por defecto de XGBoost; se fija para numerar las épocas del log.
RANDOM_STATE = 42


class LogEstiloKeras(TrainingCallback):
    """
    Imprime cada árbol como una época de Keras con verbose=2:
    "Epoch n/total" y "1/1 - Xs - Yms/step - accuracy: … - loss: … - val_accuracy: … - val_loss: …".

    Args:
        epoca_inicial (int): Épocas ya impresas por los eventos anteriores.

        total_epocas (int): Épocas de todo el script (N_ARBOLES por evento).
    """

    def __init__(self, epoca_inicial, total_epocas):
        super().__init__()
        self.epoca_inicial = epoca_inicial
        self.total_epocas = total_epocas
        self.mejor = float("inf")
        self.transcurrido = 0.0
        self.impreso = 0

    def before_iteration(self, model, epoch, evals_log):
        self.inicio = time.perf_counter()
        return False

    def after_iteration(self, model, epoch, evals_log):
        duracion = time.perf_counter() - self.inicio
        # Keras imprime segundos enteros: se acumula el resto para que la suma sea el tiempo real.
        self.transcurrido += duracion
        segundos = int(self.transcurrido) - self.impreso
        self.impreso += segundos

        train, val = evals_log["validation_0"], evals_log["validation_1"]
        numero = self.epoca_inicial + epoch + 1
        print(f"Epoch {numero}/{self.total_epocas}")
        print(f"1/1 - {segundos}s - {int(duracion * 1000)}ms/step - "
              f"accuracy: {1 - train['error'][-1]:.4f} - loss: {train['logloss'][-1]:.4f} - "
              f"val_accuracy: {1 - val['error'][-1]:.4f} - val_loss: {val['logloss'][-1]:.4f}")
        if val["logloss"][-1] < self.mejor:
            print(f"Epoch {numero}: val_loss improved from {self.mejor:.5f} to {val['logloss'][-1]:.5f}")
            self.mejor = val["logloss"][-1]

        return False


def cargar_sesiones(ruta):
    """
    Lee el dataset integrado y deja solo los ordeños.

    Args:
        ruta (Path): Ruta a dataset_vacas_integrado.csv.

    Returns:
        pd.DataFrame: Sesiones de ordeño con inicio como fecha y hora. Los rechazos
            y echadas sin ordeñar se quitan porque integrate_derivates.py les pone
            las etiquetas en 0 y entrarían como ordeños sin evento.
    """
    sesiones = pd.read_csv(ruta, dtype={"id_vaca": str}, parse_dates=["inicio"], low_memory=False)
    no_ordeno = sesiones[["accion_rechazada", "accion_echada_sin_ordenar"]].eq(1).any(axis=1)
    print(f"[datos] {len(sesiones):,} visitas; se quitan {no_ordeno.sum():,} rechazos y echadas sin ordeñar")

    return sesiones[~no_ordeno].reset_index(drop=True)


def construir_atributos(sesiones):
    """
    Arma la matriz de atributos y las etiquetas de los tres eventos.

    Args:
        sesiones (pd.DataFrame): Salida de cargar_sesiones.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            X: ATRIBUTOS, las dummies maquina_*, la hora de inicio (0 a 23) y las
                banderas <atributo>_sin_dato de clean_data.py que marcan un 0 de
                relleno. Los NaN se dejan para que XGBoost los maneje.

            y: Una columna 0/1 por evento de EVENTOS.
    """
    maquinas = [columna for columna in sesiones.columns if columna.startswith("maquina_")]
    banderas = [f"{columna}_sin_dato" for columna in ATRIBUTOS if f"{columna}_sin_dato" in sesiones]

    X = sesiones[ATRIBUTOS + maquinas + banderas].copy()
    X["hora"] = sesiones["inicio"].dt.hour
    print(f"[atributos] {X.shape[1]} columnas: {len(maquinas)} máquinas y banderas {banderas}")

    return X, sesiones[EVENTOS]


def separar_por_tiempo(inicio, val_size=VAL_SIZE, test_size=TEST_SIZE):
    """
    Separa las sesiones en train, val y test por orden cronológico.

    Args:
        inicio (pd.Series): Fecha y hora de inicio de cada sesión.

        val_size (float): Fracción para validación, justo después de train.

        test_size (float): Fracción más reciente, reservada para test.

    Returns:
        dict[str, numpy.ndarray]: Posiciones de las filas de "train", "val" y "test".
    """
    orden = np.argsort(inicio.to_numpy(), kind="stable")
    fin_train = int(len(orden) * (1 - val_size - test_size))
    fin_val = int(len(orden) * (1 - test_size))
    idx = {
        "train": orden[:fin_train],
        "val": orden[fin_train:fin_val],
        "test": orden[fin_val:],
    }
    for nombre, filas in idx.items():
        fechas = inicio.iloc[filas]
        print(f"[split] {nombre}: {len(filas):,} sesiones, {fechas.min():%d/%m/%Y} a {fechas.max():%d/%m/%Y}")

    return idx


def entrenar_evento(X, y, idx, epoca_inicial=0):
    """
    Entrena un XGBoost binario con los valores por defecto de la librería.

    Args:
        X (pd.DataFrame): Atributos de todas las sesiones.

        y (pd.Series): Etiqueta 0/1 del evento.

        idx (dict[str, numpy.ndarray]): Posiciones de separar_por_tiempo.

        epoca_inicial (int): Épocas ya impresas en el log por los eventos anteriores.

    Returns:
        XGBClassifier: Modelo entrenado con train. scale_pos_weight (negativos /
            positivos de train) compensa el desbalance del evento. Val solo se
            evalúa para el log; no detiene ni cambia el entrenamiento.
    """
    X_train, y_train = X.iloc[idx["train"]], y.iloc[idx["train"]]
    modelo = XGBClassifier(
        objective="binary:logistic",
        n_estimators=N_ARBOLES,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        eval_metric=["error", "logloss"],
        callbacks=[LogEstiloKeras(epoca_inicial, N_ARBOLES * len(EVENTOS))],
        random_state=RANDOM_STATE,
    )

    return modelo.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X.iloc[idx["val"]], y.iloc[idx["val"]])],
        verbose=False,
    )


def evaluar(modelo, X, y, idx):
    """
    Calcula las métricas del modelo en train, val y test con dos umbrales.

    Args:
        modelo (XGBClassifier): Modelo de entrenar_evento.

        X (pd.DataFrame): Atributos de todas las sesiones.

        y (pd.Series): Etiqueta 0/1 del evento.

        idx (dict[str, numpy.ndarray]): Posiciones de separar_por_tiempo.

    Returns:
        dict: Por conjunto, la prevalencia, la accuracy de responder siempre
            "sin evento", ROC-AUC, PR-AUC y, por umbral, accuracy, precisión,
            recall y F1. Los umbrales son 0.5 y el mayor que en val alcanza
            RECALL_OBJETIVO.
    """
    probabilidades = {nombre: modelo.predict_proba(X.iloc[filas])[:, 1] for nombre, filas in idx.items()}
    reales = {nombre: y.iloc[filas].to_numpy() for nombre, filas in idx.items()}

    # recall baja al subir el umbral; recall[:-1] corresponde a cada umbral.
    _, recall, umbrales = precision_recall_curve(reales["val"], probabilidades["val"])
    umbral_recall = float(umbrales[recall[:-1] >= RECALL_OBJETIVO].max())

    metricas = {"umbral_recall_val": umbral_recall}
    for nombre in idx:
        real, probabilidad = reales[nombre], probabilidades[nombre]
        metricas[nombre] = {
            "prevalencia": float(real.mean()),
            "accuracy_sin_evento": float(1 - real.mean()),
            "roc_auc": float(roc_auc_score(real, probabilidad)),
            "pr_auc": float(average_precision_score(real, probabilidad)),
        }
        for etiqueta, umbral in [("umbral_0.5", 0.5), ("umbral_recall", umbral_recall)]:
            prediccion = (probabilidad >= umbral).astype(int)
            metricas[nombre][etiqueta] = {
                "accuracy": float(accuracy_score(real, prediccion)),
                "precision": float(precision_score(real, prediccion, zero_division=0)),
                "recall": float(recall_score(real, prediccion)),
                "f1": float(f1_score(real, prediccion)),
            }

    return metricas


def main():
    """
    Entrena y evalúa un modelo por evento, guarda modelos y métricas e imprime un resumen.
    """
    sesiones = cargar_sesiones(INPUT_PATH)
    X, etiquetas = construir_atributos(sesiones)
    idx = separar_por_tiempo(sesiones["inicio"])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    resultados = {}
    for numero, evento in enumerate(EVENTOS):
        print(f"\n[entrenamiento] {evento}")
        modelo = entrenar_evento(X, etiquetas[evento], idx, numero * N_ARBOLES)
        modelo.save_model(OUTPUT_DIR / f"{evento}.json")
        resultados[evento] = evaluar(modelo, X, etiquetas[evento], idx)

        print(f"\n{evento} (umbral para recall >= {RECALL_OBJETIVO:.0%} en val: "
              f"{resultados[evento]['umbral_recall_val']:.3f})")
        for nombre in idx:
            m = resultados[evento][nombre]
            print(f"  {nombre:5}: prevalencia {m['prevalencia']:.1%} | ROC-AUC {m['roc_auc']:.3f} | "
                  f"PR-AUC {m['pr_auc']:.3f} | accuracy sin evento {m['accuracy_sin_evento']:.1%}")
            for etiqueta in ("umbral_0.5", "umbral_recall"):
                u = m[etiqueta]
                print(f"         {etiqueta:13}: accuracy {u['accuracy']:.1%} | precisión {u['precision']:.1%} | "
                      f"recall {u['recall']:.1%} | F1 {u['f1']:.3f}")

    with open(OUTPUT_DIR / "metricas.json", "w", encoding="utf-8") as archivo:
        json.dump(resultados, archivo, indent=2, ensure_ascii=False)
    print(f"\nModelos y métricas en {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
