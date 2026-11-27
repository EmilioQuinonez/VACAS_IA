"""
Cuánto se alarga el ordeño si ocurre cada evento, con las métricas de src/EVALUATION/duracion_por_evento.py
(metrics/duracion_por_evento.json): minutos extra de los ordeños de test con el evento frente a la mediana
de los ordeños sin evento de la misma vaca, y la espera estimada en la fila de cada máquina.

Lo normal de la vaca se calcula aquí con su historial previo a la sesión (mediana de sus últimos ordeños
sin evento), y la duración esperada con el evento es esa mediana más los minutos extra típicos.
"""

import json

import pandas as pd

import nueva_sesion as ns

METRICS_PATH = ns.ROOT / "src" / "EVALUATION" / "metrics" / "duracion_por_evento.json"
BASE_EVENTS = ["mechanical_connection_failure_event", "health_anomaly_event", "behavior_interruption_event"]
# Categoría de duracion_por_evento.json de cada salida del modelo.
CATEGORIES = {
    "mechanical_connection_failure_event": "Solo falla mecánica",
    "health_anomaly_event": "Solo salud",
    "behavior_interruption_event": "Solo conducta",
    "alguna_anomalia": "Cualquier evento",
}
LAST_NORMAL = 30  # Ordeños sin evento más recientes de la vaca con los que se calcula lo normal


def load(visits):
    """Métricas de duración por evento y espera por máquina, o None si no se ha corrido el script."""
    if not METRICS_PATH.exists():
        return None
    with open(METRICS_PATH, encoding="utf-8") as file:
        record = json.load(file)
    summary = {row["categoria"]: row for row in record["resumen"]}
    waits = {row["maquina"]: row for row in record["espera_por_maquina"]}
    # La máquina de referencia de la interfaz (sin dummy) es la del archivo que no tiene columna.
    dummies = {c.removeprefix("maquina_") for c in visits if c.startswith("maquina_")}
    reference = [code for code in waits if code not in dummies]
    if len(reference) == 1:
        waits[ns.REFERENCE_MACHINE] = waits[reference[0]]
    return {"periodo": [d[:10] for d in record["periodo_test"]], "ordenos": record["ordenos_test"],
            "resumen": summary, "espera": waits}


def normal_minutes(sessions):
    """Mediana de duración (min) de los últimos ordeños sin evento de la vaca, antes de la sesión."""
    past = sessions.iloc[:-1]
    clean = past[ns.is_milking(past) & past[BASE_EVENTS].eq(0).all(axis=1)]
    if clean.empty:
        return None, 0
    recent = clean["duracion_seg"].tail(LAST_NORMAL) / 60
    return float(recent.median()), len(recent)


def describe(metrics, sessions, machine, real_row=None):
    """
    Tiempo del ordeño para la respuesta de /api/predecir: lo normal de la vaca, los minutos extra típicos
    de cada evento y la duración esperada si ocurre, la espera de la máquina de la sesión y, en un ordeño
    del dataset, lo que duró de verdad.
    """
    if metrics is None:
        return None
    normal, count = normal_minutes(sessions)
    wait = metrics["espera"].get(machine)
    events = {}
    for event, category in CATEGORIES.items():
        row = metrics["resumen"][category]
        extra = row["minutos_extra_mediana"]
        events[event] = {
            "extra_mediana": extra, "extra_media": row["minutos_extra_media"],
            "esperado": None if normal is None else normal + extra,
            "ahorro_espera": None if wait is None else wait.get(f"ahorro_{category}"),
        }
    result = {
        "normal": normal, "ordenos_normales": count, "eventos": events,
        "espera_maquina": None if wait is None else wait["espera_min"],
        "periodo": metrics["periodo"], "ordenos_test": metrics["ordenos"],
    }
    if real_row is not None and pd.notna(real_row["duracion_seg"]):
        result["real"] = float(real_row["duracion_seg"]) / 60
    return result
