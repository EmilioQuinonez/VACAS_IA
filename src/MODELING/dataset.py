"""Carga del dataset integrado, atributos del modelo y partición cronológica."""

import numpy as np
import pandas as pd

EVENTS = [
    "mechanical_connection_failure_event", "health_anomaly_event",
    "behavior_interruption_event",
]
EVENT_KEYS = ("mecanico", "salud", "conducta")
QUARTERS = ("di", "dd", "ti", "td")
# Tabla de selección de datos del Doc, columna XGBoost; las máquinas (maquina_*) se agregan aparte.
FEATURES = [
    "intervalo_horas", "intervalo_horas_sin_dato", "regreso_tras_periodo_seco", "estado_numero_de_ordeno",
    "t_sesion", "thi_sesion", "horas_thi_alto_24h", "sangre_tasa_maquina_30d",
    "fme_connection", "fme_milk_production", "fme_incomplete", "kicks_ratio_7d",
    *[f"blood_diff_{quarter}" for quarter in QUARTERS],
    "box_time_to_yield_ratio", "flies_proxy",
    # Creadas por src/DP/build_history_features.py.
    "dias_en_leche", "dias_en_hato", "mes", "hora", "dia_semana",
    *[f"{key}_vaca_30d" for key in EVENT_KEYS], *[f"{key}_hato_7d" for key in EVENT_KEYS],
    "vacas_activas_7d", *[f"{key}_previo" for key in EVENT_KEYS],
    "mdi_previo", "conductividad_max_previa", "produccion_previa", "cuarto_inactivo_previo",
]


def load_sessions(path):
    """
    Lee el dataset integrado y deja solo los ordeños: integrate_derivates.py pone las
    etiquetas en 0 a rechazos y echadas, que entrarían como ordeños sin evento.
    """
    sessions = pd.read_csv(path, dtype={"id_vaca": str}, parse_dates=["inicio"], low_memory=False)
    not_milking = sessions[["accion_rechazada", "accion_echada_sin_ordenar"]].eq(1).any(axis=1)
    print(f"[data] {len(sessions):,} visits; {not_milking.sum():,} rejections and lie-downs removed")

    return sessions[~not_milking].reset_index(drop=True)


def build_features(sessions):
    """X (FEATURES y dummies maquina_*; los NaN se conservan) e y (una columna 0/1 por evento)."""
    machines = [column for column in sessions.columns if column.startswith("maquina_")]
    X = sessions[FEATURES + machines]
    print(f"[features] {X.shape[1]} columns ({len(machines)} machines)")

    return X, sessions[EVENTS]


def split_by_time(start, train_fraction, val_fraction):
    """
    Posiciones de train (las sesiones más antiguas, train_fraction del total), val (las
    siguientes, val_fraction) y test (el resto, las más recientes).
    """
    train_end = start.quantile(train_fraction)
    val_end = start.quantile(train_fraction + val_fraction)
    idx = {
        "train": np.flatnonzero(start < train_end),
        "val": np.flatnonzero((start >= train_end) & (start < val_end)),
        "test": np.flatnonzero(start >= val_end),
    }
    for name, rows in idx.items():
        dates = start.iloc[rows]
        print(f"[split] {name}: {len(rows):,} sessions, {dates.min():%d/%m/%Y} to {dates.max():%d/%m/%Y}")

    return idx
