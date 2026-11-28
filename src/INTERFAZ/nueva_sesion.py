"""
Arma la fila de una sesión capturada en la interfaz con las mismas funciones del pipeline de
datos. De la captura vienen solo los datos que se conocen al entrar al robot (fecha y hora,
máquina, clima y número de ordeño); el resto se calcula con el historial de la vaca en
Data/dataset_vacas_integrado.csv.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "Data" / "dataset_vacas_integrado.csv"


def load_module(name, path):
    """Importa un script del repo por su ruta, con un nombre propio para no chocar con otros."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


derivates = load_module("integrate_derivates", ROOT / "src" / "DP" / "integrate_derivates.py")
# La versión del XGBoost también agrega las banderas de MDI del ordeño previo.
history = load_module("xgb_build_history_features", ROOT / "src" / "MODELING" / "XGBoost" / "build_history_features.py")

ACTIONS = ["accion_rechazada", "accion_echada_sin_ordenar"]
QUARTERS = ("di", "dd", "ti", "td")
LONG_INTERVAL_HOURS = 48  # corregir_intervalos de clean_data.py
RESTART_DAYS = 45  # preparar_historial de clean_data.py
BLOOD_PEAK_RATE = 0.15  # marcar_sangre de clean_data.py
REFERENCE_MACHINE = "referencia"  # La máquina más frecuente no tiene dummy (clean_data.py)


def load_visits(path=DATA_PATH):
    """
    Todas las visitas del dataset integrado, ordenadas por vaca e inicio. Si faltan variables de
    historial del XGBoost (no se corrió su build_history_features.py), se agregan aquí.
    """
    visits = pd.read_csv(path, dtype={"id_vaca": str}, parse_dates=["inicio"], low_memory=False)
    visits = visits.sort_values(["id_vaca", "inicio"], kind="stable").reset_index(drop=True)
    missing = [column for column in history.NEW_COLUMNS if column not in visits]
    if missing:
        new = pd.DataFrame(index=visits.index)
        history.add_previous_session(visits, is_milking(visits), new)
        visits[missing] = new[missing]
    return visits


def is_milking(visits):
    return visits[ACTIONS].eq(0).all(axis=1)


def machines(visits):
    """Nombres de las máquinas: las que tienen dummy y la de referencia."""
    return [REFERENCE_MACHINE] + sorted(c.removeprefix("maquina_") for c in visits if c.startswith("maquina_"))


def machine_of(row):
    dummies = [c for c in row.index if c.startswith("maquina_")]
    active = [c.removeprefix("maquina_") for c in dummies if row[c] == 1]
    return active[0] if active else REFERENCE_MACHINE


def thi(temperature, humidity):
    """THI con la fórmula de integrar_clima.py (temperatura en °C y humedad relativa en %)."""
    return (1.8 * temperature + 32) - (0.55 - 0.0055 * humidity) * (1.8 * temperature - 26)


def humidity_from_thi(temperature, value):
    """Humedad relativa que da ese THI a esa temperatura (para prellenar la captura)."""
    slope = 1.8 * temperature - 26
    if abs(slope) < 1e-6:
        return 60.0
    return float(np.clip(((value - (1.8 * temperature + 32)) / slope + 0.55) / 0.0055, 0, 100))


def machine_blood_rate(visits, machine, start):
    """Tasa de visitas con sangre de la máquina en [inicio - 30 d, inicio), como marcar_sangre."""
    dummies = [c for c in visits if c.startswith("maquina_")]
    if machine == REFERENCE_MACHINE:
        same = visits[dummies].eq(0).all(axis=1)
    else:
        same = visits[f"maquina_{machine}"].eq(1)
    window = same & (visits["inicio"] >= start - pd.Timedelta(days=30)) & (visits["inicio"] < start)
    blood = visits.loc[window, [f"sangre_ppm_{q}" for q in QUARTERS]].max(axis=1) > 0
    return float(blood.mean()) if len(blood) else 0.0


def build_session(visits, cow, capture):
    """
    Historial de la vaca anterior a la captura y, al final, la fila nueva con todas las variables
    del modelo. capture: inicio (Timestamp), maquina, t_sesion, thi_sesion,
    horas_thi_alto_24h y estado_numero_de_ordeno.
    """
    start = pd.Timestamp(capture["inicio"])
    latest = last_date(visits)
    if start.normalize() > latest:
        raise ValueError(f"La fecha debe ser a más tardar el {latest:%d/%m/%Y}: el dataset termina el "
                         f"{visits['inicio'].max():%d/%m/%Y} y las tasas del hato usan los 7 días anteriores.")
    past = visits[(visits["id_vaca"] == cow) & (visits["inicio"] < start)]
    milkings = past[is_milking(past)]
    if milkings.empty:
        raise ValueError("La vaca no tiene ordeños anteriores a esa fecha.")
    last = milkings.iloc[-1]

    # La plantilla es el último ordeño: sus mediciones no entran al modelo, que solo usa el pasado.
    row = last.copy()
    row["inicio"] = start
    row[ACTIONS] = 0
    for column in [c for c in visits if c.startswith("maquina_")]:
        row[column] = int(column == f"maquina_{capture['maquina']}")
    for column in ("t_sesion", "thi_sesion", "horas_thi_alto_24h", "estado_numero_de_ordeno"):
        row[column] = capture[column]

    # Intervalo desde el fin del ordeño anterior; más de 48 h es regreso de periodo seco y se
    # sustituye por la mediana previa de la vaca (corregir_intervalos y tratar_nulos).
    end = last["inicio"] + pd.Timedelta(seconds=float(last["duracion_seg"]))
    interval = (start - end).total_seconds() / 3600
    row["regreso_tras_periodo_seco"] = int(interval > LONG_INTERVAL_HOURS)
    if interval > LONG_INTERVAL_HOURS:
        interval = milkings["intervalo_horas"].median()
    row["intervalo_horas"], row["intervalo_horas_sin_dato"] = interval, 0
    if start - past["inicio"].iloc[-1] >= pd.Timedelta(days=RESTART_DAYS):
        row["bloque_historial"] = past["bloque_historial"].iloc[-1] + 1

    rate = machine_blood_rate(visits, capture["maquina"], start)
    row["sangre_tasa_maquina_30d"], row["sangre_pico_maquina"] = rate, int(rate > BLOOD_PEAK_RATE)

    sessions = pd.concat([past, row.to_frame().T], ignore_index=True).astype(past.dtypes.to_dict())
    new = sessions.index[-1]

    # FME, sangre, tiempo/producción, patadas y moscas del ordeño anterior (integrate_derivates.py).
    derived = derivates.FEATURES + derivates.EVENTS
    recomputed = derivates.add_derived_features(sessions.drop(columns=derived))
    sessions.loc[new, derivates.FEATURES] = recomputed.loc[new, derivates.FEATURES]

    # Calendario, ordeño previo y tasas de la vaca (build_history_features.py).
    milking = is_milking(sessions)
    computed = pd.DataFrame(index=sessions.index)
    history.add_calendar(sessions, computed)
    history.add_previous_session(sessions, milking, computed)
    history.add_cow_rates(sessions, milking, computed)
    # Las tasas del hato usan los 7 días anteriores de todas las vacas.
    herd = pd.concat([visits[visits["inicio"] < start.normalize()], sessions.loc[[new]]], ignore_index=True)
    herd_rates = pd.DataFrame(index=herd.index)
    history.add_herd_rates(herd, is_milking(herd), herd_rates)
    computed.loc[new, herd_rates.columns] = herd_rates.iloc[-1]
    sessions.loc[new, history.NEW_COLUMNS] = computed.loc[new, history.NEW_COLUMNS]

    return sessions


def existing_session(visits, cow, start):
    """
    Historial de la vaca hasta un ordeño del dataset, que queda al final. Si una visita sin ordeño
    tiene el mismo inicio, se deja fuera para que la última fila sea el ordeño.
    """
    cow_visits = visits[visits["id_vaca"] == cow]
    start = pd.Timestamp(start)
    keep = (cow_visits["inicio"] < start) | ((cow_visits["inicio"] == start) & is_milking(cow_visits))
    return cow_visits[keep].reset_index(drop=True)


def last_date(visits):
    """Último día que se puede capturar: el siguiente al fin del dataset, el último con tasas del hato."""
    return visits["inicio"].max().normalize() + pd.Timedelta(days=1)


def notices(sessions):
    """Avisos sobre la sesión capturada (la última fila), para leer sus probabilidades con cuidado."""
    row, notes = sessions.iloc[-1], []
    if len(sessions) > 1 and row["bloque_historial"] != sessions["bloque_historial"].iloc[-2]:
        notes.append(f"Pasaron {RESTART_DAYS} días o más desde la última visita: empieza un bloque de historial "
                     "nuevo, así que no hay ordeño previo.")
    elif row["regreso_tras_periodo_seco"] == 1:
        notes.append(f"Pasaron más de {LONG_INTERVAL_HOURS} h desde el último ordeño: la sesión se marca como "
                     "regreso tras periodo seco, una situación poco común en los datos.")
    return notes
