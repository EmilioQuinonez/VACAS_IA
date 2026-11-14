"""Carga del dataset integrado, secuencias de la LSTM y partición cronológica."""

import numpy as np
import pandas as pd

EVENTS = [
    "mechanical_connection_failure_event", "health_anomaly_event",
    "behavior_interruption_event", "alguna_anomalia",
]
EVENT_KEYS = ("mecanico", "salud", "conducta")
QUARTERS = ("di", "dd", "ti", "td")
GROUP = ["id_vaca", "bloque_historial"]

# Tabla de selección de datos, columna LSTM: "entra en cada paso de la secuencia". En la fila r
# todas describen la sesión r - 1 (ya vienen desfasadas o se desfasan en load_sessions).
SEQUENCE_FEATURES = [
    "fme_connection", "fme_milk_production", "fme_incomplete",
    *[f"blood_diff_{quarter}" for quarter in QUARTERS], "box_time_to_yield_ratio",
    # Creadas por src/MODELING/LSTM/build_history_features.py.
    "produccion_previa", "mdi_previo", "conductividad_max_previa", "cuarto_inactivo_previo",
    *[f"{key}_previo" for key in EVENT_KEYS],
    # Desfasadas en load_sessions.
    "mdi_faltante_previo", "estado_mdi_sin_dato_previo", "intervalo_horas_previo",
]
# Contexto de la sesión a predecir: se conoce al entrar al robot y se une después de la LSTM.
CONTEXT_FEATURES = [
    "intervalo_horas", "intervalo_horas_sin_dato", "regreso_tras_periodo_seco", "estado_numero_de_ordeno",
    "maquina_7063e4b1", "maquina_86696b37", "t_sesion", "thi_sesion", "horas_thi_alto_24h",
    "sangre_tasa_maquina_30d", "sangre_pico_maquina", "dias_en_leche", "dias_en_hato",
    "mes_sin", "mes_cos", "hora_sin", "hora_cos", "dia_semana_sin", "dia_semana_cos",
    *[f"{key}_hato_7d" for key in EVENT_KEYS], "vacas_activas_7d",
]


def load_sessions(path):
    """
    Lee el dataset integrado y deja solo los ordeños, ordenados por vaca e inicio. Agrega
    alguna_anomalia (objetivo invertido de successful_milking_event), las variables desfasadas
    de la secuencia, el logaritmo de box_time_to_yield_ratio y el seno y coseno del calendario.
    """
    sessions = pd.read_csv(path, dtype={"id_vaca": str}, parse_dates=["inicio"], low_memory=False)
    not_milking = sessions[["accion_rechazada", "accion_echada_sin_ordenar"]].eq(1).any(axis=1)
    print(f"[data] {len(sessions):,} visits; {not_milking.sum():,} rejections and lie-downs removed")
    sessions = sessions[~not_milking].sort_values(["id_vaca", "inicio"], kind="stable").reset_index(drop=True)

    sessions["alguna_anomalia"] = 1 - sessions["successful_milking_event"]
    previous = sessions.groupby(GROUP)[["mdi_faltante", "estado_mdi_sin_dato", "intervalo_horas"]].shift(1)
    sessions[[f"{column}_previo" for column in previous]] = previous.to_numpy()
    sessions["box_time_to_yield_ratio"] = np.log1p(sessions["box_time_to_yield_ratio"])
    # Seno y coseno para que diciembre y enero, o las 23 h y las 0 h, queden juntos.
    for column, period in [("mes", 12), ("hora", 24), ("dia_semana", 7)]:
        sessions[f"{column}_sin"] = np.sin(2 * np.pi * sessions[column] / period)
        sessions[f"{column}_cos"] = np.cos(2 * np.pi * sessions[column] / period)

    return sessions


def split_by_time(start, train_fraction, val_fraction):
    """
    Posiciones de train (las sesiones más antiguas, train_fraction del total), val (las
    siguientes, val_fraction) y test (el resto, las más recientes), sin espacio entre periodos.
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


def split_test_by_cow(cows, idx):
    """
    Divide test como el reporte y la regresión logística: vacas conocidas (aparecen en train o val)
    y nuevas (su primer ordeño es en test; el modelo no las vio ni en train ni en val).
    """
    test_cows = cows.iloc[idx["test"]]
    seen = test_cows.isin(cows.iloc[np.concatenate([idx["train"], idx["val"]])]).to_numpy()
    groups = {"test_vacas_conocidas": idx["test"][seen], "test_vacas_nuevas": idx["test"][~seen]}
    for name, rows in groups.items():
        print(f"[split] {name}: {len(rows):,} sessions of {cows.iloc[rows].nunique()} cows")

    return groups


def build_inputs(sessions, train_rows, k):
    """
    X_seq (sesiones, k, variables): para la sesión t, las filas t - k + 1 ... t de la misma vaca y
    bloque, que describen las sesiones t - k ... t - 1. Los pasos sin sesión previa quedan en 0
    para la capa Masking. X_ctx: contexto de la sesión t. Ambos se imputan con la mediana y se
    estandarizan con la media y desviación de train. y: una columna 0/1 por evento. scaler: la
    mediana, media y desviación de train de cada variable, para preparar sesiones nuevas igual.
    """
    scaler = {}

    def scale(columns):
        values = sessions[columns].astype(float)
        median = values.iloc[train_rows].median()
        values = values.fillna(median)
        mean, std = values.iloc[train_rows].mean(), values.iloc[train_rows].std().replace(0, 1)
        scaler.update({column: {"median": median[column], "mean": mean[column], "std": std[column]}
                       for column in columns})
        return (values - mean) / std

    steps = scale(SEQUENCE_FEATURES)
    # Sin produccion_previa no hay sesión previa en el bloque: el paso es relleno.
    steps.loc[sessions["produccion_previa"].isna()] = 0.0
    X_seq = np.zeros((len(sessions), k, len(SEQUENCE_FEATURES)), dtype="float32")
    groups = [sessions[column] for column in GROUP]
    for lag in range(k):
        # El paso más reciente va al final; fuera de la vaca o del bloque, shift deja NaN.
        X_seq[:, k - 1 - lag] = steps.groupby(groups).shift(lag).fillna(0.0).to_numpy()

    X_ctx = scale(CONTEXT_FEATURES).to_numpy(dtype="float32")
    y = sessions[EVENTS].to_numpy(dtype="float32")
    print(f"[features] sequence {X_seq.shape}, context {X_ctx.shape}, targets {y.shape}")

    return X_seq, X_ctx, y, scaler
