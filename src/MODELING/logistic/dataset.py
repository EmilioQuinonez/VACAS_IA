"""Datos e historial disponible antes de cada sesión; no modifica el CSV original."""

import numpy as np
import pandas as pd

QUARTERS = ("di", "dd", "ti", "td")
EVENT_KEYS = ("mecanico", "salud", "conducta")
EVENTS = (
    "mechanical_connection_failure_event", "health_anomaly_event",
    "behavior_interruption_event", "alguna_anomalia",
)
EVENT_NAMES = ("Falla mecánica", "Salud", "Conducta", "Alguna anomalía")
MILK_BINS = (0, 30, 60, 90, 150, 210, 305, np.inf)
HERD_BINS = (0, 7, 30, 90, 365, np.inf)
CONTEXT = (
    "intervalo_horas", "intervalo_horas_sin_dato", "regreso_tras_periodo_seco",
    "estado_numero_de_ordeno", "thi_sesion", "horas_thi_alto_24h",
)
DP_FEATURES = (
    "fme_connection", "fme_milk_production", "fme_incomplete",
    "kicks_ratio_7d", "flies_proxy",
    *[f"blood_diff_{q}" for q in QUARTERS[:3]],
    "box_time_to_yield_ratio", "sangre_tasa_maquina_30d", "sangre_pico_maquina",
)

def load_sessions(path, labeled=True):
    """Lee ordeños elegibles y normaliza el índice para construir sus atributos."""
    
    frame = pd.read_csv(path, dtype={"id_vaca": str}, low_memory=False)
    frame["inicio"] = pd.to_datetime(frame["inicio"], errors="raise")
    actions = ["accion_rechazada", "accion_echada_sin_ordenar"]
    
    if labeled or any(c in frame for c in actions):
        keep = frame[actions].eq(0).all(axis=1)
        print(f"[data] {len(frame):,} visitas; {(~keep).sum():,} no son eventos de ordeño, por ello se excluyen", flush=True)
        frame = frame.loc[keep].copy()
    frame = frame.reset_index(drop=True)
    
    return frame

def get_labels(sessions):
    labels = sessions[[*EVENTS[:3], "successful_milking_event"]]
    if not labels.isin([0, 1]).all().all():
        raise ValueError("Las cuatro etiquetas originales deben ser binarias y sin nulos.")
    anomaly = 1 - labels["successful_milking_event"]
    
    return labels[list(EVENTS[:3])].assign(alguna_anomalia=anomaly).astype("int8")

def split_by_time(start, train_fraction=0.70, val_fraction=0.15):
    """Divide por cuantiles sin intervalos excluidos entre train, val y test.

    Las ventanas históricas de 30 y 7 días se conservan.
    """
    train_end, val_end = start.quantile([train_fraction, train_fraction + val_fraction])
    
    parts = {
        "train": np.flatnonzero(start < train_end),
        "val": np.flatnonzero((start >= train_end) & (start < val_end)),
        "test": np.flatnonzero(start >= val_end),
    }
    
    if any(len(rows) == 0 for rows in parts.values()):
        raise ValueError("La partición temporal por cuantiles deja un conjunto vacío.")
    for name, rows in parts.items():
        dates = start.iloc[rows]
        print(f"[split] {name}: {len(rows):,}; {dates.min()} a {dates.max()}", flush=True)
    
    return parts, {
        "train_fraction": train_fraction, "val_fraction": val_fraction,
        "train_end": train_end.isoformat(), "val_start": train_end.isoformat(),
        "val_end": val_end.isoformat(), "test_start": val_end.isoformat(),
        "counts": {name: len(rows) for name, rows in parts.items()},
        "eligible_count": len(start),
        "history_policy": "Atributos de DP conservados; antecedentes adicionales de Modeling disponibles antes del inicio; ventanas de 30 y 7 días conservadas.",
    }

def _source_values(history):
    values = history[["id_vaca", "inicio", "bloque_historial"]].copy()
    duration = pd.to_numeric(history["duracion_seg"], errors="raise")
    values["disponible"] = history["inicio"] + pd.to_timedelta(duration, unit="s")
    for event, key in zip(EVENTS[:3], EVENT_KEYS):
        values[f"{key}_previo"] = history[event]
    
    for base in ("mdi_faltante", "estado_mdi_sin_dato"):
        values[f"{base}_previo"] = history[base]
    
    values["mdi_previo"] = history["estado_mdi"]
    values["produccion_previa"] = history["main_produccion_kg"]
    inactive = history[[f"cuarto_inactivo_{q}" for q in QUARTERS]].eq(1)
    values["cuarto_inactivo_previo"] = inactive.any(axis=1).astype(float)
    conductivity = history[[f"conductividad_ms_cm_{q}" for q in QUARTERS]]
    values["conductividad_max_previa"] = conductivity.max(axis=1)
    values["conductividad_ms_cm_sin_dato_previa"] = history["conductividad_ms_cm_sin_dato"]
    for q in QUARTERS:
        values[f"conductividad_estimada_{q}_previa"] = history[f"conductividad_estimada_{q}"]
    
    return values

def _previous(sessions, values):
    left = sessions[["id_vaca", "bloque_historial", "inicio"]].copy()
    left["_position"] = np.arange(len(left))
    # Empates de disponibilidad: el rezago no elige una fila arbitrariamente.
    # Los indicadores usan any y las mediciones continuas la mediana del grupo.
    right = values.copy()
    keys = ["id_vaca", "bloque_historial", "disponible"]
    flags = binary_features(right.drop(columns=[*keys, "inicio"]).columns)
    aggregations = {c: ("max" if c in flags else "median") for c in right if c not in keys}
    right = right.groupby(keys, as_index=False).agg(aggregations)
    right = right.drop(columns="inicio").rename(columns={"disponible": "inicio"})

    # Selecciona la información anterior usando su fecha como referencia
    merged = pd.merge_asof(
        left.sort_values("inicio", kind="stable"), 
        right.sort_values("inicio", kind="stable"),
        on="inicio", by=["id_vaca", "bloque_historial"], 
        direction="backward", 
        allow_exact_matches=False,
    )

    # Vuelve a ordenar los ordeños y elimina las columnas temporales
    return (
        merged.sort_values("_position")
            .drop(columns=["id_vaca", "bloque_historial", "inicio", "_position"])
            .reset_index(drop=True)
    )

def _cow_rates(sessions, values, X):
    source_groups = dict(tuple(values.groupby(["id_vaca", "bloque_historial"], sort=False)))
    event_columns = [f"{key}_previo" for key in EVENT_KEYS]
    
    for key, group in sessions.groupby(["id_vaca", "bloque_historial"], sort=False):
        source = source_groups.get(key)
        if source is None or source.empty:
            continue
        daily = source.set_index("disponible")[event_columns].copy()
        daily["count"] = 1.0
        daily = daily.groupby(level=0).sum().sort_index()
        times = pd.DatetimeIndex(group["inicio"])
        grid = daily.index.union(times).unique().sort_values()
        daily = daily.reindex(grid, fill_value=0)
        sums = daily.rolling("30D", closed="left", min_periods=1).sum()
        
        for event, name in zip(event_columns, EVENT_KEYS):
            X.loc[group.index, f"{name}_vaca_30d"] = (sums[event] / sums["count"].replace(0, np.nan)).reindex(times).to_numpy()
        
def _herd_rates(sessions, history, values, X):
    if history.empty:
        for key in EVENT_KEYS:
            X[f"{key}_hato_7d"] = np.nan
        X["vacas_activas_7d"] = 0.0
        return
    
    # Solo días completos de resultados ya disponibles.
    days = values["disponible"].dt.normalize()
    query_days = sessions["inicio"].dt.normalize()

    calendar = pd.date_range(
        min(days.min(), 
        query_days.min()), 
        max(days.max(), 
        query_days.max()), 
        freq="D"
    )

    def previous_days(series):
        return series.reindex(calendar, fill_value=0).rolling(7).sum().shift(1)

    denominator = previous_days(days.value_counts())
    for event, key in zip(EVENTS[:3], EVENT_KEYS):
        rate = previous_days(history[event].groupby(days).sum()) / denominator.replace(0, np.nan)
        X[f"{key}_hato_7d"] = query_days.map(rate)
    
    presence = pd.crosstab(days, history["id_vaca"]).gt(0).reindex(calendar, fill_value=False)
    active = presence.rolling(7).max().shift(1).sum(axis=1, min_count=1)
    X["vacas_activas_7d"] = query_days.map(active)

def _calendar(sessions, history, X):
    """Aproximaciones desde primera observación; no son fechas de parto/alta reales."""
    start = sessions["inicio"]
    block_start = history.groupby(["id_vaca", "bloque_historial"])["inicio"].min()
    keys = pd.MultiIndex.from_frame(sessions[["id_vaca", "bloque_historial"]])
    first = pd.Series(block_start.reindex(keys).to_numpy(), index=sessions.index)
    
    herd_start = pd.Series(history.groupby("id_vaca")["inicio"].min().reindex(sessions["id_vaca"]).to_numpy(), index=sessions.index)
    for name, first_seen, bins in (("dias_en_leche", first, MILK_BINS), ("dias_en_hato", herd_start, HERD_BINS)):
        first_seen = first_seen.where(first_seen.le(start), start)
        elapsed = ((start - first_seen).dt.total_seconds() / 86400).clip(lower=0).fillna(0)
        codes = pd.cut(elapsed, bins=bins, right=False, labels=False)
        for category in range(1, len(bins) - 1):
            X[f"{name}_rango_{category}"] = codes.eq(category).astype(float)
    
    month = start.dt.month - 1
    hour = start.dt.hour + start.dt.minute / 60 + start.dt.second / 3600
    X["mes_sin"], X["mes_cos"] = np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)
    X["hora_sin"], X["hora_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    X["es_domingo"] = start.dt.dayofweek.eq(6).astype(float)

def build_features(sessions, history=None, machine_columns=None):
    """Conserva atributos de DP y construye solo el historial adicional de Modeling."""
    sessions = sessions.copy().reset_index(drop=True)
    history = sessions.copy() if history is None else history.copy().reset_index(drop=True)
    machines = sorted(c for c in sessions if c.startswith("maquina_")) if machine_columns is None else list(machine_columns)
    X = sessions[[*CONTEXT, *machines, *DP_FEATURES]].astype(float).copy()

    # Obtiene las mediciones previas de cada vaca y bloque.
    values = _source_values(history)
    previous = _previous(sessions, values)

    X = pd.concat([X, previous], axis=1)

    # Prepara columnas para los valores historicos
    for column in [f"{key}_vaca_30d" for key in EVENT_KEYS]:
        X[column] = np.nan

    # Agrega tasas históricas y variables de calendario.
    _cow_rates(sessions, values, X)
    _herd_rates(sessions, history, values, X)
    _calendar(sessions, history, X)
    X["produccion_previa"] = np.log1p(X["produccion_previa"])
    
    return X.astype(float)


def feature_group(column):
    """Agrupa dummies, ciclos e indicadores con su variable original para explicar alertas."""
    quality_groups = {
        "intervalo_horas_sin_dato": "intervalo_horas",
        "mdi_faltante_previo": "mdi_previo",
        "estado_mdi_sin_dato_previo": "mdi_previo",
        "conductividad_ms_cm_sin_dato_previa": "conductividad_max_previa",
    }
    if column in quality_groups:
        return quality_groups[column]
    if column.startswith("conductividad_estimada_") and column.endswith("_previa"):
        return "conductividad_max_previa"
    for name in ("dias_en_leche", "dias_en_hato", "mes", "hora", "maquina"):
        if column.startswith(name + "_"):
            return name
    return column


def binary_features(columns):
    continuous = {
        "intervalo_horas", "estado_numero_de_ordeno", "thi_sesion", "horas_thi_alto_24h",
        "mdi_previo", "conductividad_max_previa", "produccion_previa", "box_time_to_yield_ratio",
        "kicks_ratio_7d", "flies_proxy", "sangre_tasa_maquina_30d", "vacas_activas_7d",
        "mes_sin", "mes_cos", "hora_sin", "hora_cos",
        *[f"blood_diff_{q}" for q in QUARTERS[:3]],
        *[f"{key}_{window}" for key in EVENT_KEYS for window in ("vaca_30d", "hato_7d")],
    }
    return [c for c in columns if c not in continuous]
