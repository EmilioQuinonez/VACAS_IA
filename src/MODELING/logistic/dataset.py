"""Datos, atributos causales y partición cronológica; no modifica el CSV original."""

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
    
    if labeled:
        # Valida las etiquetas antes de construir atributos o usar el historial.
        get_labels(frame)
    return frame

def get_labels(sessions):
    labels = sessions[[*EVENTS[:3], "successful_milking_event"]]
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
        "history_policy": "Antecedentes estrictamente anteriores; ventanas de 30 y 7 días conservadas.",
    }

def _source_values(history):
    values = history[["id_vaca", "inicio", "bloque_historial"]].copy()
    for event, key in zip(EVENTS[:3], EVENT_KEYS):
        values[f"{key}_previo"] = history[event]
    
    for base in ("mdi_faltante", "estado_mdi_sin_dato"):
        values[f"{base}_previo"] = history[base]
    
    values["mdi_previo"] = history["estado_mdi"].mask(history["estado_mdi_sin_dato"].eq(1))
    values["produccion_previa"] = history["main_produccion_kg"]
    inactive = history[[f"cuarto_inactivo_{q}" for q in QUARTERS]].eq(1)
    values["cuarto_inactivo_previo"] = inactive.any(axis=1).astype(float)
    conductivity = history[[f"conductividad_ms_cm_{q}" for q in QUARTERS]]
    estimated = history[[f"conductividad_estimada_{q}" for q in QUARTERS]].eq(1).to_numpy()
    values["conductividad_max_previa"] = conductivity.mask(estimated | conductivity.le(0)).max(axis=1)
    
    for status, feature in (("pezones_no_encontrados", "fme_connection"), ("incompleto", "fme_incomplete")):
        values[feature] = history[[f"{status}_{q}" for q in QUARTERS]].eq(1).any(axis=1).astype(float)
    
    production = history[[f"producciones_kg_{q}" for q in QUARTERS]]
    values["fme_milk_production"] = (production.eq(0).to_numpy() & ~inactive.to_numpy()).any(axis=1).astype(float)
    blood = history[[f"sangre_ppm_{q}" for q in QUARTERS]]
    differences = blood.sub(blood.mean(axis=1), axis=0)
    
    for q in QUARTERS[:3]:
        values[f"blood_diff_{q}"] = differences[f"sangre_ppm_{q}"]
    
    values["box_time_to_yield_ratio"] = history["duracion_seg"] / 60 / history["main_produccion_kg"].where(history["main_produccion_kg"].gt(0))
    values["temperatura_previa"] = history["t_sesion"]
    values["cuartos_patada"] = history[[f"patada_{q}" for q in QUARTERS]].eq(1).sum(axis=1)
    values["con_sangre"] = blood.max(axis=1).gt(0).astype(float)
    
    return values

def _previous(sessions, values):
    left = sessions[["id_vaca", "bloque_historial", "inicio"]].copy()
    left["_position"] = np.arange(len(left))
    right = values.drop(columns=["cuartos_patada", "con_sangre"])

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
        daily = source.set_index("inicio")[[*event_columns, "cuartos_patada"]].copy()
        daily["count"] = 1.0
        times = pd.DatetimeIndex(group["inicio"])
        grid = daily.index.union(times).unique().sort_values()
        daily = daily.reindex(grid, fill_value=0)
        sums = daily.rolling("30D", closed="left", min_periods=1).sum()
        
        for event, name in zip(event_columns, EVENT_KEYS):
            X.loc[group.index, f"{name}_vaca_30d"] = (sums[event] / sums["count"].replace(0, np.nan)).reindex(times).to_numpy()
        
        sums7 = daily[["cuartos_patada", "count"]].rolling("7D", closed="left", min_periods=1).sum()
        X.loc[group.index, "kicks_ratio_7d"] = (sums7["cuartos_patada"] / sums7["count"].replace(0, np.nan)).reindex(times).to_numpy()

def _herd_rates(sessions, history, X):
    days = history["inicio"].dt.normalize()
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

def _machine_rates(sessions, history, values, machines, X):
    def machine_id(frame):
        flags = frame[machines]
        if not flags.isin([0, 1]).all().all() or flags.sum(axis=1).gt(1).any():
            raise ValueError("Dummies de máquina inválidas.")
        return flags.astype(int).astype(str).agg("|".join, axis=1)

    source_machine, query_machine = machine_id(history), machine_id(sessions)
    for machine in query_machine.unique():
        rows = query_machine.index[query_machine.eq(machine)]
        source_rows = source_machine.index[source_machine.eq(machine)]
        if len(source_rows) == 0:
            X.loc[rows, "sangre_tasa_maquina_30d"] = 0.0
            continue
        source = values.loc[source_rows].set_index("inicio")[["con_sangre"]].assign(count=1.0).groupby(level=0).sum()
        times = pd.DatetimeIndex(sessions.loc[rows, "inicio"])
        grid = source.index.union(times).unique().sort_values()
        sums = source.reindex(grid, fill_value=0).rolling("30D", closed="left", min_periods=1).sum()
        rate = (sums["con_sangre"] / sums["count"].replace(0, np.nan)).fillna(0)
        X.loc[rows, "sangre_tasa_maquina_30d"] = rate.reindex(times).to_numpy()
    X["sangre_pico_maquina"] = X["sangre_tasa_maquina_30d"].gt(0.15).astype(float)

def _calendar(sessions, history, X):
    start = sessions["inicio"]
    block_start = history.groupby(["id_vaca", "bloque_historial"])["inicio"].min()
    keys = pd.MultiIndex.from_frame(sessions[["id_vaca", "bloque_historial"]])
    first = pd.Series(block_start.reindex(keys).to_numpy(), index=sessions.index)
    
    herd_start = sessions["id_vaca"].map(history.groupby("id_vaca")["inicio"].min())
    for name, first_seen, bins in (("dias_en_leche", first, MILK_BINS), ("dias_en_hato", herd_start, HERD_BINS)):
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
    """Construye X. En inferencia solo history aporta mediciones y etiquetas, nunca las consultas."""
    sessions = sessions.copy().reset_index(drop=True)
    history = sessions.copy() if history is None else history.copy().reset_index(drop=True)
    machines = machine_columns or sorted(c for c in sessions if c.startswith("maquina_"))
    
    for name, frame in (("sesiones", sessions), ("historial", history)):
        blocks = pd.to_numeric(frame["bloque_historial"], errors="raise")
        if blocks.isna().any() or not np.isfinite(blocks).all() or blocks.mod(1).ne(0).any():
            raise ValueError(f"bloque_historial debe contener enteros sin nulos en {name}.")
        
        frame["bloque_historial"] = blocks.astype("int64")
        unknown = [c for c in frame if c.startswith("maquina_") and c not in machines and frame[c].ne(0).any()]
        if unknown:
            raise ValueError(f"Máquinas fuera del esquema guardado en {name}: {unknown}")
    
    X = sessions[[*CONTEXT, *machines]].astype(float).copy()
    X["intervalo_horas"] = X["intervalo_horas"].mask(X["intervalo_horas_sin_dato"].eq(1))

    # Obtiene las mediciones previas de cada vaca y bloque.
    values = _source_values(history)
    previous = _previous(sessions, values)

    # La temperatura previa se reserva para construir flies_proxy.
    X = pd.concat([X, previous.drop(columns="temperatura_previa")], axis=1)

    # Prepara columnas para los valores historicos
    for column in [*[f"{key}_vaca_30d" for key in EVENT_KEYS], "kicks_ratio_7d"]:
        X[column] = np.nan

    # Agrega tasas históricas y variables de calendario.
    _cow_rates(sessions, values, X)
    _herd_rates(sessions, history, X)
    _machine_rates(sessions, history, values, machines, X)
    _calendar(sessions, history, X)
    afternoon = sessions["inicio"].dt.hour.between(13, 17).astype(float)
    
    X["flies_proxy"] = afternoon * previous["temperatura_previa"] * X["kicks_ratio_7d"]
    
    for column in ("produccion_previa", "box_time_to_yield_ratio"):
        if X[column].dropna().lt(0).any():
            raise ValueError(f"{column} contiene valores negativos.")
        X[column] = np.log1p(X[column])
    
    if np.isinf(X.to_numpy(dtype=float)).any():
        raise ValueError("Los predictores contienen infinitos.")
    
    return X.astype(float)


def feature_group(column):
    """Agrupa dummies, ciclos e indicadores con su variable original para explicar alertas."""
    quality_groups = {
        "intervalo_horas_sin_dato": "intervalo_horas",
        "mdi_faltante_previo": "mdi_previo",
        "estado_mdi_sin_dato_previo": "mdi_previo",
    }
    if column in quality_groups:
        return quality_groups[column]
    for name in ("dias_en_leche", "dias_en_hato", "mes", "hora"):
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
