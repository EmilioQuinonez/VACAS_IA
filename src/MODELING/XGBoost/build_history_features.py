"""
Agrega a Data/dataset_vacas_integrado.csv las variables nuevas de la tabla de selección de
datos (pestaña DP del Doc): calendario, ordeño previo, tasas de la vaca y del hato.

Todas usan solo información anterior al inicio de la sesión, y el historial se toma solo de
ordeños (sin rechazos ni echadas). Se conservan todas las filas; si las columnas ya existen,
se reemplazan.

Uso:
    python src/MODELING/XGBoost/build_history_features.py
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
INPUT_PATH = ROOT / "Data" / "dataset_vacas_integrado.csv"
OUTPUT_PATH = INPUT_PATH

QUARTERS = ("di", "dd", "ti", "td")
EVENTS = {
    "mechanical_connection_failure_event": "mecanico",
    "health_anomaly_event": "salud",
    "behavior_interruption_event": "conducta",
}
COW_WINDOW = "30D"
HERD_DAYS = 7

PREVIOUS = [
    *[f"{key}_previo" for key in EVENTS.values()],
    "mdi_previo", "conductividad_max_previa", "produccion_previa", "cuarto_inactivo_previo",
]
NEW_COLUMNS = [
    "mes", "dia_semana", "hora", "dias_en_leche", "dias_en_hato", *PREVIOUS,
    *[f"{key}_vaca_30d" for key in EVENTS.values()],
    *[f"{key}_hato_7d" for key in EVENTS.values()],
    "vacas_activas_7d",
]


def add_calendar(sessions, new):
    """Mes, día de la semana (0 = lunes), hora y días desde la llegada al bloque y al hato."""
    start = sessions["inicio"]
    new["mes"] = start.dt.month
    new["dia_semana"] = start.dt.day_of_week
    new["hora"] = start.dt.hour

    # bloque_historial se reinicia tras 45 días de ausencia: se toma como inicio de lactancia.
    block_start = start.groupby([sessions["id_vaca"], sessions["bloque_historial"]]).transform("min")
    herd_start = start.groupby(sessions["id_vaca"]).transform("min")
    new["dias_en_leche"] = (start - block_start).dt.total_seconds() / 86400
    new["dias_en_hato"] = (start - herd_start).dt.total_seconds() / 86400


def add_previous_session(sessions, milkings, new):
    """Valores del último ordeño de la vaca, en el mismo bloque, con inicio estrictamente anterior."""
    m = sessions.loc[milkings]
    values = pd.DataFrame({
        "id_vaca": m["id_vaca"], "bloque_historial": m["bloque_historial"], "inicio": m["inicio"],
        "mdi_previo": m["estado_mdi"], "produccion_previa": m["main_produccion_kg"],
    })
    for event, key in EVENTS.items():
        values[f"{key}_previo"] = m[event]

    # Solo cuartos con lectura real: la conductividad estimada en limpieza no cuenta.
    conductivity = m[[f"conductividad_ms_cm_{q}" for q in QUARTERS]]
    estimated = m[[f"conductividad_estimada_{q}" for q in QUARTERS]].eq(1).to_numpy()
    values["conductividad_max_previa"] = conductivity.mask(estimated).max(axis=1)
    values["cuarto_inactivo_previo"] = m[[f"cuarto_inactivo_{q}" for q in QUARTERS]].eq(1).any(axis=1).astype(int)

    left = sessions[["id_vaca", "bloque_historial", "inicio"]].reset_index().sort_values("inicio", kind="stable")
    merged = pd.merge_asof(
        left, values.sort_values("inicio", kind="stable"),
        on="inicio", by=["id_vaca", "bloque_historial"],
        direction="backward", allow_exact_matches=False,
    )
    new[PREVIOUS] = merged.set_index("index")[PREVIOUS].reindex(sessions.index)


def add_cow_rates(sessions, milkings, new):
    """Proporción de ordeños con cada evento de la vaca en [inicio - 30 d, inicio); solo en ordeños."""
    m = sessions.loc[milkings, ["id_vaca", "inicio", *EVENTS]].sort_values(["id_vaca", "inicio"], kind="stable")
    rates = m.set_index("inicio").groupby("id_vaca")[list(EVENTS)].rolling(COW_WINDOW, closed="left").mean()
    for event, key in EVENTS.items():
        new[f"{key}_vaca_30d"] = pd.Series(rates[event].to_numpy(), index=m.index)


def add_herd_rates(sessions, milkings, new):
    """
    Tasa de cada evento y vacas distintas del hato en los 7 días naturales anteriores al día
    de la sesión. Por día y no por hora, para no usar ordeños de otras vacas aún en curso.
    """
    m = sessions.loc[milkings]
    day = m["inicio"].dt.normalize()
    calendar = pd.date_range(day.min(), day.max(), freq="D")

    def previous_days(daily):
        return daily.reindex(calendar, fill_value=0).rolling(HERD_DAYS).sum().shift(1)

    sessions_7d = previous_days(day.value_counts())
    session_day = sessions["inicio"].dt.normalize()
    for event, key in EVENTS.items():
        rate = previous_days(m[event].groupby(day).sum()) / sessions_7d
        new[f"{key}_hato_7d"] = session_day.map(rate)

    presence = pd.crosstab(day, m["id_vaca"]).gt(0).reindex(calendar, fill_value=False)
    active_cows = presence.rolling(HERD_DAYS).max().shift(1).sum(axis=1, min_count=1)
    new["vacas_activas_7d"] = session_day.map(active_cows)


def main():
    """Agrega las variables nuevas al dataset integrado e imprime un resumen."""
    sessions = pd.read_csv(INPUT_PATH, dtype={"id_vaca": str}, parse_dates=["inicio"], low_memory=False)
    sessions = sessions.drop(columns=[c for c in NEW_COLUMNS if c in sessions])
    milkings = sessions[["accion_rechazada", "accion_echada_sin_ordenar"]].eq(0).all(axis=1)

    new = pd.DataFrame(index=sessions.index)
    add_calendar(sessions, new)
    add_previous_session(sessions, milkings, new)
    add_cow_rates(sessions, milkings, new)
    add_herd_rates(sessions, milkings, new)
    sessions = pd.concat([sessions, new[NEW_COLUMNS]], axis=1)
    sessions.to_csv(OUTPUT_PATH, index=False)

    print(f"Sessions: {len(sessions):,} | milkings: {milkings.sum():,} | columns: {sessions.shape[1]}")
    missing = sessions.loc[milkings, NEW_COLUMNS].isna().sum()
    print("Missing values in milkings:", {c: int(n) for c, n in missing.items() if n})
    print(f"Saved to {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
