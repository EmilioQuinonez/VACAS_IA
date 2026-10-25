"""
Lee Data/dataset_vacas_integrado.csv, generado primero por `integrar_clima.py` a partir del dataset
limpio de `clean_data.py`, y agrega los atributos y eventos guardando el resultado sobre ese mismo archivo.
Conserva todas las filas, su orden y sus columnas originales, incluidas las de clima.
Los FME, diferencias de sangre y tiempo/producción usan
la sesión anterior de cada vaca dentro de su bloque de historial (se reinicia tras 45 días de ausencia);
la tasa de patadas usa [inicio - 7 días, inicio). Las cuatro etiquetas describen la sesión actual y pueden
coexistir, salvo éxito; los rechazos y echadas sin ordeñar no son ordeño, así que no tienen etiqueta.

Uso:
    python src/DP/integrate_derivates.py
"""
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = ROOT / "Data" / "dataset_vacas_integrado.csv"
OUTPUT_PATH = INPUT_PATH

QUARTERS = ("di", "dd", "ti", "td")
# Banderas 0/1 por cuarto de clean_data.py: <estado>_<cuarto>.
STATUSES = ("patada", "incompleto", "pezones_no_encontrados", "amd_pezon")
# clean_data.py deja estado_mdi y sangre_ppm_* en logaritmo (log1p): los umbrales se convierten con la misma función.
MDI_THRESHOLD = math.log1p(1.4)  # Alerta MDi >= 1.4 en la escala original.
BLOOD_PPM_THRESHOLD = math.log1p(0)  # Presencia de sangre: cualquier cuarto > 0 ppm (log1p(0) = 0).
AFTERNOON_START_HOUR = 13
AFTERNOON_END_HOUR = 17  # Extremo excluido: [13:00, 17:00).

# Todos los atributos derivados, en el orden de salida.
FEATURES = [
    "flies_proxy", "fme_connection", "fme_milk_production", "fme_incomplete",
    "kicks_ratio_7d",
    *[f"blood_diff_{quarter}" for quarter in QUARTERS],
    "box_time_to_yield_ratio",
]
EVENTS = [
    "mechanical_connection_failure_event", "health_anomaly_event",
    "behavior_interruption_event", "successful_milking_event",
]

def add_derived_features(
        milky_sessions, 
        mdi_threshold=MDI_THRESHOLD, 
        blood_threshold=BLOOD_PPM_THRESHOLD
):
    """
    Construye los predictores y etiquetas de evento para cada sesión.

    Los nulos de estados y sangre se interpretan como ausencia. Sin sesión 
    anterior o sin visitas en los siete días previos, los atributos históricos 
    quedan nulos.

    Las sesiones con el mismo inicio no se usan entre sí como antecedentes.
    Para la visita siguiente se toma la última fila original de ese inicio.
    La ventana cuenta todas las filas anteriores, sin eliminar duplicados.

    Args:
        milky_sessions (pd.DataFrame): Sesiones del dataset limpio (clean_data.py) con
            t_sesion en grados Celsius. Debe incluir identificación, inicio (fecha y hora),
            duración en segundos, producción, banderas por cuarto de cada estado y sangre.
            La sangre y el MDI vienen en logaritmo (log1p), igual que los umbrales.

        mdi_threshold (float): Límite inclusivo del índice MDI que activa la
            etiqueta de salud; por defecto, MDI >= 1.4.

        blood_threshold (float): Límite exclusivo de sangre por cuarto en ppm
            que activa la etiqueta de salud; por defecto, sangre > 0 ppm.

    Returns:
        pd.DataFrame: Copia de milky_sessions que conserva sus filas, índice, orden y
            columnas originales, y añade las columnas FEATURES y EVENTS.
            Los predictores usan la sesión anterior o el historial de siete días;
            flies_proxy combina la temperatura anterior con la hora actual.
    """
    required_columns = {
        "id_vaca", "inicio", "bloque_historial", "duracion_seg", "main_produccion_kg", "t_sesion",
        "estado_mdi", "estado_amd_ubre", "accion_rechazada", "accion_echada_sin_ordenar",
        *[f"{status}_{quarter}" for status in STATUSES for quarter in QUARTERS],
        *[f"producciones_kg_{quarter}" for quarter in QUARTERS],
        *[f"sangre_ppm_{quarter}" for quarter in QUARTERS],
    }
    missing_columns = required_columns.difference(milky_sessions.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing_columns))}")
    existing_columns = set(FEATURES + EVENTS).intersection(milky_sessions.columns)
    if existing_columns:
        raise ValueError(f"Derived columns already exist: {', '.join(sorted(existing_columns))}")

    # Índice posicional interno para conservar incluso índices originales repetidos.
    base = milky_sessions.reset_index(drop=True)
    start_time = pd.to_datetime(base["inicio"])
    is_milking = base[["accion_rechazada", "accion_echada_sin_ordenar"]].eq(0).all(axis=1)
    if base["id_vaca"].isna().any() or start_time.isna().any():
        raise ValueError("Each session must have id_vaca and a valid start time.")

    quarter_yields = base[[f"producciones_kg_{quarter}" for quarter in QUARTERS]].apply(pd.to_numeric)
    blood = base[[f"sangre_ppm_{quarter}" for quarter in QUARTERS]].apply(pd.to_numeric).fillna(0)
    total_yield = pd.to_numeric(base["main_produccion_kg"])
    status = {name: base[[f"{name}_{quarter}" for quarter in QUARTERS]].eq(1) for name in STATUSES}
    kicked_quarters = status["patada"].sum(axis=1)

    # Primero se calculan los valores de cada sesión, después se desfasan por vaca.
    current_features = pd.DataFrame(index=base.index)

    # fme_connection: presencia de algún pezón no encontrado.
    current_features["fme_connection"] = status["pezones_no_encontrados"].any(axis=1).astype(int)

    # fme_milk_production: alguna producción por cuarto exactamente igual a cero (solo ordeños).
    current_features["fme_milk_production"] = (quarter_yields.eq(0).any(axis=1) & is_milking).astype(int)

    # fme_incomplete: estado de ordeño incompleto activo.
    current_features["fme_incomplete"] = status["incompleto"].any(axis=1).astype(int)

    # blood_diff_di/dd/ti/td: sangre del cuarto menos la media de los cuatro cuartos.
    blood_differences = blood.sub(blood.mean(axis=1), axis=0)
    current_features[[f"blood_diff_{quarter}" for quarter in QUARTERS]] = blood_differences.to_numpy()

    # box_time_to_yield_ratio: duración en segundos / producción total en kg (s/kg).
    current_features["box_time_to_yield_ratio"] = (
        pd.to_numeric(base["duracion_seg"])
        / total_yield.where(total_yield.gt(0))
    )

    derived_features = pd.DataFrame(index=base.index, columns=FEATURES, dtype=float)
    previous_temperature = pd.Series(float("nan"), index=base.index, dtype=float)
    session_order = pd.DataFrame({
        "cow_id": base["id_vaca"], "block": base["bloque_historial"], "start_time": start_time
    })
    session_order = session_order.sort_values(["cow_id", "start_time"], kind="stable")

    # El historial se reinicia en cada bloque: tras 45 días de ausencia no hay sesión anterior.
    for _, group in session_order.groupby(["cow_id", "block"], sort=False):
        rows = group.index
        # Desfase de FME, blood_diff_*, box_time_to_yield_ratio y temperatura:
        # sesión previa de la misma vaca con un inicio estrictamente anterior.
        previous_features = current_features.loc[rows].set_axis(
            pd.DatetimeIndex(group["start_time"])
        )
        previous_features["previous_temperature"] = pd.to_numeric(
            base.loc[rows, "t_sesion"]
        ).to_numpy()
        previous_features = previous_features.loc[
            ~previous_features.index.duplicated(keep="last")
        ]
        # Desplaza los valores una sesión hacia adelante
        previous_features = previous_features.shift(1).reindex(group["start_time"]).set_axis(rows)
        derived_features.loc[
            rows, 
            current_features.columns
        ] = previous_features[current_features.columns]

        previous_temperature.loc[rows] = previous_features["previous_temperature"]

        # Agrupar instantes iguales impide contar otra fila del mismo inicio.
        history = pd.DataFrame({
            "kicked_quarters": kicked_quarters.loc[rows].to_numpy(), 
            "session_count": 1,
        }, index=pd.DatetimeIndex(group["start_time"]))

        history = history.groupby(level=0).sum()

        # kicks_ratio_7d: suma de cuartos afectados / número de sesiones de la vaca
        # en [inicio - 7 días, inicio), excluyendo todas las filas del inicio actual.
        window = history.rolling(
            "7D", 
            closed="left", 
            min_periods=1
        ).sum()

        kick_rate = window["kicked_quarters"] / window["session_count"]
        derived_features.loc[rows, "kicks_ratio_7d"] = (
            kick_rate.reindex(group["start_time"]).to_numpy()
        )

    # flies_proxy: indicador de tarde [13:00, 17:00) de la sesión actual *
    # temperatura de la sesión anterior en grados Celsius * tasa de patadas de 7 días.
    # Sin temperatura de la sesión anterior, flies_proxy queda faltante.
    is_afternoon = (
        start_time.dt.hour.ge(AFTERNOON_START_HOUR) & 
        start_time.dt.hour.lt(AFTERNOON_END_HOUR)
    )

    derived_features["flies_proxy"] = (
        is_afternoon.astype(int) * 
        previous_temperature * 
        derived_features["kicks_ratio_7d"]
    )

    # Las etiquetas usan la sesión actual
    events = pd.DataFrame(index=base.index)

    # mechanical_connection_failure_event: cualquiera de los tres FME actuales.
    events[EVENTS[0]] = current_features[[
        "fme_connection", "fme_milk_production", "fme_incomplete"
    ]].any(axis=1)

    # health_anomaly_event: sangre > umbral en algún cuarto, MDI >= umbral
    # o estado AMD activo de ubre o pezón.
    events[EVENTS[1]] = (
        blood.gt(blood_threshold).any(axis=1)
        | pd.to_numeric(base["estado_mdi"]).ge(mdi_threshold)
        | pd.to_numeric(base["estado_amd_ubre"]).gt(0)
        | status["amd_pezon"].any(axis=1)
    )

    # behavior_interruption_event: al menos un cuarto afectado por patada.
    events[EVENTS[2]] = kicked_quarters.gt(0)

    # Los rechazos y echadas sin ordeñar no son ordeño: no tienen etiqueta (sus mediciones son de relleno).
    events = events.mask(~is_milking, False)

    # successful_milking_event: ordeño sin ninguno de los tres eventos anteriores.
    events[EVENTS[3]] = is_milking & ~events.any(axis=1)

    result = milky_sessions.copy()
    result[FEATURES] = derived_features.to_numpy()
    result[EVENTS] = events.astype("int8").to_numpy()

    return result


def main():
    """
    Lee las sesiones, agrega atributos y eventos, guarda el CSV e imprime un 
    resumen.
    """
    milky_sessions = pd.read_csv(INPUT_PATH, dtype={"id_vaca": str}, parse_dates=["inicio"], low_memory=False)
    integrated = add_derived_features(milky_sessions)
    integrated.to_csv(OUTPUT_PATH, index=False)

    print(f"Sessions: {len(integrated):,} | cows: {integrated['id_vaca'].nunique()}")
    print(f"Health: MDI >= {MDI_THRESHOLD}, blood > {BLOOD_PPM_THRESHOLD} ppm or active AMD alert")
    for event in EVENTS:
        print(f"{event}: {integrated[event].sum():,} ({integrated[event].mean():.1%})")
    print(f"Without a previous session: {integrated['fme_connection'].isna().sum():,} | "
          f"without 7-day kick history: {integrated['kicks_ratio_7d'].isna().sum():,}")
    print(f"Columns: {integrated.shape[1]} | saved to {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
