"""
Lee Data/dataset_vacas_integrado.csv, generado primero por `integrar_clima.py`,
y agrega los atributos y eventos guardando el resultado sobre ese mismo archivo.
Conserva todas las filas, su orden y sus columnas originales, incluidas las de clima.
Los FME, diferencias de sangre y tiempo/producción usan
la sesión anterior de cada vaca; la tasa de patadas usa [inicio - 7 días, inicio).
Las cuatro etiquetas describen la sesión actual y pueden coexistir, salvo éxito.

Uso:
    python src/DP/integrate_derivates.py
"""
from pathlib import Path

import pandas as pd

try:
    from .integrar_clima import hora_de_inicio
except ImportError:
    from integrar_clima import hora_de_inicio

ROOT = Path(__file__).resolve().parent.parent.parent
INPUT_PATH = ROOT / "Data" / "dataset_vacas_integrado.csv"
OUTPUT_PATH = INPUT_PATH

QUARTERS = ("di", "dd", "ti", "td")
MDI_THRESHOLD = 1.4  # Alerta MDi >= 1.4.
BLOOD_PPM_THRESHOLD = 0  # Presencia de sangre: cualquier cuarto > 0 ppm.
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

def is_status_active(status):
    """
    Identifica estados activos; vacío o cero significa ausencia de evento.

    Args:
        status (pd.Series): Estados originales, numéricos o listas de cuartos.
            Los valores faltantes y las cadenas vacías se consideran inactivos.

    Returns:
        pd.Series: Serie booleana con el mismo índice; True indica estado activo.
    """
    text = status.fillna("").astype(str).str.strip()
    number = pd.to_numeric(text, errors="coerce")

    return text.ne("") & number.ne(0)

def count_kicked_quarters(kicks):
    """
    Cuenta los cuartos distintos afectados por patadas en cada sesión.

    Args:
        kicks (pd.Series): Columna estado_patada; contiene listas como 'DI,TD',
            'Todos', valores vacíos o ceros. 'Todos' representa cuatro cuartos.

    Returns:
        pd.Series: Número entero de cuartos afectados por sesión, de 0 a 4,
            conservando el índice original.
    """
    text = kicks.fillna("").astype(str).str.strip().str.upper()
    count = pd.DataFrame({
        quarter: text.str.contains(rf"\b{quarter.upper()}\b", regex=True)
        for quarter in QUARTERS
    }).sum(axis=1)
    # Se conserva 'TODOS' porque es un valor del dataset de entrada.
    count = count.mask(text.eq("TODOS"), 4)

    return count

def parse_duration_minutes(duration):
    """
    Convierte las duraciones mm:ss y hh:mm:ss a minutos con parte decimal.

    Args:
        duration (pd.Series): Columna main_duracion_mm_ss con las duraciones
            originales; puede contener valores faltantes.

    Returns:
        pd.Series: Duración numérica en minutos, con el mismo índice. Los
            segundos se conservan como fracción de minuto y los nulos permanecen.
    """
    duration = duration.astype("string").str.strip()
    duration = duration.where(duration.str.count(":").ne(1), "00:" + duration)
    
    return pd.to_timedelta(duration).dt.total_seconds() / 60

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
        milky_sessions (pd.DataFrame): Sesiones con sus columnas originales en español
            y t_sesion en grados Celsius. Debe incluir identificación, inicio, 
            duración, producción, estados y sangre.

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
        "id_vaca", "main_hora_de_inicio", "main_duracion_mm_ss", "main_produccion_kg",
        "t_sesion", "estado_patada", "estado_incompleto", "estado_pezones_no_encontrados",
        "estado_mdi", "estado_amd_ubre", "estado_amd_pezon",
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
    start_time = hora_de_inicio(base["main_hora_de_inicio"])
    if base["id_vaca"].isna().any() or start_time.isna().any():
        raise ValueError("Each session must have id_vaca and a valid start time.")

    quarter_yields = base[[f"producciones_kg_{quarter}" for quarter in QUARTERS]].apply(pd.to_numeric)
    blood = base[[f"sangre_ppm_{quarter}" for quarter in QUARTERS]].apply(pd.to_numeric).fillna(0)
    total_yield = pd.to_numeric(base["main_produccion_kg"])
    kicked_quarters = count_kicked_quarters(base["estado_patada"])

    # Primero se calculan los valores de cada sesión, después se desfasan por vaca.
    current_features = pd.DataFrame(index=base.index)

    # fme_connection: presencia de algún pezón no encontrado.
    current_features["fme_connection"] = is_status_active(base["estado_pezones_no_encontrados"]).astype(int)

    # fme_milk_production: alguna producción por cuarto exactamente igual a cero.
    current_features["fme_milk_production"] = quarter_yields.eq(0).any(axis=1).astype(int)

    # fme_incomplete: estado de ordeño incompleto activo.
    current_features["fme_incomplete"] = is_status_active(base["estado_incompleto"]).astype(int)

    # blood_diff_di/dd/ti/td: sangre del cuarto menos la media de los cuatro cuartos.
    blood_differences = blood.sub(blood.mean(axis=1), axis=0)
    current_features[[f"blood_diff_{quarter}" for quarter in QUARTERS]] = blood_differences.to_numpy()

    # box_time_to_yield_ratio: duración en minutos / producción total en kg.
    current_features["box_time_to_yield_ratio"] = (
        parse_duration_minutes(base["main_duracion_mm_ss"])
        / total_yield.where(total_yield.gt(0))
    )

    derived_features = pd.DataFrame(index=base.index, columns=FEATURES, dtype=float)
    previous_temperature = pd.Series(float("nan"), index=base.index, dtype=float)
    session_order = pd.DataFrame({"cow_id": base["id_vaca"], "start_time": start_time})
    session_order = session_order.sort_values(["cow_id", "start_time"], kind="stable")
    
    for _, group in session_order.groupby("cow_id", sort=False):
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
        | is_status_active(base["estado_amd_ubre"])
        | is_status_active(base["estado_amd_pezon"])
    )

    # behavior_interruption_event: al menos un cuarto afectado por patada.
    events[EVENTS[2]] = kicked_quarters.gt(0)

    # successful_milking_event: ausencia de los tres eventos anteriores.
    events[EVENTS[3]] = ~events.any(axis=1)

    result = milky_sessions.copy()
    result[FEATURES] = derived_features.to_numpy()
    result[EVENTS] = events.astype("int8").to_numpy()

    return result


def main():
    """
    Lee las sesiones, agrega atributos y eventos, guarda el CSV e imprime un 
    resumen.
    """
    milky_sessions = pd.read_csv(INPUT_PATH, dtype={"id_vaca": str, "misc_eo_po": str}, low_memory=False)
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
