import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
RUTA_VACAS = RAIZ / "Data" / "dataset_vacas_anonimo.csv"
RUTA_CLIMA = RAIZ / "Data" / "clima" / "clima_nasa_power_por_hora.csv"


def _a_nombre_columna(texto):
    """'Echada sin ordeñar' -> 'echada_sin_ordenar' (minúsculas, sin acentos ni espacios)."""
    sin_acentos = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return "_".join(sin_acentos.lower().split())


def convertir_fechas_y_duraciones(vacas):
    """Acción 1: convierte hora de inicio, duración e intervalo a datetime y número."""
    vacas = vacas.copy()

    # El archivo junta dos fuentes con formatos distintos:
    #   csv  (2025):      hora "dd/mm/aaaa hh:mm a. m.", duración "mm:ss"
    #   xlsx (2022-2026): hora "aaaa-mm-dd hh:mm:ss",    duración "hh:mm:ss"
    # La fuente se decide una sola vez, por el formato de la hora (el csv trae "a. m." / "p. m.")
    vacas["fuente"] = np.where(vacas["main_hora_de_inicio"].str.contains(r"[ap]\. m\.", regex=True), "csv", "xlsx")
    es_csv = vacas["fuente"] == "csv"
    print(f"[fuente] {es_csv.sum():,} filas del .csv y {(~es_csv).sum():,} del .xlsx")

    vacas["inicio"] = pd.Series(pd.NaT, index=vacas.index, dtype="datetime64[ns]")
    vacas.loc[es_csv, "inicio"] = pd.to_datetime(
        vacas.loc[es_csv, "main_hora_de_inicio"].str.replace("a. m.", "AM", regex=False).str.replace("p. m.", "PM", regex=False),
        format="%d/%m/%Y %I:%M %p")
    vacas.loc[~es_csv, "inicio"] = pd.to_datetime(vacas.loc[~es_csv, "main_hora_de_inicio"], format="%Y-%m-%d %H:%M:%S")
    print(f"[inicio] {es_csv.sum():,} fechas formateadas desde el formato .csv (dd/mm/aaaa hh:mm a. m.)")
    print(f"[inicio] {(~es_csv).sum():,} fechas formateadas desde el formato .xlsx (aaaa-mm-dd hh:mm:ss)")
    print(f"[inicio] rango: {vacas['inicio'].min()} a {vacas['inicio'].max()}")

    # La duración del csv es "mm:ss" y la del xlsx "hh:mm:ss"
    duracion = vacas["main_duracion_mm_ss"].where(~es_csv, "00:" + vacas["main_duracion_mm_ss"])
    vacas["duracion_seg"] = pd.to_timedelta(duracion).dt.total_seconds()
    print(f"[duracion_seg] {es_csv.sum():,} duraciones convertidas desde mm:ss (.csv) y {(~es_csv).sum():,} desde hh:mm:ss (.xlsx)")

    # Excel guarda los intervalos de 24 h o más como fecha ("1900-01-02 06:30:00"), no como "hh:mm:ss".
    # Con época 1899-12-31 la fecha equivale a la duración (se verificó contra la diferencia entre inicios consecutivos).
    intervalo = vacas["main_intervalo_de_ordeno_hh_mm"]
    es_fecha = intervalo.str.match(r"^\d{4}-\d{2}-\d{2} \d{1,2}:\d{2}:\d{2}$").fillna(False)
    es_hms = intervalo.str.match(r"^\d{1,2}:\d{2}:\d{2}$").fillna(False)
    sin_formato = intervalo.notna() & ~es_fecha & ~es_hms
    if sin_formato.any():
        raise ValueError(f"{sin_formato.sum()} intervalos con formato desconocido, ej.: {intervalo[sin_formato].iloc[0]!r}")
    print(f"[intervalo_horas] {es_hms.sum():,} intervalos convertidos desde hh:mm:ss (menos de 24 h)")
    print(f"[intervalo_horas] {es_fecha.sum():,} intervalos convertidos desde fecha de Excel (24 h o más)")
    horas = pd.Series(np.nan, index=vacas.index)
    horas[es_fecha] = (pd.to_datetime(intervalo[es_fecha]) - pd.Timestamp("1899-12-31")).dt.total_seconds() / 3600
    horas[es_hms] = pd.to_timedelta(intervalo[es_hms]).dt.total_seconds() / 3600
    vacas["intervalo_horas"] = horas
    print(f"[intervalo_horas] {horas.isna().sum():,} sin intervalo (el .csv no lo trae)")
    print("[columnas] se eliminan las 3 columnas de texto originales (ya convertidas)")

    return vacas.drop(columns=["main_hora_de_inicio", "main_duracion_mm_ss", "main_intervalo_de_ordeno_hh_mm"])


def quitar_duplicados(vacas):
    """Acción 2: quita las sesiones repetidas entre csv y xlsx y los duplicados exactos."""
    n_inicial = len(vacas)

    # Los registros de 2025 están en las dos fuentes: se quitan del csv los que ya vienen en el xlsx
    # (misma vaca y mismo minuto de inicio)
    vacas = vacas.assign(minuto=vacas["inicio"].dt.floor("min"))
    xlsx = vacas[vacas["fuente"] == "xlsx"].drop_duplicates(["id_vaca", "minuto"])
    cruce = vacas.reset_index().merge(xlsx, on=["id_vaca", "minuto"], suffixes=("", "_xlsx")).set_index("index")
    # El xlsx solo registra ordeños: un rechazo o una echada sin ordeñar del csv nunca es repetido, aunque caiga en el mismo minuto
    cruce = cruce[(cruce["fuente"] == "csv") & (cruce["main_accion"] == "Ordeño")]
    repetido = vacas.index.isin(cruce.index)
    print(f"[duplicados] {repetido.sum():,} ordeños del .csv que ya estaban en el .xlsx (misma vaca y mismo minuto)")

    # El csv no guarda segundos en la hora de inicio, así que se confirma con la duración (que sí los tiene)
    misma_duracion = ((cruce["duracion_seg"] - cruce["duracion_seg_xlsx"]).abs() <= 1).sum()
    print(f"[duplicados] {misma_duracion:,} de {len(cruce):,} coinciden además en duración (±1 s)")
    vacas = vacas[~repetido].drop(columns="minuto")

    exactos = vacas.duplicated()
    print(f"[duplicados] {exactos.sum():,} filas idénticas en todas las columnas")
    vacas = vacas[~exactos]

    print(f"[duplicados] {n_inicial:,} -> {len(vacas):,} filas ({n_inicial - len(vacas):,} eliminadas)")
    print(f"[duplicados] quedan {(vacas['fuente'] == 'csv').sum():,} filas del .csv y {(vacas['fuente'] == 'xlsx').sum():,} del .xlsx")
    return vacas.reset_index(drop=True)


def quitar_columnas_vacias_o_constantes(vacas):
    """Acción 3: elimina las columnas sin datos o con un solo valor, que no aportan al modelo."""
    n_valores = vacas.nunique(dropna=True)
    vacias = n_valores[n_valores == 0].index.tolist()
    constantes = n_valores[n_valores == 1].index.tolist()

    for col in vacias:
        print(f"[columnas] '{col}' vacía: sin ningún dato")
    for col in constantes:
        valor = vacas[col].dropna().iloc[0]
        print(f"[columnas] '{col}' constante: siempre {valor!r} ({vacas[col].notna().sum():,} registros, el resto nulo)")

    print(f"[columnas] {len(vacias) + len(constantes)} columnas eliminadas, quedan {vacas.shape[1] - len(vacias) - len(constantes)}")
    return vacas.drop(columns=vacias + constantes)


def marcar_acciones(vacas, base="Ordeño", min_registros_maquina=100):
    """Acción 4: convierte main_accion en dummies 0/1, una por cada acción distinta de la categoría base
    (la base queda con todas en 0), y deja fuera las máquinas con muy pocos registros."""
    vacas = vacas.copy()

    # El xlsx solo registra ordeños y no trae main_accion: sus nulos son la categoría base
    n_xlsx = vacas["main_accion"].isna().sum()
    vacas["main_accion"] = vacas["main_accion"].fillna(base)
    print(f"[acción] {n_xlsx:,} filas del .xlsx sin main_accion se marcan como '{base}'")
    print(f"[acción] {(vacas['main_accion'] == base).sum():,} visitas '{base}' (categoría base: todas las dummies en 0)")

    # drop_first quitaría la primera en orden alfabético, no necesariamente la base: se quita por nombre
    dummies = pd.get_dummies(vacas["main_accion"], dtype=int).drop(columns=base)
    dummies.columns = ["accion_" + _a_nombre_columna(c) for c in dummies.columns]
    for columna in dummies:
        print(f"[acción] {columna}: {dummies[columna].sum():,} visitas")
    vacas = pd.concat([vacas.drop(columns="main_accion"), dummies], axis=1)

    # Máquinas con muy pocos registros no permiten comparar nada: se dejan fuera
    por_maquina = vacas["misc_eo_po"].value_counts()
    raras = por_maquina[por_maquina < min_registros_maquina].index
    for maquina in raras:
        print(f"[máquina] '{maquina}' solo tiene {por_maquina[maquina]:,} registros: se deja fuera")
    vacas = vacas[~vacas["misc_eo_po"].isin(raras)].reset_index(drop=True)

    print(f"[acción] quedan {len(vacas):,} visitas en {vacas['misc_eo_po'].nunique()} máquinas")
    return vacas


if __name__ == "__main__":
    vacas = pd.read_csv(RUTA_VACAS, dtype={"id_vaca": str, "misc_eo_po": str}, low_memory=False)
    clima = pd.read_csv(RUTA_CLIMA, parse_dates=["hora"])

    vacas = convertir_fechas_y_duraciones(vacas)
    vacas = quitar_duplicados(vacas)
    vacas = quitar_columnas_vacias_o_constantes(vacas)
    vacas = marcar_acciones(vacas)
