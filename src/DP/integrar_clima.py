"""
Une las sesiones de ordeño con el clima por hora de NASA POWER (fase de Data Preparation).

Lee Data/dataset_vacas_anonimo.csv (vacas/3.0) y Data/clima/clima_nasa_power_por_hora.csv
(descargar_clima.py), aplica la limpieza decidida en el entendimiento de los datos, agrega el
clima de la hora en que inicia cada sesion y las horas con THI alto de las 24 h previas, y guarda
Data/dataset_vacas_clima.csv (vacas/5.0).

Uso:
    python src/DP/integrar_clima.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

raiz = Path(__file__).resolve().parent.parent.parent
ORIGEN = raiz / "Data" / "dataset_vacas_anonimo.csv"
CLIMA = raiz / "Data" / "clima" / "clima_nasa_power_por_hora.csv"
SALIDA = raiz / "Data" / "dataset_vacas_clima.csv"

CUARTOS = ["di", "dd", "ti", "td"]
# NASA POWER etiqueta cada hora una hora antes del clima que describe
DESFASE_POWER_H = 1
UMBRAL_THI = 68
# Un intervalo de mas de 48 h es el regreso tras una brecha (periodo seco), no un intervalo de ordeño
UMBRAL_INTERVALO_H = 48
ALERTAS = {"estado_patada": "patada", "estado_incompleto": "incompleto",
           "estado_pezones_no_encontrados": "pezon_no_encontrado", "estado_amd_pezon": "amd_pezon"}
# Columnas vacias o constantes, y el intervalo del sistema, que solo trae una de las dos fuentes
DESCARTADAS = ["estado_rcs_1000_celulas_ml", "misc_usuario", "misc_programa_de_lavado",
               "main_intervalo_de_ordeno_hh_mm"]


def cargar_sesiones(ruta):
    """
        Lee el dataset anonimizado y aplica la limpieza del reporte de datos (seccion 3.3, pasos 1 a 7):
        fechas y duraciones a numero, sin sesiones repetidas entre csv y xlsx ni duplicados exactos,
        solo las tres maquinas, alertas por cuarto como banderas 0/1, conductividad 0 como faltante
        e intervalos de mas de 48 h como faltantes con bandera.

        Args:
            ruta (Path): CSV generado por anonymize_data.py.

        Returns:
            DataFrame: Una fila por visita de una vaca a una de las tres maquinas, ordenada por vaca e inicio.
    """
    bruto = pd.read_csv(ruta, dtype={"id_vaca": str, "misc_eo_po": str}, low_memory=False)

    # El archivo junta dos fuentes con formatos distintos:
    #   csv  (mar-jul 2025): hora "dd/mm/aaaa hh:mm a. m.", duracion "mm:ss", trae main_accion
    #   xlsx (2022-2026):    hora "aaaa-mm-dd hh:mm:ss",   duracion "hh:mm:ss", no trae main_accion
    es_csv = bruto["main_hora_de_inicio"].str.len() == 22
    bruto["inicio"] = pd.Series(pd.NaT, index=bruto.index, dtype="datetime64[ns]")
    bruto.loc[es_csv, "inicio"] = pd.to_datetime(
        bruto.loc[es_csv, "main_hora_de_inicio"].str.replace("a. m.", "AM", regex=False).str.replace("p. m.", "PM", regex=False),
        format="%d/%m/%Y %I:%M %p")
    bruto.loc[~es_csv, "inicio"] = pd.to_datetime(bruto.loc[~es_csv, "main_hora_de_inicio"], format="%Y-%m-%d %H:%M:%S")

    duracion = bruto["main_duracion_mm_ss"].where(bruto["main_duracion_mm_ss"].str.len() == 8, "00:" + bruto["main_duracion_mm_ss"])
    bruto["duracion_seg"] = pd.to_timedelta(duracion).dt.total_seconds()

    # Los registros de 2025 estan en las dos fuentes: se quitan del csv los que ya vienen en el xlsx (misma vaca y mismo minuto)
    llave = pd.MultiIndex.from_arrays([bruto["id_vaca"], bruto["inicio"].dt.floor("min")])
    repetido = es_csv & llave.isin(llave[~es_csv])
    sesiones = bruto[~repetido].drop_duplicates().copy()
    sesiones["main_accion"] = sesiones["main_accion"].fillna("Ordeño")   # el xlsx solo trae ordeños
    sesiones = sesiones.sort_values(["id_vaca", "inicio"])

    # Horas desde el ordeño anterior de la misma vaca (solo entre ordeños). Se calcula antes de quitar la
    # cuarta maquina porque sus 2 registros tambien son ordeños de la vaca
    es_ordeno = sesiones["main_accion"] == "Ordeño"
    sesiones.loc[es_ordeno, "intervalo_h"] = sesiones[es_ordeno].groupby("id_vaca")["inicio"].diff().dt.total_seconds() / 3600
    sesiones["regreso_periodo_seco"] = (sesiones["intervalo_h"] > UMBRAL_INTERVALO_H).astype(int)
    sesiones["intervalo_h"] = sesiones["intervalo_h"].where(sesiones["intervalo_h"] <= UMBRAL_INTERVALO_H)

    # Una "maquina" del archivo tiene solo 2 registros: se deja fuera
    maquinas = sesiones["misc_eo_po"].value_counts().loc[lambda s: s > 100].index
    sesiones = sesiones[sesiones["misc_eo_po"].isin(maquinas)]

    # Las alertas traen la lista de cuartos afectados (DI,TD o Todos): una columna 0/1 por cuarto
    bloques = []
    for columna, nombre in ALERTAS.items():
        texto = sesiones[columna].fillna("").str.upper().replace("TODOS", "DI,DD,TI,TD")
        bloque = texto.str.get_dummies(sep=",").reindex(columns=[c.upper() for c in CUARTOS], fill_value=0)
        bloques.append(bloque.rename(columns=lambda c: f"{nombre}_{c.lower()}"))

    # Una conductividad de 0 es imposible: es una lectura faltante
    conductividad = [f"conductividad_ms_cm_{c}" for c in CUARTOS]
    sesiones[conductividad] = sesiones[conductividad].replace(0, np.nan)

    sesiones = sesiones.drop(columns=["main_hora_de_inicio", "main_duracion_mm_ss", *ALERTAS, *DESCARTADAS])
    return pd.concat([sesiones, *bloques], axis=1).reset_index(drop=True)


def cargar_clima(ruta):
    """
        Lee la serie horaria de NASA POWER (descargar_clima.py), que ya viene en hora local, corrige su
        desfase y le agrega el indice temperatura-humedad:
        THI = (1.8 T + 32) - (0.55 - 0.0055 HR) * (1.8 T - 26)

        La fila etiquetada t de POWER corresponde al clima de t + 1 h (confirmado contra una estacion
        observada), asi que se corre el indice una hora: una sesion a la hora T toma la fila T - 1 h.

        Args:
            ruta (Path): CSV generado por descargar_clima.py.

        Returns:
            DataFrame: Serie horaria local, ya alineada, con las variables de POWER y el THI.
    """
    clima = pd.read_csv(ruta, index_col="hora", parse_dates=True)
    clima.index = clima.index + pd.Timedelta(hours=DESFASE_POWER_H)
    clima["thi"] = (1.8 * clima["t2m"] + 32) - (0.55 - 0.0055 * clima["rh2m"]) * (1.8 * clima["t2m"] - 26)
    return clima


def agregar_clima(sesiones, clima):
    """
        Une a cada sesion el clima de la hora en que inicia y el numero de horas con THI alto en las
        24 h previas. La ventana termina justo antes de la hora de la sesion, asi que solo usa clima que
        ya ocurrio; si no esta completa, el atributo queda faltante.

        Args:
            sesiones (DataFrame): Tabla de cargar_sesiones.
            clima (DataFrame): Serie horaria de cargar_clima.

        Returns:
            DataFrame: Las sesiones con las variables de POWER, t_sesion, thi_sesion y horas_thi_alto_24h.
    """
    clima = clima.asfreq("h")
    clima["horas_thi_alto_24h"] = (clima["thi"] >= UMBRAL_THI).astype(float).where(clima["thi"].notna()).shift(1).rolling(24).sum()
    clima = clima.rename(columns={"t2m": "t_sesion", "thi": "thi_sesion"})
    return sesiones.merge(clima, left_on=sesiones["inicio"].dt.floor("h"), right_index=True, how="left").drop(columns="key_0", errors="ignore")


def main():
    sesiones = cargar_sesiones(ORIGEN)
    integrado = agregar_clima(sesiones, cargar_clima(CLIMA))
    integrado.to_csv(SALIDA, index=False)

    ordenos = integrado["main_accion"] == "Ordeño"
    print(f"Sesiones: {len(integrado):,} ({ordenos.sum():,} ordeños) | vacas: {integrado['id_vaca'].nunique()} | "
          f"{integrado['inicio'].min():%d/%m/%Y} a {integrado['inicio'].max():%d/%m/%Y}")
    print(f"Intervalos de más de {UMBRAL_INTERVALO_H} h (regreso tras periodo seco): {integrado['regreso_periodo_seco'].sum():,}")
    print(f"Sin clima de su hora: {integrado['thi_sesion'].isna().sum():,} | "
          f"sin las 24 h previas completas: {integrado['horas_thi_alto_24h'].isna().sum():,}")
    print(f"Sesiones con THI ≥ {UMBRAL_THI}: {(integrado['thi_sesion'] >= UMBRAL_THI).mean():.1%}")
    print(f"Columnas: {integrado.shape[1]} | guardado en {SALIDA.relative_to(raiz)}")


if __name__ == "__main__":
    main()
