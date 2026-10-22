from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
RUTA_VACAS = RAIZ / "Data" / "dataset_vacas_anonimo.csv"
RUTA_CLIMA = RAIZ / "Data" / "clima" / "clima_nasa_power_por_hora.csv"


def convertir_fechas_y_duraciones(vacas):
    """Acción 1: convierte hora de inicio, duración e intervalo a datetime y número."""
    vacas = vacas.copy()

    # El archivo junta dos fuentes con formatos distintos:
    #   csv  (2025):      hora "dd/mm/aaaa hh:mm a. m.", duración "mm:ss"
    #   xlsx (2022-2026): hora "aaaa-mm-dd hh:mm:ss",    duración "hh:mm:ss"
    # La fuente se decide una sola vez, por el formato de la hora (el csv trae "a. m." / "p. m.")
    vacas["fuente"] = np.where(vacas["main_hora_de_inicio"].str.contains("m.", regex=False), "csv", "xlsx")
    es_csv = vacas["fuente"] == "csv"

    vacas["inicio"] = pd.Series(pd.NaT, index=vacas.index, dtype="datetime64[ns]")
    vacas.loc[es_csv, "inicio"] = pd.to_datetime(
        vacas.loc[es_csv, "main_hora_de_inicio"].str.replace("a. m.", "AM", regex=False).str.replace("p. m.", "PM", regex=False),
        format="%d/%m/%Y %I:%M %p")
    vacas.loc[~es_csv, "inicio"] = pd.to_datetime(vacas.loc[~es_csv, "main_hora_de_inicio"], format="%Y-%m-%d %H:%M:%S")

    # La duración del csv es "mm:ss" y la del xlsx "hh:mm:ss"
    duracion = vacas["main_duracion_mm_ss"].where(~es_csv, "00:" + vacas["main_duracion_mm_ss"])
    vacas["duracion_seg"] = pd.to_timedelta(duracion).dt.total_seconds()

    # Excel guarda los intervalos de 24 h o más como fecha ("1900-01-02 06:30:00"), no como "hh:mm:ss".
    # Con época 1899-12-31 la fecha equivale a la duración (se verificó contra la diferencia entre inicios consecutivos).
    intervalo = vacas["main_intervalo_de_ordeno_hh_mm"]
    es_fecha = intervalo.str.len() == 19
    es_hms = intervalo.str.len() == 8
    horas = pd.Series(np.nan, index=vacas.index)
    horas[es_fecha] = (pd.to_datetime(intervalo[es_fecha]) - pd.Timestamp("1899-12-31")).dt.total_seconds() / 3600
    horas[es_hms] = pd.to_timedelta(intervalo[es_hms]).dt.total_seconds() / 3600
    vacas["intervalo_horas"] = horas

    return vacas.drop(columns=["main_hora_de_inicio", "main_duracion_mm_ss", "main_intervalo_de_ordeno_hh_mm"])


if __name__ == "__main__":
    vacas = pd.read_csv(RUTA_VACAS, dtype={"id_vaca": str, "misc_eo_po": str}, low_memory=False)
    clima = pd.read_csv(RUTA_CLIMA, parse_dates=["hora"])

    vacas = convertir_fechas_y_duraciones(vacas)
