"""Consolida eventos de concentrado de los XLSX con ID de vaca en el nombre."""

import re
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
CARPETA = RAIZ / "Data" / "Datos vacas historico hasta enero 2026" / "TEC-UAQ-UNAM"
DESTINO = RAIZ / "Data" / "dataset_concentrado.csv"
PATRON = re.compile(r"Concentrado consumido\s*(\d+)\.xlsx", re.IGNORECASE)
COLUMNAS_ORIGEN = ["Alimento", "Hora evento", "Consumido", "Consumido Mat. Seca"]
COLUMNAS_SALIDA = [
    "id_vaca",
    "alimento",
    "hora_evento",
    "consumido",
    "consumido_mat_seca",
]


def main():
    archivos = sorted(CARPETA.glob("Concentrado consumido*.xlsx"))
    if not archivos:
        raise FileNotFoundError(f"No hay XLSX de concentrado en {CARPETA}")

    tablas = []
    omitidos = []
    for archivo in archivos:
        coincidencia = PATRON.fullmatch(archivo.name)
        if not coincidencia:
            omitidos.append(archivo.name)
            continue
        tabla = pd.read_excel(archivo, engine="openpyxl")
        if list(tabla.columns) != COLUMNAS_ORIGEN:
            raise ValueError(
                f"Encabezados inesperados en {archivo.name}: {list(tabla.columns)}"
            )
        tabla.columns = COLUMNAS_SALIDA[1:]
        tabla.insert(0, "id_vaca", coincidencia.group(1))
        tabla["hora_evento"] = pd.to_datetime(tabla["hora_evento"], errors="raise")
        for columna in ["consumido", "consumido_mat_seca"]:
            tabla[columna] = pd.to_numeric(tabla[columna], errors="raise")
        tablas.append(tabla)

    if not tablas:
        raise ValueError("Ningún archivo tiene un ID de vaca en su nombre")
    salida = pd.concat(tablas, ignore_index=True)
    salida.to_csv(DESTINO, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(
        f"Guardado: {DESTINO} ({len(salida)} eventos, {len(tablas)} archivos, {salida.id_vaca.nunique()} vacas)"
    )
    print(f"Duplicados exactos conservados: {salida.duplicated().sum()}")
    for nombre in omitidos:
        print(f"Omitido por no tener ID de vaca: {nombre}")


if __name__ == "__main__":
    main()
