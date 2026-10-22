"""Consolida los XLSX de concentrado sin alterar los registros originales."""

import re
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
CARPETA = RAIZ / "Data" / "concentrado"
SALIDA = RAIZ / "Data" / "dataset_concentrado.csv"
PATRON = re.compile(r"Concentrado consumido\s*(\d+)\.xlsx", re.IGNORECASE)
ENCABEZADOS = ["Alimento", "Hora evento", "Consumido", "Consumido Mat. Seca"]
COLUMNAS = ["id_vaca", "alimento", "hora_evento", "consumido", "consumido_mat_seca"]


def main():
    archivos = sorted(CARPETA.glob("Concentrado consumido*.xlsx"))
    if not archivos:
        raise FileNotFoundError(f"No hay archivos de concentrado en {CARPETA}")

    tablas = []
    omitidos = []
    for archivo in archivos:
        coincidencia = PATRON.fullmatch(archivo.name)
        if not coincidencia:
            omitidos.append(archivo.name)
            continue
        tabla = pd.read_excel(archivo, engine="openpyxl")
        if list(tabla.columns) != ENCABEZADOS:
            raise ValueError(
                f"Encabezados inesperados en {archivo.name}: {list(tabla.columns)}"
            )
        tabla.columns = COLUMNAS[1:]
        tabla.insert(0, "id_vaca", coincidencia.group(1))
        tabla["hora_evento"] = pd.to_datetime(tabla["hora_evento"], errors="raise")
        for columna in ("consumido", "consumido_mat_seca"):
            tabla[columna] = pd.to_numeric(tabla[columna], errors="raise")
        if tabla[["hora_evento", "consumido"]].isna().any().any():
            raise ValueError(f"Fecha o consumo faltante en {archivo.name}")
        tablas.append(tabla)

    if not tablas:
        raise ValueError("Ningún archivo de concentrado tiene un ID de vaca")
    salida = pd.concat(tablas, ignore_index=True)
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(SALIDA, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print(
        f"Concentrado: {len(salida):,} registros, {len(tablas)} archivos, {salida.id_vaca.nunique()} vacas"
    )
    print(
        f"Duplicados exactos de origen: {salida.duplicated().sum():,} (se eliminan al integrar)"
    )
    print(f"Guardado: {SALIDA.relative_to(RAIZ)}")
    for nombre in omitidos:
        print(f"Omitido por no tener ID de vaca: {nombre}")


if __name__ == "__main__":
    main()
