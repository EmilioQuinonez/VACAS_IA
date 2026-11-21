"""
Cifras de volumen, variedad y veracidad del dataset de ordeño para el análisis de las 5 V (reto de datos).

Mide los archivos originales (cuántos hay, de qué formato y cuánto ocupan) y los datasets generados por el
pipeline (filas, columnas, vacas, periodo, memoria en Pandas y nulos). Solo lee archivos; no modifica nada.

Guarda los resultados en src/RETO_DATOS/Rendimiento/resultados/estadisticas_dataset.json.

Uso, desde la raíz del repo:
    python src/RETO_DATOS/Rendimiento/estadisticas_dataset.py
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "Data"
SALIDA = Path(__file__).resolve().parent / "resultados" / "estadisticas_dataset.json"
ORIGINALES = {"ordeno": DATA / "vacas", "concentrado": DATA / "concentrado", "clima": DATA / "clima"}
DATASETS = {
    "crudo": DATA / "dataset_vacas.csv",
    "anonimizado": DATA / "dataset_vacas_anonimo.csv",
    "integrado": DATA / "dataset_vacas_integrado.csv",
}
ALERTAS = ["estado_patada", "estado_incompleto", "estado_pezones_no_encontrados", "estado_amd_pezon"]
VACIAS = ["estado_rcs_1000_celulas_ml", "misc_usuario"]
CONDUCTIVIDAD = [f"conductividad_ms_cm_{q}" for q in ("di", "dd", "ti", "td")]


def mb(bytes_):
    return round(bytes_ / 1024 ** 2, 1)


def originales():
    """Número de archivos por formato y tamaño de cada carpeta de datos originales."""
    salida = {}
    for nombre, carpeta in ORIGINALES.items():
        archivos = [f for f in carpeta.rglob("*") if f.is_file() and not f.name.startswith(".")]
        salida[nombre] = {"archivos": len(archivos), "formatos": dict(Counter(f.suffix.lower() for f in archivos)),
                          "tamano_mb": mb(sum(f.stat().st_size for f in archivos))}
    return salida


def describir(ruta):
    """Filas, columnas, vacas, periodo, tamaño en disco y en memoria, y porcentaje de celdas nulas."""
    datos = pd.read_csv(ruta, low_memory=False)
    salida = {"filas": len(datos), "columnas": datos.shape[1], "vacas": int(datos["id_vaca"].nunique()),
              "tamano_disco_mb": mb(ruta.stat().st_size),
              "memoria_pandas_mb": mb(datos.memory_usage(deep=True).sum()),
              "celdas_nulas_pct": round(float(datos.isna().mean().mean() * 100), 1)}
    if "inicio" in datos:
        salida["periodo"] = [str(datos["inicio"].min()), str(datos["inicio"].max())]
    return salida, datos


def calidad(crudo):
    """Problemas de calidad del dataset crudo que resuelve la limpieza."""
    salida = {
        "nulos_alertas_pct": {c: round(float(crudo[c].isna().mean() * 100), 1) for c in ALERTAS if c in crudo},
        "columnas_vacias_pct": {c: round(float(crudo[c].isna().mean() * 100), 1) for c in VACIAS if c in crudo},
        "conductividad_en_cero_pct": {c: round(float((crudo[c] == 0).mean() * 100), 1) for c in CONDUCTIVIDAD if c in crudo},
        "filas_identicas": int(crudo.duplicated().sum()),
    }
    if "estado_mdi" in crudo:
        salida["mdi_nulo_pct"] = round(float(crudo["estado_mdi"].isna().mean() * 100), 1)
    return salida


def main():
    resultados = {"originales": originales()}
    for nombre, ruta in DATASETS.items():
        resultados[nombre], datos = describir(ruta)
        if nombre == "crudo":
            resultados["calidad_crudo"] = calidad(datos)
        print(f"[datos] {nombre}: {resultados[nombre]['filas']:,} filas x {resultados[nombre]['columnas']} columnas, "
              f"{resultados[nombre]['tamano_disco_mb']} MB en disco, {resultados[nombre]['memoria_pandas_mb']} MB en memoria")
    for nombre, info in resultados["originales"].items():
        print(f"[originales] {nombre}: {info['archivos']} archivos {info['formatos']}, {info['tamano_mb']} MB")

    SALIDA.parent.mkdir(exist_ok=True)
    with open(SALIDA, "w", encoding="utf-8") as archivo:
        json.dump(resultados, archivo, indent=2, ensure_ascii=False)
    print(f"Resultados en {SALIDA.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
