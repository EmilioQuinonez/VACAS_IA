import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parent.parent.parent
CARPETA = RAIZ / "Data" / "vacas"
SALIDA = RAIZ / "Data" / "dataset_vacas.csv"

# Se usan los nombres del CSV para conservar las columnas de salida existentes.
EQUIVALENCIAS = {
    "main_numero_de_ordeno": "estado_numero_de_ordeno",
    "main_rcs_1000_celulas_ml": "estado_rcs_1000_celulas_ml",
    "misc_ms": "misc_eo_po",
}


def normalizar_nombre(texto):
    """
        Normaliza un texto eliminando acentos, convirtiendo a minúsculas y 
        reemplazando caracteres no alfanuméricos por guiones bajos.
        Ejemplo: Media de los flujos (kg/min) -> media_de_los_flujos_kg_min

        Args:
            texto (str): El texto a normalizar.
        
        Returns:
            str: El texto normalizado.
    """
    texto = unicodedata.normalize("NFKD", str(texto))
    texto = texto.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "_", texto).strip("_")


def leer_archivo(archivo):
    """
        Lee las dos filas de encabezado y los datos de un CSV o XLSX.

        Args:
            archivo (Path): Ruta del archivo a leer.

        Returns:
            pd.DataFrame: DataFrame con los datos del archivo, con nombres de 
            columnas normalizados
    """
    if archivo.suffix.lower() == ".xlsx":
        lector = pd.read_excel
        opciones = {"engine": "openpyxl"}
    else:
        lector = pd.read_csv
        opciones = {"sep": ",", "engine": "python"}

    encabezados = lector(archivo, header=None, nrows=2, **opciones)
    grupo_actual = ""
    columnas = []

    for grupo, subcolumna in zip(encabezados.iloc[0], encabezados.iloc[1]):
        if pd.notna(grupo):
            grupo_actual = normalizar_nombre(grupo)
        nombre = f"{grupo_actual}_{normalizar_nombre(subcolumna)}"
        columnas.append(EQUIVALENCIAS.get(nombre, nombre))

    if len(columnas) != len(set(columnas)):
        raise ValueError(f"Hay columnas repetidas en los encabezados de {archivo.name}")

    return lector(archivo, header=None, names=columnas, skiprows=2, **opciones)

def id_de_vaca(archivo):
    """
        Obtiene el identificador de la vaca a partir del nombre del archivo.

        Args:
            archivo (Path): Ruta del archivo.
        
        Returns:
            str: Identificador de la vaca.
    """
    if archivo.suffix.lower() == ".csv":
        return archivo.stem

    coincidencia = re.search(r"(\d+)$", archivo.stem)
    if not coincidencia:
        raise ValueError(f"No se encontró un identificador de vaca en {archivo.name}")
    return coincidencia.group(1)


def main():
    archivos = sorted(CARPETA.glob("*.csv")) + sorted(
        archivo for archivo in CARPETA.glob("*.xlsx")
        if not archivo.name.startswith("~$")
    )
    clean = []
    total_duplicados = 0
    total_nulos = 0

    for archivo in archivos:
        print(f"\nLeyendo: {archivo.name} ")
        data = leer_archivo(archivo)

        total_duplicados += data.duplicated().sum()
        data = data.replace("?", np.nan)
        total_nulos += data.isnull().sum().sum()

        data.insert(0, "id_vaca", id_de_vaca(archivo))
        clean.append(data)

    print("============================= DF completo =============================")
    df_salida = pd.concat(clean, ignore_index=True)
    print("Total de archivos procesados: ", len(archivos))
    print("Observaciones duplicadas: ", total_duplicados)
    print("Valores nulos: ", total_nulos)

    print("... guardando ...")
    df_salida.to_csv(SALIDA, index=False)
    print("Guardado!")


if __name__ == "__main__":
    main()
