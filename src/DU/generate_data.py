import pandas as pd
import numpy as np
import re
import unicodedata
from pathlib import Path

raiz = Path(__file__).resolve().parent.parent.parent
carpeta = raiz / "Data" / "vacas"
clean = []
total_duplicados = 0
total_nulos = 0
total_archivos = len(list(carpeta.glob("*.csv")))


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


for archivo in carpeta.glob("*.csv"):
    # Carga de datos
    print(f"\nLeyendo: {archivo.name} ")

    encabezados = pd.read_csv(archivo, sep=',', header=None, nrows=2, engine='python')
    grupo_actual = ""
    columnas = []

    for grupo, subcolumna in zip(encabezados.iloc[0], encabezados.iloc[1]):
        if pd.notna(grupo):
            grupo_actual = normalizar_nombre(grupo)
        columnas.append(f"{grupo_actual}_{normalizar_nombre(subcolumna)}")

    data = pd.read_csv(
        archivo,
        sep=',',
        header=None,
        names=columnas,
        skiprows=2,
        engine='python'
    )

    # Conteo de datos duplicados
    duplicados = data.duplicated().sum()
    total_duplicados += duplicados

    # Revision de datos faltantes
    data = data.replace('?', np.nan)
    nulos = data.isnull().sum().sum()
    total_nulos += nulos

    nombre_archivo = archivo.stem

    # Crear columna al inicio
    data.insert(0, "id_vaca", nombre_archivo)

    clean.append(data)

print("============================= DF completo =============================")
df_salida = pd.concat(clean, ignore_index=True)
print("Total de archivos procesados: ", total_archivos)
print("Observaciones duplicadas: ", total_duplicados)
print("Valores nulos: ", total_nulos)

print("... guardando ...")
df_salida.to_csv(raiz / "Data" / "dataset_vacas.csv", index=False)
print("Guardado!")