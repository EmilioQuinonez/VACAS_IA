import hashlib
import hmac
import os
from pathlib import Path

import pandas as pd
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
raiz = Path(__file__).resolve().parent.parent.parent

# Columnas que identifican a la vaca y a la maquina
columnas = ["id_vaca", "misc_eo_po"]
largo_hash = 8


def leer_variable(variable):
    """
        Lee una variable del .env y falla si no existe.

        Args:
            variable (str): Nombre de la variable en el .env.

        Returns:
            str: El valor de la variable.
    """
    valor = os.getenv(variable)
    if not valor:
        raise RuntimeError(f"Falta la variable {variable} en el archivo .env")
    return valor


def aplicar_hash(valor, llave):
    """
        Calcula el hash de un valor usando una llave secreta (HMAC-SHA256).
        Sin la llave no se puede recalcular el hash a partir del valor original.

        Args:
            valor (str): El valor original.
            llave (str): La llave secreta del .env.

        Returns:
            str: Los primeros caracteres del hash en hexadecimal.
    """
    resumen = hmac.new(llave.encode(), str(valor).encode(), hashlib.sha256)
    return resumen.hexdigest()[:largo_hash]


origen = raiz / "Data" / "dataset_vacas.csv"
destino = raiz / "Data" / "dataset_vacas_anonimo.csv"
ruta_mapa = raiz / "Data" / "mapa_id_vacas.csv"

# La llave es secreta: es lo unico que se lee del .env
llave = leer_variable("COWSKEY")

data = pd.read_csv(origen, dtype={"id_vaca": str})
mapa = []

for columna in columnas:
    equivalencias = {valor: aplicar_hash(valor, llave) for valor in data[columna].unique()}

    # Dos valores distintos no pueden terminar con el mismo hash
    if len(set(equivalencias.values())) < len(equivalencias):
        raise RuntimeError(f"Hay hashes repetidos en {columna}: aumenta largo_hash")

    data[columna] = data[columna].map(equivalencias)
    mapa += [(columna, original, anonimo) for original, anonimo in equivalencias.items()]
    print(f"{columna}: {len(equivalencias)} valores anonimizados")

# El mapa permite rastrear de vuelta: no debe salir de la carpeta de datos
pd.DataFrame(mapa, columns=["columna", "original", "anonimo"]).to_csv(ruta_mapa, index=False)

print("... guardando ...")
data.to_csv(destino, index=False)
print("Guardado!")
