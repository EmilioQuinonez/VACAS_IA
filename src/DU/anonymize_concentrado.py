"""Anonimiza concentrado con la llave de ordeño, sin modificar el mapa."""

import hashlib
import hmac
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

RAIZ = Path(__file__).resolve().parents[2]
RUTA_MAPA = RAIZ / "Data" / "mapa_id_vacas.csv"
LARGO_HASH = 8


def leer_llave():
    load_dotenv(RAIZ / ".env")
    llave = os.getenv("COWSKEY")
    if not llave:
        raise RuntimeError("Falta COWSKEY en el archivo .env")
    return llave


def aplicar_hash(valor, llave):
    return hmac.new(llave.encode(), str(valor).encode(), hashlib.sha256).hexdigest()[
        :LARGO_HASH
    ]


def anonimizar_ids_vaca(data):
    """Usa los IDs existentes como referencia y calcula los que aún no estén en el mapa."""
    if not RUTA_MAPA.is_file():
        raise FileNotFoundError(f"No existe el mapa de referencia: {RUTA_MAPA}")

    # keep_default_na=False conserva la fila heredada con original vacío en
    # misc_eo_po. Esa fila no participa en la anonimización de id_vaca.
    mapa = pd.read_csv(RUTA_MAPA, dtype=str, keep_default_na=False)
    if list(mapa.columns) != ["columna", "original", "anonimo"]:
        raise ValueError(f"Columnas inesperadas en el mapa: {RUTA_MAPA}")
    vacas = mapa.loc[mapa["columna"] == "id_vaca", ["original", "anonimo"]]
    if (vacas.empty or vacas.eq("").any().any()
            or vacas.duplicated("original").any()
            or vacas.duplicated("anonimo").any()):
        raise ValueError(f"IDs de vaca inválidos o repetidos en: {RUTA_MAPA}")

    llave = leer_llave()
    equivalencias = dict(vacas.itertuples(index=False, name=None))
    for original, anonimo in equivalencias.items():
        if aplicar_hash(original, llave) != anonimo:
            raise ValueError(f"COWSKEY no coincide con los IDs de vaca en: {RUTA_MAPA}")

    originales = data["id_vaca"].astype("string")
    if originales.isna().any():
        raise ValueError("El concentrado contiene filas sin id_vaca")
    anonimos_usados = {anonimo: original for original, anonimo in equivalencias.items()}
    nuevos = 0
    for original in originales.unique():
        if original not in equivalencias:
            anonimo = aplicar_hash(original, llave)
            if anonimo in anonimos_usados and anonimos_usados[anonimo] != original:
                raise RuntimeError(f"Colisión de hash para id_vaca {original}")
            equivalencias[original] = anonimo
            anonimos_usados[anonimo] = original
            nuevos += 1

    data["id_vaca"] = originales.map(equivalencias)
    print(f"id_vaca: {originales.nunique()} IDs anonimizados")
    print(f"IDs nuevos calculados sin modificar el mapa: {nuevos}")
    return data


def main():
    origen = RAIZ / "Data" / "dataset_concentrado.csv"
    destino = RAIZ / "Data" / "dataset_concentrado_anonimo.csv"
    data = pd.read_csv(origen, dtype={"id_vaca": str})
    anonimizar_ids_vaca(data)
    data.to_csv(destino, index=False)
    print(f"Guardado: {destino}")


if __name__ == "__main__":
    main()
