"""Anonimiza el concentrado con COWSKEY y verifica el mapa de ordeño."""

import hashlib
import hmac
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


RAIZ = Path(__file__).resolve().parents[2]
ORIGEN = RAIZ / "Data" / "dataset_concentrado.csv"
SALIDA = RAIZ / "Data" / "dataset_concentrado_anonimo.csv"
MAPA = RAIZ / "Data" / "mapa_id_vacas.csv"


def aplicar_hash(valor, llave):
    return hmac.new(llave.encode(), str(valor).encode(), hashlib.sha256).hexdigest()[:8]


def main():
    load_dotenv(RAIZ / ".env")
    llave = os.getenv("COWSKEY")
    if not llave:
        raise RuntimeError("Falta COWSKEY en .env")

    mapa = pd.read_csv(MAPA, dtype=str, keep_default_na=False)
    if list(mapa.columns) != ["columna", "original", "anonimo"]:
        raise ValueError("El mapa de IDs tiene columnas inesperadas")
    vacas = mapa.loc[mapa["columna"].eq("id_vaca"), ["original", "anonimo"]]
    if vacas.empty or vacas.eq("").any().any() or vacas.duplicated("original").any() or vacas.duplicated("anonimo").any():
        raise ValueError("El mapa contiene IDs de vaca vacíos o repetidos")
    equivalencias = dict(vacas.itertuples(index=False, name=None))
    if any(aplicar_hash(original, llave) != anonimo for original, anonimo in equivalencias.items()):
        raise ValueError("COWSKEY no coincide con el mapa de IDs de ordeño")

    datos = pd.read_csv(ORIGEN, dtype={"id_vaca": str})
    if datos["id_vaca"].isna().any():
        raise ValueError("Hay registros de concentrado sin ID")
    hashes_usados = {anonimo: original for original, anonimo in equivalencias.items()}
    nuevos = 0
    for original in datos["id_vaca"].unique():
        if original not in equivalencias:
            anonimo = aplicar_hash(original, llave)
            if anonimo in hashes_usados and hashes_usados[anonimo] != original:
                raise ValueError(f"Colisión de hash para la vaca {original}")
            equivalencias[original] = anonimo
            hashes_usados[anonimo] = original
            nuevos += 1
    datos["id_vaca"] = datos["id_vaca"].map(equivalencias)
    datos.to_csv(SALIDA, index=False)
    print(f"Concentrado anonimizado: {len(datos):,} registros, {datos.id_vaca.nunique()} vacas; IDs fuera del mapa: {nuevos}")
    print(f"Guardado: {SALIDA.relative_to(RAIZ)}; mapa de IDs intacto")


if __name__ == "__main__":
    main()
