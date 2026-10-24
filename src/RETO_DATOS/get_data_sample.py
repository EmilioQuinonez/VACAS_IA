"""
Calcula el tamaño con Cochran y selecciona una muestra estratificada.

Cochran para estimar proporciones: n0 = Z² * p * (1 - p) / e².
Corrección para población finita: n = n0 / (1 + (n0 - 1) / N).
El resultado se redondea hacia arriba.

Los registros se distribuyen proporcionalmente entre estratos mediante
restos mayores y se seleccionan al azar sin reemplazo dentro de cada uno.
El margen de error corresponde a la estimación de proporciones bajo los
supuestos de Cochran, no garantiza esa precisión para todas las variables.

Uso: 
    python src/RETO_DATOS/get_data_sample.py
"""

from math import ceil
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "Data" / "dataset_vacas_integrado.csv"
OUTPUT_PATH = ROOT / "src" / "RETO_DATOS" / "dataset_vacas_muestra.csv"

# Ajusta estos parámetros para cambiar el tamaño calculado automáticamente.
CONFIDENCE_LEVEL = 0.95
MARGIN_OF_ERROR = 0.05
EXPECTED_PROPORTION = 0.5
STRATIFY_COLUMN = "id_vaca"
RANDOM_STATE = 42

def get_sample_size(
        total, confidence=CONFIDENCE_LEVEL,
        error=MARGIN_OF_ERROR, proportion=EXPECTED_PROPORTION
):
    """
    Calcula el tamaño de muestra con Cochran y corrección por población finita.

    Args:
        total (int): Número total de registros del dataset.

        confidence (float): Nivel de confianza entre 0 y 1, sin incluir los
            extremos; por defecto, 0.95 (95%).

        error (float): Margen de error absoluto para estimar proporciones,
            mayor que 0 y menor que 1; por defecto, 0.05 (5 puntos porcentuales).

        proportion (float): Proporción esperada de la característica de interés,
            entre 0 y 1, sin incluir los extremos. Por defecto, 0.5, que utiliza
            la máxima variabilidad cuando se desconoce esta proporción.

    Returns:
        tuple[float, int]: Tamaño teórico n0 sin corrección y tamaño corregido
            por población finita, redondeado hacia arriba y limitado al total
            de registros.
    """
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    n0 = z**2 * proportion * (1 - proportion) / error**2
    size = ceil(n0 / (1 + (n0 - 1) / total)) if total else 0
    return n0, min(size, total)


def stratified_sample(data, size, column=STRATIFY_COLUMN, seed=RANDOM_STATE):
    """
    Selecciona registros aleatorios por estrato con asignación proporcional.

    Distribuye los lugares restantes por restos mayores para obtener exactamente
    size registros. Los empates se resuelven por orden de aparición del estrato.
    Los valores faltantes forman un estrato propio.

    Args:
        data (pd.DataFrame): Dataset original con la columna de estratificación.

        size (int): Total de registros a seleccionar, entre 0 y len(data).

        column (str): Nombre de la columna que define los estratos;
            por defecto, id_vaca.

        seed (int): Semilla del generador aleatorio; por defecto, 42.
            Permite repetir la selección con los mismos datos y parámetros.

    Returns:
        pd.DataFrame: Copia de los registros seleccionados sin reemplazo,
            conservando todas las columnas, el índice y el orden original.
            Si data está vacío, devuelve una copia vacía.
    """
    if data.empty:
        return data.copy()

    # Factorizar incluye los identificadores faltantes como un estrato propio.
    codes, _ = pd.factorize(data[column], sort=False)
    groups = pd.DataFrame({"stratum": codes}).groupby("stratum", sort=False).indices
    counts = np.array([len(positions) for positions in groups.values()])

    quotas = counts * size // len(data)
    remainders = counts * size % len(data)
    remaining = size - int(quotas.sum())
    priority = np.argsort(-remainders, kind="stable")
    quotas[priority[:remaining]] += 1

    rng = np.random.default_rng(seed)
    selected = [rng.choice(positions, size=int(quota), replace=False)
                for positions, quota in zip(groups.values(), quotas)]
    return data.iloc[np.sort(np.concatenate(selected))].copy()

def main():
    data = pd.read_csv(INPUT_PATH, dtype={"id_vaca": str, "misc_eo_po": str},
                       low_memory=False)

    n0, size = get_sample_size(len(data))
    print(f"Dataset: {len(data):,} registros")
    print(f"Confianza: {CONFIDENCE_LEVEL:.0%} | "
          f"Margen de error: {MARGIN_OF_ERROR:.1%} | "
          f"Proporción esperada: {EXPECTED_PROPORTION:.2f}")
    print(f"Cochran sin corrección: {n0:.2f} | "
          f"Tamaño con corrección finita: {size:,}")

    sample = stratified_sample(data, size)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUTPUT_PATH, index=False)
    total_strata = data[STRATIFY_COLUMN].nunique(dropna=False)
    sampled_strata = sample[STRATIFY_COLUMN].nunique(dropna=False)
    print(f"Estrato: {STRATIFY_COLUMN} | Representados: {sampled_strata}/{total_strata}")
    print(f"Muestra aleatoria estratificada: {len(sample):,} registros | Semilla: {RANDOM_STATE}")

if __name__ == "__main__":
    main()
