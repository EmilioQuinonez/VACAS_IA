"""
Separa un CSV en train, val y test mediante indices, sin escribir archivos.

Para usarlo desde otro archivo y separar en conjuntos usando k fold, se importa 
la funcion get_k_folds y se llama con los parametros deseados.

Ejemplo: 
    from src.DP.k_fold import get_k_folds

    dataset, folds = get_k_folds(INPUT_DIR)
    fold = folds[0]

    train = dataset.iloc[fold["train"]]
    val = dataset.iloc[fold["val"]]
    test = dataset.iloc[fold["test"]]

Uso: 
    python src/DP/k_fold.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "src" / "RETO_DATOS" / "dataset_vacas_muestra.csv"
N_SPLITS = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42


def get_k_folds(input_dir, n_splits=5, test_size=0.2, random_state=42):
    """Carga un CSV y genera indices de train/val con un test fijo separado.

    Args:
        input_dir (str): Ruta al archivo CSV.

        n_splits (int): Numero de folds sobre las filas ajenas a test.

        test_size (float): Fraccion del dataset reservada para test.

        random_state (int): Semilla para las dos mezclas aleatorias.

    Returns:
        tuple[pandas.DataFrame, list[dict[str, numpy.ndarray]]]:
            dataset: DataFrame completo con todas las filas y columnas del CSV.

            folds: Lista de longitud n_splits. Cada elemento es un
                diccionario con tres arrays unidimensionales de enteros:
                - "train": posiciones de filas originales para entrenar.
                - "val": posiciones de filas originales para validar.
                - "test": posiciones de filas originales para evaluar.
    """
    dataset = pd.read_csv(Path(input_dir), dtype=str, keep_default_na=False)

    n_test = int(np.ceil(len(dataset) * test_size))
    if len(dataset) - n_test < n_splits:
        raise ValueError(
            "No hay suficientes filas fuera de test para el numero de folds indicado."
        )

    train, test = train_test_split(
        np.arange(len(dataset)), 
        test_size=test_size, 
        random_state=random_state,
        shuffle=True,
    )
    # split devuelve posiciones dentro de desarrollo, no del CSV original.
    k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []

    for k_train, k_val in k_fold.split(train):
        # Traducir posiciones locales a posiciones del dataset original.
        folds.append({
            "train": train[k_train],
            "val": train[k_val],
            "test": test,
        })

    return dataset, folds


if __name__ == "__main__":
    # Probar dentro del archivo, si se importa la función en otro script, no se
    # ejecuta este bloque.

    dataset, folds = get_k_folds(INPUT_DIR, N_SPLITS, TEST_SIZE, RANDOM_STATE)
    
    for numero, fold in enumerate(folds, start=1):
        print(
            f"Fold {numero}: train={len(fold['train'])}, "
            f"val={len(fold['val'])}, test={len(fold['test'])}"
        )

    fold = folds[0]

    train = dataset.iloc[fold["train"]]
    val = dataset.iloc[fold["val"]]
    test = dataset.iloc[fold["test"]]

    print("===================================================================")
    print("Primero 5 registros del conjunto de entrenamiento: ")
    print(train.head())
    print("===================================================================")
    print("Primero 5 registros del conjunto de validación: ")
    print(val.head())
    print("===================================================================")
    print("Primero 5 registros del conjunto de prueba: ")
    print(test.head())
    print("===================================================================")
