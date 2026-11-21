"""
Separa un CSV en folds temporales con train creciente y test fijo.

Para usarlo desde otro archivo y separar en conjuntos usando k fold, se importa 
la funcion get_k_folds y se llama con los parametros deseados.

Ejemplo: 
    from src.RETO_DATOS.k_fold import INPUT_DIR, get_k_folds

    dataset, folds = get_k_folds(INPUT_DIR)
    fold = folds[0]

    train = dataset.iloc[fold["train"]]
    val = dataset.iloc[fold["val"]]
    test = dataset.iloc[fold["test"]]

Uso: 
    python src/RETO_DATOS/k_fold.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "src" / "RETO_DATOS" / "dataset_vacas_muestra.csv"
N_SPLITS = 5
TEST_SIZE = 0.15
RANDOM_STATE = 42


def _horarios(dataset):
    """Obtiene los instantes de inicio y fin de cada sesión.

    Args:
        dataset (pd.DataFrame): Filas originales del CSV.

    Returns:
        tuple[pd.Series, pd.Series]: Inicio y fin de las sesiones.
    """
    inicio = pd.to_datetime(dataset["inicio"], errors="coerce", format="mixed")
    duracion = pd.to_numeric(dataset["duracion_seg"], errors="coerce")
    fin = inicio + pd.to_timedelta(duracion, unit="s", errors="raise")
    
    return inicio, fin


def get_k_folds(input_dir, n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Carga un CSV y genera folds cronológicos con un test fijo.

    Args:
        input_dir (str): Ruta al archivo CSV.

        n_splits (int): Número de folds con train creciente.

        test_size (float): Fracción de sesiones más recientes reservada para test.
            Las sesiones con la misma hora de inicio quedan juntas.

        random_state (int): Se conserva por compatibilidad. Los cortes son deterministas.

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

    inicio, fin = _horarios(dataset)
    orden = inicio.sort_values(kind="stable").index.to_numpy()
    n_test = int(np.ceil(len(dataset) * test_size))
    inicio_test = inicio.iloc[orden[-n_test]]
    test = orden[inicio.iloc[orden].ge(inicio_test)]
    train_val = inicio.lt(inicio_test) & fin.lt(inicio_test)
    train_val_ordenado = orden[train_val.iloc[orden]]
    fechas_train_val = inicio.iloc[train_val_ordenado].unique()
    if len(fechas_train_val) < n_splits + 1:
        raise ValueError(
            f"Se necesitan al menos {n_splits + 1} horas de inicio distintas para train y val "
            "después de excluir las sesiones que alcanzan el inicio de test."
        )

    bloques = np.array_split(np.asarray(fechas_train_val), n_splits + 1)
    folds = []
    inicios_train_val = inicio.iloc[train_val_ordenado]
    fines_train_val = fin.iloc[train_val_ordenado]
    for numero, bloque_val in enumerate(bloques[1:], start=1):
        inicio_val = bloque_val[0]
        train = train_val_ordenado[inicios_train_val.lt(inicio_val) & fines_train_val.lt(inicio_val)]
        val = train_val_ordenado[inicios_train_val.isin(bloque_val)]
        if not len(train):
            raise ValueError(f"El fold {numero} queda sin train por sesiones que alcanzan el inicio de val.")
        folds.append({"train": train, "val": val, "test": test})

    return dataset, folds


def _mostrar_vacas(cows, evaluacion, referencia):
    """Imprime sesiones y vacas vistas y no vistas respecto a un conjunto.

    Args:
        cows (pd.Series): Identificadores alineados con las filas del CSV.
        evaluacion (np.ndarray): Posiciones de las sesiones evaluadas.
        referencia (np.ndarray): Posiciones donde se buscan antecedentes de las vacas.
    """
    evaluadas = cows.iloc[evaluacion]
    identificadas = evaluadas[evaluadas.ne("")]
    vistas = identificadas.isin(cows.iloc[referencia])
    for etiqueta, grupo in (("Vistas", identificadas[vistas]), ("No vistas", identificadas[~vistas])):
        print(f"  {etiqueta}: {len(grupo)} sesiones de {grupo.nunique()} vacas")
    if len(identificadas) < len(evaluadas):
        print(f"  Sin identificador: {len(evaluadas) - len(identificadas)} sesiones")


if __name__ == "__main__":
    # Probar dentro del archivo, si se importa la función en otro script, no se
    # ejecuta este bloque.

    dataset, folds = get_k_folds(INPUT_DIR, N_SPLITS, TEST_SIZE, RANDOM_STATE)
    inicio, fin = _horarios(dataset)
    cows = dataset.get("id_vaca", pd.Series("", index=dataset.index)).str.strip()
    test_rows = folds[0]["test"]
    inicio_test = inicio.iloc[test_rows].min()
    train_val = np.flatnonzero(inicio.lt(inicio_test) & fin.lt(inicio_test))
    print(f"Sesiones excluidas de train y val por alcanzar el inicio de test: {len(dataset) - len(train_val) - len(test_rows)}")
    for numero, fold in enumerate(folds, start=1):
        print(
            f"Fold {numero}: train={len(fold['train'])}, "
            f"val={len(fold['val'])}, test={len(fold['test'])}"
        )
        for conjunto in ("train", "val", "test"):
            fechas = inicio.iloc[fold[conjunto]]
            print(f"  {conjunto}: {fechas.min()} a {fechas.max()}")
        print("Vacas en val respecto a train:")
        _mostrar_vacas(cows, fold["val"], fold["train"])
        print("===============================================================")

    print("Vacas en test respecto a todas las sesiones disponibles para train y val:")
    _mostrar_vacas(cows, test_rows, train_val)

    fold = folds[0]

    train = dataset.iloc[fold["train"]]
    val = dataset.iloc[fold["val"]]
    test = dataset.iloc[fold["test"]]

    print("===================================================================")
    print("Primeros 5 registros de train: ")
    print(train.head())
    print("===================================================================")
    print("Primeros 5 registros de val: ")
    print(val.head())
    print("===================================================================")
    print("Primeros 5 registros de test: ")
    print(test.head())
    print("===================================================================")
