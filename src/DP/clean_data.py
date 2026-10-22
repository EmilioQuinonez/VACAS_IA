from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
RUTA_VACAS = RAIZ / "Data" / "dataset_vacas_anonimo.csv"
RUTA_CLIMA = RAIZ / "Data" / "clima" / "clima_nasa_power_por_hora.csv"

vacas = pd.read_csv(RUTA_VACAS, dtype={"id_vaca": str, "misc_eo_po": str}, low_memory=False)
clima = pd.read_csv(RUTA_CLIMA, parse_dates=["hora"])


