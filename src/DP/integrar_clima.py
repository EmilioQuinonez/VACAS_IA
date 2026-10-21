"""
Agrega a las sesiones de ordeño los atributos de clima de NASA POWER (fase de Data Preparation).

Lee Data/dataset_vacas_anonimo.csv (vacas/3.0) y Data/clima/clima_nasa_power_por_hora.csv
(descargar_clima.py) y le agrega a cada sesion la temperatura y el THI de la hora en que inicia
y las horas con THI alto de las 24 h previas. Las sesiones no se limpian ni se modifican: solo
se agregan esas tres columnas. Guarda Data/dataset_vacas_clima.csv (vacas/5.0).

Uso:
    python src/DP/integrar_clima.py
"""
from pathlib import Path

import pandas as pd

raiz = Path(__file__).resolve().parent.parent.parent
ORIGEN = raiz / "Data" / "dataset_vacas_anonimo.csv"
CLIMA = raiz / "Data" / "clima" / "clima_nasa_power_por_hora.csv"
SALIDA = raiz / "Data" / "dataset_vacas_clima.csv"

# NASA POWER etiqueta cada hora una hora antes del clima que describe
DESFASE_POWER_H = 1
UMBRAL_THI = 68


def hora_de_inicio(texto):
    """
        Convierte main_hora_de_inicio a fecha y hora. generate_data.py unio en un solo CSV las filas
        de los CSV crudos (mar-jul 2025) y de los xlsx crudos (2022-2026), y cada fila conserva el
        formato de su origen: "dd/mm/aaaa hh:mm a. m." (22 caracteres) o "aaaa-mm-dd hh:mm:ss".

        Args:
            texto (Series): Columna main_hora_de_inicio.

        Returns:
            Series: Fecha y hora de inicio de cada sesion.
    """
    de_csv = texto.str.len() == 22
    inicio = pd.Series(pd.NaT, index=texto.index, dtype="datetime64[ns]")
    inicio[de_csv] = pd.to_datetime(
        texto[de_csv].str.replace("a. m.", "AM", regex=False).str.replace("p. m.", "PM", regex=False),
        format="%d/%m/%Y %I:%M %p")
    inicio[~de_csv] = pd.to_datetime(texto[~de_csv], format="%Y-%m-%d %H:%M:%S")
    return inicio


def cargar_clima(ruta):
    """
        Lee la serie horaria de NASA POWER (descargar_clima.py), que ya viene en hora local, corrige su
        desfase y le agrega el indice temperatura-humedad:
        THI = (1.8 T + 32) - (0.55 - 0.0055 HR) * (1.8 T - 26)

        La fila etiquetada t de POWER corresponde al clima de t + 1 h (confirmado contra una estacion
        observada), asi que se corre el indice una hora: una sesion a la hora T toma la fila T - 1 h.

        Args:
            ruta (Path): CSV generado por descargar_clima.py.

        Returns:
            DataFrame: Serie horaria local, ya alineada, con las variables de POWER y el THI.
    """
    clima = pd.read_csv(ruta, index_col="hora", parse_dates=True)
    clima.index = clima.index + pd.Timedelta(hours=DESFASE_POWER_H)
    clima["thi"] = (1.8 * clima["t2m"] + 32) - (0.55 - 0.0055 * clima["rh2m"]) * (1.8 * clima["t2m"] - 26)
    return clima


def agregar_clima(sesiones, clima):
    """
        Une a cada sesion la temperatura y el THI de la hora en que inicia y el numero de horas con THI
        alto en las 24 h previas. La ventana termina justo antes de la hora de la sesion, asi que solo
        usa clima que ya ocurrio; si no esta completa, el atributo queda faltante. Las demas variables
        de POWER (viento, radiacion, lluvia, nubosidad...) no se usan en el modelo y no se agregan.

        Args:
            sesiones (DataFrame): Dataset anonimizado de sesiones.
            clima (DataFrame): Serie horaria de cargar_clima.

        Returns:
            DataFrame: Las sesiones con t_sesion, thi_sesion y horas_thi_alto_24h.
    """
    clima = clima.asfreq("h")
    clima["horas_thi_alto_24h"] = (clima["thi"] >= UMBRAL_THI).astype(float).where(clima["thi"].notna()).shift(1).rolling(24).sum()
    clima = clima[["t2m", "thi", "horas_thi_alto_24h"]].rename(columns={"t2m": "t_sesion", "thi": "thi_sesion"})
    hora = hora_de_inicio(sesiones["main_hora_de_inicio"]).dt.floor("h")
    return sesiones.join(clima.reindex(hora).set_axis(sesiones.index))


def main():
    sesiones = pd.read_csv(ORIGEN, dtype={"id_vaca": str, "misc_eo_po": str}, low_memory=False)
    integrado = agregar_clima(sesiones, cargar_clima(CLIMA))
    integrado.to_csv(SALIDA, index=False)

    print(f"Sesiones: {len(integrado):,} | vacas: {integrado['id_vaca'].nunique()}")
    print(f"Sin clima de su hora: {integrado['thi_sesion'].isna().sum():,} | "
          f"sin las 24 h previas completas: {integrado['horas_thi_alto_24h'].isna().sum():,}")
    print(f"Sesiones con THI ≥ {UMBRAL_THI}: {(integrado['thi_sesion'] >= UMBRAL_THI).mean():.1%}")
    print(f"Columnas: {integrado.shape[1]} | guardado en {SALIDA.relative_to(raiz)}")


if __name__ == "__main__":
    main()
