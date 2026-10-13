import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

raiz = Path(__file__).resolve().parent.parent.parent
tablas_defecto = raiz.parent / "analisis_descriptivo_datos_vacas" / "tablas"
era5_defecto = (raiz.parent / "investigacion_factibilidad" / "datos"
                / "clima_era5_land_2025-01-28_2026-01-28.json")
clima_defecto = raiz / "Data" / "clima"
salida_defecto = raiz / "docs" / "DU" / "figuras"

URL_POWER = "https://power.larc.nasa.gov/api/temporal/hourly/point"
PARAMETROS = "T2M,RH2M,PRECTOTCORR,WS10M"
# Centro de la celda de NASA POWER (MERRA-2: 0.5 x 0.625 grados), no la ubicacion exacta del establo
LATITUD_CELDA = 20.5
LONGITUD_CELDA = -100.0
ZONA_HORARIA = "America/Mexico_City"
UMBRAL_THI = 68

COLOR_BASE = "#2a7a8c"
COLOR_ALERTA = "#b5473a"


def calcular_thi(temperatura, humedad):
    """
        Calcula el indice temperatura-humedad (THI) para ganado.
        THI = (1.8 T + 32) - (0.55 - 0.0055 HR) * (1.8 T - 26)

        Args:
            temperatura (Series): Temperatura del aire en grados Celsius.
            humedad (Series): Humedad relativa en porcentaje.

        Returns:
            Series: El THI de cada hora.
    """
    return (1.8 * temperatura + 32) - (0.55 - 0.0055 * humedad) * (1.8 * temperatura - 26)


def descargar_power(inicio, fin, destino):
    """
        Descarga de NASA POWER la serie horaria (UTC) entre dos fechas, en bloques de un ano.
        Cada bloque se guarda como JSON en la carpeta de datos y no se vuelve a pedir si ya existe.

        Args:
            inicio (Timestamp): Primer dia a descargar.
            fin (Timestamp): Ultimo dia a descargar.
            destino (Path): Carpeta donde se guardan las respuestas originales.

        Returns:
            DataFrame: Serie horaria en UTC con t2m, rh2m, prectotcorr y ws10m.
    """
    destino.mkdir(parents=True, exist_ok=True)
    bloques = []
    desde = inicio
    while desde <= fin:
        hasta = min(desde + pd.DateOffset(years=1) - pd.Timedelta(days=1), fin)
        archivo = destino / f"nasa_power_{desde:%Y%m%d}_{hasta:%Y%m%d}.json"

        if not archivo.exists():
            print(f"Descargando NASA POWER {desde:%Y-%m-%d} a {hasta:%Y-%m-%d} ...")
            respuesta = requests.get(URL_POWER, timeout=120, params={
                "parameters": PARAMETROS, "community": "AG", "format": "JSON",
                "latitude": LATITUD_CELDA, "longitude": LONGITUD_CELDA,
                "start": f"{desde:%Y%m%d}", "end": f"{hasta:%Y%m%d}", "time-standard": "UTC",
            })
            respuesta.raise_for_status()
            archivo.write_text(respuesta.text)
            time.sleep(2)

        crudo = json.loads(archivo.read_text())
        valores = pd.DataFrame(crudo["properties"]["parameter"])
        valores.index = pd.to_datetime(valores.index, format="%Y%m%d%H")
        bloques.append(valores.replace(crudo["header"]["fill_value"], np.nan))
        desde = hasta + pd.Timedelta(days=1)

    serie = pd.concat(bloques).sort_index()
    serie.index.name = "hora_utc"
    return serie.rename(columns=str.lower)


def a_hora_local(serie_utc):
    """
        Convierte una serie horaria de UTC a hora local de Mexico, sin zona horaria.
        Considera el horario de verano vigente hasta el 30/10/2022: esa hora aparece dos veces
        en la hora local y se promedia.

        Args:
            serie_utc (DataFrame): Serie con indice en UTC.

        Returns:
            DataFrame: Serie con indice de hora local.
    """
    local = serie_utc.copy()
    local.index = local.index.tz_localize("UTC").tz_convert(ZONA_HORARIA).tz_localize(None)
    local.index.name = "hora"
    return local.groupby(level=0).mean()


def leer_era5_land(ruta):
    """
        Lee la descarga exploratoria de ERA5-Land (Open-Meteo) hecha en el estudio de factibilidad.

        Args:
            ruta (Path): Archivo JSON con la respuesta original de Open-Meteo.

        Returns:
            DataFrame: Serie horaria local con temperatura y humedad.
    """
    crudo = json.loads(ruta.read_text())["hourly"]
    serie = pd.DataFrame({
        "t2m": crudo["temperature_2m"],
        "rh2m": crudo["relative_humidity_2m"],
    }, index=pd.to_datetime(crudo["time"]))
    serie.index.name = "hora"
    return serie


def metricas(referencia, comparada):
    """
        Calcula correlacion, sesgo y error cuadratico medio entre dos series alineadas.

        Args:
            referencia (Series): Serie de referencia (ERA5-Land).
            comparada (Series): Serie a evaluar (NASA POWER).

        Returns:
            dict: Pearson, sesgo (comparada - referencia), RMSE y RMSE sin el sesgo medio.
    """
    diferencia = (comparada - referencia).dropna()
    return {
        "pearson": round(float(referencia.corr(comparada)), 3),
        "sesgo": round(float(diferencia.mean()), 2),
        "rmse": round(float(np.sqrt((diferencia ** 2).mean())), 2),
        "rmse_sin_sesgo": round(float(diferencia.std(ddof=0)), 2),
    }


def evaluar_desfases(era5, power, maximo=3):
    """
        Mide la correlacion de la temperatura entre ERA5-Land y NASA POWER al correr NASA POWER
        varias horas. NASA POWER entrega promedios horarios y ERA5-Land valores instantaneos,
        por lo que puede haber un desfase de una hora.

        Args:
            era5 (DataFrame): Serie horaria local de ERA5-Land.
            power (DataFrame): Serie horaria local de NASA POWER.
            maximo (int): Horas maximas a correr en cada sentido.

        Returns:
            Series: Correlacion de Pearson de la temperatura por desfase en horas.
    """
    resultado = {}
    for desfase in range(-maximo, maximo + 1):
        corrida = power.copy()
        corrida.index = corrida.index + pd.Timedelta(hours=desfase)
        alineada = era5[["t2m"]].join(corrida[["t2m"]], how="inner", lsuffix="_era5", rsuffix="_power").dropna()
        resultado[desfase] = alineada["t2m_era5"].corr(alineada["t2m_power"])
    return pd.Series(resultado, name="pearson_t2m")


def graficar_comparacion(conjunto, desfases, destino):
    """
        Grafica la dispersion de NASA POWER contra ERA5-Land y el sesgo por hora del dia.

        Args:
            conjunto (DataFrame): Series alineadas con columnas *_era5 y *_power.
            desfases (Series): Correlacion de la temperatura por desfase en horas.
            destino (Path): Carpeta donde se guarda la imagen.
    """
    fig, ejes = plt.subplots(1, 4, figsize=(18, 4.2))
    for eje, variable, etiqueta in zip(ejes[:2], ["t2m", "rh2m"], ["Temperatura (°C)", "Humedad relativa (%)"]):
        x, y = conjunto[f"{variable}_era5"], conjunto[f"{variable}_power"]
        eje.hexbin(x, y, gridsize=45, cmap="viridis", mincnt=1, bins="log")
        limites = [min(x.min(), y.min()), max(x.max(), y.max())]
        eje.plot(limites, limites, color=COLOR_ALERTA, linestyle="--", linewidth=1.2)
        eje.set_xlabel(f"{etiqueta} - ERA5-Land")
        eje.set_ylabel(f"{etiqueta} - NASA POWER")
        eje.set_title(f"r = {x.corr(y):.2f}")
        eje.spines[["top", "right"]].set_visible(False)

    sesgo = (conjunto["thi_power"] - conjunto["thi_era5"]).groupby(conjunto.index.hour).mean()
    ejes[2].bar(sesgo.index, sesgo.values, color=COLOR_BASE)
    ejes[2].axhline(0, color="black", linewidth=0.8)
    ejes[2].set_xlabel("Hora local del día")
    ejes[2].set_ylabel("THI NASA POWER - THI ERA5-Land")
    ejes[2].set_title("Sesgo del THI por hora")
    ejes[2].spines[["top", "right"]].set_visible(False)

    colores = [COLOR_ALERTA if h == desfases.idxmax() else COLOR_BASE for h in desfases.index]
    ejes[3].bar(desfases.index, desfases.values, color=colores)
    ejes[3].set_ylim(desfases.min() - 0.05, 1)
    ejes[3].set_xlabel("Horas que se corre NASA POWER")
    ejes[3].set_ylabel("Correlación de la temperatura")
    ejes[3].set_title("Desfase entre las series")
    ejes[3].spines[["top", "right"]].set_visible(False)

    fig.suptitle("NASA POWER vs ERA5-Land (28/01/2025 a 27/01/2026), tal como se entrega NASA POWER")
    fig.tight_layout()
    fig.savefig(destino / "clima_comparacion.png", dpi=200)
    plt.close(fig)


def graficar_cobertura(sesiones_por_mes, destino):
    """
        Grafica las sesiones de ordeno por mes y marca cuales tienen clima de cada fuente.

        Args:
            sesiones_por_mes (DataFrame): Columnas total, con_power y con_era5 por mes.
            destino (Path): Carpeta donde se guarda la imagen.
    """
    fig, eje = plt.subplots(figsize=(10, 4))
    eje.bar(sesiones_por_mes.index, sesiones_por_mes["total"], width=25, color="#c9d6d9", label="Sesiones de ordeño")
    eje.bar(sesiones_por_mes.index, sesiones_por_mes["con_era5"], width=25, color=COLOR_ALERTA,
            label="Con clima ERA5-Land (descarga actual)")
    eje.plot(sesiones_por_mes.index, sesiones_por_mes["con_power"], color=COLOR_BASE, linewidth=2.2,
             label="Con clima NASA POWER")
    eje.set_ylabel("Sesiones por mes")
    eje.set_title("Cobertura de clima sobre las sesiones de ordeño")
    eje.legend(frameon=False)
    eje.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(destino / "clima_cobertura.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compara NASA POWER con ERA5-Land y mide su cobertura sobre los ordeños.")
    parser.add_argument("--tablas", type=Path, default=tablas_defecto,
                        help="Carpeta con leche_eventos_normalizados.csv.gz")
    parser.add_argument("--era5", type=Path, default=era5_defecto, help="JSON de ERA5-Land de Open-Meteo")
    parser.add_argument("--clima", type=Path, default=clima_defecto, help="Carpeta local (fuera de Git) para el clima")
    parser.add_argument("--salida", type=Path, default=salida_defecto, help="Carpeta de salida de las imágenes")
    args = parser.parse_args()

    args.salida.mkdir(parents=True, exist_ok=True)
    leche = pd.read_csv(args.tablas / "leche_eventos_normalizados.csv.gz", usecols=["timestamp"], parse_dates=["timestamp"])
    print(f"Sesiones leídas: {len(leche):,} ({leche['timestamp'].min():%d/%m/%Y} a {leche['timestamp'].max():%d/%m/%Y})")

    # Un dia de margen en cada extremo para no perder horas al pasar de UTC a hora local
    power = a_hora_local(descargar_power(leche["timestamp"].min().normalize() - pd.Timedelta(days=1),
                                         leche["timestamp"].max().normalize() + pd.Timedelta(days=1), args.clima))
    power["thi"] = calcular_thi(power["t2m"], power["rh2m"])
    era5 = leer_era5_land(args.era5)
    era5["thi"] = calcular_thi(era5["t2m"], era5["rh2m"])

    power.to_csv(args.clima / "clima_nasa_power_por_hora.csv")
    print(f"NASA POWER: {len(power):,} horas, faltantes T2M {power['t2m'].isna().sum()}, RH2M {power['rh2m'].isna().sum()}")

    conjunto = era5.join(power, how="inner", lsuffix="_era5", rsuffix="_power").dropna(
        subset=["t2m_era5", "t2m_power", "rh2m_era5", "rh2m_power"])
    print(f"Horas en común: {len(conjunto):,}")

    resultados = {"horas_en_comun": len(conjunto)}
    for variable in ["t2m", "rh2m", "thi"]:
        resultados[variable] = metricas(conjunto[f"{variable}_era5"], conjunto[f"{variable}_power"])
        print(variable, resultados[variable])

    # NASA POWER son promedios horarios: se busca el desfase que mejor alinea las series
    desfases = evaluar_desfases(era5, power)
    mejor = int(desfases.idxmax())
    resultados["pearson_t2m_por_desfase"] = {int(h): round(float(r), 3) for h, r in desfases.items()}
    resultados["mejor_desfase_horas"] = mejor
    print("Correlación de T2M por desfase (h):", resultados["pearson_t2m_por_desfase"])

    corrida = power.copy()
    corrida.index = corrida.index + pd.Timedelta(hours=mejor)
    corrida["thi"] = calcular_thi(corrida["t2m"], corrida["rh2m"])
    alineado = era5.join(corrida, how="inner", lsuffix="_era5", rsuffix="_power").dropna(
        subset=["t2m_era5", "t2m_power", "rh2m_era5", "rh2m_power"])
    resultados["con_desfase_corregido"] = {v: metricas(alineado[f"{v}_era5"], alineado[f"{v}_power"])
                                           for v in ["t2m", "rh2m", "thi"]}
    print(f"Métricas con NASA POWER corrida {mejor} h:", resultados["con_desfase_corregido"])

    # Las dos fuentes deben coincidir en la dinamica diaria aunque difieran en el nivel
    diario = conjunto.resample("D").mean()
    resultados["pearson_diario"] = {v: round(float(diario[f"{v}_era5"].corr(diario[f"{v}_power"])), 3)
                                    for v in ["t2m", "rh2m", "thi"]}
    print("Pearson de promedios diarios:", resultados["pearson_diario"])

    resultados["pct_horas_thi_alto"] = {
        "era5_land": round(float((conjunto["thi_era5"] >= UMBRAL_THI).mean() * 100), 1),
        "nasa_power": round(float((conjunto["thi_power"] >= UMBRAL_THI).mean() * 100), 1),
    }
    print(f"% de horas con THI >= {UMBRAL_THI}:", resultados["pct_horas_thi_alto"])

    # Cobertura de clima sobre las sesiones, uniendo por hora
    hora = leche["timestamp"].dt.floor("h")
    con_power = hora.isin(power.dropna(subset=["t2m", "rh2m"]).index)
    con_era5 = hora.isin(era5.index)
    resultados["cobertura_sesiones_pct"] = {
        "nasa_power": round(float(con_power.mean() * 100), 2),
        "era5_land_actual": round(float(con_era5.mean() * 100), 2),
    }
    print("Cobertura de sesiones con clima:", resultados["cobertura_sesiones_pct"])

    por_mes = pd.DataFrame({"total": 1, "con_power": con_power.astype(int), "con_era5": con_era5.astype(int)})
    por_mes = por_mes.groupby(leche["timestamp"].dt.to_period("M").dt.to_timestamp()).sum()

    graficar_comparacion(conjunto, desfases, args.salida)
    graficar_cobertura(por_mes, args.salida)
    (args.clima / "resultados_comparacion_clima.json").write_text(json.dumps(resultados, indent=2))
    print(f"Figuras guardadas en: {args.salida}")


if __name__ == "__main__":
    main()
