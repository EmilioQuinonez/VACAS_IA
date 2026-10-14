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

from clima_comparacion import (ZONA_HORARIA, UMBRAL_THI, LATITUD_CELDA, LONGITUD_CELDA, calcular_thi,
                               descargar_power, a_hora_local)

raiz = Path(__file__).resolve().parent.parent.parent
tablas_defecto = raiz.parent / "analisis_descriptivo_datos_vacas" / "tablas"
clima_defecto = raiz / "Data" / "clima"
salida_defecto = raiz / "docs" / "DU" / "figuras"

URL_OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
URL_IEM = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
# Punto de la malla de ERA5-Land (0.1 grados) usado en el estudio de factibilidad; no es la ubicacion exacta del establo
LATITUD_ERA5 = 20.5
LONGITUD_ERA5 = -100.2
ESTACION = "MMQT"
UMBRAL_THI_DIARIO = 70
MARGEN_DIAS = 7
MAXIMO_DESFASE = 3
MIN_OBS_POR_DIA = 18

# Regla de decision fijada antes de ver los resultados (ver docs/DU/Investigacion_Clima.md):
# NASA POWER se queda como fuente salvo que otra candidata reduzca el RMSE del THI diario medio
# al menos esta fraccion y no empeore el F1 del umbral de THI por hora.
MEJORA_RELATIVA_MINIMA = 0.10

COLOR_BASE = "#2a7a8c"
COLOR_ALERTA = "#b5473a"
COLOR_AUXILIAR = "#e0a458"
COLORES_FUENTE = {"nasa_power": COLOR_BASE, "era5_land": COLOR_ALERTA, "era5": COLOR_AUXILIAR}
NOMBRES_FUENTE = {"nasa_power": "NASA POWER", "era5_land": "ERA5-Land", "era5": "ERA5"}


def descargar_open_meteo(modelo, inicio, fin, destino):
    """
        Descarga de la API historica de Open-Meteo la serie horaria (UTC) de un reanalisis, en bloques de un ano.
        Cada bloque se guarda como JSON y no se vuelve a pedir si ya existe.

        Args:
            modelo (str): Modelo de reanalisis, "era5_land" o "era5".
            inicio (Timestamp): Primer dia a descargar.
            fin (Timestamp): Ultimo dia a descargar.
            destino (Path): Carpeta donde se guardan las respuestas originales.

        Returns:
            DataFrame: Serie horaria en UTC con t2m y rh2m. ERA5-Land no entrega viento, radiacion ni
                precipitacion por esta API (llegan nulos), por lo que solo se piden temperatura y humedad.
    """
    destino.mkdir(parents=True, exist_ok=True)
    bloques = []
    desde = inicio
    while desde <= fin:
        hasta = min(desde + pd.DateOffset(years=1) - pd.Timedelta(days=1), fin)
        archivo = destino / f"open_meteo_{modelo}_{desde:%Y%m%d}_{hasta:%Y%m%d}.json"

        if not archivo.exists():
            print(f"Descargando Open-Meteo {modelo} {desde:%Y-%m-%d} a {hasta:%Y-%m-%d} ...")
            respuesta = requests.get(URL_OPEN_METEO, timeout=120, params={
                "latitude": LATITUD_ERA5, "longitude": LONGITUD_ERA5, "models": modelo,
                "start_date": f"{desde:%Y-%m-%d}", "end_date": f"{hasta:%Y-%m-%d}", "timezone": "GMT",
                "hourly": "temperature_2m,relative_humidity_2m",
            })
            respuesta.raise_for_status()
            archivo.write_text(respuesta.text)
            time.sleep(2)

        crudo = json.loads(archivo.read_text())["hourly"]
        bloque = pd.DataFrame({"t2m": crudo["temperature_2m"], "rh2m": crudo["relative_humidity_2m"]},
                              index=pd.to_datetime(crudo["time"]))
        bloques.append(bloque)
        desde = hasta + pd.Timedelta(days=1)

    serie = pd.concat(bloques).sort_index()
    serie.index.name = "hora_utc"
    return serie.astype(float)


def descargar_estacion(inicio, fin, destino):
    """
        Descarga del Iowa Environmental Mesonet las observaciones METAR de la estacion de referencia.
        La estacion solo sirve para validar: sus datos no entran a la tabla final de clima.

        Args:
            inicio (Timestamp): Primer dia a descargar.
            fin (Timestamp): Ultimo dia a descargar.
            destino (Path): Carpeta donde se guarda el CSV original.

        Returns:
            DataFrame: Observaciones con indice en UTC y columnas t2m y rh2m.
    """
    destino.mkdir(parents=True, exist_ok=True)
    archivo = destino / f"iem_{ESTACION}_{inicio:%Y%m%d}_{fin:%Y%m%d}.csv"
    if not archivo.exists():
        print(f"Descargando observaciones de {ESTACION} del Iowa Mesonet ...")
        respuesta = requests.get(URL_IEM, timeout=180, params={
            "station": ESTACION, "data": ["tmpc", "relh"], "tz": "UTC", "format": "onlycomma",
            "missing": "M", "trace": "T", "direct": "yes", "report_type": [3, 4],
            "year1": inicio.year, "month1": inicio.month, "day1": inicio.day,
            "year2": fin.year, "month2": fin.month, "day2": fin.day,
        })
        respuesta.raise_for_status()
        archivo.write_text(respuesta.text)

    crudo = pd.read_csv(archivo, na_values=["M"], parse_dates=["valid"])
    estacion = crudo.set_index("valid")[["tmpc", "relh"]].rename(columns={"tmpc": "t2m", "relh": "rh2m"})
    estacion.index.name = "hora_utc"
    return estacion.dropna().sort_index()


def interpolar(serie, tiempos, desfase_horas):
    """
        Evalua una serie horaria en instantes arbitrarios por interpolacion lineal, despues de correrla
        desfase_horas. Los instantes fuera del rango de la serie quedan como nulos.

        Args:
            serie (Series): Serie horaria con indice en UTC.
            tiempos (DatetimeIndex): Instantes (UTC) donde se quiere el valor.
            desfase_horas (int): Horas que se corre la serie hacia adelante.

        Returns:
            ndarray: Valor interpolado en cada instante.
    """
    valida = serie.dropna()
    x = (valida.index + pd.Timedelta(hours=desfase_horas)).values.astype("datetime64[s]").astype("int64")
    t = tiempos.values.astype("datetime64[s]").astype("int64")
    return np.interp(t, x, valida.values, left=np.nan, right=np.nan)


def alinear_con_estacion(fuente, estacion, desfase_horas):
    """
        Evalua una fuente en los instantes en que la estacion reporto y calcula el THI de ambas.

        Args:
            fuente (DataFrame): Serie horaria de la fuente (UTC) con t2m y rh2m.
            estacion (DataFrame): Observaciones de la estacion (UTC) con t2m y rh2m.
            desfase_horas (int): Horas que se corre la fuente.

        Returns:
            DataFrame: Una fila por observacion con columnas est_* y fte_* de t2m, rh2m y thi.
    """
    alineado = pd.DataFrame({"est_t2m": estacion["t2m"], "est_rh2m": estacion["rh2m"]})
    alineado["fte_t2m"] = interpolar(fuente["t2m"], estacion.index, desfase_horas)
    alineado["fte_rh2m"] = interpolar(fuente["rh2m"], estacion.index, desfase_horas)
    alineado = alineado.dropna()
    for prefijo in ["est", "fte"]:
        alineado[f"{prefijo}_thi"] = calcular_thi(alineado[f"{prefijo}_t2m"], alineado[f"{prefijo}_rh2m"])
    return alineado


def errores(referencia, comparada):
    """
        Calcula sesgo, error absoluto medio, RMSE, RMSE sin el sesgo medio y correlacion de una serie contra su referencia.
        El RMSE sin sesgo mide solo cuanto se parece la dinamica de la fuente a la de la estacion.

        Args:
            referencia (Series): Valores de la estacion.
            comparada (Series): Valores de la fuente, alineados con la referencia.

        Returns:
            dict: sesgo (fuente - estacion), mae, rmse, rmse_sin_sesgo y pearson.
    """
    diferencia = comparada - referencia
    return {"sesgo": round(float(diferencia.mean()), 3), "mae": round(float(diferencia.abs().mean()), 3),
            "rmse": round(float(np.sqrt((diferencia ** 2).mean())), 3),
            "rmse_sin_sesgo": round(float(diferencia.std(ddof=0)), 3),
            "pearson": round(float(referencia.corr(comparada)), 3)}


def acuerdo_umbral(referencia, comparada, umbral):
    """
        Mide si la fuente cruza el umbral de THI cuando lo cruza la estacion.

        Args:
            referencia (Series): THI de la estacion.
            comparada (Series): THI de la fuente.
            umbral (float): THI a partir del cual hay estres termico.

        Returns:
            dict: sensibilidad, precision y F1 de la deteccion del umbral.
    """
    real, predicho = referencia >= umbral, comparada >= umbral
    verdaderos = float((real & predicho).sum())
    sensibilidad = verdaderos / max(float(real.sum()), 1.0)
    precision = verdaderos / max(float(predicho.sum()), 1.0)
    f1 = 2 * sensibilidad * precision / max(sensibilidad + precision, 1e-9)
    return {"sensibilidad": round(sensibilidad, 3), "precision": round(precision, 3), "f1": round(f1, 3)}


def resumir_fuente(alineado):
    """
        Calcula las metricas por hora y por dia de una fuente contra la estacion.
        Los resumenes diarios usan los mismos instantes en las dos series, para que difieran solo
        por la fuente, y exigen al menos MIN_OBS_POR_DIA observaciones en el dia local.

        Args:
            alineado (DataFrame): Salida de alinear_con_estacion.

        Returns:
            dict: Metricas horarias, diarias y de acuerdo del umbral.
    """
    resumen = {"observaciones": int(len(alineado))}
    resumen["horario"] = {v: errores(alineado[f"est_{v}"], alineado[f"fte_{v}"]) for v in ["t2m", "rh2m", "thi"]}

    dia_local = alineado.index.tz_localize("UTC").tz_convert(ZONA_HORARIA).tz_localize(None).normalize()
    por_dia = alineado.groupby(dia_local)
    completos = por_dia.size() >= MIN_OBS_POR_DIA
    medias = por_dia[[c for c in alineado.columns]].mean()[completos]
    maximos = por_dia[["est_thi", "fte_thi"]].max()[completos]
    resumen["dias"] = int(completos.sum())
    resumen["diario_medio"] = {v: errores(medias[f"est_{v}"], medias[f"fte_{v}"]) for v in ["t2m", "rh2m", "thi"]}
    resumen["diario_thi_max"] = errores(maximos["est_thi"], maximos["fte_thi"])
    resumen["umbral_thi_horario"] = acuerdo_umbral(alineado["est_thi"], alineado["fte_thi"], UMBRAL_THI)
    resumen["umbral_thi_max_diario"] = acuerdo_umbral(maximos["est_thi"], maximos["fte_thi"], UMBRAL_THI_DIARIO)
    return resumen


def buscar_desfase(fuente, estacion):
    """
        Busca cuantas horas hay que correr la fuente para que su temperatura coincida mejor con la estacion.

        Args:
            fuente (DataFrame): Serie horaria de la fuente (UTC).
            estacion (DataFrame): Observaciones de la estacion (UTC).

        Returns:
            Series: Correlacion de la temperatura por desfase en horas.
    """
    correlaciones = {}
    for desfase in range(-MAXIMO_DESFASE, MAXIMO_DESFASE + 1):
        alineado = alinear_con_estacion(fuente, estacion, desfase)
        correlaciones[desfase] = alineado["est_t2m"].corr(alineado["fte_t2m"])
    return pd.Series(correlaciones)


def cobertura_sesiones(sesiones, fuente, desfase_horas):
    """
        Calcula el porcentaje de sesiones cuya hora local tiene temperatura y humedad en la fuente.

        Args:
            sesiones (Series): Hora de inicio de cada sesion (hora local).
            fuente (DataFrame): Serie horaria de la fuente (UTC).
            desfase_horas (int): Horas que se corre la fuente.

        Returns:
            float: Porcentaje de sesiones con clima.
    """
    local = a_hora_local(fuente.dropna(subset=["t2m", "rh2m"]).shift(desfase_horas, freq="h"))
    return float(sesiones.dt.floor("h").isin(local.index).mean() * 100)


def elegir_fuente(resumenes):
    """
        Aplica la regla de decision: gana NASA POWER salvo que otra candidata mejore con claridad.

        Args:
            resumenes (dict): Metricas de cada fuente, con llaves nasa_power, era5_land, etc.

        Returns:
            tuple: Nombre de la fuente elegida y texto con la justificacion.
    """
    base = resumenes["nasa_power"]["diario_medio"]["thi"]["rmse"]
    aspirantes = []
    for nombre, resumen in resumenes.items():
        if nombre == "nasa_power":
            continue
        mejora = (base - resumen["diario_medio"]["thi"]["rmse"]) / base
        f1_igual_o_mayor = resumen["umbral_thi_horario"]["f1"] >= resumenes["nasa_power"]["umbral_thi_horario"]["f1"]
        if mejora >= MEJORA_RELATIVA_MINIMA and f1_igual_o_mayor:
            aspirantes.append((mejora, nombre))
    if aspirantes:
        mejora, nombre = max(aspirantes)
        return nombre, f"{NOMBRES_FUENTE[nombre]} reduce {mejora * 100:.0f} % el RMSE del THI diario respecto a NASA POWER"
    return "nasa_power", "ninguna candidata supera a NASA POWER por el margen fijado; se queda la fuente sugerida por el socio"


def construir_variables_clima(clima):
    """
        Calcula, para cada hora, las variables de clima que se conocen antes de esa hora:
        horas con THI alto en las 24 y 48 h previas, resumen del dia anterior y THI medio de 1 a 4 dias antes.

        Args:
            clima (DataFrame): Serie horaria local y continua con t2m, rh2m y thi.

        Returns:
            DataFrame: La serie original mas las variables derivadas, con el mismo indice.
    """
    clima = clima.copy()
    alto = (clima["thi"] >= UMBRAL_THI).astype(float).where(clima["thi"].notna())
    for horas in [24, 48]:
        clima[f"horas_thi_alto_{horas}h"] = alto.rolling(horas, min_periods=horas).sum().shift(1)

    dia = clima.index.normalize()
    diario = clima.groupby(dia).agg(thi_max=("thi", "max"), thi_medio=("thi", "mean"), thi_min=("thi", "min"),
                                    t_max=("t2m", "max"), t_min=("t2m", "min"))
    diario["amplitud_t"] = diario["t_max"] - diario["t_min"]
    previo = diario[["thi_max", "thi_medio", "thi_min", "amplitud_t"]].shift(1, freq="D")
    previo.columns = [f"{c}_dia_previo" for c in previo.columns]
    rezagos = pd.concat({f"thi_medio_rezago_{k}d": diario["thi_medio"].shift(k, freq="D") for k in range(1, 5)}, axis=1)

    derivadas = pd.concat([previo, rezagos], axis=1)
    return clima.join(derivadas.reindex(dia).set_axis(clima.index))


def graficar_validacion(alineados, desfases, resumenes, elegida, destino):
    """
        Grafica la validacion de las fuentes contra la estacion: desfase, sesgo del THI por hora,
        error del THI diario y acuerdo del umbral de estres termico.

        Args:
            alineados (dict): Observaciones alineadas de cada fuente.
            desfases (dict): Correlacion de la temperatura por desfase de cada fuente.
            resumenes (dict): Metricas de cada fuente.
            elegida (str): Fuente elegida.
            destino (Path): Carpeta donde se guarda la imagen.
    """
    fig, ejes = plt.subplots(1, 4, figsize=(19, 4.3))
    for nombre, serie in desfases.items():
        ejes[0].plot(serie.index, serie.values, marker="o", color=COLORES_FUENTE[nombre], label=NOMBRES_FUENTE[nombre])
    ejes[0].set_xlabel("Horas que se corre la fuente")
    ejes[0].set_ylabel("Correlación de la temperatura con la estación")
    ejes[0].set_title("Desfase horario")
    ejes[0].legend(frameon=False, fontsize=8)

    for nombre, alineado in alineados.items():
        hora = alineado.index.tz_localize("UTC").tz_convert(ZONA_HORARIA).hour
        sesgo = (alineado["fte_thi"] - alineado["est_thi"]).groupby(hora).mean()
        ejes[1].plot(sesgo.index, sesgo.values, color=COLORES_FUENTE[nombre], linewidth=2, label=NOMBRES_FUENTE[nombre])
    ejes[1].axhline(0, color="black", linewidth=0.8)
    ejes[1].set_xlabel("Hora local del día")
    ejes[1].set_ylabel("THI fuente - THI estación")
    ejes[1].set_title("Sesgo del THI por hora")

    x = np.arange(3)
    ancho = 0.8 / len(resumenes)
    for i, (nombre, resumen) in enumerate(resumenes.items()):
        valores = [resumen["diario_medio"][v]["rmse"] for v in ["t2m", "rh2m", "thi"]]
        ejes[2].bar(x + i * ancho, valores, ancho, color=COLORES_FUENTE[nombre], label=NOMBRES_FUENTE[nombre])
    ejes[2].set_xticks(x + ancho * (len(resumenes) - 1) / 2)
    ejes[2].set_xticklabels(["Temperatura (°C)", "Humedad (%)", "THI"])
    ejes[2].set_ylabel("RMSE de promedios diarios")
    ejes[2].set_title("Error diario contra la estación")

    metricas = [("umbral_thi_horario", f"THI hora ≥ {UMBRAL_THI}"),
                ("umbral_thi_max_diario", f"THI máx. día ≥ {UMBRAL_THI_DIARIO}")]
    x = np.arange(len(metricas))
    for i, (nombre, resumen) in enumerate(resumenes.items()):
        ejes[3].bar(x + i * ancho, [resumen[m]["f1"] for m, _ in metricas], ancho, color=COLORES_FUENTE[nombre],
                    edgecolor="black" if nombre == elegida else "none", linewidth=1.5)
    ejes[3].set_xticks(x + ancho * (len(resumenes) - 1) / 2)
    ejes[3].set_xticklabels([etiqueta for _, etiqueta in metricas])
    ejes[3].set_ylim(0, 1)
    ejes[3].set_ylabel("F1 al detectar estrés térmico")
    ejes[3].set_title("Acuerdo del umbral")

    fig.suptitle(f"Fuentes de clima contra la estación de referencia ({ESTACION}), 2022-2026. "
                 f"Fuente elegida: {NOMBRES_FUENTE[elegida]}")
    for eje in ejes:
        eje.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(destino / "clima_validacion.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Valida las fuentes de clima contra una estación y deja la tabla "
                                                 "horaria de la fuente elegida.")
    parser.add_argument("--tablas", type=Path, default=tablas_defecto,
                        help="Carpeta con leche_eventos_normalizados.csv.gz")
    parser.add_argument("--clima", type=Path, default=clima_defecto, help="Carpeta local (fuera de Git) para el clima")
    parser.add_argument("--salida", type=Path, default=salida_defecto, help="Carpeta de salida de las imágenes")
    args = parser.parse_args()

    args.salida.mkdir(parents=True, exist_ok=True)
    sesiones = pd.read_csv(args.tablas / "leche_eventos_normalizados.csv.gz", usecols=["timestamp"],
                           parse_dates=["timestamp"])["timestamp"]
    inicio = sesiones.min().normalize() - pd.Timedelta(days=MARGEN_DIAS)
    fin = sesiones.max().normalize() + pd.Timedelta(days=1)
    print(f"Sesiones: {len(sesiones):,} ({sesiones.min():%d/%m/%Y} a {sesiones.max():%d/%m/%Y})")

    fuentes = {
        "nasa_power": descargar_power(inicio, fin, args.clima),
        "era5_land": descargar_open_meteo("era5_land", inicio, fin, args.clima),
        "era5": descargar_open_meteo("era5", inicio, fin, args.clima),
    }
    estacion = descargar_estacion(inicio, fin, args.clima)
    print(f"Estación {ESTACION}: {len(estacion):,} observaciones, {estacion.index.min()} a {estacion.index.max()}")

    resultados = {"estacion": ESTACION, "observaciones_estacion": int(len(estacion)),
                  "regla": {"mejora_relativa_minima_rmse_thi_diario": MEJORA_RELATIVA_MINIMA,
                            "umbral_thi_horario": UMBRAL_THI, "umbral_thi_max_diario": UMBRAL_THI_DIARIO},
                  "fuentes": {}}
    alineados, desfases, resumenes, elegidos_desfase = {}, {}, {}, {}
    for nombre, fuente in fuentes.items():
        desfases[nombre] = buscar_desfase(fuente, estacion)
        elegidos_desfase[nombre] = int(desfases[nombre].idxmax())
        alineados[nombre] = alinear_con_estacion(fuente, estacion, elegidos_desfase[nombre])
        resumenes[nombre] = resumir_fuente(alineados[nombre])
        sin_desfase = resumir_fuente(alinear_con_estacion(fuente, estacion, 0))
        resultados["fuentes"][nombre] = {
            "desfase_horas": elegidos_desfase[nombre],
            "pearson_t2m_por_desfase": {int(h): round(float(r), 3) for h, r in desfases[nombre].items()},
            "cobertura_sesiones_pct": round(cobertura_sesiones(sesiones, fuente, elegidos_desfase[nombre]), 2),
            "faltantes_t2m_rh2m": int(fuente[["t2m", "rh2m"]].isna().any(axis=1).sum()),
            "con_desfase": resumenes[nombre],
            "sin_desfase": sin_desfase,
        }
        print(f"{nombre}: desfase {elegidos_desfase[nombre]} h, "
              f"RMSE THI diario {resumenes[nombre]['diario_medio']['thi']['rmse']}, "
              f"F1 umbral horario {resumenes[nombre]['umbral_thi_horario']['f1']}")

    elegida, motivo = elegir_fuente(resumenes)
    resultados["fuente_elegida"], resultados["motivo"] = elegida, motivo
    print(f"Fuente elegida: {NOMBRES_FUENTE[elegida]} ({motivo})")

    # Tabla final: la fuente elegida, corrida su desfase, en hora local y continua
    final = fuentes[elegida].shift(elegidos_desfase[elegida], freq="h")
    final = a_hora_local(final)
    final = final[["t2m", "rh2m"]].reindex(pd.date_range(final.index.min(), final.index.max(), freq="h"))
    final["thi"] = calcular_thi(final["t2m"], final["rh2m"])
    final.index.name = "hora_local"
    final = construir_variables_clima(final)
    final.to_csv(args.clima / "clima_horario.csv")
    resultados["tabla_final"] = {"horas": int(len(final)), "inicio": str(final.index.min()), "fin": str(final.index.max()),
                                 "columnas": list(final.columns)}

    graficar_validacion(alineados, desfases, resumenes, elegida, args.salida)
    (args.clima / "validacion_clima.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False))
    print(f"Tabla horaria: {args.clima / 'clima_horario.csv'}")
    print(f"Figura: {args.salida / 'clima_validacion.png'}")


if __name__ == "__main__":
    main()
