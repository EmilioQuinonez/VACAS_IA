"""
Descarga el clima por hora de NASA POWER para las fechas de las sesiones de ordeno.

Cubre desde el dato mas antiguo hasta el ultimo de las sesiones: del 13/07/2022 al 28/01/2026.
Requiere: numpy, pandas y requests (ver requirements.txt).

Uso:
    python src/scripts/descargar_clima.py

Salidas (en Data/clima/, que Git ignora):
    nasa_power_<inicio>_<fin>.json   respuesta original de cada bloque anual
    clima_nasa_power_por_hora.csv    serie horaria en hora local (America/Mexico_City)
    nasa_power_manifest.json         version de la API, fechas de descarga, unidades y faltantes

NASA POWER no pide llave ni cuenta: el script no lee el .env ni maneja credenciales.
El CSV conserva la hora tal como la entrega POWER (promedio horario); no aplica ningun
desfase. Al terminar imprime la fila para docs/data_audit_logs.csv y el texto de citacion.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

raiz = Path(__file__).resolve().parent.parent.parent
destino = raiz / "Data" / "clima"

URL_POWER = "https://power.larc.nasa.gov/api/temporal/hourly/point"
# Centro de la celda de NASA POWER (MERRA-2: 0.5 x 0.625 grados). POWER devuelve el mismo
# valor para cualquier punto de la celda, asi que no se necesita una ubicacion mas precisa
LATITUD_CELDA = 20.5
LONGITUD_CELDA = -100.0
ZONA_HORARIA = "America/Mexico_City"
COMUNIDAD = "AG"

# Primera y ultima fecha (hora local) de las sesiones de ordeno
INICIO_LOCAL = "2022-07-13"
FIN_LOCAL = "2026-01-28"
# NASA se pide en UTC: el ultimo dia local termina el 29/01 en UTC
INICIO_UTC = "2022-07-13"
FIN_UTC = "2026-01-29"

# Variables horarias de la comunidad AG
PARAMETROS = [
    "T2M", "RH2M", "T2MDEW", "T2MWET", "QV2M", "PS", "WS2M", "WS10M", "WD10M",
    "PRECTOTCORR", "ALLSKY_SFC_SW_DWN", "ALLSKY_SFC_LW_DWN", "CLOUD_AMT",
]

ESTADOS_REINTENTO = {429, 500, 502, 503, 504}
REINTENTOS = 3
PAUSA_ENTRE_PETICIONES = 2
TIMEOUT = 120
USER_AGENT = "VacasIA-descarga-clima/1.0"

ARCHIVO_CSV = "clima_nasa_power_por_hora.csv"
ARCHIVO_MANIFIESTO = "nasa_power_manifest.json"


def pedir_bloque(desde, hasta, parametros):
    """
        Pide a NASA POWER un bloque de fechas, con reintentos y espera creciente ante
        limites de uso (429) o fallos del servidor.

        Args:
            desde (Timestamp): Primer dia del bloque.
            hasta (Timestamp): Ultimo dia del bloque.
            parametros (list): Parametros a pedir.

        Returns:
            str: El texto JSON de la respuesta.
    """
    consulta = {
        "parameters": ",".join(parametros), "community": COMUNIDAD, "format": "JSON",
        "latitude": LATITUD_CELDA, "longitude": LONGITUD_CELDA,
        "start": f"{desde:%Y%m%d}", "end": f"{hasta:%Y%m%d}", "time-standard": "UTC",
    }

    for intento in range(1, REINTENTOS + 1):
        try:
            respuesta = requests.get(URL_POWER, params=consulta, timeout=TIMEOUT,
                                     headers={"User-Agent": USER_AGENT})
        except (requests.ConnectionError, requests.Timeout):
            if intento == REINTENTOS:
                raise RuntimeError("No se pudo conectar con NASA POWER") from None
        else:
            if respuesta.ok:
                return respuesta.text
            if respuesta.status_code not in ESTADOS_REINTENTO or intento == REINTENTOS:
                raise RuntimeError(
                    f"NASA POWER respondio {respuesta.status_code}: {respuesta.text[:200]}"
                )
        time.sleep(PAUSA_ENTRE_PETICIONES * 2 ** intento)


def bloque_completo(archivo, parametros):
    """
        Dice si un bloque ya descargado se puede reutilizar: debe ser un JSON legible
        que traiga todos los parametros pedidos.

        Args:
            archivo (Path): Ruta del bloque.
            parametros (list): Parametros que se necesitan.

        Returns:
            bool: True si el bloque sirve tal como esta.
    """
    if not archivo.exists():
        return False
    try:
        disponibles = json.loads(archivo.read_text())["properties"]["parameter"]
    except (ValueError, KeyError, TypeError):
        return False
    return all(parametro in disponibles for parametro in parametros)


def descargar_bloques(inicio, fin, parametros, destino):
    """
        Descarga en bloques de un ano (UTC) las fechas pedidas. Cada bloque se guarda como
        JSON y solo se vuelve a pedir si no existe o si le faltan parametros.

        Args:
            inicio (Timestamp): Primer dia a descargar.
            fin (Timestamp): Ultimo dia a descargar.
            parametros (list): Parametros a descargar.
            destino (Path): Carpeta donde se guardan las respuestas originales.

        Returns:
            list: Rutas de los bloques, en orden cronologico.
    """
    destino.mkdir(parents=True, exist_ok=True)
    archivos = []
    desde = inicio
    while desde <= fin:
        hasta = min(desde + pd.DateOffset(years=1) - pd.Timedelta(days=1), fin)
        archivo = destino / f"nasa_power_{desde:%Y%m%d}_{hasta:%Y%m%d}.json"

        if bloque_completo(archivo, parametros):
            print(f"Ya existe {archivo.name}, no se vuelve a pedir")
        else:
            print(f"Descargando NASA POWER {desde:%Y-%m-%d} a {hasta:%Y-%m-%d} ...")
            texto = pedir_bloque(desde, hasta, parametros)

            # Se valida antes de guardar y se escribe en un temporal para no dejar JSON a medias
            disponibles = json.loads(texto)["properties"]["parameter"]
            faltan = [p for p in parametros if p not in disponibles]
            if faltan:
                raise RuntimeError(f"La respuesta no trae los parametros {faltan}")
            temporal = archivo.with_suffix(".json.tmp")
            temporal.write_text(texto)
            temporal.replace(archivo)
            time.sleep(PAUSA_ENTRE_PETICIONES)

        archivos.append(archivo)
        desde = hasta + pd.Timedelta(days=1)
    return archivos


def leer_bloque(archivo, parametros):
    """
        Lee un bloque descargado y lo convierte en una tabla horaria en UTC.

        Args:
            archivo (Path): Respuesta original de NASA POWER.
            parametros (list): Parametros a conservar.

        Returns:
            tuple: (DataFrame horario en UTC con los valores faltantes como NaN, dict con
            la version de la API y las unidades de cada parametro)
    """
    crudo = json.loads(archivo.read_text())
    valores = pd.DataFrame(crudo["properties"]["parameter"])[parametros]
    valores.index = pd.to_datetime(valores.index, format="%Y%m%d%H")
    valores = valores.replace(crudo["header"]["fill_value"], np.nan)

    info = {
        "version": crudo["header"].get("api", {}).get("version"),
        "unidades": {p: crudo.get("parameters", {}).get(p, {}).get("units") for p in parametros},
    }
    return valores, info


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


def escribir_manifiesto(destino, archivos, parametros, infos, serie_local):
    """
        Guarda los metadatos de la corrida para citar la fuente y reconstruir la descarga.

        Args:
            destino (Path): Carpeta de salida.
            archivos (list): Bloques usados.
            parametros (list): Parametros descargados.
            infos (list): Version y unidades leidas de cada bloque.
            serie_local (DataFrame): Serie horaria final en hora local.

        Returns:
            dict: El contenido del manifiesto.
    """
    fechas = [datetime.fromtimestamp(a.stat().st_mtime) for a in archivos]
    manifiesto = {
        "fuente": "NASA POWER, API horaria, comunidad " + COMUNIDAD,
        "versiones_api": sorted({i["version"] for i in infos if i["version"]}),
        "fecha_descarga_primer_bloque": f"{min(fechas):%Y-%m-%d}",
        "fecha_descarga_ultimo_bloque": f"{max(fechas):%Y-%m-%d}",
        "rango_pedido_utc": [archivos[0].stem[-17:-9], archivos[-1].stem[-8:]],
        "zona_horaria_salida": ZONA_HORARIA,
        "bloques": [a.name for a in archivos],
        "horas_locales": int(len(serie_local)),
        "primera_hora_local": f"{serie_local.index.min():%Y-%m-%d %H:%M}",
        "ultima_hora_local": f"{serie_local.index.max():%Y-%m-%d %H:%M}",
        "parametros": {
            p: {"unidades": infos[-1]["unidades"][p], "faltantes": int(serie_local[p.lower()].isna().sum())}
            for p in parametros
        },
        "nota": "Promedios horarios tal como los entrega POWER, sin desfase aplicado.",
    }
    ruta = destino / ARCHIVO_MANIFIESTO
    ruta.write_text(json.dumps(manifiesto, indent=2, ensure_ascii=False))
    return manifiesto


def imprimir_resumen(manifiesto):
    """
        Imprime la cita de la fuente y la fila lista para docs/data_audit_logs.csv.
        La fila no se agrega sola: quien descargo llena su usuario y la version vacas/x.y.

        Args:
            manifiesto (dict): Metadatos de la corrida.
    """
    # Sin comas: la version va tambien en una fila de CSV
    version = " / ".join(manifiesto["versiones_api"]) or "desconocida"
    fecha = manifiesto["fecha_descarga_ultimo_bloque"].replace("-", "/")
    print("\nCitacion:")
    print(f"  The data was obtained from the POWER Project's Hourly {version} version on {fecha}")

    ahora = datetime.now()
    parametros = " ".join(manifiesto["parametros"])
    observaciones = (
        f"{manifiesto['horas_locales']} horas locales del {manifiesto['primera_hora_local']} "
        f"al {manifiesto['ultima_hora_local']}; variables {parametros}; "
        f"NASA POWER {version}; respuestas originales en Data/clima"
    )
    print("\nFila para docs/data_audit_logs.csv (agregala debajo del encabezado y llena Persona y version):")
    print(f"  {ahora:%d/%m/%Y %H:%M},,,Integración,"
          f"Descarga del clima por hora de NASA POWER para el analisis de las sesiones,"
          f"Completado,{observaciones}")


def main():
    inicio, fin = pd.Timestamp(INICIO_UTC), pd.Timestamp(FIN_UTC)

    archivos = descargar_bloques(inicio, fin, PARAMETROS, destino)

    bloques, infos = zip(*(leer_bloque(a, PARAMETROS) for a in archivos))
    serie_local = a_hora_local(pd.concat(bloques).sort_index()).rename(columns=str.lower)
    # Solo las horas locales del rango de las sesiones
    serie_local = serie_local.loc[INICIO_LOCAL:FIN_LOCAL]

    print("... guardando ...")
    serie_local.to_csv(destino / ARCHIVO_CSV)
    manifiesto = escribir_manifiesto(destino, archivos, PARAMETROS, list(infos), serie_local)
    print(f"Guardado! {len(serie_local)} horas locales en {destino / ARCHIVO_CSV}")
    imprimir_resumen(manifiesto)


if __name__ == "__main__":
    main()
