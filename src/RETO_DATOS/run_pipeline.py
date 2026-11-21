"""Ejecuta diez veces la preparación y resume sus tiempos.

Uso:
    python src/RETO_DATOS/run_pipeline.py
"""

from datetime import datetime
from pathlib import Path
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

import psutil
import requests


ROOT = Path(__file__).resolve().parents[2]
N_RUNS = 10
CLIMA = ROOT / "Data" / "clima"
SCRIPTS = [
    "src/scripts/generate_data.py",
    "src/scripts/anonymize_data.py",
    "src/scripts/descargar_clima.py",
    "src/DP/clean_data.py",
    "src/DP/integrar_clima.py",
    "src/DP/integrate_derivates.py",
    "src/scripts/generate_concentrado.py",
    "src/scripts/anonymize_concentrado.py",
    "src/DP/integrar_concentrado.py",
    "src/RETO_DATOS/get_data_sample.py",
    "src/RETO_DATOS/k_fold.py",
]


def mostrar_equipo():
    """Imprime los datos del equipo y del intérprete disponibles."""
    def consultar(funcion):
        """Obtiene un dato del equipo o indica que no está disponible.

        Args:
            funcion (callable): Función que consulta el dato.

        Returns:
            object: Valor consultado o texto de disponibilidad.
        """
        try:
            valor = funcion()
            return valor if valor else "No disponible"
        except (OSError, RuntimeError, ValueError, NotImplementedError):
            return "No disponible"

    print(f"Fecha: {datetime.now().astimezone().isoformat(timespec='seconds')}")
    print(f"Sistema: {platform.platform()} | Arquitectura: {platform.machine()}")
    print(f"Python: {platform.python_version()} | Intérprete: {sys.executable}")
    print(f"Procesador: {consultar(platform.processor)}")
    print(f"Núcleos físicos: {consultar(lambda: psutil.cpu_count(logical=False))}")
    print(f"Procesadores lógicos: {consultar(lambda: psutil.cpu_count(logical=True))}")
    print(f"RAM total (GiB): {consultar(lambda: round(psutil.virtual_memory().total / 1024**3, 2))}")


def medir_internet():
    """Mide e imprime una estimación de descarga hacia Cloudflare."""
    inicio = time.perf_counter()
    recibidos = 0
    try:
        with requests.get(
            "https://speed.cloudflare.com/__down",
            params={"bytes": 10000000, "nonce": time.time_ns()},
            headers={"Accept-Encoding": "identity", "Cache-Control": "no-cache"},
            stream=True, timeout=(5, 5),
        ) as respuesta:
            respuesta.raise_for_status()
            for bloque in respuesta.iter_content(chunk_size=65536):
                recibidos += len(bloque)
                if recibidos >= 10000000 or time.perf_counter() - inicio >= 10:
                    break
        segundos = time.perf_counter() - inicio
        if not recibidos:
            raise ValueError("La respuesta no contiene datos")
        print(f"Internet (Cloudflare): {recibidos * 8 / segundos / 1_000_000:.2f} Mbps | "
              f"{recibidos:,} bytes | {segundos:.2f} s", flush=True)
    except (requests.RequestException, OSError, ValueError) as error:
        print(f"Internet: No disponible ({error})", flush=True)


def ejecutar_script(ruta):
    """Ejecuta un script completo y mide su duración.

    Args:
        ruta (str): Ruta del script relativa a la raíz.

    Returns:
        tuple[int, float]: Código de salida y tiempo en segundos.
    """
    print(f"\nEjecutando {ruta}", flush=True)
    inicio = time.perf_counter()
    try:
        proceso = subprocess.run([sys.executable, "-u", str(ROOT / ruta)], cwd=ROOT)
        return proceso.returncode, time.perf_counter() - inicio
    finally:
        duracion = time.perf_counter() - inicio
        print(f"Tiempo de {Path(ruta).stem}: {duracion:.2f} s ({duracion / 60:.2f} min)", flush=True)


def ejecutar_clima(ruta):
    """Ejecuta la descarga con la caché de bloques apartada.

    Args:
        ruta (str): Ruta del script relativa a la raíz.

    Returns:
        tuple[int, float]: Código de salida y tiempo en segundos.
    """
    CLIMA.mkdir(parents=True, exist_ok=True)
    patron = re.compile(r"nasa_power_\d{8}_\d{8}\.json\Z")
    archivos = [archivo for archivo in CLIMA.iterdir() if patron.fullmatch(archivo.name)]
    respaldo = Path(tempfile.mkdtemp(prefix="respaldo_clima_", dir=CLIMA))
    try:
        movidos = []
        exito = False
        try:
            for archivo in archivos:
                shutil.move(str(archivo), str(respaldo / archivo.name))
                movidos.append(archivo.name)
            codigo, segundos = ejecutar_script(ruta)
            exito = codigo == 0
        finally:
            if not exito:
                for nombre in movidos:
                    os.replace(respaldo / nombre, CLIMA / nombre)
    except BaseException:
        if any(respaldo.iterdir()):
            print(f"Respaldo de clima conservado en: {respaldo}", flush=True)
        else:
            respaldo.rmdir()
        raise
    else:
        shutil.rmtree(respaldo)
    return codigo, segundos


def mostrar_resumen(tiempos, totales):
    """Imprime estadísticas de las ejecuciones terminadas correctamente.

    Args:
        tiempos (dict[str, list[float]]): Duraciones válidas por script.
        totales (list[float]): Suma de duraciones por vuelta completa.
    """
    print("\nResumen de tiempos (segundos)", flush=True)
    for nombre, valores in list(tiempos.items()) + [("Pipeline completo", totales)]:
        if not valores:
            print(f"{nombre}: sin ejecuciones completas")
            continue
        desviacion = f"{statistics.stdev(valores):.2f}" if len(valores) > 1 else "No disponible"
        print(f"{nombre}: n={len(valores)}/{N_RUNS} | promedio={statistics.mean(valores):.2f} | "
              f"desviación={desviacion} | mínimo={min(valores):.2f} | máximo={max(valores):.2f}")


def main():
    """Ejecuta el pipeline y devuelve su estado.

    Returns:
        int: Cero si todas las vueltas terminaron; otro valor si hubo error.
    """
    tiempos = {Path(ruta).stem: [] for ruta in SCRIPTS}
    totales = []
    mostrar_equipo()
    try:
        for vuelta in range(1, N_RUNS + 1):
            print(f"\n========== Vuelta {vuelta}/{N_RUNS} ==========", flush=True)
            medir_internet()
            total = 0.0
            for ruta in SCRIPTS:
                codigo, segundos = (ejecutar_clima(ruta) if Path(ruta).stem == "descargar_clima"
                                    else ejecutar_script(ruta))
                if codigo:
                    print(f"Falló {ruta} con código {codigo}; se detiene el pipeline.", flush=True)
                    return codigo if codigo > 0 else 1
                tiempos[Path(ruta).stem].append(segundos)
                total += segundos
            totales.append(total)
            print(f"Total de la vuelta {vuelta}: {total:.2f} s ({total / 60:.2f} min)", flush=True)
    except KeyboardInterrupt:
        print("\nEjecución interrumpida.", flush=True)
        return 130
    except OSError as error:
        print(f"\nError al ejecutar el pipeline: {error}", flush=True)
        return 1
    finally:
        mostrar_resumen(tiempos, totales)
    return 0


if __name__ == "__main__":
    sys.exit(main())
