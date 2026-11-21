"""
Mide el tiempo y la memoria de cada etapa del pipeline, desde la extracción hasta el entrenamiento de los
tres modelos, para justificar con datos la decisión sobre Big Data (reto de datos).

Cada etapa corre en un proceso aparte (etapas.py), así que la memoria de una no se suma a la siguiente.
Las salidas intermedias van a una carpeta temporal que se borra al terminar: nunca se escribe en Data/
ni se sobrescriben modelos o métricas. El tiempo es el tiempo real del proceso, incluida la carga de
librerías; la memoria es la residente máxima del proceso (no disponible en Windows).

Requiere los datos originales en Data/vacas/, el archivo .env con COWSKEY y el clima descargado en
Data/clima/. La etapa de la LSTM requiere TensorFlow.

Guarda en src/RETO_DATOS/Rendimiento/resultados/:
    rendimiento.csv   una fila por etapa: tiempo (s) y memoria máxima (MB)
    equipo.json       procesador, núcleos, RAM, sistema operativo y versiones de las librerías

Uso, desde la raíz del repo:
    python src/RETO_DATOS/Rendimiento/medir_rendimiento.py
    python src/RETO_DATOS/Rendimiento/medir_rendimiento.py --sin-lstm
"""

import argparse
import csv
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

CARPETA = Path(__file__).resolve().parent
RESULTADOS = CARPETA / "resultados"
sys.path.insert(0, str(CARPETA))
from etapas import ETAPAS  # noqa: E402

DESCRIPCION = {
    "extraccion": "Extracción y carga de 235 archivos CSV y XLSX (generate_data.py)",
    "anonimizacion": "Anonimización (anonymize_data.py)",
    "limpieza": "Limpieza de ordeños y clima (clean_data.py)",
    "clima": "Integración climática (integrar_clima.py)",
    "derivadas": "Variables derivadas y etiquetas (integrate_derivates.py)",
    "historial": "Variables de historial (build_history_features.py)",
    "regresion_logistica": "Entrenamiento y evaluación de la regresión logística, 4 salidas (logistic/main.py)",
    "xgboost": "Entrenamiento y calibración de XGBoost v5, 4 salidas",
    "lstm": "Entrenamiento y calibración de la LSTM v5, 4 salidas",
}


def procesador():
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return platform.processor() or platform.machine()


def ram_gb():
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3, 1)
    except (AttributeError, ValueError, OSError):
        return None


def versiones():
    salida = {"python": platform.python_version()}
    for paquete in ("pandas", "numpy", "scikit-learn", "xgboost", "tensorflow"):
        try:
            salida[paquete] = version(paquete)
        except PackageNotFoundError:
            salida[paquete] = None
    return salida


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sin-lstm", action="store_true", help="no mide la LSTM (no requiere TensorFlow)")
    etapas = [e for e in ETAPAS if not (parser.parse_args().sin_lstm and e == "lstm")]

    temporal = Path(tempfile.mkdtemp(prefix="rendimiento_vacas_"))
    filas = []
    try:
        for etapa in etapas:
            print(f"[rendimiento] {etapa}...", flush=True)
            inicio = time.perf_counter()
            proceso = subprocess.run([sys.executable, "-W", "ignore", str(CARPETA / "etapas.py"), str(temporal), etapa],
                                     capture_output=True, text=True)
            segundos = time.perf_counter() - inicio
            if proceso.returncode != 0:
                print(proceso.stderr[-2000:], file=sys.stderr)
                raise SystemExit(f"La etapa {etapa} falló; las siguientes dependen de ella.")
            memoria = json.loads(proceso.stdout.strip().splitlines()[-1])["memoria_maxima_mb"]
            filas.append({"etapa": etapa, "descripcion": DESCRIPCION[etapa], "tiempo_s": round(segundos, 1),
                          "memoria_maxima_mb": None if memoria is None else round(memoria)})
            print(f"  {segundos:,.1f} s | memoria máxima {memoria:,.0f} MB" if memoria else f"  {segundos:,.1f} s", flush=True)
    finally:
        shutil.rmtree(temporal, ignore_errors=True)  # incluye copias con identificadores reales

    RESULTADOS.mkdir(exist_ok=True)
    with open(RESULTADOS / "rendimiento.csv", "w", newline="", encoding="utf-8") as archivo:
        writer = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        writer.writeheader()
        writer.writerows(filas)
    equipo = {"procesador": procesador(), "nucleos": os.cpu_count(), "ram_gb": ram_gb(),
              "sistema": f"{platform.system()} {platform.release()}", "fecha": time.strftime("%Y-%m-%d"),
              "versiones": versiones()}
    with open(RESULTADOS / "equipo.json", "w", encoding="utf-8") as archivo:
        json.dump(equipo, archivo, indent=2, ensure_ascii=False)
    datos = [f for f in filas if f["etapa"] not in ("regresion_logistica", "xgboost", "lstm")]
    print(f"\nPipeline de datos: {sum(f['tiempo_s'] for f in datos):,.1f} s en total")
    print(f"Resultados en {(RESULTADOS / 'rendimiento.csv').relative_to(CARPETA.parents[2])}")


if __name__ == "__main__":
    main()
