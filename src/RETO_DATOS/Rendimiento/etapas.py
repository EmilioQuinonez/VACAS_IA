"""
Corre una sola etapa del pipeline con sus salidas en una carpeta temporal, para medir su rendimiento.

Cada etapa llama a las mismas funciones del repositorio que se usan en el flujo normal, pero cambia las
rutas de salida a `salida`, así que nunca escribe en Data/ ni sobrescribe modelos o métricas guardados.
Las etapas se encadenan: cada una lee lo que dejó la anterior en `salida`. Al terminar imprime una línea
JSON con la memoria máxima del proceso, que lee medir_rendimiento.py.

Uso (normalmente lo llama medir_rendimiento.py):
    python src/RETO_DATOS/Rendimiento/etapas.py <carpeta_salida> <etapa>
"""

import contextlib
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "Data"

ETAPAS = [
    "extraccion", "anonimizacion", "limpieza", "clima", "derivadas", "historial",
    "regresion_logistica", "xgboost", "lstm",
]


def memoria_maxima_mb():
    """Memoria residente máxima del proceso en MB (None en Windows, que no tiene el módulo resource)."""
    try:
        import resource
    except ImportError:
        return None
    maxima = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maxima / 1024 ** 2 if sys.platform == "darwin" else maxima / 1024  # macOS: bytes; Linux: KB


def extraccion(salida):
    """generate_data.py: une los CSV y XLSX de cada vaca en un solo dataset."""
    sys.path.insert(0, str(ROOT / "src" / "scripts"))
    import generate_data as m
    m.CARPETA, m.SALIDA = DATA / "vacas", salida / "dataset_vacas.csv"
    m.main()


def anonimizacion(salida):
    """anonymize_data.py: corre a nivel de módulo, así que se ejecuta su código con las rutas cambiadas."""
    ruta = ROOT / "src" / "scripts" / "anonymize_data.py"
    codigo = ruta.read_text(encoding="utf-8")
    for archivo in ("dataset_vacas.csv", "dataset_vacas_anonimo.csv", "mapa_id_vacas.csv"):
        original = f'raiz / "Data" / "{archivo}"'
        if original not in codigo:
            raise SystemExit(f"anonymize_data.py cambió: no se encontró {original}; no se corre para no escribir en Data/")
        codigo = codigo.replace(original, f"Path(r'{salida / archivo}')")
    exec(compile(codigo, str(ruta), "exec"), {"__name__": "__main__", "__file__": str(ruta)})


def limpieza(salida):
    """clean_data.py: corre su bloque principal con las rutas de salida cambiadas."""
    sys.path.insert(0, str(ROOT / "src" / "DP"))
    import clean_data as m
    (salida / "clima").mkdir(exist_ok=True)
    m.RAIZ = salida
    m.RUTA_VACAS = salida / "dataset_vacas_anonimo.csv"
    m.SALIDA_VACAS = salida / "dataset_vacas_limpio.csv"
    m.SALIDA_CLIMA = salida / "clima" / "clima_nasa_power_limpio.csv"
    principal = (ROOT / "src" / "DP" / "clean_data.py").read_text(encoding="utf-8").split('if __name__ == "__main__":')[1]
    exec(re.sub(r"^    ", "", principal, flags=re.M), vars(m))


def clima(salida):
    """integrar_clima.py: temperatura, THI y horas con THI alto de cada sesión."""
    sys.path.insert(0, str(ROOT / "src" / "DP"))
    import integrar_clima as m
    m.raiz = salida
    m.ORIGEN = salida / "dataset_vacas_limpio.csv"
    m.CLIMA = salida / "clima" / "clima_nasa_power_limpio.csv"
    m.SALIDA = salida / "dataset_vacas_integrado.csv"
    m.main()


def derivadas(salida):
    """integrate_derivates.py: atributos derivados y etiquetas de evento."""
    sys.path.insert(0, str(ROOT / "src" / "DP"))
    import integrate_derivates as m
    m.ROOT = salida
    m.INPUT_PATH = m.OUTPUT_PATH = salida / "dataset_vacas_integrado.csv"
    m.main()


def historial(salida):
    """build_history_features.py de XGBoost: incluye todas las variables de historial de los tres modelos."""
    sys.path.insert(0, str(ROOT / "src" / "MODELING" / "XGBoost"))
    import build_history_features as m
    m.ROOT = salida
    m.INPUT_PATH = m.OUTPUT_PATH = salida / "dataset_vacas_integrado.csv"
    m.main()


def regresion_logistica(salida):
    """logistic/main.py (v4): entrena las 4 salidas y las evalúa; sus modelos y gráficas van a `salida`."""
    sys.path.insert(0, str(ROOT))
    from src.MODELING.logistic import main as m
    m.run(input_path=salida / "dataset_vacas_integrado.csv", output_dir=salida / "logistic")


def xgboost(salida):
    """XGBoost v5: entrena, calibra y fija el umbral de las 4 salidas, como main.py, sin guardar modelos."""
    sys.path.insert(0, str(ROOT / "src" / "MODELING" / "XGBoost"))
    from dataset import EVENTS, build_features, load_sessions, split_by_time
    from evaluation import choose_threshold
    from versions import VERSIONS
    from xgboost_model import calibrate, train_event
    sessions = load_sessions(salida / "dataset_vacas_integrado.csv")
    X, labels = build_features(sessions)
    idx = split_by_time(sessions["inicio"], 0.70, 0.15)
    for event in EVENTS:
        model = train_event(X, labels[event], idx["train"], idx["val"], params=VERSIONS["v5"], random_state=42)
        calibrated = calibrate(model, X, labels[event], idx["val"])
        choose_threshold(calibrated, X, labels[event], idx["val"], 0.80)


def lstm(salida):
    """LSTM v5: arma las secuencias, entrena con early stopping y calibra, como main.py, sin guardar el modelo.

    No incluye las explicaciones por alerta ni las gráficas, que en main.py van después del entrenamiento.
    """
    sys.path.insert(0, str(ROOT / "src" / "MODELING" / "LSTM"))
    import main as m
    from dataset import build_inputs, load_sessions, split_by_time
    from evaluation import fit_calibration
    sessions = load_sessions(salida / "dataset_vacas_integrado.csv")
    idx = split_by_time(sessions["inicio"], m.TRAIN_FRACTION, m.VAL_FRACTION)
    X_seq, X_ctx, y, _ = build_inputs(sessions, idx["train"], m.CONFIG["k"])
    model, history = m.train(m.CONFIG, X_seq, X_ctx, y, idx)
    rows = idx["val"]
    fit_calibration(y[rows], model.predict([X_seq[rows], X_ctx[rows]], batch_size=m.CONFIG["batch_size"], verbose=0))
    print(f"[lstm] épocas ejecutadas: {len(history['loss'])}", file=sys.stderr)


def main():
    salida, etapa = Path(sys.argv[1]), sys.argv[2]
    if etapa not in ETAPAS:
        raise SystemExit(f"Etapa desconocida: {etapa}. Opciones: {', '.join(ETAPAS)}")
    salida.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):  # oculta lo que imprime cada script
        globals()[etapa](salida)
    print(json.dumps({"etapa": etapa, "memoria_maxima_mb": memoria_maxima_mb()}))


if __name__ == "__main__":
    main()
