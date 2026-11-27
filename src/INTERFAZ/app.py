"""
Servidor Flask de la interfaz para probar el modelo final (XGBoost): sirve la página de static/
(index.html, styles.css, script.js e íconos) y una API JSON.
Con el dataset integrado ya cargado, se elige una vaca y se captura una sesión nueva (o se elige
una del dataset), y el XGBoost da la probabilidad de cada evento y la variable que
más aumenta el riesgo, traducida a lenguaje no técnico con qué revisar (traduccion.py). Si existe
Data/mapa_id_vacas.csv, las vacas y las máquinas se muestran con su nombre real (VMS 1, VMS 2, VMS 3).

Uso, desde la raíz del repo y después de preparar los datos y src/MODELING/XGBoost/build_history_features.py:
    python src/INTERFAZ/app.py
y abrir http://localhost:8000
"""

import threading

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

import modelos
import nueva_sesion as ns
import traduccion

PORT = 8000
CAPTURE_FIELDS = {"temperatura": float, "humedad": float, "horas_thi_alto_24h": float, "numero_ordeno": int}
state = {}
lock = threading.Lock()  # Una predicción a la vez: los DataFrames se comparten

app = Flask(__name__, static_folder="static", static_url_path="")
app.json.sort_keys = False  # La página usa el orden de eventos, métricas y variables


class ApiError(Exception):
    """Error que la página muestra tal cual: se responde como {"detail": mensaje}."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


@app.errorhandler(ApiError)
def api_error(error):
    return jsonify(detail=str(error)), error.status


def load():
    print("[app] Cargando el dataset integrado...", flush=True)
    visits = ns.load_visits()

    labels = traduccion.load_labels(visits)
    if labels["vacas"]:
        print(f"[app] Nombres reales de {len(labels['vacas'])} vacas y {len(labels['maquinas'])} máquinas "
              f"({traduccion.MAP_PATH.name})", flush=True)
    else:
        print(f"[app] Sin {traduccion.MAP_PATH.name}: se muestran los identificadores anonimizados", flush=True)
    print("[app] Cargando el modelo...", flush=True)
    state.update(visits=visits, herd=visits[ns.is_milking(visits)], model=modelos.load_model(), labels=labels)


def plain(value):
    """Valor que se puede mandar en JSON (sin tipos de numpy ni NaN)."""
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return value


def read_capture():
    """
    El JSON de /api/predecir validado: modo, vaca e inicio siempre. En una sesión nueva agrega
    session, la captura con los nombres de columna que espera ns.build_session.
    """
    data = request.get_json(silent=True) or {}
    if data.get("modo") not in ("nueva", "existente") or not data.get("vaca") or not data.get("inicio"):
        raise ApiError(422, "Faltan el modo, la vaca o la fecha.")
    try:
        data["inicio"] = pd.Timestamp(data["inicio"])
    except ValueError:
        raise ApiError(422, "La fecha no es válida.")
    if data["modo"] == "nueva":
        if any(data.get(field) is None for field in ["maquina", *CAPTURE_FIELDS]):
            raise ApiError(422, "Faltan datos de la captura.")
        try:
            values = {field: kind(data[field]) for field, kind in CAPTURE_FIELDS.items()}
        except (TypeError, ValueError):
            raise ApiError(422, "La temperatura, la humedad, las horas y el número de ordeño deben ser números.")
        data["session"] = {
            "inicio": data["inicio"], "maquina": data["maquina"], "t_sesion": values["temperatura"],
            "thi_sesion": ns.thi(values["temperatura"], values["humedad"]),
            "horas_thi_alto_24h": values["horas_thi_alto_24h"], "estado_numero_de_ordeno": values["numero_ordeno"],
        }
    return data


@app.get("/")
def index():
    return app.send_static_file("index.html")


@app.get("/api/info")
def info():
    herd, model = state["herd"], state["model"]
    last = herd.groupby("id_vaca")["inicio"].max().sort_values(ascending=False)
    return {
        "vacas": [{"id": cow, "ultimo": plain(start)} for cow, start in last.items()],
        "maquinas": ns.machines(state["visits"]),
        "eventos": [{"id": event, "nombre": modelos.EVENT_NAMES[event]} for event in modelos.EVENTS],
        "modelo": {"nombre": model.name, "desempeno": model.performance},
        "linea_base": modelos.baseline(),
        "fecha_maxima": f"{ns.last_date(state['visits']):%Y-%m-%d}",
        "etiquetas": state["labels"],
        "glosario": traduccion.glossary(),
    }


@app.get("/api/vacas/<cow>")
def cow_defaults(cow):
    milkings = state["herd"][state["herd"]["id_vaca"] == cow]
    if milkings.empty:
        raise ApiError(404, "Vaca no encontrada")
    last = milkings.iloc[-1]
    interval = last["intervalo_horas"] if pd.notna(last["intervalo_horas"]) and last["intervalo_horas"] > 0 else 10
    return {
        "ultimo": plain(last["inicio"]),
        "sugerido": plain((last["inicio"] + pd.Timedelta(hours=float(interval))).floor("min")),
        "maquina": ns.machine_of(last),
        "temperatura": round(float(last["t_sesion"]), 1),
        "humedad": round(ns.humidity_from_thi(float(last["t_sesion"]), float(last["thi_sesion"]))),
        "horas_thi_alto_24h": plain(last["horas_thi_alto_24h"]) or 0,
        "numero_ordeno": int(last["estado_numero_de_ordeno"]) or 1,
        "ordenos": [plain(start) for start in milkings["inicio"].iloc[::-1]],
    }


@app.post("/api/predecir")
def predict():
    capture = read_capture()
    visits = state["visits"]
    start = capture["inicio"]
    actual, notes = None, []
    with lock:
        if capture["modo"] == "nueva":
            try:
                sessions = ns.build_session(visits, capture["vaca"], capture["session"])
            except ValueError as error:
                raise ApiError(422, str(error))
            notes = ns.notices(sessions)
        else:
            sessions = ns.existing_session(visits, capture["vaca"], start)
            if sessions.empty or sessions["inicio"].iloc[-1] != start:
                raise ApiError(404, "No existe ese ordeño de la vaca.")
            row = sessions.iloc[-1]
            actual = {event: bool(row[event]) for event in modelos.EVENTS[:3]}
            actual["alguna_anomalia"] = not bool(row["successful_milking_event"])

        results = state["model"].predict(sessions)

    row = sessions.iloc[-1]
    return {
        "vaca": capture["vaca"], "inicio": plain(row["inicio"]), "real": actual, "avisos": notes,
        "resultados": traduccion.describe_results(results, state["labels"]),
        "variables": traduccion.session_variables(row, modelos.xgb_dataset.FEATURES, ns.machine_of(row), state["labels"]),
    }


if __name__ == "__main__":
    load()
    print(f"[app] Lista en http://localhost:{PORT}", flush=True)
    app.run(host="127.0.0.1", port=PORT, threaded=True)
