"""
Servidor de la interfaz para probar los modelos: sirve la página de static/ (index.html,
styles.css y app.js) y una API JSON.
Con el dataset integrado ya cargado, se elige una vaca y se captura una sesión nueva (o se elige
una del dataset), y los tres modelos dan la probabilidad de cada evento y las 3 variables que
más influyeron.

Uso, desde la raíz del repo y después de preparar los datos y build_history_features.py:
    python src/INTERFAZ/app.py
y abrir http://localhost:8000
"""

import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import nueva_sesion as ns

STATIC_DIR = Path(__file__).resolve().parent / "static"
PORT = 8000
state = {}
lock = threading.Lock()  # Una predicción a la vez: TensorFlow y los DataFrames se comparten


@asynccontextmanager
async def lifespan(app):
    print("[app] Cargando el dataset integrado...", flush=True)
    visits = ns.load_visits()
    import modelos  # TensorFlow tarda en importar

    print("[app] Cargando los modelos...", flush=True)
    state.update(visits=visits, herd=visits[ns.is_milking(visits)], modelos=modelos, models=modelos.load_models())
    print(f"[app] Lista en http://localhost:{PORT}", flush=True)
    yield


app = FastAPI(title="Predicción de eventos de ordeño", lifespan=lifespan)
# Permite abrir index.html desde otro lado de esta computadora (doble clic, cuyo origen es "null",
# o Live Server) y llamar a esta API; páginas de otros sitios no pueden leerla.
app.add_middleware(CORSMiddleware, allow_origin_regex=r"(null|https?://(localhost|127\.0\.0\.1)(:\d+)?)",
                   allow_methods=["*"], allow_headers=["*"])


class Capture(BaseModel):
    modo: Literal["nueva", "existente"]
    vaca: str
    inicio: str  # "AAAA-MM-DD HH:MM[:SS]"
    maquina: Optional[str] = None
    temperatura: Optional[float] = None
    humedad: Optional[float] = None
    horas_thi_alto_24h: Optional[float] = None
    numero_ordeno: Optional[int] = None
    modelos: Optional[list] = None


def plain(value):
    """Valor que se puede mandar en JSON (sin tipos de numpy ni NaN)."""
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return value


@app.get("/api/info")
def info():
    herd, modelos = state["herd"], state["modelos"]
    last = herd.groupby("id_vaca")["inicio"].max().sort_values(ascending=False)
    return {
        "vacas": [{"id": cow, "ultimo": plain(start)} for cow, start in last.items()],
        "maquinas": ns.machines(state["visits"]),
        "eventos": [{"id": event, "nombre": modelos.EVENT_NAMES[event]} for event in modelos.EVENTS],
        "modelos": [{"nombre": m.name, "desempeno": {e: {k: plain(v) for k, v in p.items()} for e, p in m.performance.items()}}
                    for m in state["models"]],
        "linea_base": modelos.baseline(),
        "fecha_maxima": f"{ns.last_date(state['visits']):%Y-%m-%d}",
    }


@app.get("/api/vacas/{cow}")
def cow_defaults(cow: str):
    milkings = state["herd"][state["herd"]["id_vaca"] == cow]
    if milkings.empty:
        raise HTTPException(404, "Vaca no encontrada")
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
def predict(capture: Capture):
    visits, herd, modelos = state["visits"], state["herd"], state["modelos"]
    start = pd.Timestamp(capture.inicio)
    actual, notes = None, []
    with lock:
        if capture.modo == "nueva":
            fields = [capture.maquina, capture.temperatura, capture.humedad,
                      capture.horas_thi_alto_24h, capture.numero_ordeno]
            if any(value is None for value in fields):
                raise HTTPException(422, "Faltan datos de la captura.")
            try:
                sessions = ns.build_session(visits, capture.vaca, {
                    "inicio": start, "maquina": capture.maquina, "t_sesion": capture.temperatura,
                    "thi_sesion": ns.thi(capture.temperatura, capture.humedad),
                    "horas_thi_alto_24h": capture.horas_thi_alto_24h,
                    "estado_numero_de_ordeno": capture.numero_ordeno,
                })
            except ValueError as error:
                raise HTTPException(422, str(error))
            notes = ns.notices(sessions)
        else:
            sessions = ns.existing_session(visits, capture.vaca, start)
            if sessions.empty or sessions["inicio"].iloc[-1] != start:
                raise HTTPException(404, "No existe ese ordeño de la vaca.")
            row = sessions.iloc[-1]
            actual = {event: bool(row[event]) for event in modelos.EVENTS[:3]}
            actual["alguna_anomalia"] = not bool(row["successful_milking_event"])

        chosen = capture.modelos or [m.name for m in state["models"]]
        results = {m.name: m.predict(sessions, herd) for m in state["models"] if m.name in chosen}

    row = sessions.iloc[-1]
    columns = list(dict.fromkeys(modelos.xgb_dataset.FEATURES + modelos.lstm_dataset.CONTEXT_FEATURES))
    return {
        "vaca": capture.vaca, "inicio": plain(row["inicio"]), "real": actual, "avisos": notes,
        "resultados": {name: {event: {**r, "factores": [{k: plain(v) for k, v in f.items()} for f in r["factores"]]}
                              for event, r in res.items()} for name, res in results.items()},
        "variables": {column: plain(row.get(column)) for column in columns},
    }


# Al final, para que /api tenga prioridad: / sirve index.html y sus archivos (styles.css, app.js).
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT)
