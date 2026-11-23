"""
Cuánto aportan los eventos a la cola (fase de Evaluation, "Impactos contra el objetivo inicial").

Con los ordeños de test (mismo filtro y corte que los modelos):
1. Minutos extra de cada ordeño con evento: su duración menos la mediana de los ordeños sin evento de la
   misma vaca en test. Categorías excluyentes: solo falla mecánica, solo salud, solo conducta, 2 o más.
2. Minutos extra al día por categoría y su porcentaje del tiempo que las máquinas estuvieron ocupadas.
3. Espera estimada con Pollaczek-Khinchine (misma fórmula que el DU II), por máquina, tal como fue y como si
   los ordeños de cada categoría hubieran durado lo normal para la vaca. La diferencia es el aporte máximo
   de esa categoría a la espera.
4. De los minutos extra, qué porcentaje cae en ordeños que cada modelo alertó (salida alguna anomalía).
   Es lo más que podría atender el personal si actuara sobre cada alerta.

Lee Data/dataset_vacas_integrado.csv y las alertas de test con "git show <ref>:<ruta>". Solo escribe en
src/EVALUATION/metrics/.

Uso, desde la raíz del repo:
    python -B src/EVALUATION/duracion_por_evento.py
    python -B src/EVALUATION/duracion_por_evento.py --lstm-ref origin/develop
"""

import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "Data" / "dataset_vacas_integrado.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "metrics"

sys.path.insert(0, str(ROOT / "src" / "MODELING" / "XGBoost"))
from dataset import split_by_time  # noqa: E402  Mismo corte que los modelos

TRAIN_FRACTION, VAL_FRACTION = 0.70, 0.15
EVENTS = {
    "mechanical_connection_failure_event": "Falla mecánica",
    "health_anomaly_event": "Salud",
    "behavior_interruption_event": "Conducta",
}
MACHINES = (
    "maquina_86696b37",
    "maquina_b96e85be",
)  # Si las dos valen 0, es la máquina base
BASE_MACHINE = "7063e4b1"
ALERTS = {
    "Regresión logística": "src/MODELING/logistic/metrics/v4/alertas_test.csv.gz",
    "XGBoost": "src/MODELING/XGBoost/models/v5/alertas_test.csv.gz",
    "LSTM": "src/MODELING/LSTM/metrics/v5/alertas_test.csv.gz",
}


def load_test():
    """Ordeños de test con duración en minutos, máquina y categoría de evento."""
    columns = [
        "id_vaca",
        "inicio",
        "duracion_seg",
        "accion_rechazada",
        "accion_echada_sin_ordenar",
        *EVENTS,
        *MACHINES,
    ]
    sessions = pd.read_csv(
        INPUT_PATH, usecols=columns, dtype={"id_vaca": str}, parse_dates=["inicio"]
    )
    milking = (
        sessions[["accion_rechazada", "accion_echada_sin_ordenar"]].eq(0).all(axis=1)
    )
    sessions = (
        sessions[milking]
        .sort_values(["id_vaca", "inicio"], kind="stable")
        .reset_index(drop=True)
    )
    test = sessions.iloc[
        split_by_time(sessions["inicio"], TRAIN_FRACTION, VAL_FRACTION)["test"]
    ].copy()

    test["minutos"] = test["duracion_seg"] / 60
    test["maquina"] = BASE_MACHINE
    for column in MACHINES:
        test.loc[test[column] == 1, "maquina"] = column.removeprefix("maquina_")
    events = test[list(EVENTS)]
    count = events.sum(axis=1)
    test["categoria"] = np.select(
        [count == 0, count >= 2] + [(count == 1) & (events[e] == 1) for e in EVENTS],
        ["Sin evento", "Dos o más"]
        + [f"Solo {name.lower()}" for name in EVENTS.values()],
        default="",
    )
    test["inicio_texto"] = test["inicio"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return test


def add_extra_minutes(test):
    """Minutos extra: duración menos la mediana de los ordeños sin evento de la misma vaca."""
    normal = (
        test[test["categoria"] == "Sin evento"].groupby("id_vaca")["minutos"].median()
    )
    test["normal_vaca"] = (
        test["id_vaca"]
        .map(normal)
        .fillna(test.loc[test["categoria"] == "Sin evento", "minutos"].median())
    )
    test["extra"] = test["minutos"] - test["normal_vaca"]
    return test


def waiting_minutes(minutes, days):
    """Espera estimada de una máquina con Pollaczek-Khinchine: rho·E[S]·(1 + CV²) / (2·(1 − rho))."""
    rho = minutes.sum() / (days * 24 * 60)
    mean = minutes.mean()
    cv = minutes.std() / mean
    return rho, rho * mean * (1 + cv**2) / (2 * (1 - rho))


def alerted_keys(ref, path):
    """Parejas (id_vaca, inicio) con alerta de alguna anomalía en el archivo de un modelo."""
    raw = subprocess.run(
        ["git", "show", f"{ref}:{path}"], cwd=ROOT, capture_output=True, check=True
    ).stdout
    alerts = pd.read_csv(
        io.BytesIO(raw), compression="gzip", dtype={"id_vaca": str, "inicio": str}
    )
    alerts = alerts[alerts["evento"].str.contains("alguna", case=False)]
    return set(zip(alerts["id_vaca"], alerts["inicio"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr-ref", default="origin/develop")
    parser.add_argument("--xgb-ref", default="origin/develop")
    parser.add_argument("--lstm-ref", default="update/lstm_model")
    args = parser.parse_args()
    refs = dict(zip(ALERTS, (args.lr_ref, args.xgb_ref, args.lstm_ref)))

    test = add_extra_minutes(load_test())
    days = (test["inicio"].max() - test["inicio"].min()).total_seconds() / 86400
    busy = test["minutos"].sum()
    keys = list(zip(test["id_vaca"], test["inicio_texto"]))
    alerted = {}
    for model, path in ALERTS.items():
        model_keys = alerted_keys(refs[model], path)  # Se lee una sola vez por modelo
        alerted[model] = np.array([k in model_keys for k in keys])
        print(
            f"[alertas] {model}: {alerted[model].sum():,} ordeños de test con alerta de alguna anomalía"
        )

    categories = ["Solo falla mecánica", "Solo salud", "Solo conducta", "Dos o más"]
    rows = []
    for category in categories + ["Cualquier evento"]:
        mask = (
            (test["categoria"] != "Sin evento")
            if category == "Cualquier evento"
            else (test["categoria"] == category)
        )
        group = test[mask]
        positive = group["extra"].clip(lower=0)
        row = {
            "categoria": category,
            "ordenos_por_dia": len(group) / days,
            "minutos_extra_mediana": group["extra"].median(),
            "minutos_extra_media": group["extra"].mean(),
            "minutos_extra_por_dia": group["extra"].sum() / days,
            "pct_tiempo_ocupado": 100 * group["extra"].sum() / busy,
        }
        for model in ALERTS:
            row[f"pct_minutos_extra_con_alerta_{model}"] = (
                100 * positive[alerted[model][mask.to_numpy()]].sum() / positive.sum()
                if positive.sum()
                else np.nan
            )
        rows.append(row)
    summary = pd.DataFrame(rows)

    # Espera por máquina: tal como fue y quitando los minutos extra de cada categoría.
    waits = []
    for machine, group in test.groupby("maquina"):
        rho, wait = waiting_minutes(group["minutos"], days)
        entry = {
            "maquina": machine,
            "ordenos_por_dia": len(group) / days,
            "ocupacion": rho,
            "duracion_media_min": group["minutos"].mean(),
            "espera_min": wait,
        }
        for category in categories + ["Cualquier evento"]:
            mask = (
                (group["categoria"] != "Sin evento")
                if category == "Cualquier evento"
                else (group["categoria"] == category)
            )
            adjusted = group["minutos"].where(
                ~mask, np.minimum(group["minutos"], group["normal_vaca"])
            )
            _, wait_without = waiting_minutes(adjusted, days)
            entry[f"espera_sin_{category}"] = wait_without
            entry[f"ahorro_{category}"] = wait - wait_without
        waits.append(entry)
    waits = pd.DataFrame(waits)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        OUTPUT_DIR / "duracion_por_evento.csv", index=False, encoding="utf-8-sig"
    )
    waits.to_csv(
        OUTPUT_DIR / "espera_por_maquina.csv", index=False, encoding="utf-8-sig"
    )
    record = {
        "periodo_test": [str(test["inicio"].min()), str(test["inicio"].max())],
        "dias": days,
        "ordenos_test": len(test),
        "ordenos_por_categoria": test["categoria"].value_counts().to_dict(),
        "minutos_ocupados_por_dia": busy / days,
        "refs_alertas": refs,
        "resumen": summary.round(3).to_dict(orient="records"),
        "espera_por_maquina": waits.round(3).to_dict(orient="records"),
    }
    with open(OUTPUT_DIR / "duracion_por_evento.json", "w", encoding="utf-8") as file:
        json.dump(record, file, indent=2, ensure_ascii=False, default=float)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(
        f"[test] {len(test):,} ordeños en {days:.1f} días; {busy / days:,.0f} min ocupados al día (3 máquinas)"
    )
    print("\n[minutos extra por categoría]")
    print(summary.round(2).to_string(index=False))
    print("\n[espera estimada por máquina, en minutos]")
    print(
        waits[
            ["maquina", "ocupacion", "espera_min"]
            + [c for c in waits if c.startswith("ahorro_")]
        ]
        .round(2)
        .to_string(index=False)
    )
    print(f"\nResultados en {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
