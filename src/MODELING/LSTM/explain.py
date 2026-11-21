"""
Tres factores por alerta con Integrated Gradients: las 3 variables que más subieron el log-odds y,
si son de la secuencia, de qué sesión previa vino su mayor aporte. La referencia es la vaca con
valores promedio de train (0 ya escalado). Son asociaciones, no causas.
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from dataset import CONTEXT_FEATURES, SEQUENCE_FEATURES

STEPS = 32  # Puntos entre la referencia y la sesión real
TOP_N = 3
NAMES = SEQUENCE_FEATURES + CONTEXT_FEATURES
MDI_FLAGS = ("mdi_faltante_previo", "estado_mdi_sin_dato_previo")
CYCLES = ("mes", "hora", "dia_semana")


def feature_group(name):
    """Une en un factor las columnas de una misma variable: dummies, seno y coseno, y banderas del MDI."""
    if name.startswith("maquina_"):
        return "maquina"
    if name in MDI_FLAGS:
        return "mdi_previo"
    for cycle in CYCLES:
        if name in (f"{cycle}_sin", f"{cycle}_cos"):
            return cycle
    return name


GROUPS = {}
for position, feature in enumerate(NAMES):
    GROUPS.setdefault(feature_group(feature), []).append(position)
GROUP_NAMES = list(GROUPS)


def integrated_gradients(model, seq, ctx, output, batch_size):
    """Aporte de cada entrada al log-odds: gradiente promedio entre la referencia y la entrada, por la entrada."""
    seq_gradients, ctx_gradients = np.zeros_like(seq), np.zeros_like(ctx)
    for alpha in (np.arange(STEPS) + 0.5) / STEPS:
        for start in range(0, len(seq), batch_size):
            rows = slice(start, start + batch_size)
            inputs = [tf.constant(alpha * seq[rows]), tf.constant(alpha * ctx[rows])]
            with tf.GradientTape() as tape:
                tape.watch(inputs)
                probability = tf.clip_by_value(model(inputs, training=False)[:, output], 1e-7, 1 - 1e-7)
                log_odds = tf.math.log(probability) - tf.math.log(1 - probability)
            gradient_seq, gradient_ctx = tape.gradient(log_odds, inputs)
            seq_gradients[rows] += gradient_seq.numpy()
            ctx_gradients[rows] += gradient_ctx.numpy()

    return seq * seq_gradients / STEPS, ctx * ctx_gradients / STEPS


def factor_value(sessions, name, row):
    """Valor de la variable agrupada en la fila row, como texto."""
    if name == "maquina":
        machines = [column for column in NAMES if column.startswith("maquina_")]
        active = [column.removeprefix("maquina_") for column in machines if sessions[column].iat[row] == 1]
        return active[0] if active else "referencia"
    if name == "mdi_previo" and any(sessions[flag].iat[row] == 1 for flag in MDI_FLAGS):
        return "faltante"
    value = sessions[name].iat[row]
    if name == "box_time_to_yield_ratio":
        value = np.expm1(value)  # Viene en logaritmo
    return "faltante" if pd.isna(value) else f"{value:.4g}"


def explain(model, X_seq, X_ctx, rows, sessions, output, batch_size, scale):
    """
    Los 3 factores de cada sesión de rows para la salida output: los mayores aportes positivos por
    variable agrupada; si hay menos de 3, los demás quedan vacíos. scale es la pendiente a de la
    calibración. Devuelve los factores y el aporte de cada columna, sin agrupar.
    """
    seq_contribution, ctx_contribution = integrated_gradients(model, X_seq[rows], X_ctx[rows], output, batch_size)
    seq_contribution, ctx_contribution = seq_contribution * scale, ctx_contribution * scale
    # La secuencia suma sus k pasos.
    contribution = np.concatenate([seq_contribution.sum(axis=1), ctx_contribution], axis=1)
    grouped = np.column_stack([contribution[:, GROUPS[name]].sum(axis=1) for name in GROUP_NAMES])
    k = X_seq.shape[1]

    factors = []
    for i, row in enumerate(rows):
        factor = {}
        top = np.argsort(-grouped[i], kind="stable")[:TOP_N]
        for rank, group in enumerate(top, start=1):
            name, amount = GROUP_NAMES[group], grouped[i, group]
            if amount <= 0:  # Solo cuentan las variables que suben el riesgo
                name = session = value_text = ""
                amount = np.nan
            elif GROUPS[name][0] < len(SEQUENCE_FEATURES):
                # El último paso describe la sesión anterior (t-1).
                by_step = seq_contribution[i][:, GROUPS[name]].sum(axis=1)
                steps_back = k - 1 - by_step.argmax()
                session = f"t-{steps_back + 1}"
                value_text = factor_value(sessions, name, row - steps_back)
            else:
                session = "actual"
                value_text = factor_value(sessions, name, row)
            factor[f"factor_{rank}"] = name
            factor[f"valor_{rank}"] = value_text
            factor[f"aporte_{rank}"] = amount
            factor[f"sesion_{rank}"] = session
        factors.append(factor)

    # Las columnas se fijan para que la tabla exista aunque no haya alertas.
    columns = [f"{field}_{rank}" for rank in range(1, TOP_N + 1)
               for field in ("factor", "valor", "aporte", "sesion")]
    return pd.DataFrame(factors, columns=columns), pd.DataFrame(contribution, columns=NAMES)
