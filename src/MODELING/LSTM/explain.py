"""
Tres factores por alerta con Integrated Gradients: cuánto sumó o restó cada variable al log-odds y,
si es de la secuencia, de qué sesión previa vino. La referencia es la vaca con valores promedio de
train (0 ya escalado). Son asociaciones, no causas.
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from dataset import CONTEXT_FEATURES, SEQUENCE_FEATURES

STEPS = 32  # Puntos entre la referencia y la sesión real
TOP_N = 3
NAMES = SEQUENCE_FEATURES + CONTEXT_FEATURES


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


def explain(model, X_seq, X_ctx, rows, sessions, output, batch_size, scale):
    """
    Los 3 factores de cada sesión de rows para la salida output; scale es la pendiente a de la
    calibración. Devuelve los factores y el aporte de cada variable.
    """
    seq_contribution, ctx_contribution = integrated_gradients(model, X_seq[rows], X_ctx[rows], output, batch_size)
    seq_contribution, ctx_contribution = seq_contribution * scale, ctx_contribution * scale
    # La secuencia suma sus k pasos.
    contribution = np.concatenate([seq_contribution.sum(axis=1), ctx_contribution], axis=1)
    k = X_seq.shape[1]

    factors = []
    for i, row in enumerate(rows):
        factor = {}
        top = np.argsort(-np.abs(contribution[i]))[:TOP_N]
        for rank, column in enumerate(top, start=1):
            name, amount = NAMES[column], contribution[i, column]
            if abs(amount) <= 1e-12:  # Sin aporte: no cuenta como factor (como en la regresión logística)
                name = session = value_text = ""
            else:
                if column < len(SEQUENCE_FEATURES):
                    # El último paso describe la sesión anterior (t-1).
                    steps_back = k - 1 - np.abs(seq_contribution[i, :, column]).argmax()
                    session, value = f"t-{steps_back + 1}", sessions[name].iat[row - steps_back]
                else:
                    session, value = "actual", sessions[name].iat[row]
                if name == "box_time_to_yield_ratio":
                    value = np.expm1(value)  # Viene en logaritmo
                value_text = "faltante" if pd.isna(value) else f"{value:.4g}"
            factor[f"factor_{rank}"] = name
            factor[f"sesion_{rank}"] = session
            factor[f"valor_{rank}"] = value_text
            factor[f"aporte_log_odds_{rank}"] = amount
            factor[f"direccion_{rank}"] = "aumenta riesgo" if amount > 0 else "reduce riesgo"
        factors.append(factor)

    # Las columnas se fijan para que la tabla exista aunque no haya alertas.
    columns = [f"{field}_{rank}" for rank in range(1, TOP_N + 1)
               for field in ("factor", "sesion", "valor", "aporte_log_odds", "direccion")]
    return pd.DataFrame(factors, columns=columns), pd.DataFrame(contribution, columns=NAMES)
