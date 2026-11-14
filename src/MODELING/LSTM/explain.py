"""
Tres factores por alerta con Integrated Gradients, como pide el reporte: cuánto sumó o restó cada
variable al log-odds de la predicción, y de qué sesión previa vino cuando es una variable de la
secuencia. La referencia es la misma vaca con valores promedio de train en sus sesiones previas y
en el contexto (0 ya escalado); los pasos de relleno siguen ignorados por Masking. Los aportes de
una sesión suman su log-odds menos el de esa referencia. Son asociaciones del modelo, no causas.
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from dataset import CONTEXT_FEATURES, SEQUENCE_FEATURES

STEPS = 32  # Puntos del camino entre la referencia y la sesión real
TOP_N = 3
NAMES = np.array(SEQUENCE_FEATURES + CONTEXT_FEATURES, dtype=object)


def integrated_gradients(model, seq, ctx, output, batch_size):
    """Aporte de cada entrada al log-odds de la salida output: (n, k, variables) y (n, variables)."""
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


def explain(model, X_seq, X_ctx, rows, sessions, output, batch_size):
    """
    Factores de las sesiones rows (posiciones en sessions) para la salida output. Una variable de la
    secuencia suma sus aportes de los k pasos y se reporta la sesión previa que más aportó (t-1 es
    el ordeño anterior).
    Devuelve los factores por sesión y el aporte de cada variable en cada sesión.
    """
    seq_contribution, ctx_contribution = integrated_gradients(model, X_seq[rows], X_ctx[rows], output, batch_size)
    contribution = np.concatenate([seq_contribution.sum(axis=1), ctx_contribution], axis=1)
    k, n_seq = X_seq.shape[1], len(SEQUENCE_FEATURES)
    # Pasos hacia atrás del paso con más aporte; el paso k - 1 es la fila de la sesión y describe t-1.
    lag = k - 1 - np.abs(seq_contribution).argmax(axis=1)

    factors = pd.DataFrame(index=range(len(rows)))
    ranking = np.argsort(-np.abs(contribution), axis=1)[:, :TOP_N]
    positions = np.arange(len(rows))
    for rank in range(TOP_N):
        column = ranking[:, rank]
        amount = contribution[positions, column]
        in_sequence = column < n_seq
        back = np.where(in_sequence, lag[positions, np.minimum(column, n_seq - 1)], 0)
        source = rows - back  # La fila t - lag describe la sesión t - lag - 1.
        values = [sessions[name].iat[row] for name, row in zip(NAMES[column], source)]
        values = [np.expm1(v) if name == "box_time_to_yield_ratio" else v for name, v in zip(NAMES[column], values)]
        factors[f"factor_{rank + 1}"] = np.where(np.abs(amount) > 1e-12, NAMES[column], "")
        factors[f"sesion_{rank + 1}"] = np.where(in_sequence, [f"t-{b + 1}" for b in back], "actual")
        factors[f"valor_{rank + 1}"] = ["faltante" if pd.isna(v) else f"{v:.4g}" for v in values]
        factors[f"aporte_log_odds_{rank + 1}"] = amount
        factors[f"direccion_{rank + 1}"] = np.where(amount > 0, "aumenta riesgo", "reduce riesgo")

    return factors, pd.DataFrame(contribution, columns=NAMES)
