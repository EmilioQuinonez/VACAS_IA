"""Clasificador XGBoost por evento y su log con formato Keras."""

import time

from xgboost import XGBClassifier
from xgboost.callback import TrainingCallback


class KerasStyleLog(TrainingCallback):
    """
    Imprime cada árbol como una época de Keras (verbose=2) para el panel /entrenamiento;
    sin val, solo accuracy y loss.
    """

    def __init__(self, first_epoch, total_epochs):
        super().__init__()
        self.first_epoch = first_epoch
        self.total_epochs = total_epochs
        self.best = float("inf")
        self.elapsed = 0.0
        self.printed = 0

    def before_iteration(self, model, epoch, evals_log):
        self.start = time.perf_counter()
        return False

    def after_iteration(self, model, epoch, evals_log):
        duration = time.perf_counter() - self.start
        # Keras imprime segundos enteros: se arrastra el resto para que la suma sea el tiempo real.
        self.elapsed += duration
        seconds = int(self.elapsed) - self.printed
        self.printed += seconds

        train, val = evals_log["validation_0"], evals_log.get("validation_1")
        number = self.first_epoch + epoch + 1
        summary = (f"1/1 - {seconds}s - {int(duration * 1000)}ms/step - "
                   f"accuracy: {1 - train['error'][-1]:.4f} - loss: {train['logloss'][-1]:.4f}")
        if val:
            summary += f" - val_accuracy: {1 - val['error'][-1]:.4f} - val_loss: {val['logloss'][-1]:.4f}"
        print(f"Epoch {number}/{self.total_epochs}")
        print(summary)
        if val and val["logloss"][-1] < self.best:
            print(f"Epoch {number}: val_loss improved from {self.best:.5f} to {val['logloss'][-1]:.5f}")
            self.best = val["logloss"][-1]

        return False


def train_event(X, y, rows, val_rows=None, first_epoch=0, *, n_trees, total_epochs, random_state):
    """
    Entrena un XGBoost con valores por defecto; scale_pos_weight compensa el desbalance.
    val_rows solo se evalúa para el log: no detiene ni cambia el entrenamiento.
    """
    X_fit, y_fit = X.iloc[rows], y.iloc[rows]
    eval_set = [(X_fit, y_fit)]
    if val_rows is not None:
        eval_set.append((X.iloc[val_rows], y.iloc[val_rows]))
    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=n_trees,
        scale_pos_weight=(y_fit == 0).sum() / (y_fit == 1).sum(),
        eval_metric=["error", "logloss"],
        callbacks=[KerasStyleLog(first_epoch, total_epochs)],
        random_state=random_state,
    )

    return model.fit(X_fit, y_fit, eval_set=eval_set, verbose=False)
