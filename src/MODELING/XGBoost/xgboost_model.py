"""Clasificador XGBoost por evento."""

from xgboost import XGBClassifier


def train_event(X, y, rows, val_rows=None, *, n_trees, random_state):
    """
    Entrena un XGBoost con valores por defecto; scale_pos_weight compensa el desbalance.
    val_rows solo se evalúa para las curvas de aprendizaje: no detiene ni cambia el entrenamiento.
    """
    X_fit, y_fit = X.iloc[rows], y.iloc[rows]
    eval_set = [(X_fit, y_fit)]
    if val_rows is not None:
        eval_set.append((X.iloc[val_rows], y.iloc[val_rows]))
    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=n_trees,
        scale_pos_weight=(y_fit == 0).sum() / (y_fit == 1).sum(),
        eval_metric="logloss",
        random_state=random_state,
    )

    return model.fit(X_fit, y_fit, eval_set=eval_set, verbose=False)
