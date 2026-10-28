"""Finite-sample split conformal prediction intervals."""
import math
import numpy as np


def conformal_radius(y_calibration, predictions, alpha=0.10):
    y = np.asarray(y_calibration, dtype=float).ravel()
    pred = np.asarray(predictions, dtype=float).ravel()
    if not 0 < alpha < 1 or len(y) == 0 or y.shape != pred.shape:
        raise ValueError("Calibration vide, dimensions incompatibles ou alpha invalide.")
    scores = np.abs(y - pred)
    if not np.isfinite(scores).all():
        raise ValueError("Scores de calibration non finis.")
    rank = math.ceil((len(scores) + 1) * (1 - alpha))
    if rank > len(scores):
        raise ValueError("Pas assez de données de calibration pour ce niveau de couverture.")
    return float(np.partition(scores, rank - 1)[rank - 1])


def prediction_interval(prediction, radius):
    pred = np.asarray(prediction, dtype=float)
    return pred - radius, pred + radius
