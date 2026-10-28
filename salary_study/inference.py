"""Load the locally trained bundle and validate predictions consistently."""
from hashlib import sha256
import json
from pathlib import Path
import joblib
import numpy as np
from .data import CATEGORICAL, FEATURES
from .uncertainty import prediction_interval


def load_bundle(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    model_path = directory / "model.joblib"
    if sha256(model_path.read_bytes()).hexdigest() != metadata["model_sha256"]:
        raise ValueError("Le modèle et ses métadonnées ne correspondent pas. Réentraîner le modèle.")
    # Load only locally produced, trusted bundles; joblib uses pickle internally.
    return joblib.load(model_path), metadata


def predict(bundle, rows):
    if set(FEATURES) - set(rows.columns):
        raise ValueError("Le profil d'entrée est incomplet.")
    if rows.empty or rows[FEATURES].isna().any().any():
        raise ValueError("Le profil est vide ou contient des valeurs manquantes.")
    for col in CATEGORICAL:
        if not rows[col].isin(bundle["categories"][col]).all():
            raise ValueError(f"Catégorie absente de l'entraînement : {col}")
    experience = rows.Experience.to_numpy(dtype=float)
    lo, hi = bundle["experience_range"]
    if not np.isfinite(experience).all() or (experience < lo).any() or (experience > hi).any():
        raise ValueError(f"L'expérience doit être comprise entre {lo:g} et {hi:g} ans.")
    prediction = bundle["pipeline"].predict(rows[FEATURES])
    lower, upper = prediction_interval(prediction, bundle["radius_mad"])
    return prediction, lower, upper
