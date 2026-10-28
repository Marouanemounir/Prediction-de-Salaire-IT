"""Schema checks, dataset audit and leakage-resistant group splits."""
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

CATEGORICAL = ["Entreprise", "Profil", "Niveau_Etude", "Technologie"]
FEATURES = ["Experience", *CATEGORICAL]
TARGET = "Salaire"


def load_data(path):
    path = Path(path)
    raw = pd.read_csv(path, encoding="utf-8-sig")
    missing = set(FEATURES + [TARGET]) - set(raw.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {sorted(missing)}")
    raw = raw[FEATURES + [TARGET]].copy()
    if raw.isna().any().any():
        raise ValueError("Le CSV contient des valeurs manquantes.")
    for col in CATEGORICAL:
        raw[col] = raw[col].astype(str).str.strip()
        if raw[col].eq("").any():
            raise ValueError(f"Catégorie vide : {col}")
    for col in ["Experience", TARGET]:
        raw[col] = pd.to_numeric(raw[col], errors="raise")
        if not np.isfinite(raw[col]).all():
            raise ValueError(f"Valeur non finie : {col}")
    if (raw.Experience < 0).any() or (raw[TARGET] <= 0).any():
        raise ValueError("Expérience négative ou salaire non positif.")
    clean = raw.drop_duplicates().copy()  # Keep original CSV row indices.
    audit = {
        "sha256": sha256(path.read_bytes()).hexdigest(),
        "raw_rows": len(raw), "usable_rows": len(clean),
        "exact_duplicates_removed": len(raw) - len(clean),
        "unique_feature_vectors": len(clean[FEATURES].drop_duplicates()),
        "salary_min": float(clean[TARGET].min()),
        "salary_max": float(clean[TARGET].max()),
        "salary_median": float(clean[TARGET].median()),
        "salary_multiple_of_1000_fraction": float(raw[TARGET].mod(1000).eq(0).mean()),
        "category_counts": {c: raw[c].value_counts().sort_index().to_dict() for c in CATEGORICAL},
        "provenance_status": "unverified",
    }
    return clean, audit


def feature_groups(data):
    """Same complete input profile stays in the same split, even with different y."""
    return pd.MultiIndex.from_frame(data[FEATURES]).factorize(sort=True)[0]


def split_data(data, seed=2025):
    groups = feature_groups(data)
    development, test = next(GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=seed
    ).split(data, groups=groups))
    fit, calibration = next(GroupShuffleSplit(
        n_splits=1, test_size=0.25, random_state=seed + 1
    ).split(data.iloc[development], groups=groups[development]))
    return {
        "train": data.iloc[development[fit]].copy(),
        "calibration": data.iloc[development[calibration]].copy(),
        "test": data.iloc[test].copy(),
    }
