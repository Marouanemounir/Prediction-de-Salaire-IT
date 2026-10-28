"""SHAP on encoded features, aggregated back to the five original variables."""
import numpy as np
import pandas as pd
import shap
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from .data import CATEGORICAL, FEATURES


def explain_predictions(pipeline, background, rows):
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    transformed = preprocess.transform(rows[FEATURES])
    reference = preprocess.transform(background[FEATURES])
    if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
        explainer = shap.TreeExplainer(model, reference, feature_perturbation="interventional")
        explanation = explainer(transformed, check_additivity=True)
        values = explanation.values
        base = np.asarray(explanation.base_values).reshape(-1)
    elif isinstance(model, DummyRegressor):
        values = np.zeros_like(transformed)
        base = np.full(len(rows), float(np.asarray(model.constant_).ravel()[0]))
    else:
        explainer = shap.LinearExplainer((model.coef_, model.intercept_),
                                        shap.maskers.Independent(reference))
        explanation = explainer(transformed)
        values = explanation.values
        base = np.asarray(explanation.base_values).reshape(-1)
    encoder = preprocess.named_transformers_["categories"]
    owners = ["Experience"]
    for name, categories, drop_idx in zip(CATEGORICAL, encoder.categories_, encoder.drop_idx_):
        owners.extend([name] * (len(categories) - int(drop_idx is not None)))
    grouped = pd.DataFrame({name: values[:, np.asarray(owners) == name].sum(axis=1)
                            for name in FEATURES}, index=rows.index)
    prediction = pipeline.predict(rows[FEATURES])
    np.testing.assert_allclose(base + grouped.sum(axis=1).to_numpy(), prediction, atol=0.05, rtol=1e-6)
    return grouped, base
