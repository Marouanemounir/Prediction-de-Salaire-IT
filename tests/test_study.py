from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from salary_study.data import FEATURES, TARGET, feature_groups, load_data, split_data
from salary_study.explain import explain_predictions
from salary_study.inference import predict
from salary_study.models import ScratchLinearRegression, make_pipeline
from salary_study.uncertainty import conformal_radius

ROOT = Path(__file__).resolve().parents[1]


def test_no_profile_leakage_and_reproducible_splits():
    data, audit = load_data(ROOT / "DATASET_IT_MAROC_2025.csv")
    assert audit["raw_rows"] == 5000
    assert audit["exact_duplicates_removed"] == 46
    parts = split_data(data)
    sets = {k: set(map(tuple, v[FEATURES].to_numpy())) for k, v in parts.items()}
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["train"].isdisjoint(sets["calibration"])
    assert sets["test"].isdisjoint(sets["calibration"])
    assert sum(map(len, parts.values())) == len(data)
    assert set().union(*(set(p.index) for p in parts.values())) == set(data.index)
    repeated = split_data(data)
    for name in parts:
        pd.testing.assert_frame_equal(parts[name], repeated[name])
    train = parts["train"]
    groups = feature_groups(train)
    for fit, val in GroupKFold(5, shuffle=True, random_state=2025).split(train, groups=groups):
        assert set(groups[fit]).isdisjoint(groups[val])


def test_preprocessing_does_not_learn_unseen_category_or_test_mean():
    frame = pd.DataFrame({"Experience": [0, 2, 4], "Entreprise": ["A"]*3,
                          "Profil": ["dev"]*3, "Niveau_Etude": ["Master"]*3, "Technologie": ["Python"]*3})
    pipe = make_pipeline(LinearRegression()).fit(frame, [5000, 7000, 9000])
    other = frame.iloc[[0]].copy()
    other["Entreprise"], other["Experience"] = "TEST_ONLY", 100
    before = pipe.named_steps["preprocess"].named_transformers_["numeric"].mean_.copy()
    with pytest.warns(UserWarning, match="unknown categories"):
        result = pipe.predict(other)
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(before, [2.0])
    np.testing.assert_array_equal(pipe.named_steps["preprocess"].named_transformers_["numeric"].mean_, before)
    assert "TEST_ONLY" not in pipe.named_steps["preprocess"].named_transformers_["categories"].categories_[0]


def test_gradient_descent_recovers_known_linear_function():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(200, 4))
    y = X @ np.array([1000, -20, 400, 90]) + 6000
    model = ScratchLinearRegression(tol=1e-13).fit(X, y)
    assert model.converged_
    np.testing.assert_allclose(model.predict(X), y, atol=0.01)


def test_conformal_uses_finite_sample_order_statistic():
    assert conformal_radius(np.arange(1, 10), np.zeros(9), alpha=.2) == 8
    assert conformal_radius(np.arange(1, 10), np.zeros(9), alpha=.1) == 9
    with pytest.raises(ValueError, match="Pas assez"):
        conformal_radius([1, 2], [0, 0], alpha=.01)
    with pytest.raises(ValueError):
        conformal_radius([1], [np.nan])
    with pytest.raises(ValueError):
        conformal_radius([1, 2], [0])


def test_schema_refuses_invalid_salary(tmp_path):
    data, _ = load_data(ROOT / "DATASET_IT_MAROC_2025.csv")
    data = data.head(2).copy()
    data.loc[data.index[0], TARGET] = -1
    path = tmp_path / "bad.csv"
    data.to_csv(path, index=False)
    with pytest.raises(ValueError, match="non positif"):
        load_data(path)


@pytest.mark.parametrize("kind", ["linear", "tree"])
def test_shap_adds_up_for_linear_and_tree(kind):
    from sklearn.ensemble import GradientBoostingRegressor
    data, _ = load_data(ROOT / "DATASET_IT_MAROC_2025.csv")
    train = data.head(120)
    model = LinearRegression() if kind == "linear" else GradientBoostingRegressor(n_estimators=10, random_state=1)
    pipe = make_pipeline(model).fit(train[FEATURES], train[TARGET])
    values, base = explain_predictions(pipe, train.head(50), train.tail(5))
    np.testing.assert_allclose(base + values.sum(axis=1), pipe.predict(train.tail(5)[FEATURES]), atol=.05)


def test_prediction_rejects_out_of_domain_and_returns_interval():
    data, _ = load_data(ROOT / "DATASET_IT_MAROC_2025.csv")
    data = data.head(100)
    pipe = make_pipeline(LinearRegression()).fit(data[FEATURES], data[TARGET])
    bundle = {"pipeline": pipe, "radius_mad": 1000., "experience_range": [0, 15],
              "categories": {c: data[c].unique() for c in FEATURES if c != "Experience"}}
    row = data.head(1)[FEATURES].copy()
    point, lo, hi = predict(bundle, row)
    np.testing.assert_allclose(lo, point - 1000)
    np.testing.assert_allclose(hi, point + 1000)
    row["Experience"] = 99
    with pytest.raises(ValueError, match="expérience"):
        predict(bundle, row)
    row["Experience"], row["Entreprise"] = 1, "INCONNUE"
    with pytest.raises(ValueError, match="Catégorie"):
        predict(bundle, row)
