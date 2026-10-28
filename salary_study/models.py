"""Common preprocessing and a NumPy gradient-descent reference estimator."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted, validate_data

from .data import CATEGORICAL


class ScratchLinearRegression(RegressorMixin, BaseEstimator):
    """Batch gradient descent on a quadratic objective, without an sklearn solver.

    Cache X.T@X and X.T@y to avoid traversing every row on every iteration.
    The spectral learning rate is safe for the convex least-squares objective.
    """
    def __init__(self, max_iter=10000, tol=1e-10):
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X, y = validate_data(self, X, y, y_numeric=True)
        design = np.column_stack([np.ones(len(X)), X])
        gram = design.T @ design / len(y)
        rhs = design.T @ y / len(y)
        rate = 1.0 / max(float(np.linalg.eigvalsh(gram)[-1]), 1e-12)
        theta = np.zeros(design.shape[1])
        previous = np.inf
        self.converged_ = False
        for iteration in range(self.max_iter):
            theta -= rate * (gram @ theta - rhs)
            loss = float(theta @ gram @ theta / 2 - theta @ rhs)
            if np.isfinite(previous) and abs(previous - loss) <= self.tol * max(1, abs(previous)):
                self.converged_ = True
                break
            previous = loss
        self.intercept_, self.coef_ = theta[0], theta[1:]
        self.n_iter_ = iteration + 1
        return self

    def predict(self, X):
        check_is_fitted(self, "coef_")
        X = validate_data(self, X, reset=False)
        return X @ self.coef_ + self.intercept_


def make_pipeline(model):
    preprocess = ColumnTransformer([
        ("numeric", StandardScaler(), ["Experience"]),
        ("categories", OneHotEncoder(handle_unknown="ignore", drop="first",
                                     sparse_output=False), CATEGORICAL),
    ], verbose_feature_names_out=False)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def candidates(seed=2025):
    """Small predeclared grids; all choices use only training CV MAE."""
    return {
        "Baseline médiane": (DummyRegressor(strategy="median"), {}),
        "Linear Regression": (LinearRegression(), {}),
        "Linear from scratch": (ScratchLinearRegression(), {}),
        "Ridge": (Ridge(), {"model__alpha": [0.1, 10.0, 100.0]}),
        "Lasso": (Lasso(max_iter=20000), {"model__alpha": [1.0, 10.0, 100.0]}),
        "Random Forest": (RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=seed),
                          {"model__min_samples_leaf": [1, 5], "model__max_features": [1.0, 0.7]}),
        "Gradient Boosting": (GradientBoostingRegressor(random_state=seed),
                              {"model__n_estimators": [150, 300], "model__max_depth": [2, 3],
                               "model__learning_rate": [0.05]}),
    }
