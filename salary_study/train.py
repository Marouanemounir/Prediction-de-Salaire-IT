"""CLI: python -m salary_study.train --data ... --output artifacts"""
import argparse
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, GroupKFold
from threadpoolctl import threadpool_limits

from .data import CATEGORICAL, FEATURES, TARGET, feature_groups, load_data, split_data
from .diagnostics import bootstrap_metrics, metrics, residual_diagnostics
from .explain import explain_predictions
from .models import candidates, make_pipeline
from .reporting import client_report, figures, save_json
from .uncertainty import conformal_radius, prediction_interval

ROOT = Path(__file__).resolve().parents[1]


def run(data_path=ROOT / "DATASET_IT_MAROC_2025.csv", output=ROOT / "artifacts", seed=2025, alpha=0.10):
    if not 0 < alpha < 1:
        raise ValueError("alpha doit être strictement entre 0 et 1.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data, audit = load_data(data_path)
    parts = split_data(data, seed)
    train, cal, test = (parts[k] for k in ["train", "calibration", "test"])
    groups = feature_groups(train)
    cv = GroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(cv.split(train, groups=groups))
    fitted, cv_details, cv_summary = {}, {}, {}
    with threadpool_limits(limits=1):
        for name, (model, grid) in candidates(seed).items():
            print(f"Validation croisée : {name}", flush=True)
            search = GridSearchCV(make_pipeline(model), grid, scoring="neg_mean_absolute_error",
                                  cv=folds, n_jobs=1, refit=True, error_score="raise")
            search.fit(train[FEATURES], train[TARGET])
            fitted[name] = search.best_estimator_
            cv_summary[name] = {"cv_mae_mad": float(-search.best_score_),
                                "cv_std_mad": float(search.cv_results_["std_test_score"][search.best_index_]),
                                "parameters": search.best_params_}
            cv_details[name] = [
                {"parameters": params, "mean_mae_mad": float(-search.cv_results_["mean_test_score"][i]),
                 "fold_mae_mad": [float(-search.cv_results_[f"split{k}_test_score"][i]) for k in range(5)]}
                for i, params in enumerate(search.cv_results_["params"])
            ]
    # Freeze the selection BEFORE evaluating calibration and test.
    selected = min(cv_summary, key=lambda name: cv_summary[name]["cv_mae_mad"])
    pipeline = fitted[selected]
    radius = conformal_radius(cal[TARGET], pipeline.predict(cal[FEATURES]), alpha)
    all_metrics, rows = {}, []
    for name, model in fitted.items():
        pred = model.predict(test[FEATURES])
        point = metrics(test[TARGET], pred)
        all_metrics[name] = {"test": point, "train": metrics(train[TARGET], model.predict(train[FEATURES])),
                             "bootstrap_95": bootstrap_metrics(test[TARGET], pred, seed),
                             "cv": cv_summary[name]}
        rows.append({"model": name, **{k: v for k, v in cv_summary[name].items() if k != "parameters"},
                     **{f"test_{k}": v for k, v in point.items()}})
    comparison = pd.DataFrame(rows).sort_values("cv_mae_mad")
    pred = pipeline.predict(test[FEATURES])
    lower, upper = prediction_interval(pred, radius)
    predictions = test[FEATURES].copy()
    predictions["actual_mad"] = test[TARGET]
    predictions["prediction_mad"] = pred
    predictions["lower_mad"], predictions["upper_mad"] = lower, upper
    predictions["residual_mad"] = test[TARGET].to_numpy() - pred
    predictions["covered"] = (test[TARGET].to_numpy() >= lower) & (test[TARGET].to_numpy() <= upper)
    predictions["experience_band"] = pd.cut(test.Experience, [-np.inf, 2, 5, 10, np.inf],
                                            labels=["0-2", "3-5", "6-10", "11+"])
    segments_test = []
    for column in ["Profil", "experience_band"]:
        for value, group in predictions.groupby(column, observed=True):
            segments_test.append({"dimension": column, "segment": str(value), "n": len(group),
                                  "mae_mad": float(group.residual_mad.abs().mean()),
                                  "mean_residual_mad": float(group.residual_mad.mean()),
                                  "coverage": float(group.covered.mean())})
    background = train.sample(min(100, len(train)), random_state=seed)[FEATURES]
    explained = test.sample(min(200, len(test)), random_state=seed)[FEATURES]
    contributions, base = explain_predictions(pipeline, background, explained)
    importance = contributions.abs().mean().rename_axis("variable").reset_index(name="mean_abs_shap_mad")
    importance = importance.sort_values("mean_abs_shap_mad")
    local = explained.join(contributions.add_prefix("shap_"))
    local["base_value_mad"] = base
    local["prediction_mad"] = pipeline.predict(explained)
    diagnostics = residual_diagnostics(test[TARGET], pred, test.Experience)
    bundle = {"pipeline": pipeline, "radius_mad": radius, "alpha": alpha,
              "background": background, "categories": {c: sorted(train[c].unique().tolist()) for c in CATEGORICAL},
              "experience_range": [float(train.Experience.min()), float(train.Experience.max())],
              "profile_technology_pairs": set(zip(train.Profil, train.Technologie)), "selected_model": selected}
    joblib.dump(bundle, output / "model.joblib", compress=3)
    metadata = {"seed": seed, "data_audit": audit, "split_sizes": {k: len(v) for k, v in parts.items()},
                "selection_rule": "Minimum training-only 5-fold group CV MAE; test never used for selection",
                "selected_model": selected, "selected_parameters": cv_summary[selected]["parameters"],
                "selected_test_metrics": all_metrics[selected]["test"],
                "selected_test_bootstrap_95": all_metrics[selected]["bootstrap_95"],
                "interval": {"alpha": alpha, "radius_mad": radius,
                             "test_coverage": float(predictions.covered.mean()), "mean_width_mad": float(2 * radius)},
                "residual_diagnostics": diagnostics, "shap_sample_size": len(explained),
                "model_sha256": sha256((output / "model.joblib").read_bytes()).hexdigest(),
                "python": platform.python_version(),
                "versions": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "shap", "statsmodels", "joblib"]}}
    manifest = pd.concat([pd.DataFrame({"csv_row_index": v.index, "split": k}) for k, v in parts.items()])
    manifest["cv_validation_fold"] = -1
    fold_by_row = {int(train.index[idx]): k for k, (_, val) in enumerate(folds) for idx in val}
    manifest["cv_validation_fold"] = manifest.csv_row_index.map(fold_by_row).fillna(-1).astype(int)
    manifest.sort_values("csv_row_index").to_csv(output / "split_manifest.csv", index=False)
    predictions.to_csv(output / "test_predictions.csv", index_label="csv_row_index")
    pd.DataFrame(segments_test).to_csv(output / "segments_test.csv", index=False)
    comparison.to_csv(output / "model_comparison.csv", index=False)
    importance.to_csv(output / "shap_global.csv", index=False)
    local.to_csv(output / "shap_local.csv", index_label="csv_row_index")
    segments = data.groupby("Profil")[TARGET].agg(n="size", median_mad="median",
                            q25_mad=lambda x: x.quantile(.25), q75_mad=lambda x: x.quantile(.75)).reset_index()
    segments.to_csv(output / "salary_segments.csv", index=False)
    for filename, obj in [("metadata.json", metadata), ("metrics.json", all_metrics),
                          ("cv_results.json", cv_details), ("diagnostics.json", diagnostics)]:
        save_json(output / filename, obj)
    figures(output, predictions, importance)
    client_report(output, metadata, comparison, segments)
    print(f"Retenu : {selected}. Rapport : {output / 'rapport_client.md'}", flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "DATASET_IT_MAROC_2025.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--alpha", type=float, default=0.10)
    args = parser.parse_args()
    run(args.data, args.output, args.seed, args.alpha)


if __name__ == "__main__":
    main()
