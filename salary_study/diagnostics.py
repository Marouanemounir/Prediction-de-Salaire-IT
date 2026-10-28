"""Held-out residual diagnostics and bootstrap uncertainty of metrics."""
import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.diagnostic import het_breuschpagan


def metrics(y, pred):
    return {"r2": float(r2_score(y, pred)),
            "mae_mad": float(mean_absolute_error(y, pred)),
            "rmse_mad": float(np.sqrt(mean_squared_error(y, pred)))}


def bootstrap_metrics(y, pred, seed=2025, repetitions=1000):
    """Conditional on this fitted model; does not include retraining variability."""
    y, pred = np.asarray(y), np.asarray(pred)
    rng = np.random.default_rng(seed)
    samples = [metrics(y[idx], pred[idx]) for idx in
               rng.integers(0, len(y), size=(repetitions, len(y)))]
    return {key: [float(x) for x in np.quantile([s[key] for s in samples], [0.025, 0.975])]
            for key in samples[0]}


def residual_diagnostics(y, pred, experience):
    residual = np.asarray(y) - np.asarray(pred)
    jb = stats.jarque_bera(residual)
    centered = stats.ttest_1samp(residual, popmean=0)
    # Deliberately low-dimensional full-rank auxiliary design, on held-out data.
    design = np.column_stack([np.ones(len(pred)), pred, experience])
    lm, p_lm, f_stat, p_f = het_breuschpagan(residual, design, robust=True)
    return {
        "mean_residual_mad": float(residual.mean()),
        "std_residual_mad": float(residual.std(ddof=1)),
        "jarque_bera": {"statistic": float(jb.statistic), "p_value": float(jb.pvalue),
                        "null": "Résidus normalement distribués"},
        "bias_t_test": {"statistic": float(centered.statistic), "p_value": float(centered.pvalue),
                        "null": "Erreur moyenne nulle"},
        "breusch_pagan_koenker": {"statistic": float(lm), "p_value": float(p_lm),
                                 "f_statistic": float(f_stat), "f_p_value": float(p_f),
                                 "null": "Variance constante par rapport à prédiction et expérience"},
        "interpretation": "Tests exploratoires, p-values non corrigées. Indépendance non vérifiable sans identifiants. Aucune conclusion causale.",
    }
