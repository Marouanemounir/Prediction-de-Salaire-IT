"""Optional integration checks against locally generated artifacts."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from salary_study.inference import load_bundle, predict

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


def test_corrupted_bundle_rejected_before_deserializing(tmp_path):
    (tmp_path / "model.joblib").write_bytes(b"corrupted")
    (tmp_path / "metadata.json").write_text(json.dumps({"model_sha256": "invalid"}), encoding="utf-8")
    with pytest.raises(ValueError, match="correspondent"):
        load_bundle(tmp_path)


@pytest.mark.skipif(not (ARTIFACTS / "model.joblib").exists(), reason="Entraînement local requis")
def test_saved_bundle_reproduces_test_predictions():
    saved = pd.read_csv(ARTIFACTS / "test_predictions.csv")
    bundle, metadata = load_bundle(ARTIFACTS)
    point, lo, hi = predict(bundle, saved)
    np.testing.assert_allclose(point, saved.prediction_mad, atol=1e-8)
    np.testing.assert_allclose(lo, saved.lower_mad, atol=1e-8)
    np.testing.assert_allclose(hi, saved.upper_mad, atol=1e-8)
    assert len(saved) == metadata["split_sizes"]["test"]


@pytest.mark.skipif(not (ARTIFACTS / "model.joblib").exists(), reason="Entraînement local requis")
def test_streamlit_study_and_prediction():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / "app_streamlit.py"), default_timeout=30).run()
    assert not app.exception
    assert len(app.tabs) == 3
    app.button[0].click().run()
    assert not app.exception
    assert any(metric.label == "Estimation (MAD/mois)" for metric in app.metric)
