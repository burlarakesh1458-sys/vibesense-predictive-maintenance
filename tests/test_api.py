"""API + exported-model tests. Use only committed artefacts (models/, demo/assets/, fixtures)."""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vibesense.api import app, parse_csv
from vibesense.cnn import normalize
from vibesense.predictor import Predictor

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"


def load_fixture(name):
    return parse_csv((FIX / f"{name}.csv").read_text())


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["window_samples"] == 2048 and len(r["classes"]) == 10


@pytest.mark.parametrize("name,ftype", [("Normal_load3_0", "Normal"), ("IR021_load3_0", "InnerRace"),
                                         ("OR007_load3_0", "OuterRace"), ("B014_load3_0", "Ball")])
@pytest.mark.parametrize("model", ["cnn", "gbm"])
def test_predict_fixture_fault_type(client, name, ftype, model):
    xs = load_fixture(name).tolist()
    r = client.post("/predict", json={"samples": xs, "model": model})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["fault_type"] == ftype
    assert 0 < out["confidence"] <= 1 and abs(sum(out["probabilities"].values()) - 1) < 1e-3
    assert out["latency_ms"] >= 0


def test_health_index_flags_fault_not_normal(client):
    n = client.post("/predict", json={"samples": load_fixture("Normal_load3_0").tolist()}).json()
    f = client.post("/predict", json={"samples": load_fixture("IR021_load3_0").tolist()}).json()
    assert f["health_index"] > n["health_index"]
    assert f["anomalous"] and not n["anomalous"]


def test_predict_rejects_short_window(client):
    r = client.post("/predict", json={"samples": [0.0] * 100})
    assert r.status_code == 422


def test_predict_rejects_bad_model(client):
    r = client.post("/predict", json={"samples": [0.0] * 2048, "model": "svm"})
    assert r.status_code == 422


def test_resampling_path(client):
    x = load_fixture("OR007_load3_0")
    up = np.interp(np.linspace(0, len(x) - 1, len(x) * 2), np.arange(len(x)), x)  # 24 kHz version
    r = client.post("/predict", json={"samples": up.tolist(), "fs": 24000}).json()
    assert r["fault_type"] == "OuterRace"


def test_csv_upload(client):
    data = (FIX / "OR007_load3_0.csv").read_bytes()
    r = client.post("/predict/csv", files={"file": ("x.csv", data, "text/csv")})
    assert r.status_code == 200 and r.json()["fault_type"] == "OuterRace"


def test_parse_csv_skips_header_and_takes_first_column():
    text = "t,acc\n" + "\n".join(f"{i},{i * 0.5}" for i in range(2100))
    v = parse_csv(text)
    assert len(v) == 2100 and v[3] == 3.0


def test_onnx_matches_python_reference_on_demo_samples():
    """The browser runs the same ONNX file; its outputs must equal the torch probabilities stored at export."""
    p = Predictor()
    s = json.loads((ROOT / "demo" / "assets" / "samples.json").read_text())
    X = np.asarray([x["samples"] for x in s["samples"]], dtype=np.float32)
    ref = np.asarray([x["py_cnn_proba"] for x in s["samples"]])
    got = p.proba(X, "cnn")
    assert np.abs(got - ref).max() < 1e-3
    assert (ROOT / "demo" / "assets" / "cnn10.onnx").read_bytes() == (ROOT / "models" / "cnn10.onnx").read_bytes()


def test_normalize_is_scale_invariant():
    x = np.random.default_rng(0).normal(size=(2, 2048)).astype(np.float32)
    assert np.allclose(normalize(x), normalize(5 * x + 2), atol=1e-4)
