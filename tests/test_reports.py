"""Committed reports are internally consistent (the README and demo only quote these files)."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"


def test_leakage_free_runs_never_share_recordings():
    for p in (R / "runs").glob("*.json"):
        r = json.loads(p.read_text())
        assert sum(map(sum, r["confusion_matrix"])) == r["n"], p.name
        if r["split"] != "random":
            # 40 recordings in total; leakage-free splits partition them
            assert r["train_recordings"] + r["test_recordings"] == 40, p.name


def test_demo_results_mirror_reports():
    demo = json.loads((ROOT / "demo" / "assets" / "results.json").read_text())
    assert demo["metrics"] == json.loads((R / "metrics.json").read_text())
    assert demo["health"] == json.loads((R / "health_index.json").read_text())
    dep = json.loads((R / "deploy_model.json").read_text())
    assert demo["deploy"]["cnn_acc_load3"] == dep["cnn_onnx"]["accuracy"]
    assert json.loads((R / "runs" / "10class_load3_cnn.json").read_text())["accuracy"] == dep["cnn_onnx"]["accuracy"]
