"""Train the served models on loads 0-2 hp only (so load-3 data stays a true hold-out
for the demo), export them, and measure hold-out metrics, ONNX parity and latency.

Outputs
  models/gbm10.joblib, models/cnn10.onnx, models/health.json, models/meta.json
  demo/assets/cnn10.onnx, demo/assets/samples.json
  tests/fixtures/*.csv
  reports/deploy_model.json, reports/health_index.json (+ reports/latency.json via bench_latency.py)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from vibesense import features  # noqa: E402
from vibesense.cnn import export_onnx, normalize, predict_proba_torch, train_cnn  # noqa: E402
from vibesense.data import CLASSES10, FS, WINDOW  # noqa: E402
from vibesense.metrics import classification_report  # noqa: E402
from vibesense.splits import split_indices  # noqa: E402

MODELS, REPORTS, ASSETS = ROOT / "models", ROOT / "reports", ROOT / "demo" / "assets"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    a = ap.parse_args()
    for d in (MODELS, REPORTS / "runs", ASSETS):
        d.mkdir(parents=True, exist_ok=True)
    ds = dict(np.load(ROOT / "data" / "processed" / "windows.npz"))
    tr, te = split_indices("load3", ds)
    X, F, y = ds["X"], ds["F"], ds["y10"]

    import joblib
    import onnxruntime as ort
    from sklearn.ensemble import HistGradientBoostingClassifier

    print("training GBM on loads 0-2")
    gbm = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(F[tr], y[tr])
    joblib.dump(gbm, MODELS / "gbm10.joblib", compress=3)

    print("training CNN on loads 0-2")
    t0 = time.perf_counter()
    cnn = train_cnn(X[tr], y[tr], len(CLASSES10), epochs=a.epochs, seed=0, log=print)
    cnn_fit_s = time.perf_counter() - t0
    export_onnx(cnn, MODELS / "cnn10.onnx", WINDOW)
    shutil.copy(MODELS / "cnn10.onnx", ASSETS / "cnn10.onnx")

    print("fitting health index on normal windows, loads 0-2 (see scripts/health_eval.py)")
    from health_eval import run as health_run

    hi, hrep = health_run(ds)
    (MODELS / "health.json").write_text(json.dumps(hi.to_dict()))
    (REPORTS / "health_index.json").write_text(json.dumps(hrep, indent=1))

    meta = {"classes": CLASSES10, "fs_hz": FS, "window": WINDOW, "trained_on": "CWRU 12k drive-end, loads 0-2 hp",
            "cnn_input": "per-window z-score of raw samples, shape (batch,1,2048)", "feature_names": features.feature_names()}
    (MODELS / "meta.json").write_text(json.dumps(meta, indent=1))

    # ---- hold-out metrics of the exact exported artefacts (load 3 hp)
    sess = ort.InferenceSession(str(MODELS / "cnn10.onnx"), providers=["CPUExecutionProvider"])
    logits = sess.run(None, {"signal": normalize(X[te])[:, None, :]})[0]
    p_onnx = logits.argmax(1)
    p_torch_proba = predict_proba_torch(cnn, X[te])
    max_abs = float(np.abs(np.exp(logits - logits.max(1, keepdims=True)) /
                           np.exp(logits - logits.max(1, keepdims=True)).sum(1, keepdims=True) - p_torch_proba).max())
    rep = {
        "split": "train loads 0-2 hp (recordings never seen at test) -> test load 3 hp",
        "cnn_onnx": classification_report(y[te], p_onnx, CLASSES10),
        "gbm": classification_report(y[te], gbm.predict(F[te]), CLASSES10),
        "onnx_vs_torch_max_abs_prob_diff": max_abs,
        "onnx_vs_torch_label_agreement": float((p_onnx == p_torch_proba.argmax(1)).mean()),
        "onnx_bytes": (MODELS / "cnn10.onnx").stat().st_size,
        "cnn_params": int(sum(p.numel() for p in cnn.parameters())),
    }
    (REPORTS / "deploy_model.json").write_text(json.dumps(rep, indent=1))
    # the exported CNN *is* the 10-class load-3 CNN of the evaluation grid (same data, recipe, seed)
    run = dict(rep["cnn_onnx"], task="10class", split="load3", model="cnn", seed=0, n_train=int(len(tr)),
               fit_seconds=round(cnn_fit_s, 1), source="scripts/train_deploy.py (ONNX export evaluated)",
               train_recordings=int(len(np.unique(ds["file_id"][tr]))), test_recordings=int(len(np.unique(ds["file_id"][te]))))
    (REPORTS / "runs" / "10class_load3_cnn.json").write_text(json.dumps(run, indent=1))
    print("deploy CNN acc", rep["cnn_onnx"]["accuracy"], "GBM acc", rep["gbm"]["accuracy"])

    print("HI AUROC", hrep["auroc_normal_vs_fault_load3"])

    # ---- demo samples + test fixtures: 2 hold-out windows per class from load 3
    samples = []
    yt = y[te]
    rng = np.random.default_rng(7)
    for c, name in enumerate(CLASSES10):
        idx = te[yt == c]
        for k, i in enumerate(sorted(rng.choice(idx, 2, replace=False))):
            x = X[i]
            samples.append({
                "id": f"{name}_load3_{k}", "label": name, "load_hp": int(ds["load"][i]), "file_id": int(ds["file_id"][i]),
                "start_sample": int(ds["pos"][i]) * 1024,
                "samples": [round(float(v), 5) for v in x],
                "py_cnn_proba": [round(float(v), 5) for v in p_torch_proba[np.searchsorted(te, i)]],
                "py_gbm_proba": [round(float(v), 5) for v in gbm.predict_proba(F[i : i + 1])[0]],
                "health_index": round(float(hi.score(F[i : i + 1])[0]), 3),
            })
    (ASSETS / "samples.json").write_text(json.dumps({"fs": FS, "window": WINDOW, "classes": CLASSES10,
                                                      "threshold": hi.threshold, "samples": samples}))
    fx = ROOT / "tests" / "fixtures"
    fx.mkdir(parents=True, exist_ok=True)
    for sname in ("Normal_load3_0", "IR021_load3_0", "OR007_load3_0", "B014_load3_0"):
        s0 = next(s for s in samples if s["id"] == sname)
        (fx / f"{sname}.csv").write_text("acc_g\n" + "\n".join(str(v) for v in s0["samples"]) + "\n")
    import subprocess

    subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--models", "none"], check=True)  # re-summarize
    subprocess.run([sys.executable, str(ROOT / "scripts" / "bench_latency.py")], check=True)
    print("done")


if __name__ == "__main__":
    main()
