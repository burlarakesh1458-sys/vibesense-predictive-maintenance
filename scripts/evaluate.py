"""Run every (task, split, model) experiment and write reports/metrics.json.

Tasks : 10-class (normal + 3 fault types x 3 sizes) and 4-class (fault type).
Splits: random (leaky window shuffle), load0..load3 (leave-one-load-out), cross_size (4-class).
Models: rf, gbm (on 56 hand-crafted features), cnn (1D-CNN on the raw window).

Resumable: each run is cached in reports/runs/<task>_<split>_<model>.json.

    python scripts/evaluate.py [--models rf,gbm,cnn] [--epochs 15]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vibesense.data import CLASSES4, CLASSES10  # noqa: E402
from vibesense.metrics import classification_report  # noqa: E402
from vibesense.splits import assert_no_recording_overlap, split_indices  # noqa: E402

RUNS = ROOT / "reports" / "runs"
TASKS = {"10class": ("y10", CLASSES10), "4class": ("y4", CLASSES4)}
SPLITS = {
    "10class": ["random", "load0", "load1", "load2", "load3"],
    "4class": ["random", "load0", "load1", "load2", "load3", "cross_size"],
}


def make_tree(name: str, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

    if name == "rf":
        return RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=seed)
    if name == "gbm":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
    raise ValueError(name)


def run_one(ds, task, split, model, epochs, seed=0) -> dict:
    ykey, classes = TASKS[task]
    tr, te = split_indices(split, ds, seed=seed)
    if split != "random":
        assert_no_recording_overlap(ds, tr, te)
    y = ds[ykey]
    t0 = time.perf_counter()
    if model == "cnn":
        from vibesense.cnn import predict_proba_torch, train_cnn

        m = train_cnn(ds["X"][tr], y[tr], len(classes), epochs=epochs, seed=seed, log=print)
        fit_s = time.perf_counter() - t0
        pred = predict_proba_torch(m, ds["X"][te]).argmax(1)
    else:
        m = make_tree(model, seed).fit(ds["F"][tr], y[tr])
        fit_s = time.perf_counter() - t0
        pred = m.predict(ds["F"][te])
    rep = classification_report(y[te], pred, classes)
    rep.update(task=task, split=split, model=model, seed=seed, n_train=int(len(tr)), fit_seconds=round(fit_s, 1),
               train_recordings=int(len(np.unique(ds["file_id"][tr]))),
               test_recordings=int(len(np.unique(ds["file_id"][te]))))
    return rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="rf,gbm,cnn")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--only", default="", help="comma list of task_split keys to run, e.g. 4class_load3 (default: all)")
    ap.add_argument("--data", default=str(ROOT / "data" / "processed" / "windows.npz"))
    a = ap.parse_args()
    ds = dict(np.load(a.data))
    RUNS.mkdir(parents=True, exist_ok=True)
    for model in [m for m in a.models.split(",") if m in ("rf", "gbm", "cnn")]:
        for task, splits in SPLITS.items():
            for split in splits:
                f = RUNS / f"{task}_{split}_{model}.json"
                if f.exists() or (a.only and f"{task}_{split}" not in a.only.split(",")):
                    continue
                print(f"== {task} {split} {model}", flush=True)
                rep = run_one(ds, task, split, model, a.epochs)
                f.write_text(json.dumps(rep, indent=1))
                print(f"   acc={rep['accuracy']:.4f} macroF1={rep['macro_f1']:.4f} fit={rep['fit_seconds']}s", flush=True)
    summarize()


def summarize() -> None:
    runs = [json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json"))]
    table: dict = {}
    for r in runs:
        table.setdefault(r["task"], {}).setdefault(r["model"], {})[r["split"]] = {
            "accuracy": r["accuracy"], "macro_f1": r["macro_f1"], "n_test": r["n"],
        }
    for _task, models in table.items():
        for _model, splits in models.items():
            lolo = [splits[f"load{k}"] for k in range(4) if f"load{k}" in splits]
            if len(lolo) == 4:
                splits["lolo_mean"] = {
                    "accuracy": float(np.mean([s["accuracy"] for s in lolo])),
                    "macro_f1": float(np.mean([s["macro_f1"] for s in lolo])),
                    "accuracy_min": float(np.min([s["accuracy"] for s in lolo])),
                }
    out = ROOT / "reports" / "metrics.json"
    out.write_text(json.dumps(table, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
