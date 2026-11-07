"""Fit + evaluate the unsupervised health index (HI) and write models/health.json,
reports/health_index.json. Uses only NORMAL windows from loads 0-2 hp to fit and to
choose the alarm threshold; load 3 hp (normal + every fault) is the untouched test.

Two threshold rules are reported:
  * in_sample_p99 - 99th percentile of the HI of the very windows the model was fit on
                    (naive; optimistic because those windows define the distribution)
  * cross_load_p99 - for each training load k, fit on the other two training loads and score
                    load-k normal windows; pool the scores and take the 99th percentile.
                    This simulates "the machine runs at a load we have not seen". Served.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vibesense.data import CLASSES10  # noqa: E402
from vibesense.health import HealthIndex  # noqa: E402


def run(ds: dict) -> tuple[HealthIndex, dict]:
    from sklearn.metrics import roc_auc_score

    F, y, load, size = ds["F"], ds["y10"], ds["load"], ds["size"]
    normal_tr = (y == 0) & (load != 3)
    hi = HealthIndex.fit(F[normal_tr])
    in_sample = hi.threshold

    pooled = []
    for k in (0, 1, 2):
        fit_m = normal_tr & (load != k)
        held = normal_tr & (load == k)
        pooled.append(HealthIndex.fit(F[fit_m]).score(F[held]))
    cross = float(np.percentile(np.concatenate(pooled), 99))
    hi.threshold = cross

    te = load == 3
    s, yt = hi.score(F[te]), y[te]

    def rates(thr):
        return {"threshold": float(thr),
                "false_alarm_rate_normal_load3": float((s[yt == 0] > thr).mean()),
                "detection_rate_faults_load3": float((s[yt != 0] > thr).mean())}

    rep = {
        "fit_on": "normal windows, loads 0-2 hp only (no fault labels used)",
        "served_threshold_rule": "cross_load_p99",
        "auroc_normal_vs_fault_load3": float(roc_auc_score(yt != 0, s)),
        "thresholds": {"in_sample_p99": rates(in_sample), "cross_load_p99": rates(cross)},
        "mean_hi_by_fault_size_mils_load3": {str(sz): float(s[size[te] == sz].mean()) for sz in (0, 7, 14, 21)},
        "min_hi_faults_load3": float(s[yt != 0].min()),
        "max_hi_normal_load3": float(s[yt == 0].max()),
        "by_class_load3": {CLASSES10[c]: {"mean_hi": float(s[yt == c].mean()), "alarm_rate": float((s[yt == c] > cross).mean())}
                           for c in range(len(CLASSES10))},
    }
    # flat aliases used by the demo page
    rep["threshold_served"] = cross
    rep["false_alarm_rate_normal_load3"] = rep["thresholds"]["cross_load_p99"]["false_alarm_rate_normal_load3"]
    rep["detection_rate_faults_load3"] = rep["thresholds"]["cross_load_p99"]["detection_rate_faults_load3"]
    return hi, rep


def main() -> None:
    ds = dict(np.load(ROOT / "data" / "processed" / "windows.npz"))
    hi, rep = run(ds)
    (ROOT / "models" / "health.json").write_text(json.dumps(hi.to_dict()))
    (ROOT / "reports" / "health_index.json").write_text(json.dumps(rep, indent=1))
    sp = ROOT / "demo" / "assets" / "samples.json"
    if sp.exists():  # HI values per sample are unchanged (same fit data); only the alarm threshold moves
        s = json.loads(sp.read_text())
        s["threshold"] = hi.threshold
        sp.write_text(json.dumps(s))
    print(json.dumps({k: rep[k] for k in ("auroc_normal_vs_fault_load3", "thresholds")}, indent=1))


if __name__ == "__main__":
    main()
