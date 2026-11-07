"""Single-window inference latency of the EXPORTED artefacts (what the API serves).

Pins every library to 1 thread (OMP/MKL/onnxruntime) so numbers are per-core, and
records the 1-minute load average so readers can judge how noisy the box was.
    python scripts/bench_latency.py [--n 500]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vibesense import features  # noqa: E402
from vibesense.api import parse_csv  # noqa: E402
from vibesense.predictor import Predictor  # noqa: E402


def bench(fn, n, warm=30) -> dict:
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {"median_ms": round(statistics.median(ts), 3), "p95_ms": round(ts[int(0.95 * n) - 1], 3), "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    p = Predictor()
    x = parse_csv((ROOT / "tests" / "fixtures" / "OR007_load3_0.csv").read_text())[:2048][None, :]
    f = features.extract(x)
    load_before = os.getloadavg()[0]
    out = {
        "note": "one 2048-sample window, 1 thread (OMP/MKL/ORT pinned), shared 2-vCPU sandbox with other jobs running",
        "cnn_onnxruntime": bench(lambda: p.proba(x, "cnn"), a.n),
        "feature_extraction": bench(lambda: features.extract(x), a.n),
        "gbm_predict_given_features": bench(lambda: p.gbm.predict_proba(f), a.n),
        "gbm_end_to_end": bench(lambda: p.proba(x, "gbm"), a.n),
        "health_index_end_to_end": bench(lambda: p.health.score(features.extract(x)), a.n),
        "full_predict_call_cnn": bench(lambda: p.predict(x[0], model="cnn"), a.n),
        "loadavg_1min_before_after": [round(load_before, 2), round(os.getloadavg()[0], 2)],
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }
    (ROOT / "reports" / "latency.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    assert np.isfinite(f).all()


if __name__ == "__main__":
    main()
