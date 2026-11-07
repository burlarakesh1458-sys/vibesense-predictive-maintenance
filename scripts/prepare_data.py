"""Download the 40 CWRU recordings (12 kHz drive-end, normal + IR/B/OR at 7/14/21 mils,
loads 0-3 hp), verify sha256 against data/cwru_sha256.json, window them and cache
features to data/processed/.

    python scripts/prepare_data.py [--raw DIR] [--write-hashes]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vibesense import features  # noqa: E402
from vibesense.data import CATALOG, build_dataset, sha256  # noqa: E402

HASHES = ROOT / "data" / "cwru_sha256.json"


def download(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for rec in CATALOG:
        dst = raw / f"{rec.file_id}.mat"
        if dst.exists() and dst.stat().st_size > 0:
            continue
        print(f"downloading {rec.url}")
        for attempt in range(3):
            try:
                urllib.request.urlretrieve(rec.url, dst)
                break
            except OSError as e:  # retry transient network errors
                print(f"  retry {attempt + 1}: {e}")
                time.sleep(2 * (attempt + 1))
        else:
            raise SystemExit(f"failed to download {rec.url}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed"))
    ap.add_argument("--write-hashes", action="store_true")
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    download(raw)

    hashes = {str(r.file_id): sha256(raw / f"{r.file_id}.mat") for r in CATALOG}
    if a.write_hashes or not HASHES.exists():
        HASHES.write_text(json.dumps(hashes, indent=1) + "\n")
    else:
        ref = json.loads(HASHES.read_text())
        bad = [k for k, v in hashes.items() if ref.get(k) != v]
        if bad:
            raise SystemExit(f"sha256 mismatch for {bad}; CWRU files changed upstream?")
        print("sha256 verified for", len(hashes), "files")

    t0 = time.perf_counter()
    ds = build_dataset(raw)
    ds["F"] = features.extract(ds["X"])
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "windows.npz", **ds)
    print(f"{len(ds['X'])} windows x {ds['X'].shape[1]} samples, {ds['F'].shape[1]} features "
          f"({time.perf_counter() - t0:.1f}s) -> {out / 'windows.npz'}")
    print("windows per class:", np.bincount(ds["y10"]).tolist())


if __name__ == "__main__":
    main()
