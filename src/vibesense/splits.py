"""Evaluation splits.

* ``random``      – window-level stratified 70/30 shuffle. Overlapping windows from the
                    SAME recording land on both sides, so this is the optimistic, leaky
                    protocol many CWRU papers report.
* ``load{k}``     – leave-one-load-out: train on the recordings at the other three motor
                    loads, test on every recording at load k (0-3 hp). No recording is
                    shared. ``load3`` is the headline "train 0-2 hp, test 3 hp" split.
* ``cross_size``  – 4-class only: faults train on 0.007" + 0.021" damage and test on the
                    unseen 0.014" damage; normal data is split by load (0-2 train / 3 test).
                    This is the hardest, most deployment-like test because the test
                    bearings were never seen in training.
"""

from __future__ import annotations

import numpy as np


def split_indices(name: str, ds: dict[str, np.ndarray], seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    n = len(ds["y10"])
    idx = np.arange(n)
    if name == "random":
        from sklearn.model_selection import train_test_split

        tr, te = train_test_split(idx, test_size=0.3, random_state=seed, stratify=ds["y10"])
        return np.sort(tr), np.sort(te)
    if name.startswith("load"):
        k = int(name[4:])
        te = ds["load"] == k
        return idx[~te], idx[te]
    if name == "cross_size":
        normal = ds["size"] == 0
        te = (ds["size"] == 14) | (normal & (ds["load"] == 3))
        tr = ((ds["size"] == 7) | (ds["size"] == 21)) | (normal & (ds["load"] != 3))
        return idx[tr], idx[te]
    raise ValueError(f"unknown split {name!r}")


def assert_no_recording_overlap(ds: dict[str, np.ndarray], tr: np.ndarray, te: np.ndarray) -> None:
    shared = set(np.unique(ds["file_id"][tr])) & set(np.unique(ds["file_id"][te]))
    if shared:
        raise AssertionError(f"recordings on both sides of split: {sorted(shared)}")
