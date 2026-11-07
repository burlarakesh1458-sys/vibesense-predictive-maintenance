"""CWRU bearing data catalogue, loading and windowing.

All recordings are the 12 kHz drive-end (DE) accelerometer channel of the CWRU
Bearing Data Center test rig (SKF 6205-2RS JEM drive-end bearing). Faults were
seeded by EDM at 0.007, 0.014 and 0.021 inch; outer-race faults are the
centred (6 o'clock) position.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FS = 12_000  # Hz, drive-end channel used for every file below
WINDOW = 2048  # samples (~0.17 s, 5.86 Hz FFT resolution)
STRIDE = 1024  # 50 % overlap
MAX_SAMPLES = 121_000  # cap per recording so the 40 s normal files don't swamp the 10 s fault files
NOMINAL_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
BASE_URL = "https://engineering.case.edu/sites/default/files/{file_id}.mat"

CLASSES10 = [
    "Normal",
    "IR007", "IR014", "IR021",
    "B007", "B014", "B021",
    "OR007", "OR014", "OR021",
]
CLASSES4 = ["Normal", "InnerRace", "Ball", "OuterRace"]
TYPE_OF = {"Normal": "Normal", "IR": "InnerRace", "B": "Ball", "OR": "OuterRace"}


@dataclass(frozen=True)
class Recording:
    file_id: int
    label10: str
    load_hp: int

    @property
    def fault_type(self) -> str:
        if self.label10 == "Normal":
            return "Normal"
        return TYPE_OF[self.label10.rstrip("0123456789")]

    @property
    def fault_size_mils(self) -> int:
        return 0 if self.label10 == "Normal" else int(self.label10[-3:])

    @property
    def url(self) -> str:
        return BASE_URL.format(file_id=self.file_id)


def _block(label: str, first_id: int) -> list[Recording]:
    return [Recording(first_id + load, label, load) for load in range(4)]


CATALOG: list[Recording] = [
    *_block("Normal", 97),
    *_block("IR007", 105), *_block("B007", 118), *_block("OR007", 130),
    *_block("IR014", 169), *_block("B014", 185), *_block("OR014", 197),
    *_block("IR021", 209), *_block("B021", 222), *_block("OR021", 234),
]


def label_to_type(label10: str) -> str:
    return Recording(0, label10, 0).fault_type


def load_signal(path: Path, file_id: int) -> tuple[np.ndarray, float | None]:
    """Return (DE signal float32, RPM or None).

    Quirks handled: 99.mat also contains the X098 arrays, and 98.mat has no RPM.
    """
    import scipy.io as sio

    mat = sio.loadmat(str(path))
    key = f"X{file_id:03d}_DE_time"
    if key not in mat:  # fall back to the single DE array in the file
        cands = [k for k in mat if k.endswith("_DE_time")]
        if len(cands) != 1:
            raise KeyError(f"{path}: cannot find {key} (have {cands})")
        key = cands[0]
    rpm_key = f"X{file_id:03d}RPM"
    rpm = float(np.asarray(mat[rpm_key]).ravel()[0]) if rpm_key in mat else None
    return np.asarray(mat[key], dtype=np.float32).ravel(), rpm


def make_windows(x: np.ndarray, window: int = WINDOW, stride: int = STRIDE) -> np.ndarray:
    """Slice a 1-D signal into overlapping windows -> (n, window)."""
    x = np.asarray(x, dtype=np.float32).ravel()
    if len(x) < window:
        raise ValueError(f"signal has {len(x)} samples, need >= {window}")
    n = 1 + (len(x) - window) // stride
    idx = np.arange(window)[None, :] + stride * np.arange(n)[:, None]
    return x[idx]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_dataset(raw_dir: Path) -> dict[str, np.ndarray]:
    """Window every catalogued recording. Returns arrays keyed by name."""
    X, y10, loads, fids, rpms, pos = [], [], [], [], [], []
    for rec in CATALOG:
        sig, rpm = load_signal(raw_dir / f"{rec.file_id}.mat", rec.file_id)
        sig = sig[:MAX_SAMPLES]
        w = make_windows(sig)
        X.append(w)
        y10 += [CLASSES10.index(rec.label10)] * len(w)
        loads += [rec.load_hp] * len(w)
        fids += [rec.file_id] * len(w)
        rpms += [rpm or NOMINAL_RPM[rec.load_hp]] * len(w)
        pos += list(range(len(w)))
    y10 = np.asarray(y10)
    y4 = np.asarray([CLASSES4.index(label_to_type(CLASSES10[i])) for i in y10])
    sizes = np.asarray([0 if c == 0 else int(CLASSES10[c][-3:]) for c in y10])
    return {
        "X": np.concatenate(X).astype(np.float32),
        "y10": y10,
        "y4": y4,
        "load": np.asarray(loads),
        "file_id": np.asarray(fids),
        "rpm": np.asarray(rpms, dtype=np.float32),
        "size": sizes,
        "pos": np.asarray(pos),
    }
