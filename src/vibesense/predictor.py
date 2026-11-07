"""Inference wrapper used by the API: feature GBM (joblib) + 1D-CNN (ONNX) + health index."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import features
from .cnn import normalize
from .data import CLASSES10, FS, WINDOW, label_to_type
from .health import HealthIndex

MODEL_DIR = Path(os.environ.get("VIBESENSE_MODELS", Path(__file__).resolve().parents[2] / "models"))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class Predictor:
    def __init__(self, model_dir: Path = MODEL_DIR):
        import joblib
        import onnxruntime as ort

        self.model_dir = Path(model_dir)
        self.meta = json.loads((self.model_dir / "meta.json").read_text())
        self.gbm = joblib.load(self.model_dir / "gbm10.joblib")
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        self.cnn = ort.InferenceSession(str(self.model_dir / "cnn10.onnx"), so, providers=["CPUExecutionProvider"])
        self.health = HealthIndex.from_dict(json.loads((self.model_dir / "health.json").read_text()))
        self.classes = self.meta.get("classes", CLASSES10)

    @staticmethod
    def prepare(samples, fs: float = FS) -> np.ndarray:
        x = np.asarray(samples, dtype=np.float32).ravel()
        if not np.all(np.isfinite(x)):
            raise ValueError("samples contain NaN/inf")
        if fs != FS:  # linear resample to the training rate
            n = int(round(len(x) * FS / fs))
            x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
        if len(x) < WINDOW:
            raise ValueError(f"need at least {WINDOW} samples at {FS} Hz (got {len(x)})")
        return x[:WINDOW]

    def proba(self, x: np.ndarray, model: str = "cnn") -> np.ndarray:
        x = np.atleast_2d(x).astype(np.float32)
        if model == "cnn":
            logits = self.cnn.run(None, {"signal": normalize(x)[:, None, :]})[0]
            return _softmax(logits)
        if model == "gbm":
            return self.gbm.predict_proba(features.extract(x))
        raise ValueError("model must be 'cnn' or 'gbm'")

    def predict(self, samples, fs: float = FS, model: str = "cnn") -> dict:
        x = self.prepare(samples, fs)
        p = self.proba(x, model)[0]
        i = int(p.argmax())
        label = self.classes[i]
        hi = float(self.health.score(features.extract(x))[0])
        type_p: dict[str, float] = {}
        for c, pc in zip(self.classes, p, strict=True):
            type_p[label_to_type(c)] = type_p.get(label_to_type(c), 0.0) + float(pc)
        return {
            "model": model,
            "label": label,
            "fault_type": label_to_type(label),
            "fault_size_inch": None if label == "Normal" else int(label[-3:]) / 1000,
            "confidence": float(p[i]),
            "probabilities": {c: round(float(v), 5) for c, v in zip(self.classes, p, strict=True)},
            "type_probabilities": {k: round(v, 5) for k, v in type_p.items()},
            "health_index": round(hi, 3),
            "health_threshold": round(self.health.threshold, 3),
            "anomalous": bool(hi > self.health.threshold),
            "window_samples": WINDOW,
            "fs_hz": FS,
        }
