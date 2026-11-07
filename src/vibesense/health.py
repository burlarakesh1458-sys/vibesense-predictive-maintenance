"""Unsupervised health index: Mahalanobis distance of a window's features from the
healthy (normal-bearing) feature distribution. Fitted on normal data only, so it
needs no fault labels and can flag damage modes it has never seen.

HI = 0 at the healthy centroid, and the alarm threshold is the 99th percentile of
the healthy training windows' distance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class HealthIndex:
    mean: np.ndarray
    scale: np.ndarray
    precision: np.ndarray
    threshold: float

    @classmethod
    def fit(cls, F_normal: np.ndarray, q: float = 99.0) -> HealthIndex:
        from sklearn.covariance import LedoitWolf

        mean = F_normal.mean(0)
        scale = F_normal.std(0) + 1e-6
        Z = (F_normal - mean) / scale
        lw = LedoitWolf().fit(Z)
        obj = cls(mean, scale, lw.precision_, 0.0)
        obj.threshold = float(np.percentile(obj.score(F_normal), q))
        return obj

    def score(self, F: np.ndarray) -> np.ndarray:
        Z = (np.atleast_2d(F) - self.mean) / self.scale
        d2 = np.einsum("ij,jk,ik->i", Z, self.precision, Z)
        return np.sqrt(np.maximum(d2, 0.0))

    def to_dict(self) -> dict:
        return {
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "precision": self.precision.tolist(),
            "threshold": self.threshold,
        }

    @classmethod
    def from_dict(cls, d: dict) -> HealthIndex:
        return cls(np.asarray(d["mean"]), np.asarray(d["scale"]), np.asarray(d["precision"]), float(d["threshold"]))


def fit_trend(t: np.ndarray, hi: np.ndarray, threshold: float) -> dict:
    """Fit log(HI) = a + b t and extrapolate to the alarm threshold.

    A deliberately simple degradation model for a trend demo: returns slope, the
    projected crossing time and the time remaining from the last sample.
    """
    t = np.asarray(t, float)
    y = np.log(np.maximum(np.asarray(hi, float), 1e-6))
    b, a = np.polyfit(t, y, 1)
    out = {"a": float(a), "b": float(b), "crossing_t": None, "remaining": None}
    if b > 0:
        tc = (np.log(threshold) - a) / b
        out["crossing_t"] = float(tc)
        out["remaining"] = float(max(tc - t[-1], 0.0))
    return out
