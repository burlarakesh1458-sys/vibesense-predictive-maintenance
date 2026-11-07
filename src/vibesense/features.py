"""Hand-crafted vibration features: time-domain statistics, FFT band energies,
and envelope-spectrum amplitudes at the bearing defect frequencies.

Everything here is plain numpy so the same code runs in training, the API and
the tests. Input is one window (or a batch of windows) sampled at FS.
"""

from __future__ import annotations

import numpy as np

from .data import FS

# SKF 6205-2RS JEM (CWRU drive-end bearing) defect frequencies as multiples of
# shaft speed, from the CWRU bearing-specification page.
DEFECT_ORDERS = {"BPFI": 5.4152, "BPFO": 3.5848, "BSF": 4.7135, "FTF": 0.39828}
DEFAULT_RPM = 1772.0  # mid-range of the 1730-1797 rpm rig; peaks are searched +-4 %
ENV_BAND = (2000.0, 5500.0)  # Hz, resonance band demodulated for the envelope
N_FFT_BANDS = 16
N_ENV_BANDS = 10

TIME_NAMES = [
    "mean", "std", "rms", "peak", "p2p", "skew", "kurtosis",
    "crest", "shape", "impulse", "clearance", "zero_cross",
]


def feature_names() -> list[str]:
    names = list(TIME_NAMES)
    names += [f"fft_band{i:02d}" for i in range(N_FFT_BANDS)]
    names += ["spec_centroid", "spec_spread", "spec_entropy", "log_total_power"]
    names += [f"env_band{i:02d}" for i in range(N_ENV_BANDS)]
    for k in DEFECT_ORDERS:
        names += [f"env_{k}_h{h}" for h in (1, 2, 3)]
    names += ["env_kurtosis", "band_rms_ratio"]
    return names


def _time_feats(x: np.ndarray) -> np.ndarray:
    mean = x.mean(1)
    xc = x - mean[:, None]
    std = xc.std(1) + 1e-12
    rms = np.sqrt((x**2).mean(1)) + 1e-12
    peak = np.abs(x).max(1)
    p2p = x.max(1) - x.min(1)
    skew = (xc**3).mean(1) / std**3
    kurt = (xc**4).mean(1) / std**4
    mabs = np.abs(x).mean(1) + 1e-12
    sqrt_amp = (np.sqrt(np.abs(x)).mean(1)) ** 2 + 1e-12
    zc = (np.diff(np.signbit(xc), axis=1) != 0).mean(1)
    return np.stack(
        [mean, std, rms, peak, p2p, skew, kurt, peak / rms, rms / mabs, peak / mabs, peak / sqrt_amp, zc],
        axis=1,
    )


def spectrum(x: np.ndarray, fs: float = FS) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed single-sided amplitude spectrum. x: (n, L) -> (freqs, (n, L//2+1))."""
    x = np.atleast_2d(x)
    L = x.shape[1]
    w = np.hanning(L)
    mag = np.abs(np.fft.rfft((x - x.mean(1, keepdims=True)) * w, axis=1)) * 2 / w.sum()
    return np.fft.rfftfreq(L, 1 / fs), mag


def envelope_spectrum(x: np.ndarray, fs: float = FS, band=ENV_BAND) -> tuple[np.ndarray, np.ndarray]:
    """Band-pass (FFT mask) + Hilbert envelope + amplitude spectrum of the envelope."""
    x = np.atleast_2d(x)
    L = x.shape[1]
    X = np.fft.fft(x - x.mean(1, keepdims=True), axis=1)
    f = np.fft.fftfreq(L, 1 / fs)
    # analytic signal restricted to the positive band: zero negatives and out-of-band
    h = np.zeros(L)
    h[(f >= band[0]) & (f <= band[1])] = 2.0
    env = np.abs(np.fft.ifft(X * h, axis=1))
    w = np.hanning(L)
    emag = np.abs(np.fft.rfft((env - env.mean(1, keepdims=True)) * w, axis=1)) * 2 / w.sum()
    return np.fft.rfftfreq(L, 1 / fs), emag


def _band_energy(mag: np.ndarray, freqs: np.ndarray, edges: np.ndarray) -> np.ndarray:
    p = mag**2
    tot = p.sum(1, keepdims=True) + 1e-20
    out = [p[:, (freqs >= lo) & (freqs < hi)].sum(1) for lo, hi in zip(edges[:-1], edges[1:], strict=True)]
    return np.log10(np.stack(out, 1) / tot + 1e-9)


def extract(x: np.ndarray, fs: float = FS, rpm: float | None = None) -> np.ndarray:
    """Feature matrix for windows x of shape (n, L) or (L,). Returns float32 (n, n_features)."""
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    shaft = (rpm or DEFAULT_RPM) / 60.0

    t = _time_feats(x)

    freqs, mag = spectrum(x, fs)
    edges = np.linspace(0, fs / 2, N_FFT_BANDS + 1)
    fb = _band_energy(mag, freqs, edges)
    p = mag**2 + 1e-20
    pn = p / p.sum(1, keepdims=True)
    centroid = (pn * freqs).sum(1)
    spread = np.sqrt((pn * (freqs - centroid[:, None]) ** 2).sum(1))
    entropy = -(pn * np.log(pn)).sum(1) / np.log(pn.shape[1])
    logp = np.log10(p.sum(1))
    spec = np.stack([centroid / (fs / 2), spread / (fs / 2), entropy, logp], 1)

    efreq, emag = envelope_spectrum(x, fs)
    eb = _band_energy(emag, efreq, np.linspace(0, 1000, N_ENV_BANDS + 1))
    ref = np.median(emag[:, (efreq > 5) & (efreq < 1000)], axis=1) + 1e-12
    defects = []
    for order in DEFECT_ORDERS.values():
        for h in (1, 2, 3):
            f0 = order * shaft * h
            m = (efreq >= f0 * 0.96) & (efreq <= f0 * 1.04)
            if not m.any():  # tolerance narrower than one bin: take the nearest bin
                m = np.zeros_like(efreq, dtype=bool)
                m[np.argmin(np.abs(efreq - f0))] = True
            defects.append(np.log10(emag[:, m].max(1) / ref + 1e-9))
    env_k = ((emag - emag.mean(1, keepdims=True)) ** 4).mean(1) / (emag.var(1) ** 2 + 1e-20)
    band_mask = (freqs >= ENV_BAND[0]) & (freqs <= ENV_BAND[1])
    band_ratio = np.sqrt((mag[:, band_mask] ** 2).sum(1) / ((mag**2).sum(1) + 1e-20))
    out = np.concatenate([t, fb, spec, eb, np.stack(defects, 1), np.stack([np.log10(env_k + 1e-9), band_ratio], 1)], 1)
    return np.nan_to_num(out).astype(np.float32)
