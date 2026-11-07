import numpy as np
import pytest

from vibesense import features
from vibesense.data import CATALOG, CLASSES10, FS, WINDOW, label_to_type, make_windows
from vibesense.health import HealthIndex, fit_trend
from vibesense.splits import assert_no_recording_overlap, split_indices


def test_catalog_is_40_unique_recordings_balanced():
    assert len(CATALOG) == 40
    assert len({r.file_id for r in CATALOG}) == 40
    for label in CLASSES10:
        assert sorted(r.load_hp for r in CATALOG if r.label10 == label) == [0, 1, 2, 3]


@pytest.mark.parametrize("label,typ", [("Normal", "Normal"), ("IR007", "InnerRace"), ("B021", "Ball"), ("OR014", "OuterRace")])
def test_label_to_type(label, typ):
    assert label_to_type(label) == typ


def test_make_windows_shapes_and_overlap():
    x = np.arange(10_000, dtype=np.float32)
    w = make_windows(x, 2048, 1024)
    assert w.shape == (1 + (10_000 - 2048) // 1024, 2048)
    assert w[1, 0] == 1024 and w[0, 1024] == w[1, 0]
    with pytest.raises(ValueError):
        make_windows(np.zeros(100), 2048, 1024)


def test_feature_vector_matches_names_and_is_finite():
    x = np.random.default_rng(0).normal(size=(3, WINDOW))
    f = features.extract(x)
    assert f.shape == (3, len(features.feature_names()))
    assert np.isfinite(f).all()


def test_time_stats_of_pure_sine():
    t = np.arange(WINDOW * 4) / FS
    x = np.sin(2 * np.pi * 93.75 * t)[:WINDOW]  # integer number of cycles
    f = dict(zip(features.feature_names(), features.extract(x)[0], strict=True))
    assert f["crest"] == pytest.approx(np.sqrt(2), rel=0.02)
    assert f["kurtosis"] == pytest.approx(1.5, rel=0.03)


def _impulse_train(rate_hz, rng):
    """Decaying 3.5 kHz resonance excited at rate_hz - a synthetic outer-race fault."""
    n = WINDOW
    t = np.arange(n) / FS
    x = 0.05 * rng.normal(size=n)
    ring = np.exp(-t[:120] * 1500) * np.sin(2 * np.pi * 3500 * t[:120])
    for k in np.arange(0, n / FS, 1 / rate_hz):
        i = int(k * FS)
        seg = ring[: n - i]
        x[i : i + len(seg)] += seg
    return x


def test_envelope_feature_finds_bpfo_impulses():
    rng = np.random.default_rng(1)
    shaft = features.DEFAULT_RPM / 60
    bpfo = features.DEFECT_ORDERS["BPFO"] * shaft
    names = features.feature_names()
    j_o, j_i = names.index("env_BPFO_h1"), names.index("env_BPFI_h1")
    f_fault = features.extract(_impulse_train(bpfo, rng))[0]
    f_noise = features.extract(0.05 * rng.normal(size=WINDOW))[0]
    assert f_fault[j_o] > f_noise[j_o] + 0.5  # >3x above the envelope-spectrum median
    assert f_fault[j_o] > f_fault[j_i]
    assert f_fault[names.index("kurtosis")] > 3.5


def _toy_ds():
    loads, fids, y10, size = [], [], [], []
    for r in CATALOG:
        for _ in range(5):
            loads.append(r.load_hp)
            fids.append(r.file_id)
            y10.append(CLASSES10.index(r.label10))
            size.append(r.fault_size_mils)
    return {k: np.asarray(v) for k, v in dict(load=loads, file_id=fids, y10=y10, size=size).items()}


@pytest.mark.parametrize("split", ["load0", "load1", "load2", "load3", "cross_size"])
def test_leakage_free_splits_share_no_recording(split):
    ds = _toy_ds()
    tr, te = split_indices(split, ds)
    assert len(tr) and len(te) and len(set(tr) & set(te)) == 0
    assert_no_recording_overlap(ds, tr, te)


def test_cross_size_holds_out_014_only():
    ds = _toy_ds()
    tr, te = split_indices("cross_size", ds)
    assert set(ds["size"][te]) == {0, 14} and 14 not in set(ds["size"][tr])


def test_random_split_does_leak_recordings():
    ds = _toy_ds()
    tr, te = split_indices("random", ds)
    with pytest.raises(AssertionError):
        assert_no_recording_overlap(ds, tr, te)


def test_health_index_threshold_and_shift():
    rng = np.random.default_rng(0)
    normal = rng.normal(size=(2000, 6))
    hi = HealthIndex.fit(normal)
    assert (hi.score(normal) > hi.threshold).mean() == pytest.approx(0.01, abs=0.005)
    assert hi.score(normal[:200] + 3).mean() > 2 * hi.score(normal[:200]).mean()
    assert np.allclose(HealthIndex.from_dict(hi.to_dict()).score(normal[:5]), hi.score(normal[:5]))


def test_trend_extrapolation_recovers_exponential():
    t = np.arange(50.0)
    out = fit_trend(t, 0.5 * np.exp(0.05 * t), threshold=10.0)
    assert out["b"] == pytest.approx(0.05, rel=1e-3)
    assert out["crossing_t"] == pytest.approx(np.log(20) / 0.05, rel=1e-3)
    assert fit_trend(t, np.ones_like(t) * 0.5, 10.0)["remaining"] is None
