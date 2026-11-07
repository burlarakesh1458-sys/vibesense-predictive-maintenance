"""The browser demo's JS DSP (demo/dsp.js) must match the Python feature code. Skipped if node is absent."""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from vibesense.cnn import normalize
from vibesense.features import envelope_spectrum, spectrum

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def test_js_spectrum_envelope_zscore_match_python():
    rng = np.random.default_rng(3)
    t = np.arange(2048) / 12000
    am = 0.2 * np.sin(2 * np.pi * 3500 * t) * (1 + np.sin(2 * np.pi * 107 * t))  # modulated resonance
    x = np.sin(2 * np.pi * 157 * t) + 0.3 * rng.normal(size=2048) + am
    js = f"""
    const d = require({json.dumps(str(ROOT / "demo" / "dsp.js"))});
    const x = {json.dumps(x.tolist())};
    const r = {{amp: Array.from(d.ampSpectrum(x)), env: Array.from(d.envelope(x)), z: Array.from(d.zscore(x))}};
    console.log(JSON.stringify(r));
    """
    out = json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)
    _, amp = spectrum(x[None])
    _, env = envelope_spectrum(x[None])
    assert np.allclose(out["amp"], amp[0], atol=1e-9)
    assert np.allclose(out["env"], env[0], atol=1e-9)
    assert np.allclose(out["z"], normalize(x)[0], atol=1e-4)
