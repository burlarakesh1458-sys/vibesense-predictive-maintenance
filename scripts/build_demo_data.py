"""Bundle the measured reports into demo/assets/results.json for the static page.
No numbers are created here - they are copied from reports/."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"


def main() -> None:
    runs = {p.stem: json.loads(p.read_text()) for p in (R / "runs").glob("*.json")}
    confusion = {}
    for task in ("10class", "4class"):
        for split in ("load3", "random", "cross_size"):
            for model in ("cnn", "gbm", "rf"):
                r = runs.get(f"{task}_{split}_{model}")
                if r:
                    confusion[f"{task}|{split}|{model}"] = {"classes": r["classes"], "matrix": r["confusion_matrix"]}
    dep = json.loads((R / "deploy_model.json").read_text())
    out = {
        "metrics": json.loads((R / "metrics.json").read_text()),
        "confusion": confusion,
        "deploy": {k: dep[k] for k in ("onnx_bytes", "cnn_params", "onnx_vs_torch_label_agreement")}
        | {"cnn_acc_load3": dep["cnn_onnx"]["accuracy"], "gbm_acc_load3": dep["gbm"]["accuracy"]},
        "health": json.loads((R / "health_index.json").read_text()),
        "latency": json.loads((R / "latency.json").read_text()),
        "web_parity": json.loads((R / "web_parity.json").read_text()) if (R / "web_parity.json").exists() else None,
    }
    dst = ROOT / "demo" / "assets" / "results.json"
    dst.write_text(json.dumps(out))
    print("wrote", dst, dst.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
