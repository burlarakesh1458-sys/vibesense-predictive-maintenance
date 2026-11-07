# Resume bullets — VibeSense (bearing fault diagnosis & predictive maintenance)

Every number comes from `reports/` (see README "Results").

- Built a bearing fault-diagnosis pipeline on the CWRU vibration dataset (40 recordings, 4,680 windows): 56 time/FFT/envelope-spectrum features with RandomForest/GBM vs a 23k-parameter 1D-CNN; the CNN reached **99.7 % accuracy on 10 fault classes at an unseen 3 hp motor load** (`reports/deploy_model.json`).
- Exposed evaluation leakage: models scoring **100 %** on the common random-window split dropped to **38–49 % fault-type accuracy** when tested on an unseen fault size (new damaged bearings), and documented why recording-level load splits still overestimate field performance (`reports/metrics.json`).
- Designed an unsupervised health index (Mahalanobis distance on healthy-only data) with a cross-load calibrated alarm threshold: **AUROC 1.000, 0 % false alarms and 100 % fault detection** at the unseen load, versus **62 % false alarms** with a naive in-sample threshold (`reports/health_index.json`).
- Exported the CNN to a **93.5 kB ONNX** model (max |Δp| 4.8e-7 vs PyTorch) serving in **0.12 ms/window** on one CPU thread via FastAPI, and in the browser with onnxruntime-web (**20/20** held-out demo windows correct, verified in node) (`reports/latency.json`, `reports/web_parity.json`).
- Shipped it as a tested product: FastAPI `/predict` + CSV upload + dashboard, Docker (non-root), GitHub Actions CI, 38 offline pytest tests including Python↔JS DSP and ONNX parity checks.
