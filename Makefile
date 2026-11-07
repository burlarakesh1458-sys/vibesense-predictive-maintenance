PY ?= python
export PYTHONPATH := src

.PHONY: setup setup-train data evaluate deploy health bench web-parity demo-smoke figures demo-data all test lint serve demo docker

setup:            ## runtime + test deps (serving uses committed ONNX/joblib models)
	$(PY) -m pip install -r requirements-dev.txt
setup-train:      ## + torch CPU, onnx, matplotlib for retraining
	$(PY) -m pip install -r requirements-train.txt
data:             ## download 40 CWRU .mat files, verify sha256, window + featurize
	$(PY) scripts/prepare_data.py
evaluate:         ## rf/gbm/cnn x random / leave-one-load-out / cross-size -> reports/metrics.json
	$(PY) scripts/evaluate.py
deploy:           ## train served models on loads 0-2, export ONNX, hold-out + latency + health reports
	$(PY) scripts/train_deploy.py
health:           ## refit/re-evaluate the health index only (cross-load calibrated threshold)
	$(PY) scripts/health_eval.py
bench:            ## single-window latency of the exported models -> reports/latency.json
	$(PY) scripts/bench_latency.py
web-parity:       ## run the browser path (onnxruntime-web WASM + demo/dsp.js) in node vs PyTorch outputs
	npm i --silent --prefix .cache/ortweb onnxruntime-web@1.22.0 && NODE_PATH=.cache/ortweb/node_modules node scripts/web_parity.cjs
demo-smoke:       ## headless jsdom test of index.html with the real onnxruntime-web model
	npm i --silent --prefix .cache/ortweb onnxruntime-web@1.22.0 jsdom@25 && NODE_PATH=.cache/ortweb/node_modules node scripts/demo_smoke.cjs
figures:          ## confusion matrices + health-index plot + RESULTS.md
	$(PY) scripts/make_figures.py
demo-data:        ## copy measured reports into demo/assets/results.json for the static page
	$(PY) scripts/build_demo_data.py
all: data evaluate deploy figures demo-data

test:
	$(PY) -m pytest
lint:
	ruff check src tests scripts
serve:
	$(PY) -m uvicorn vibesense.api:app --host 0.0.0.0 --port 8000
demo:             ## API + dashboard on http://localhost:8000 (static page also works via any file server)
	$(PY) -m uvicorn vibesense.api:app --host 127.0.0.1 --port 8000
docker:
	docker build -t vibesense:latest . && docker run --rm -p 8000:8000 vibesense:latest
