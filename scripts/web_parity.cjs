// Run the demo's exact browser path (demo/dsp.js z-score + onnxruntime-web WASM + demo/assets/cnn10.onnx)
// in node on the 20 demo samples and compare with the PyTorch probabilities stored at export.
//   npm i onnxruntime-web@1.22.0 && NODE_PATH=./node_modules node scripts/web_parity.cjs
const ort = require("onnxruntime-web");
const fs = require("fs");
const path = require("path");
const ROOT = path.resolve(__dirname, "..");
const dsp = require(path.join(ROOT, "demo", "dsp.js"));

(async () => {
  ort.env.wasm.numThreads = 1;
  const sess = await ort.InferenceSession.create(path.join(ROOT, "demo", "assets", "cnn10.onnx"), { executionProviders: ["wasm"] });
  const S = JSON.parse(fs.readFileSync(path.join(ROOT, "demo", "assets", "samples.json")));
  let maxDiff = 0, agree = 0, correct = 0;
  const ms = [];
  for (const s of S.samples) {
    const t0 = performance.now();
    const out = await sess.run({ signal: new ort.Tensor("float32", dsp.zscore(s.samples), [1, 1, S.window]) });
    ms.push(performance.now() - t0);
    const p = dsp.softmax(Array.from(out.logits.data));
    p.forEach((v, i) => { maxDiff = Math.max(maxDiff, Math.abs(v - s.py_cnn_proba[i])); });
    const a = p.indexOf(Math.max(...p)), b = s.py_cnn_proba.indexOf(Math.max(...s.py_cnn_proba));
    agree += a === b; correct += S.classes[a] === s.label;
  }
  ms.sort((x, y) => x - y);
  const rep = {
    runtime: `onnxruntime-web ${(ort.env.versions && ort.env.versions.web) || "unknown"} (wasm, 1 thread) in node ${process.version}`,
    n_samples: S.samples.length,
    max_abs_prob_diff_vs_pytorch: maxDiff,
    label_agreement_vs_pytorch: agree / S.samples.length,
    correct_vs_ground_truth: correct / S.samples.length,
    median_ms_per_window: ms[Math.floor(ms.length / 2)],
  };
  fs.writeFileSync(path.join(ROOT, "reports", "web_parity.json"), JSON.stringify(rep, null, 1));
  console.log(JSON.stringify(rep));
})().catch((e) => { console.error(e); process.exit(1); });
