// Headless smoke test of the static page: loads index.html in jsdom with the real demo/dsp.js,
// demo/app.js and onnxruntime-web (from NODE_PATH), serves demo/assets from disk, and checks that
// KPIs, the results table and a live prediction render without errors.
//   npm i onnxruntime-web@1.22.0 jsdom@25 && NODE_PATH=./node_modules node scripts/demo_smoke.cjs
const fs = require("fs");
const path = require("path");
const { JSDOM, VirtualConsole } = require("jsdom");
const ort = require("onnxruntime-web");
const ROOT = path.resolve(__dirname, "..");

const errors = [];
const vc = new VirtualConsole();
vc.on("jsdomError", (e) => errors.push(String(e)));
vc.on("error", (e) => errors.push(String(e)));
const html = fs.readFileSync(path.join(ROOT, "index.html"), "utf8").replace(/<script[^>]*src=[^>]*><\/script>/g, "");
const dom = new JSDOM(html, { runScripts: "outside-only", pretendToBeVisual: true, virtualConsole: vc });
const w = dom.window;
w.ort = ort;
ort.env.wasm.numThreads = 1;
// minimal canvas + fetch shims (jsdom has neither)
w.HTMLCanvasElement.prototype.getContext = () => new Proxy({}, { get: (t, k) => (k in t ? t[k] : () => {}), set: (t, k, v) => ((t[k] = v), true) });
w.fetch = async (u) => ({ json: async () => JSON.parse(fs.readFileSync(path.join(ROOT, u), "utf8")) });
const origCreate = ort.InferenceSession.create.bind(ort.InferenceSession);
ort.InferenceSession.create = (u, o) => origCreate(path.join(ROOT, u), o);
w.performance = performance;
// jsdom runs scripts in its own realm; share node typed arrays so ort sees the same constructors (a real browser has one realm)
for (const T of ["Float32Array", "Float64Array", "Uint8Array", "Int32Array"]) w[T] = globalThis[T];
w.eval(fs.readFileSync(path.join(ROOT, "demo", "dsp.js"), "utf8"));
w.eval(fs.readFileSync(path.join(ROOT, "demo", "app.js"), "utf8"));

const wait = (ms) => new Promise((r) => setTimeout(r, ms));
(async () => {
  for (let i = 0; i < 100 && !/confidence/.test(w.document.getElementById("result").textContent); i++) await wait(100);
  const doc = w.document;
  const out = {
    kpis: [...doc.querySelectorAll(".kpi .v")].map((e) => e.textContent),
    table_rows: doc.querySelectorAll("#resTable tr").length,
    options: doc.querySelectorAll("#sampleSel option").length,
    result: doc.getElementById("result").textContent.replace(/\s+/g, " ").trim().slice(0, 160),
    button: doc.getElementById("runBtn").textContent,
    errors,
  };
  console.log(JSON.stringify(out, null, 1));
  const ok = out.kpis.length === 4 && out.table_rows >= 4 && out.options === 20 && /correct/.test(out.result) && !errors.length;
  process.exit(ok ? 0 : 1);
})();
