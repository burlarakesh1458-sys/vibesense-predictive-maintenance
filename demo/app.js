/* VibeSense in-browser demo: onnxruntime-web 1D-CNN + JS FFT / envelope spectrum. */
"use strict";
const { FS, fft, ampSpectrum, envelope, zscore, resample, softmax } = window.DSP;
const WIN = 2048, RPM = 1772;
const ORDERS = { BPFO: 3.5848, BPFI: 5.4152, BSF: 4.7135 };
const COLORS = { BPFO: "#ffb547", BPFI: "#ff6b6b", BSF: "#b28dff" };
const $ = (id) => document.getElementById(id);
let session = null, SAMPLES = null, RESULTS = null, current = null;

/* ---------- drawing ---------- */
function setupCanvas(c) {
  const dpr = window.devicePixelRatio || 1, w = c.clientWidth || c.width, h = Math.round(w * c.height / c.width);
  if (c._w !== w) { c.style.height = h + "px"; c.width = w * dpr; c.height = h * dpr; c._w = w; c._h = h; }
  const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  return [g, w, c._h || h];
}
function line(c, ys, { xmax, xlabel, log = false, markers = null, color = "#4cc2ff" } = {}) {
  const [g, w, h] = setupCanvas(c), padL = 38, padB = 18, W = w - padL - 6, H = h - padB - 6;
  let v = Array.from(ys); if (log) v = v.map((a) => Math.log10(a + 1e-9));
  let lo = Math.min(...v), hi = Math.max(...v); if (hi - lo < 1e-12) hi = lo + 1;
  g.strokeStyle = "#1f2a36"; g.lineWidth = 1;
  for (let i = 0; i <= 4; i++) { const y = 6 + H * i / 4; g.beginPath(); g.moveTo(padL, y); g.lineTo(padL + W, y); g.stroke(); }
  g.fillStyle = "#8b98a7"; g.font = "10px sans-serif";
  if (xmax) for (let i = 0; i <= 4; i++) { const xv = xmax * i / 4; g.fillText((xv >= 1000 ? (xv / 1000) + "k" : Math.round(xv)) + (i === 4 ? " " + xlabel : ""), padL + W * i / 4 - (i === 4 ? 30 : 6), h - 4); }
  g.fillText(log ? "log" : hi.toFixed(2), 2, 14);
  if (markers) for (const [name, f] of markers) {
    if (f > xmax) continue; const x = padL + W * f / xmax;
    g.strokeStyle = COLORS[name.replace(/\d$/, "")]; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(x, 6); g.lineTo(x, 6 + H); g.stroke(); g.setLineDash([]);
  }
  g.strokeStyle = color; g.lineWidth = 1.2; g.beginPath();
  for (let i = 0; i < v.length; i++) { const x = padL + W * i / (v.length - 1), y = 6 + H * (1 - (v[i] - lo) / (hi - lo)); i ? g.lineTo(x, y) : g.moveTo(x, y); }
  g.stroke();
}
function drawSignal(x) {
  line($("cWave"), x, { xmax: x.length / FS * 1000, xlabel: "ms" });
  const sp = ampSpectrum(x); line($("cFft"), sp, { xmax: FS / 2, xlabel: "Hz", log: true });
  const env = envelope(x), df = FS / x.length, nmax = Math.floor(500 / df);
  const shaft = RPM / 60, mk = [];
  for (const [k, o] of Object.entries(ORDERS)) for (const h of [1, 2]) mk.push([k + h, o * shaft * h]);
  line($("cEnv"), env.slice(1, nmax + 1), { xmax: nmax * df, xlabel: "Hz", markers: mk, color: "#3ddc97" });
}
function bars(probs, classes) {
  const order = classes.map((c, i) => [c, probs[i]]).sort((a, b) => b[1] - a[1]);
  return '<div class="bars">' + order.map(([c, p], i) =>
    `<div class="bar ${i === 0 ? "top" : ""}"><span>${c}</span><div class="t"><div class="f" style="width:${(p * 100).toFixed(1)}%"></div></div><span>${(p * 100).toFixed(1)}%</span></div>`).join("") + "</div>";
}
const TYPE = { Normal: "Normal", IR: "Inner race", B: "Ball", OR: "Outer race" };
const describe = (lbl) => lbl === "Normal" ? "Healthy bearing" : `${TYPE[lbl.replace(/\d+$/, "")]} fault · ${(+lbl.slice(-3) / 1000).toFixed(3)} in`;

/* ---------- inference ---------- */
async function run() {
  if (!current || !session) return;
  const btn = $("runBtn"); btn.disabled = true; btn.textContent = "Running…";
  try {
    const fs = +$("fsIn").value || FS;
    let x = resample(current.samples, current.fromSample ? FS : fs);
    if (x.length < WIN) throw new Error(`need ≥ ${WIN} samples at 12 kHz after resampling (have ${x.length})`);
    x = Array.from(x.slice(0, WIN));
    drawSignal(x);
    const t0 = performance.now();
    const out = await session.run({ signal: new ort.Tensor("float32", zscore(x), [1, 1, WIN]) });
    const ms = performance.now() - t0;
    const p = softmax(Array.from(out.logits.data)), classes = SAMPLES.classes;
    const i = p.indexOf(Math.max(...p)), lbl = classes[i];
    let html = `<div class="sub">1D-CNN (ONNX, in-browser) · ${ms.toFixed(1)} ms</div>
      <div class="label">${lbl}</div><div>${describe(lbl)}</div>
      <div style="margin-top:6px">confidence <b>${(p[i] * 100).toFixed(1)}%</b>
      <span class="badge ${lbl === "Normal" ? "b-ok" : "b-bad"}">${lbl === "Normal" ? "OK" : "FAULT"}</span></div>`;
    if (current.meta) {
      const m = current.meta, ok = m.label === lbl;
      const gi = m.py_gbm_proba.indexOf(Math.max(...m.py_gbm_proba));
      html += `<div class="sub" style="margin-top:10px">Ground truth: <b>${m.label}</b> (CWRU file ${m.file_id}.mat, 3 hp, sample ${m.start_sample}) —
        <span class="badge ${ok ? "b-ok" : "b-bad"}">${ok ? "correct" : "wrong"}</span></div>
        <div class="sub">Python GBM on 56 features: <b>${classes[gi]}</b> (${(m.py_gbm_proba[gi] * 100).toFixed(1)}%) ·
        health index <b>${m.health_index.toFixed(1)}</b> <span class="badge ${m.health_index > SAMPLES.threshold ? "b-warn" : "b-ok"}">${m.health_index > SAMPLES.threshold ? "above" : "below"} alarm ${SAMPLES.threshold.toFixed(1)}</span></div>`;
    } else {
      html += `<div class="sub" style="margin-top:10px">Uploaded signal · first 2048 samples after resampling ${fs} → 12000 Hz. Model trained on CWRU drive-end data only; other machines are out of distribution.</div>`;
    }
    html += bars(p, classes);
    $("result").innerHTML = html;
  } catch (e) {
    $("result").innerHTML = `<div class="b-bad badge">error</div><div class="sub">${e.message}</div>`;
  } finally { btn.disabled = false; btn.textContent = "Run diagnosis"; }
}

function pickSample() {
  const s = SAMPLES.samples[+$("sampleSel").value];
  current = { samples: s.samples, meta: s, fromSample: true };
  $("fsIn").value = FS; drawSignal(s.samples); run();
}
function onCsv(ev) {
  const f = ev.target.files[0]; if (!f) return;
  f.text().then((t) => {
    const vals = [];
    for (const ln of t.split(/\r?\n/)) { const c = ln.replace(/[;\t]/g, ",").split(",")[0].trim(); if (c && !isNaN(+c)) vals.push(+c); }
    if (vals.length < 256) { $("result").innerHTML = `<div class="sub">CSV has only ${vals.length} numeric rows.</div>`; return; }
    current = { samples: vals, meta: null, fromSample: false }; run();
  });
}

/* ---------- results explorer ---------- */
const pct = (v) => (v * 100).toFixed(1) + "%";
const SPLIT_NAMES = { random: "Random windows (leaky)", load3: "Train 0–2 hp → test 3 hp", lolo_mean: "Leave-one-load-out (mean of 4)", cross_size: "Unseen fault size (14 mil)" };
const MODEL_NAMES = { rf: "Random forest (features)", gbm: "Gradient boosting (features)", cnn: "1D-CNN (raw signal)" };
function renderTable(task) {
  const t = RESULTS.metrics[task]; if (!t) return;
  const cols = ["random", "load3", "lolo_mean"].concat(task === "4class" ? ["cross_size"] : []);
  const best = {};
  for (const c of cols) best[c] = Math.max(...Object.values(t).map((m) => m[c] ? m[c].macro_f1 : 0));
  let h = "<tr><th>Model</th>" + cols.map((c) => `<th>${SPLIT_NAMES[c]}<br><span class="muted">acc · macro-F1</span></th>`).join("") + "</tr>";
  for (const m of ["rf", "gbm", "cnn"]) {
    if (!t[m]) continue;
    h += `<tr><td>${MODEL_NAMES[m]}</td>` + cols.map((c) => {
      const r = t[m][c]; if (!r) return "<td>—</td>";
      const cls = c === "random" ? "leaky" : (r.macro_f1 === best[c] ? "best" : "");
      return `<td class="${cls}">${pct(r.accuracy)} · ${pct(r.macro_f1)}</td>`;
    }).join("") + "</tr>";
  }
  $("resTable").innerHTML = h;
  document.querySelectorAll("#taskTabs button").forEach((b) => b.classList.toggle("on", b.dataset.t === task));
}
function renderCm(key) {
  const cm = RESULTS.confusion[key]; if (!cm) return;
  const c = $("cCm"), [g, w] = setupCanvas(c), n = cm.classes.length, pad = 64, cell = (w - pad - 8) / n;
  const rows = cm.matrix.map((r) => { const s = r.reduce((a, b) => a + b, 0) || 1; return r.map((v) => v / s); });
  g.font = "11px sans-serif";
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const v = rows[i][j]; g.fillStyle = i === j ? `rgba(61,220,151,${0.12 + 0.88 * v})` : `rgba(255,107,107,${v > 0 ? 0.12 + 0.88 * v : 0.04})`;
    g.fillRect(pad + j * cell + 1, pad + i * cell + 1, cell - 2, cell - 2);
    if (cm.matrix[i][j]) { g.fillStyle = "#e6edf3"; g.textAlign = "center"; g.fillText(cm.matrix[i][j], pad + j * cell + cell / 2, pad + i * cell + cell / 2 + 4); }
  }
  g.fillStyle = "#8b98a7"; g.textAlign = "right";
  cm.classes.forEach((nm, i) => g.fillText(nm, pad - 4, pad + i * cell + cell / 2 + 4));
  g.save(); g.textAlign = "left";
  cm.classes.forEach((nm, j) => { g.save(); g.translate(pad + j * cell + cell / 2 + 4, pad - 4); g.rotate(-Math.PI / 3); g.fillText(nm, 0, 0); g.restore(); });
  g.restore(); g.textAlign = "left"; g.fillText("true ↓  predicted →", 2, 12);
}
function renderSide() {
  const h = RESULTS.health, l = RESULTS.latency;
  $("hiBox").innerHTML = `<div class="kv">
    <div>Fit on</div><div>healthy windows, 0–2 hp</div>
    <div>Alarm threshold (cross-load p99)</div><div>${h.threshold_served.toFixed(1)}</div>
    <div>AUROC healthy vs faulty (3 hp)</div><div><b>${h.auroc_normal_vs_fault_load3.toFixed(3)}</b></div>
    <div>False-alarm rate, healthy 3 hp</div><div>${pct(h.false_alarm_rate_normal_load3)}</div>
    <div>Fault detection rate at threshold</div><div>${pct(h.detection_rate_faults_load3)}</div>
    ${Object.entries(h.mean_hi_by_fault_size_mils_load3).map(([s, v]) => `<div>Mean HI, ${s === "0" ? "healthy" : s + " mil faults"}</div><div>${v.toFixed(1)}</div>`).join("")}
  </div>`;
  const row = (k, nm) => l[k] ? `<div>${nm}</div><div>${l[k].median_ms.toFixed(2)} ms <span class="muted">(p95 ${l[k].p95_ms.toFixed(2)})</span></div>` : "";
  $("latBox").innerHTML = `<div class="kv">${row("cnn_onnxruntime", "CNN, onnxruntime (Python)")}${row("full_predict_call_cnn", "Full /predict path (CNN + health index)")}
    ${row("feature_extraction", "56-feature extraction")}${row("gbm_end_to_end", "Features + GBM")}</div>
    <div class="sub muted" style="margin-top:6px">${l.note}; load average ${l.loadavg_1min_before_after[0]} on ${l.cpu_count} vCPUs.</div>
    ${RESULTS.web_parity ? `<div class="sub muted" style="margin-top:6px">Browser path check (${RESULTS.web_parity.runtime}): ${RESULTS.web_parity.n_samples}/${RESULTS.web_parity.n_samples} demo samples give the same label as PyTorch, max probability difference ${RESULTS.web_parity.max_abs_prob_diff_vs_pytorch.toExponential(1)}.</div>` : ""}`;
}
function renderKpis() {
  const t = RESULTS.metrics["10class"], d = RESULTS.deploy;
  const k = [
    [pct(t.cnn.load3.accuracy), "CNN accuracy, unseen 3 hp load", "10 classes · no shared recordings"],
    [pct(t.cnn.random.accuracy), "same CNN, leaky random split", "what most CWRU papers report"],
    [pct(RESULTS.metrics["4class"].gbm.cross_size ? Math.max(...["rf", "gbm", "cnn"].map((m) => RESULTS.metrics["4class"][m].cross_size.accuracy)) : 0), "best fault-TYPE accuracy, unseen 14-mil faults", "hardest split: new damaged bearings"],
    [RESULTS.latency.cnn_onnxruntime.median_ms.toFixed(2) + " ms", "CNN inference / window (1 CPU thread)", `${(d.onnx_bytes / 1024).toFixed(0)} KB ONNX · ${d.cnn_params.toLocaleString()} params`],
  ];
  $("kpis").innerHTML = k.map(([v, a, s]) => `<div class="kpi"><div class="v">${v}</div><div class="k">${a}</div><div class="s">${s}</div></div>`).join("");
}

async function init() {
  [SAMPLES, RESULTS] = await Promise.all([fetch("demo/assets/samples.json").then((r) => r.json()), fetch("demo/assets/results.json").then((r) => r.json())]);
  $("sampleSel").innerHTML = SAMPLES.samples.map((s, i) => `<option value="${i}">${s.label} — ${describe(s.label)} (#${s.id.slice(-1)})</option>`).join("");
  renderKpis(); renderSide();
  $("taskTabs").innerHTML = ["10class", "4class"].map((t) => `<button data-t="${t}">${t === "10class" ? "10 classes (type + size)" : "4 classes (fault type)"}</button>`).join("");
  document.querySelectorAll("#taskTabs button").forEach((b) => b.onclick = () => renderTable(b.dataset.t));
  renderTable("10class");
  $("cmSel").innerHTML = Object.keys(RESULTS.confusion).map((k) => { const [t, s, m] = k.split("|"); return `<option value="${k}">${MODEL_NAMES[m]} · ${t} · ${SPLIT_NAMES[s]}</option>`; }).join("");
  $("cmSel").value = "10class|load3|cnn"; $("cmSel").onchange = () => renderCm($("cmSel").value); renderCm($("cmSel").value);
  $("sampleSel").onchange = pickSample; $("csvIn").onchange = onCsv; $("runBtn").onclick = run;
  try {
    ort.env.wasm.numThreads = 1;
    session = await ort.InferenceSession.create("demo/assets/cnn10.onnx", { executionProviders: ["wasm"] });
    $("runBtn").disabled = false; $("runBtn").textContent = "Run diagnosis";
    $("sampleSel").value = "2"; pickSample();
  } catch (e) {
    $("runBtn").textContent = "Model failed to load"; $("result").innerHTML = `<div class="sub">onnxruntime-web error: ${e.message}</div>`;
  }
}
window.addEventListener("resize", () => { if (current) drawSignal(current.samples.slice(0, WIN)); if (RESULTS) renderCm($("cmSel").value); });
init();
