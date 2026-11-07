/* Shared DSP for the demo (browser) and the JS-parity test (node). */
"use strict";
(function (root) {
const FS = 12000;

function fft(re, im) { // in-place radix-2, length power of 2
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2;
        const xr = re[b] * cr - im[b] * ci, xi = re[b] * ci + im[b] * cr;
        re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
}
const mean = (x) => x.reduce((a, b) => a + b, 0) / x.length;
function ampSpectrum(x) {
  const n = x.length, m = mean(x), re = new Float64Array(n), im = new Float64Array(n);
  let wsum = 0;
  for (let i = 0; i < n; i++) { const w = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1)); wsum += w; re[i] = (x[i] - m) * w; }
  fft(re, im);
  const out = new Float64Array(n / 2 + 1);
  for (let k = 0; k <= n / 2; k++) out[k] = Math.hypot(re[k], im[k]) * 2 / wsum;
  return out;
}
function envelope(x, lo = 2000, hi = 5500) {
  const n = x.length, m = mean(x), re = new Float64Array(n), im = new Float64Array(n);
  for (let i = 0; i < n; i++) re[i] = x[i] - m;
  fft(re, im);
  for (let k = 0; k < n; k++) { // analytic band: keep positive in-band bins x2
    const f = (k <= n / 2 ? k : k - n) * FS / n;
    const h = (f >= lo && f <= hi) ? 2 : 0;
    re[k] *= h; im[k] *= h;
  }
  // inverse FFT via conj trick
  for (let k = 0; k < n; k++) im[k] = -im[k];
  fft(re, im);
  const env = new Float64Array(n);
  for (let i = 0; i < n; i++) env[i] = Math.hypot(re[i], -im[i]) / n;
  return ampSpectrum(env);
}
function zscore(x) {
  const m = mean(x); let v = 0;
  for (const a of x) v += (a - m) ** 2;
  const s = Math.sqrt(v / x.length) + 1e-6;
  return Float32Array.from(x, (a) => (a - m) / s);
}
function resample(x, fs) {
  if (fs === FS) return x;
  const n = Math.round(x.length * FS / fs), out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const t = i * (x.length - 1) / (n - 1), j = Math.floor(t), f = t - j;
    out[i] = j + 1 < x.length ? x[j] * (1 - f) + x[j + 1] * f : x[j];
  }
  return out;
}
const softmax = (z) => { const m = Math.max(...z), e = z.map((v) => Math.exp(v - m)), s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); };


const api = { FS, fft, ampSpectrum, envelope, zscore, resample, softmax };
if (typeof module !== "undefined" && module.exports) module.exports = api; else root.DSP = api;
})(typeof window !== "undefined" ? window : globalThis);
