# Data provenance

## Case Western Reserve University (CWRU) Bearing Data Center

- Source: official CWRU site, `https://engineering.case.edu/bearingdatacenter/download-data-file`
  (files fetched directly from `https://engineering.case.edu/sites/default/files/<id>.mat`).
- Downloaded: 2026-11-07 by `scripts/prepare_data.py`; SHA-256 of all 40 files pinned in
  `data/cwru_sha256.json` and verified on every run.
- Terms: CWRU makes the data freely available for research/education; the site asks users to
  acknowledge the CWRU Bearing Data Center. The `.mat` files are **not** redistributed in this repo;
  only 4 short 2048-sample CSV excerpts (`tests/fixtures/`) and 20 excerpts in
  `demo/assets/samples.json` (each ~0.17 s from a single recording) are committed for tests and the demo.
- Rig: 2 hp Reliance motor, SKF 6205-2RS JEM deep-groove ball bearing at the drive end; single-point
  faults seeded by electro-discharge machining.

### Files used (12 kHz drive-end accelerometer, `X<id>_DE_time`)

| class | fault diameter | 0 hp | 1 hp | 2 hp | 3 hp |
|---|---|---|---|---|---|
| Normal | – | 97 | 98 | 99 | 100 |
| Inner race | 0.007" | 105 | 106 | 107 | 108 |
| Ball | 0.007" | 118 | 119 | 120 | 121 |
| Outer race (centred, @6:00) | 0.007" | 130 | 131 | 132 | 133 |
| Inner race | 0.014" | 169 | 170 | 171 | 172 |
| Ball | 0.014" | 185 | 186 | 187 | 188 |
| Outer race (centred, @6:00) | 0.014" | 197 | 198 | 199 | 200 |
| Inner race | 0.021" | 209 | 210 | 211 | 212 |
| Ball | 0.021" | 222 | 223 | 224 | 225 |
| Outer race (centred, @6:00) | 0.021" | 234 | 235 | 236 | 237 |

### Quirks found and handled
- `99.mat` contains both the `X098_*` and `X099_*` arrays; the loader picks the key matching the file id.
- `98.mat` has no RPM variable; nominal rig speed for the load is used where RPM is needed (only metadata - features use a fixed 1772 rpm reference with a +-4 % peak search).
- Normal recordings are ~20-40 s while fault recordings are ~10 s. Every recording is cut to its first
  121,000 samples so each of the 10 classes has exactly 468 windows (2048 samples, 50 % overlap).
- The normal files are sometimes described as 48 kHz in third-party write-ups. A check of 97.mat
  shows the dominant low-frequency peak at 89.9 Hz under a 12 kHz assumption (= 3x the 29.95 Hz shaft
  rate at 1797 rpm), versus 161.9 Hz (no shaft-harmonic match) under 48 kHz, so they are treated as 12 kHz.

## Not used
- IMS / NASA PCoE run-to-failure bearing data: not downloaded (multi-GB archive on a shared 2-CPU box).
  The remaining-life trend code (`vibesense.health.fit_trend`) is unit-tested on synthetic data only.
