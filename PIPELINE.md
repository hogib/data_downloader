# The Dataset Generation Pipeline

**A technical report on `data_downloader/seismic_cli/`**

*Companion documents: `cnn_earthquake/report.md` (full investigation),
`accuracy_summary.md` (results), `spectrogram_classifier_report.md` (best
detector), `catalog_report.md` (forecasting). This one covers only how the
data is built. Section references of the form "§12 defect N" point at
`report.md`'s defect changelog.*

---

## 1. What this is

Eleven CLI commands produce ten different dataset formats, but there is
essentially **one pipeline**, parameterised two ways: by a pluggable *encoder*
that decides what a window becomes, and by a *split discipline* that is derived
from where the label lives. Almost everything else — gap rejection, per-station
sampling rates, station caps, the manifest — is shared machinery that every
format inherits.

That sharing is the point. An earlier standalone copy of this logic
(`spectrograph.py`, still in the tree) carried its own splits-by-file,
splits-per-class, and no-station-cap implementations, and accumulated the whole
family of leakage defects independently. It is now dead code — nothing imports
it, and its docstring is a list of the bugs it has that `seismic_cli` fixed.
Keeping representations as *encoders* inside one pipeline rather than as
parallel scripts is what stops that from recurring.

```
FDSN ──► download.py ──► anchor-windows ──► generate-*-dataset ──► manifest.csv
         (60 s raw)      (arrival-cut)      (encode + split)       (+ tensors)
```

---

## 2. The two spines

### 2.1 The encoder protocol is the seam

Every window encoder implements one call signature:

```python
encoder(cleaned_win, fs_station, sta_key, selection,
        station_baselines, out_dir, stem) -> filename
```

plus optional attributes `ext` (`.png` or `.pt`), `requires_spawn`, and
`target_samples()`. The orchestrator handles reading, cleaning, windowing,
splitting and manifesting; the encoder only decides what one cleaned window
*becomes*.

| Encoder | Module | Output keys |
|---|---|---|
| `RamImageEncoder` | `core.py` | RGB PNG (R=Z, G=N, B=E) |
| `SpectrogramEncoder` | `spectrogram.py` | `(3, freq, time)` dB tensor |
| `RamDualEncoder` | `ram_dual.py` | `{seq, img}` |
| `SpectrogramDualEncoder` | `spectrogram.py` | `{seq, img}` |
| `RamAuxEncoder` | `ram_aux.py` | `{img, aux}` |
| `RamDualAuxEncoder` | `ram_dual.py` | `{seq, img, aux}` |
| `SpectrogramDualAuxEncoder` | `spectrogram.py` | `{seq, img, aux}` |
| `*EncoderV2` (×3) | various | as above, `aux` is 6 per-component scalars |

Two consequences worth naming. `SpectrogramDualEncoder` wraps
`SpectrogramEncoder` *by composition*, reusing `normalize_spec` directly rather
than keeping a second copy that could drift. And the `V2` per-component
variants are subclasses overriding only `__call__`, so the default path stays
byte-identical — this project's standard way of adding behaviour without
disturbing reproducibility.

**The one real exception:** `catalog.py` does not use this protocol. It writes
tensors directly in `encode_and_write`, because its input is a catalog of
discrete events rather than a waveform file, so nothing upstream of the encoder
applies. Sections 3–5 below describe the waveform pipeline; §6 covers the
catalog path separately.

### 2.2 The split discipline is derived, not chosen

This is the pipeline's actual intellectual content. **The leak vector follows
from where the label lives**, and each dataset gets the discipline that closes
its own leak:

| Dataset | Label lives on | Dominant leak | Discipline |
|---|---|---|---|
| Detection | the window | station identity — model scores on the instrument | **Station-disjoint, unified across classes** |
| Magnitude regression | the *event* | one event at two stations puts the same target in train and test | **Event-disjoint** (`--split-by event`, default) |
| Three-class risk | window + event | noise class present, so station leakage returns as the bigger risk | **Station-disjoint across all three classes**; event overlap *measured*, not enforced |
| Catalog forecasting | the *future* | the label looks forward, so a window can be labelled by an event in a later split | **Chronological + label-aware embargo**, or LOEO |

Detection's rule exists because roughly 97 % of earthquake stations also supply
noise; allocating classes independently let nearly every station appear as
train-earthquake *and* test-noise (§12 defect 1). Regression's differs because
its target is per-event: station-disjointness would not stop the identical
magnitude label appearing on both sides. Riskclass could not enforce both
simultaneously — an event recorded at N stations would force all N into one
split — so it enforces the station rule and *reports* event overlap as a
diagnostic. Catalog's is different in kind: no spatial rule helps when the leak
is temporal.

**Enforced vs. measured is a real distinction here.** "Every station occupies
exactly one split" is guaranteed by construction and printed as INFO.
Riskclass's zero station leakage and regression's event-disjointness are
*verified after the fact* by re-reading the manifest. Both appear in output;
only the first is a guarantee.

---

## 3. Stage 1 — Acquisition (`src/download.py`)

For each catalog event, stations within `SEARCH_RADIUS_DEG = 0.5°` (~55 km) are
resolved via FDSN (default KOERI). Station lookups are cached on a ~1.1 km
rounded coordinate grid, so co-located events share one metadata query, and all
windows for an event go out as a single bulk request then get sliced in memory.

- **Earthquake:** 60 s from origin time.
- **Noise:** 300 s slices at −3 h and −6 h.
- **Contamination check:** a noise window is discarded if *any* event in the
  **unfiltered** catalog falls within ±300 s. Checking the filtered catalog
  would let sub-threshold events pass silently into the noise class; the buffer
  is wide because coda can ring for minutes. The check is purely temporal, so a
  distant event still vetoes a window — over-conservative, and it discards
  otherwise-scarce noise.

Existing files are skipped, so reruns resume cleanly.

`src/extract.py` prepares the download catalog. Its one subtlety: the `Type`
column (ML, MW, Mb…) names the magnitude *scale*, not the size — filtering on
`Type` alone, as an earlier version did, does not filter out small events.

> **A limit worth knowing before planning any acquisition.** Of 726
> catalog-listed M ≥ 4 events lacking waveforms, only **76 (10.5 %)** were
> retrievable. Station metadata resolves normally; the archive returns
> `HTTP 204` for the rest. A catalog entry does not imply a recording, and the
> gap is an order of magnitude, not a margin.

## 4. Stage 2 — Arrival anchoring (`anchor-windows`)

Short windows cut from origin time can miss the P-wave entirely at distant
stations — at 6 s, an arrival later than 6 s after origin means the "earthquake"
window contains no earthquake. Anchoring re-derives short windows from
already-downloaded 60 s data, no redownload.

The pick uses the classic STA/LTA characteristic function, taking the first
`trigger_onset` crossing of `trigger_on = 3.5` as arrival sample $a$, then cuts

$$[\,a - fT,\ a - fT + T\,), \qquad f = \texttt{pre\_arrival\_fraction} = 0.2$$

so 20 % of the window precedes the arrival. Two correctness requirements were
violated before being fixed: the trace **must be detrended first** — a large DC
offset pins the characteristic function near 1, making a pick impossible
(§12 defect 6) — and the pick should prefer the **vertical** component with
fallback, since `sorted(traces)[0]` selects E before Z alphabetically
(§12 defect 7).

That 20 % buffer is load-bearing downstream: it is exactly what
`eval-sta-lta`'s auto-derived LTA violated, putting the arrival inside
`classic_sta_lta`'s forced-zero warm-up and scoring the baseline at AUC 0.51
(§12 defect 14).

A per-run diagnostic reports stations seen / skipped / picked-on-Z /
picked-via-fallback / unpicked, plus how close failed picks came to threshold —
so "no pick" can be distinguished from "nearly picked".

## 5. Stage 3 — Dataset generation

### 5.1 Per-window processing

Detrend (linear + constant) → 5 % Hann taper → 4th-order zero-phase Butterworth
bandpass 1–45 Hz → encoder.

### 5.2 The five shared guarantees

**Station-disjoint splits, unified across classes** — §2.2 above.

**Per-window station caps** (`--max-windows-per-station`). Enforced as a *quota*
per (station, file), not by dropping files. The distinction matters: one 300 s
noise file yields ~200 windows at 3 s, so a file-granularity cap of 20 silently
passed all 200 through (§12 defect 2). Filenames keep the original window index
$w$, so the sample range $[ws,\ ws+T)$ stays recoverable for
`eval-sta-lta`'s exact reconstruction even after subsampling.

**Maximum-size balanced mode** (`--max`). Assigns *every* usable station to the
split with the largest relative deficit, then trims the surplus class per split
by largest-remainder proportional rounding over per-file quotas. Without it,
generation stops once global targets fill and discards every remaining station —
costly exactly where station diversity is already scarce. Balancing works on
scan *estimates*, so actual counts can differ slightly where windows are later
rejected.

**Gap rejection.** Traces merge *without* interpolation fill, so gaps stay
masked; gaps are then filled for filtering while a boolean mask records which
samples are synthetic. Any window whose worst channel exceeds 5 % synthetic
samples is rejected. Previously `fill_value='interpolate'` turned telemetry gaps
into linear ramps that entered training as real signal (§12 defect 11).

**Per-station sampling rates.** Window sizing uses each station's own rate, read
per-trace, rather than assuming the first trace's rate applies to every station
in the file (§12 defect 12). Component selection is by role, requiring a
vertical — `sorted(keys)[:3]` could return `['1','2','E']`, two horizontals and
no vertical (§12 defect 13).

### 5.3 Station noise baselines

`compute_station_noise_baselines` streams every noise file, applies the *same*
cleaning as training windows, and accumulates running mean/std per
(station, component) via sum/sum-of-squares. A pair qualifies only with ≥ 60 s
of usable data; otherwise it is omitted and callers fall back to per-window
self-standardization.

These feed three distinct things: `--baseline` standardization (which §6.2 of
`report.md` shows is a near-no-op for RAM, since RAM is scale-invariant);
`log_snr` in the aux vectors; and station-relative spectrogram normalization,
where `normalize="station"` subtracts the station's own noise profile in dB, so
pixel values read as *dB above this station's noise floor*.

### 5.4 Execution

Header-only scan (`scan_single_mseed`) counts extractable windows per
(file, station) without reading sample data, which is what makes split
allocation cheap. Work then fans out one task per *file* — each file read once,
only its assigned stations written — across a `ProcessPoolExecutor`.

Torch-based encoders set `requires_spawn = True` and get a `spawn` context:
torch's threading/OpenMP state does not survive `fork()`, and the workers
deadlock **silently** at 0 % CPU rather than erroring.

### 5.5 Output

A `manifest.csv` (columns vary by generator, always enough to reconstruct the
exact source samples) plus tensors or PNGs. Recent generators also carry label
and feature columns directly — riskclass writes `risk_class`, `magnitude`,
`log_snr`, `distance_km`; catalog writes all nine aux features — which is why
scalar baselines on those tasks need no tensor loading at all.

### 5.6 Data-quality filters specific to the risk dataset

`riskclass.py` adds `--min-log-snr` (default −3.0), rejecting windows whose RMS
falls below 5 % of their own station's noise floor. This is instrument-fault
rejection, not outlier trimming: one station's traces spanned ~58 counts on a
~5.38-million-count DC offset with ~50 unique values across 30,001 samples — a
stuck digitizer (§12 defect 15). Gap masking catches missing data; it does not
catch a digitizer frozen at a constant.

`--balance-ratio` (default 4.0) caps the abundant classes at 4× the rare class
per split, applied at file level *before* encoding so discarded windows are
never encoded.

## 6. The catalog path (`catalog.py`)

Structurally different: input is a catalog of discrete events, not waveforms.
Column names are auto-detected across AFAD/Kandilli export conventions
(including Turkish headers — `enlem`, `boylam`, `büyüklük`).

A fixed-**event-count** window (default 64 events, stride 8) slides along the
catalog. Fixed count rather than fixed duration keeps sequence length constant
for the LSTM and the RAM reshape; the window's *duration* becomes a feature,
and is informative — 64 events in 3 days is a different state from 64 over 3
years.

Three correctness properties:

- **Gardner–Knopoff declustering** separates independent mainshocks from
  aftershocks *for target selection only*. Dependent events stay in the window
  features, since that seismicity is real. The window is applied symmetrically
  in time so a foreshock is also absorbed.
- **Windows containing an M ≥ threshold event are dropped** — they describe
  aftermath, not precursor state, and would let the model read the answer off
  its own input.
- **Label-aware embargo** drops exactly those windows whose target event falls
  beyond their own split's boundary, rather than applying a blanket time gap
  wide enough to empty a split.

`--split-mode loeo` skips splitting entirely and writes one flat `all` bucket,
leaving folds to be formed per target event at training time.

**The diagnostic that matters most** is the count of distinct target events —
that, not the window count, is the real sample size. A region with one
qualifying event cannot support a split no matter how many thousands of windows
slide out of it, and the code says so explicitly with remediation options.

> **A defect fixed this session, not yet in report.md's changelog.** Class names
> were hardcoded `lt_1y / 1_5y / gt_5y` while `assign_risk_classes` derives
> *terciles*. On the pooled catalog those land at 26 d and 71 d — so `gt_5y`
> actually meant 71–817 **days**. Names are now generated from the boundaries in
> force. See `catalog_report.md` §2.

## 7. Command reference

| Command | Orchestrator | Encoder | Output | Split |
|---|---|---|---|---|
| `anchor-windows` | `anchor.py` | — | anchored mseed | — |
| `generate-dataset` | `core.run_balanced_preprocessing` | `RamImageEncoder` | RGB PNG | station |
| `generate-spectrogram-dataset` | same | `SpectrogramEncoder` | `.pt` | station |
| `generate-dual-dataset` | same | `RamDualEncoder` | `{seq,img}` | station |
| `generate-spec-dual-dataset` | same | `SpectrogramDualEncoder` | `{seq,img}` | station |
| `generate-ram-aux-dataset` | same | `RamAuxEncoder` | `{img,aux}` | station |
| `generate-dual-aux-dataset` | same | `RamDualAuxEncoder` | `{seq,img,aux}` | station |
| `generate-spec-dual-aux-dataset` | same | `SpectrogramDualAuxEncoder` | `{seq,img,aux}` | station |
| `generate-regression-dataset` | `regression.py` | spectrogram or RAM | `.pt` + magnitude | **event** |
| `generate-riskclass-dataset` | `riskclass.py` | spectrogram or RAM | `.pt` + 3-class | station (all 3 classes) |
| `generate-catalog-dataset` | `catalog.py` | *(direct write)* | `{seq,img,aux}` | **chronological / LOEO** |
| `eval-sta-lta` | `eval_baseline.py` | — | baseline metrics | — |

The seven station-split commands share one orchestrator and differ only by
encoder — that is why a leakage fix applies to all of them at once, and why the
list can grow without the guarantees drifting.

## 8. Known limitations

- **Noise-station diversity is the binding constraint.** The risk dataset runs
  on single-digit noise-station counts per split, which is few enough that one
  faulty instrument distorted an entire result.
- The noise-contamination check is time-only; adding a distance term would
  recover usable noise.
- `--max` balancing works on scan estimates, so realised counts drift slightly.
- `riskclass` cannot enforce station- and event-disjointness simultaneously; it
  enforces the former and measures the latter.
- `spectrograph.py` is dead code, retained only as a record of the defects that
  motivated consolidating into `seismic_cli`.
