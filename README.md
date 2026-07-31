# Seismic Waveform Downloader & RAM-Image Dataset Pipeline

Tools for building machine-learning datasets from raw seismic waveforms:

1. **Download** earthquake and ambient-noise MiniSEED data in bulk from FDSN
   servers (multithreaded, station-cached, contamination-checked).
2. **Anchor** short windows (3s/6s/10s) on the actual P-wave arrival, derived
   from already-downloaded longer windows — no redownload needed.
3. **Generate** balanced, station-disjoint train/val/test datasets of RGB
   images via the Relative Angle Matrix (RAM) transform, with a manifest for
   exact window reconstruction.
4. **Evaluate** a classic STA/LTA trigger baseline on the *exact same* test
   windows a CNN trained on the dataset sees.

The `seismic_cli` package is the canonical pipeline. The scripts under `src/`
are earlier iterations kept for reference — prefer the CLI wherever a CLI
command exists (`src/download.py` and `src/extract.py` are still the intended
tools for downloading and catalog filtering).

---

## Installation

Requires **Python 3.12+**. Dependencies: ObsPy, NumPy, SciPy, pandas,
scikit-learn, Pillow, Typer (and PyTorch/torchaudio for the legacy
spectrogram script only). Everything installs from prebuilt wheels on all
three platforms — no compiler needed.

### Option A — uv (recommended)

[uv](https://docs.astral.sh/uv/) reads `pyproject.toml`/`uv.lock` and
reproduces the exact locked environment, including the `seismic-cli`
entry point.

**Linux / macOS**

```bash
# install uv (skip if you have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/hogib/data_downloader.git
cd data_downloader
uv sync                      # creates .venv/ and installs everything

uv run seismic-cli --help    # run commands through uv...
# ...or activate the venv and call seismic-cli directly:
source .venv/bin/activate
seismic-cli --help
```

**Windows (PowerShell)**

```powershell
# install uv (skip if you have it)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

git clone https://github.com/hogib/data_downloader.git
cd data_downloader
uv sync

uv run seismic-cli --help
# ...or activate the venv:
.venv\Scripts\Activate.ps1
seismic-cli --help
```

> On `cmd.exe` use `.venv\Scripts\activate.bat` instead of the PowerShell
> activation script.

### Option B — plain pip / venv

**Linux / macOS**

```bash
git clone https://github.com/hogib/data_downloader.git
cd data_downloader
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .             # installs deps + the seismic-cli entry point
seismic-cli --help
```

**Windows (PowerShell)**

```powershell
git clone https://github.com/hogib/data_downloader.git
cd data_downloader
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .
seismic-cli --help
```

> **Editable install matters:** `pip install -e .` (and `uv sync`, which
> installs the project editable by default) means edits to the source take
> effect immediately. A plain `pip install .` snapshots the code — after
> editing you must reinstall, or the CLI silently runs the stale copy.

### Platform notes

- **Linux:** any recent distro works; wheels cover x86_64 and aarch64.
- **macOS:** both Apple Silicon and Intel are covered by wheels. If you use
  Homebrew Python, make sure `python3.12 --version` reports ≥ 3.12.
- **Windows:** ObsPy ships Windows wheels; no MSVC build tools required.
  If `py -3.12` isn't found, install Python 3.12 from python.org or
  `winget install Python.Python.3.12`. Long path support is recommended
  (`git config --global core.longpaths true`) since dataset trees nest deep.
- **Conda (any OS):** `conda create -n seismic python=3.12`, activate it,
  then follow Option B inside the environment.

---

## Pipeline overview

```
catalog CSV ──► src/extract.py ──► filtered catalog
                                        │
                                        ▼
                              src/download.py  (FDSN bulk download)
                                        │
              ┌─────────────────────────┴───────────────────────┐
              ▼                                                 ▼
   data/batched_waveforms/window_post_60s/        data/batched_noise_waveforms/
              │                                                 │
              │  (short windows only)                           │
              ▼                                                 │
   seismic-cli anchor-windows  ──► window_post_6s_anchored/     │
              │                                                 │
              └───────────────┬─────────────────────────────────┘
                              ▼
                 seismic-cli generate-dataset
                              │
              ┌───────────────┴────────────────┐
              ▼                                ▼
   dataset_*/train|val|test/<class>/*.png   dataset_*/manifest.csv
              │                                │
              ▼                                ▼
   CNN training (cnn_earthquake repo)   seismic-cli eval-sta-lta
```

---

# Usage reference manual

## 1. Catalog preparation — `src/extract.py`

Filters a raw event catalog down to the events worth downloading.

```bash
python src/extract.py
```

Edit the `__main__` block to set the input CSV and `MIN_MAGNITUDE`. The
filter drops rows with missing/unrated magnitude scales (`Type` column) and
events below the magnitude threshold. Keep the **unfiltered** catalog too —
the downloader needs it for noise contamination checks.

**Expected catalog columns:** `Latitude`, `Longitude`, `Date`
(`DD/MM/YYYY HH:MM:SS` or ISO), `Magnitude`, `Type`, and optionally
`EventID` (falls back to row index).

## 2. Downloading waveforms — `src/download.py`

```bash
python src/download.py
```

Configuration lives in the globals at the top of the file:

| Setting | Default | Meaning |
|---|---|---|
| `EARTHQUAKE_BATCHES` | `[("window_post_60s", 0, 60)]` | Earthquake windows as `(folder, start_offset_s, end_offset_s)` relative to origin time. |
| `NOISE_BATCHES` | 300s slices at −3h and −6h | Noise windows relative to origin time. |
| `CATALOG_FILE` | `catalogs/extracted_earthquakes.csv` | Which events to download (the filtered catalog). |
| `FULL_CATALOG_FILE` | `catalogs/deprem_katalog_utc.csv` | Unfiltered catalog used **only** for contamination checks — small filtered-out quakes still count as contamination. |
| `SEARCH_RADIUS_DEG` | `0.5` | Station search radius around each epicenter (~55 km). |
| `FDSN_CLIENT` | `"KOERI"` | FDSN node (`"IRIS"`, `"GFZ"`, ... also work). |
| `MAX_WORKERS` | `15` | Concurrent download threads. Lower it if the server throttles you. |
| `FILE_LIMIT` | `100000` | Cap on events processed; set small for a test run. |
| `NOISE_CONTAMINATION_BUFFER_SEC` | `300` | Padding around *any* cataloged event that disqualifies a noise window. |

Behavior worth knowing:

- Station lookups are cached per rounded coordinate (~1.1 km grid), so
  events in the same area share one FDSN metadata query.
- All windows for one event go out as a single bulk request, then get
  sliced in memory.
- Already-existing output files are skipped, so reruns resume cleanly.
- Catalog rows with unparseable dates are excluded from contamination
  checking (with a warning) rather than breaking the run.

Output layout:

```
data/batched_waveforms/<window_name>/event_<EventID>_raw.mseed
data/batched_noise_waveforms/<window_name>/noise_event_<EventID>_raw.mseed
```

## 3. Arrival-anchored short windows — `seismic-cli anchor-windows`

Short windows sliced at event *origin* time can miss the P-wave arrival
entirely at distant stations. This command re-derives short windows from
already-downloaded 60s data, anchored on a coarse STA/LTA arrival pick.

```bash
seismic-cli anchor-windows \
    --source-dir data/batched_waveforms/window_post_60s \
    --output-base-dir data/batched_waveforms \
    -t 3 -t 6 -t 10
```

| Option | Default | Meaning |
|---|---|---|
| `--source-dir` | required | Directory of long-window (e.g. 60s) mseed files. |
| `--output-base-dir` | required | Where `window_post_<N>s_anchored/` subfolders get written. |
| `--target-seconds`, `-t` | required, repeatable | Short window length(s) to derive. |
| `--pick-sta-seconds` | `1.0` | STA length for the arrival pick. |
| `--pick-lta-seconds` | `10.0` | LTA length for the arrival pick. |
| `--trigger-on` | `3.5` | STA/LTA ratio that declares an arrival. |
| `--trigger-off` | `1.0` | Ratio that ends the trigger. |
| `--pre-arrival-fraction` | `0.2` | Fraction of the window placed *before* the arrival (e.g. 6s window → 1.2s pre, 4.8s post). |
| `--limit-files` | none | Process only the first N source files (quick test). |

Picking details: data is detrended before STA/LTA (raw MiniSEED counts carry
DC offsets that pin the ratio near 1), the vertical (Z) component is tried
first with fallback to the horizontals, and a `[PICK DIAGNOSTICS]` summary
prints at the end — stations seen / skipped / picked on Z / picked via
fallback / no pick, plus how close failed stations came to `--trigger-on`.
**If you don't see that diagnostics block, you're running a stale install**
(see the editable-install note above). If failures cluster just below the
trigger threshold, lower `--trigger-on`; if they sit near 1.0, inspect the
input data.

## 4. Dataset generation — `seismic-cli generate-dataset`

Converts earthquake + noise mseed directories into a balanced,
station-disjoint RGB RAM-image dataset with a reconstruction manifest.

```bash
# 60s dataset, defaults
seismic-cli generate-dataset \
    --eq-dir data/batched_waveforms/window_post_60s \
    --noise-dir data/batched_noise_waveforms \
    --output-dir dataset_60s \
    --window-seconds 60 --overlap 0.25

# 6s dataset from anchored windows, maximum size, station cap
seismic-cli generate-dataset \
    --eq-dir data/batched_waveforms/window_post_6s_anchored \
    --noise-dir data/batched_noise_waveforms \
    --output-dir dataset_6s_max \
    --window-seconds 6 --overlap 0.5 \
    --max --max-windows-per-station 20
```

| Option | Default | Meaning |
|---|---|---|
| `--eq-dir` / `--noise-dir` / `--output-dir` | required | Input mseed directories and output dataset root. |
| `--window-seconds` | `60.0` | Window length. Must match the eval later. |
| `--overlap` | `0.5` | Sliding-window overlap fraction. |
| `--target-n` | `64` | RAM image resolution (n×n pixels). The reshape depth `d` is derived per window as `ceil(samples / n)`. |
| `--train-ratio` / `--val-ratio` / `--test-ratio` | `0.70/0.15/0.15` | Split ratios (by window count, at station granularity). |
| `--limit-pictures` | none | Cap total images across both classes. Mutually exclusive with `--max`. |
| `--max` | off | **Maximum balanced dataset**: every usable station is assigned to a split; the surplus class is then trimmed *per split* to match the smaller class (evenly-spaced window subsampling). Without it, generation stops at `min(eq_total, noise_total)` targets and drops leftover stations. |
| `--max-windows-per-station` | none | Cap any single station's contribution, enforced per *window* (a single long file can't blow past it). Strongly recommended for short windows. |
| `--baseline` / `--no-baseline` | off | Standardize each channel against that station's long-term noise mean/std (STA/LTA-style memory) instead of per-window statistics. Stations lacking `--min-baseline-seconds` of usable noise fall back to self-standardization. |
| `--freqmin` / `--freqmax` | `1.0` / `45.0` | Bandpass corners used in cleaning (and baseline computation). |
| `--min-baseline-seconds` | `60.0` | Minimum usable noise per station/component to trust a baseline. |
| `--num-cores` | CPU count − 1 | Worker processes. |

Guarantees enforced during generation:

- **Station-disjoint splits, unified across classes** — every station lands
  in exactly one of train/val/test, for *both* its earthquake and noise
  windows. No instrument ever appears in train under one label and test
  under the other.
- **Component-role channel selection** — Red=Z, Green=N/1, Blue=E/2 for
  every station; stations without a usable vertical are skipped.
- **Per-station sampling rates** — each station's windows are sized with its
  own sampling rate (recorded in the manifest `fs` column); components with
  mismatched rates are skipped.
- **Gap rejection** — telemetry gaps are tracked through merging; windows
  with more than 5% interpolated samples are discarded rather than being
  labeled as real signal.
- Class balance is computed on header-scan estimates; actual counts can
  differ slightly where windows get rejected at generation time. The
  manifest is the ground truth for what was written.

Output layout:

```
dataset_60s/
├── train/01_earthquake/*.png     train/00_noise/*.png
├── val/  01_earthquake/*.png     val/  00_noise/*.png
├── test/ 01_earthquake/*.png     test/ 00_noise/*.png
└── manifest.csv
```

**Manifest schema** (`manifest.csv`): one row per image.

| Column | Meaning |
|---|---|
| `split` | `train` / `val` / `test` |
| `class_name` | `01_earthquake` / `00_noise` |
| `station_key` | `NETWORK.STATION` |
| `file_path` | Source mseed file (as given at generation time — usually relative, so run downstream tools from the same working directory). |
| `filename` | Image name: `<file>_<station>_win<NNN>.png`. `NNN` is the window's *original* index: window start sample = `NNN × step`, so the exact sample range is always recoverable even when windows were subsampled by caps or `--max` trimming. |
| `fs` | Sampling rate the station's windows were generated with. |

## 5. STA/LTA baseline — `seismic-cli eval-sta-lta`

Scores the classic STA/LTA trigger on the exact windows a CNN sees
(reconstructed from raw mseed via the manifest: same file, station, and
window index).

```bash
seismic-cli eval-sta-lta \
    --manifest-path dataset_6s_max/manifest.csv \
    --split test --window-seconds 6 --overlap 0.5
```

| Option | Default | Meaning |
|---|---|---|
| `--manifest-path` | required | The `manifest.csv` from generate-dataset. |
| `--split` | `test` | Which split to evaluate. |
| `--window-seconds` / `--overlap` | `60.0` / `0.5` | **Must match generate-dataset.** |
| `--sta-seconds` | auto | Short-term window. Auto-derived: `LTA/10`, floored at 0.05s. |
| `--lta-seconds` | auto | Long-term window. Auto-derived: `window/3`, capped at 10s. |

Auto-derived parameters by window length: 60s → STA 1.0 / LTA 10.0 (the
classic defaults), 6s → 0.2 / 2.0, 3s → 0.1 / 1.0. Every run prints the
parameters used and whether they were derived or explicitly set. Traces are
detrended before scoring.

Reported metrics: **AUC** (threshold-free — use this to compare against the
CNN), plus accuracy/precision/recall at the Youden's-J threshold. That
threshold is chosen *on the evaluated split*, so the thresholded numbers are
STA/LTA's upper bound, not a fair head-to-head accuracy comparison.

Run this from the same working directory generate-dataset ran in — the
manifest stores `file_path` as given, and the tool prints a path sanity
check at startup.

## 6. Maintenance utilities

| Script | Purpose |
|---|---|
| `src/delete_corrupt.py` | Scans a dataset tree for 0-byte or unreadable PNGs. Deletion line is commented out — review first. |
| `src/delete_size.py` | Finds images whose size differs from the majority. Dry run by default; set `auto_delete=True` after reviewing. |

## 7. Legacy scripts (`src/`)

`src/process.py`, `src/reg.py`, `src/spectrograph.py`,
`src/standard_per_station.py`, `src/sta-lta.py`, and
`src/arrival_for_small.py` are earlier iterations of what is now
`seismic_cli` (dataset generation, baseline standardization, STA/LTA
comparison, arrival anchoring). They predate several bug fixes that live
only in the CLI package — **use the CLI**, and treat these as read-only
history.

---

## End-to-end example

```bash
# 1. Filter the catalog (edit thresholds in the __main__ block first)
python src/extract.py

# 2. Bulk-download 60s earthquake windows + noise (configure globals first)
python src/download.py

# 3. Derive arrival-anchored 6s windows from the 60s data
seismic-cli anchor-windows \
    --source-dir data/batched_waveforms/window_post_60s \
    --output-base-dir data/batched_waveforms -t 6

# 4. Generate the maximum balanced 6s dataset
seismic-cli generate-dataset \
    --eq-dir data/batched_waveforms/window_post_6s_anchored \
    --noise-dir data/batched_noise_waveforms \
    --output-dir dataset_6s_max \
    --window-seconds 6 --overlap 0.5 --max --max-windows-per-station 20

# 5. Train the CNN on it (cnn_earthquake repo)
python cnn_train.py --dataset-dir dataset_6s_max --window-seconds 6

# 6. Score the STA/LTA baseline on the identical test windows
seismic-cli eval-sta-lta \
    --manifest-path dataset_6s_max/manifest.csv \
    --split test --window-seconds 6 --overlap 0.5
```

## Troubleshooting

- **`anchor-windows` writes 0 files** → read the `[PICK DIAGNOSTICS]` block.
  No block printed at all means a stale (non-editable) install is running.
- **`eval-sta-lta` skips every entry** → `--window-seconds`/`--overlap`
  don't match generation, or you're in a different working directory than
  generate-dataset ran from (check the printed path sanity check).
- **FDSN downloads failing or throttled** → lower `MAX_WORKERS`, verify the
  `FDSN_CLIENT` node serves your region, and test with a small `FILE_LIMIT`.
- **A split comes out empty in `--max` mode** → that split received stations
  of only one class (the run warns about this); add stations or adjust
  ratios/caps.
