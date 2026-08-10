"""
Magnitude-regression dataset generation.

Produces labelled windows from earthquake mseed only (no noise class), each
carrying the source magnitude plus the two physical predictors magnitude
actually depends on:

    log_snr      = log( sigma_window / sigma_station_noise )
    distance_km  = epicentral distance from event to station

Local magnitude is essentially log peak amplitude corrected for distance, so
these two are not "extra features" -- they are the classical ML relation.
They are emitted explicitly because the RAM transform is exactly
scale-invariant (see cnn_earthquake/report.md 8.2): amplitude information is
annihilated by the encoding, so no amount of standardization can carry it
into the image. Spectrogram encoders with `normalize="station"` do retain
amplitude, but the scalars are cheap, physically interpretable, and give the
model a direct path to the dominant term either way.

**Splitting.** For classification the leak that matters is station identity.
For regression the label is per-EVENT, so the same earthquake recorded at two
stations carries an identical target -- putting one in train and the other in
test leaks the answer directly. `split_by="event"` (default) therefore keeps
whole events together. `split_by="station"` is offered for continuity with
the classifier; either way the residual overlap along the *other* dimension is
measured and reported, because one of the two always leaks somewhat and
pretending otherwise is how the earlier results went wrong.
"""

import concurrent.futures
import csv
import math
import multiprocessing
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from seismic_cli.core import (_cap_station_windows, _masked_to_filled,
                              clean_and_filter_1d,
                              compute_station_noise_baselines, scan_single_mseed,
                              select_components, window_array_indexed)

EVENT_ID_RE = re.compile(r"^(?:noise_)?event_(.+?)_raw$")


def parse_event_id(file_stem: str) -> Optional[str]:
    """`event_<EventID>_raw` -> `<EventID>` (the layout download.py writes)."""
    m = EVENT_ID_RE.match(file_stem)
    return m.group(1) if m else None


def _pick_column(df: pd.DataFrame, candidates) -> Optional[str]:
    lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def load_event_catalog(catalog_path: str) -> Dict[str, Dict[str, float]]:
    """EventID -> {magnitude, lat, lon}. Rows without a usable magnitude are dropped."""
    df = pd.read_csv(catalog_path)
    c_id = _pick_column(df, ["eventid", "event_id", "id"])
    c_mag = _pick_column(df, ["magnitude", "mag", "ml", "m"])
    c_lat = _pick_column(df, ["latitude", "lat"])
    c_lon = _pick_column(df, ["longitude", "lon", "long"])
    if c_mag is None:
        raise ValueError(f"No magnitude column found in {catalog_path}; saw {list(df.columns)}")

    out: Dict[str, Dict[str, float]] = {}
    for idx, row in df.iterrows():
        key = str(row[c_id]).strip() if c_id else f"idx_{idx}"
        try:
            mag = float(row[c_mag])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(mag):
            continue
        entry = {"magnitude": mag}
        for name, col in (("lat", c_lat), ("lon", c_lon)):
            try:
                entry[name] = float(row[col]) if col else np.nan
            except (TypeError, ValueError):
                entry[name] = np.nan
        out[key] = entry
    print(f"[catalog] Loaded {len(out)} events with usable magnitudes from {catalog_path}")
    return out


def load_station_coords(path: Optional[str]) -> Dict[str, Tuple[float, float]]:
    """NET.STA -> (lat, lon). Optional; without it distance_km is left NaN."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Station catalog not found: {p} -- distances will be NaN.")
        return {}
    df = pd.read_csv(p)
    c_net = _pick_column(df, ["network", "net", "network_code"])
    c_sta = _pick_column(df, ["station", "sta", "station_code", "code"])
    c_lat = _pick_column(df, ["latitude", "lat"])
    c_lon = _pick_column(df, ["longitude", "lon", "long"])
    if not (c_sta and c_lat and c_lon):
        print(f"[WARN] Could not identify station/lat/lon columns in {p} "
              f"(saw {list(df.columns)}) -- distances will be NaN.")
        return {}
    coords = {}
    for _, row in df.iterrows():
        sta = str(row[c_sta]).strip()
        net = str(row[c_net]).strip() if c_net else ""
        try:
            lat, lon = float(row[c_lat]), float(row[c_lon])
        except (TypeError, ValueError):
            continue
        if net:
            coords[f"{net}.{sta}"] = (lat, lon)
        coords[sta] = (lat, lon)   # also key bare station, for NET mismatches
    print(f"[catalog] Loaded coordinates for {len(coords)} station keys from {p}")
    return coords


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    if any(v is None or not np.isfinite(v) for v in (lat1, lon1, lat2, lon2)):
        return float("nan")
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _station_coord(coords, sta_key):
    if sta_key in coords:
        return coords[sta_key]
    bare = sta_key.split(".")[-1]
    return coords.get(bare, (np.nan, np.nan))


# ---------------------------------------------------------------------------
# Per-file worker
# ---------------------------------------------------------------------------
#
# The read-only data shared by every task (event_meta, station_coords,
# station_baselines, encoder, and the scalar options) is set ONCE per worker
# process via `_init_regression_worker`, an `initializer=` passed to
# ProcessPoolExecutor -- not threaded through each task's arguments. On the
# real catalog, `event_meta` alone (every event with a usable magnitude, no
# magnitude floor applied here) is ~480k entries / ~22 MB pickled; passing it
# inside every one of ~31k per-file tasks means `ex.map` pickles and unpickles
# that blob 31k times over IPC (~0.2s to pickle ALONE, before unpickling or
# the queue itself), which dominates total runtime by orders of magnitude
# over the actual per-window encoding cost (sub-millisecond once a worker's
# lazy torch/torchaudio import has run once). An `initializer` sends each
# large object to each worker exactly once, at process startup.

_worker: Dict[str, object] = {}


def _init_regression_worker(station_baselines, encoder, fs, window_seconds, overlap,
                            max_gap_fraction, freqmin, freqmax, event_meta, station_coords,
                            per_component_aux=False):
    _worker.update(station_baselines=station_baselines, encoder=encoder, fs=fs,
                   window_seconds=window_seconds, overlap=overlap,
                   max_gap_fraction=max_gap_fraction, freqmin=freqmin, freqmax=freqmax,
                   event_meta=event_meta, station_coords=station_coords,
                   per_component_aux=per_component_aux)


def _process_regression_file(args):
    file_path, assignments = args
    station_baselines = _worker["station_baselines"]
    encoder = _worker["encoder"]
    fs = _worker["fs"]
    window_seconds = _worker["window_seconds"]
    overlap = _worker["overlap"]
    max_gap_fraction = _worker["max_gap_fraction"]
    freqmin = _worker["freqmin"]
    freqmax = _worker["freqmax"]
    event_meta = _worker["event_meta"]
    station_coords = _worker["station_coords"]
    per_component_aux = _worker["per_component_aux"]
    from obspy import read
    try:
        st = read(str(file_path))
        try:
            st.merge(method=1)
        except Exception as e:
            return f"[WARN] merge failed for {file_path.name}: {e}"

        event_id = parse_event_id(file_path.stem)
        meta = event_meta.get(event_id)
        if meta is None:
            return []          # no magnitude for this event -> nothing to label

        stations: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, float]]] = {}
        for tr in st:
            sta_key = f"{tr.stats.network}.{tr.stats.station}"
            if sta_key not in assignments:
                continue
            chan = tr.stats.channel[-1].upper()
            existing = stations.setdefault(sta_key, {}).get(chan)
            if existing is not None and len(existing[0]) >= tr.stats.npts:
                continue
            data, gap_mask = _masked_to_filled(tr.data)
            stations[sta_key][chan] = (data, gap_mask, tr.stats.sampling_rate)

        rows = []
        for sta_key, channels in stations.items():
            selection = select_components(channels.keys())
            if selection is None:
                continue
            rates = {channels[c][2] for c in selection}
            if len(rates) != 1:
                continue
            fs_station = rates.pop()

            target_samples = int(fs_station * window_seconds)
            tolerance = int(target_samples * 0.05)
            raw = [channels[c][0] for c in selection]
            masks = [channels[c][1] for c in selection]
            min_len = min(len(c) for c in raw)
            if min_len < (target_samples - tolerance):
                continue

            event_data = np.column_stack([c[:min_len] for c in raw])
            gap_mask = np.column_stack([m[:min_len] for m in masks])
            windows, _ = window_array_indexed(
                event_data, gap_mask, fs=fs_station, window_seconds=window_seconds,
                overlap=overlap, max_gap_fraction=max_gap_fraction,
            )

            split_name, out_dir, quota = assignments[sta_key]
            if quota is not None and len(windows) > quota:
                sel = np.linspace(0, len(windows) - 1, quota).round().astype(int)
                windows = [windows[i] for i in sorted(set(sel.tolist()))]

            sta_lat, sta_lon = _station_coord(station_coords, sta_key)
            dist_km = haversine_km(meta.get("lat"), meta.get("lon"), sta_lat, sta_lon)

            for w_idx, win in windows:
                cleaned = np.zeros_like(win, dtype=np.float64)
                for i in range(win.shape[1]):
                    cleaned[:, i] = clean_and_filter_1d(win[:, i], fs_station, freqmin, freqmax)

                # log SNR against the station's own long-term noise level --
                # the amplitude term the encoding may not preserve, so it is
                # recorded regardless. Per-component (one scalar per Z/N/E,
                # NaN per-slot on missing baseline/dead channel) mirrors
                # ram_aux.RamAuxEncoderV2's math instead of collapsing to one
                # Z/N/E-averaged value.
                if per_component_aux:
                    snr_per_component = [float("nan")] * len(selection)
                    for i, comp in enumerate(selection):
                        _mu, sigma_noise = station_baselines.get((sta_key, comp), (None, None))
                        sigma_win = float(np.std(cleaned[:, i]))
                        if sigma_noise and sigma_noise > 0 and sigma_win > 0:
                            snr_per_component[i] = math.log(sigma_win / sigma_noise)
                    aux_vals = snr_per_component
                else:
                    snrs = []
                    for i, comp in enumerate(selection):
                        _mu, sigma_noise = station_baselines.get((sta_key, comp), (None, None))
                        sigma_win = float(np.std(cleaned[:, i]))
                        if sigma_noise and sigma_noise > 0 and sigma_win > 0:
                            snrs.append(math.log(sigma_win / sigma_noise))
                    aux_vals = [float(np.mean(snrs)) if snrs else float("nan")]

                stem = f"{file_path.stem}_{sta_key}_win{w_idx:03d}"
                filename = encoder(cleaned, fs_station, sta_key, selection,
                                   station_baselines, out_dir, stem)
                rows.append((split_name, sta_key, str(event_id), str(file_path), filename,
                             fs_station, meta["magnitude"], *aux_vals, dist_km))
        return rows
    except Exception as e:
        return f"[WARN] Failed file {file_path.stem}: {e}"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_regression_preprocessing(
    eq_dir: str,
    noise_dir: str,
    catalog_path: str,
    output_dir: str,
    encoder,
    station_catalog: Optional[str] = None,
    split_ratios: tuple = (0.70, 0.15, 0.15),
    fs: float = 100.0,
    window_seconds: float = 60.0,
    overlap: float = 0.5,
    max_windows_per_station: Optional[int] = None,
    split_by: str = "event",
    freqmin: float = 1.0,
    freqmax: float = 45.0,
    min_baseline_seconds: float = 60.0,
    max_gap_fraction: float = 0.05,
    num_cores: Optional[int] = None,
    seed: int = 42,
    per_component_aux: bool = False,
):
    if split_by not in ("event", "station"):
        raise ValueError("split_by must be 'event' or 'station'")

    print("=" * 60)
    print(f"MAGNITUDE REGRESSION DATASET  (split_by={split_by})")
    print("=" * 60)

    event_meta = load_event_catalog(catalog_path)
    station_coords = load_station_coords(station_catalog)

    # Amplitude reference: reuse the classifier's station noise baselines.
    station_baselines, _ = compute_station_noise_baselines(
        noise_dir, fs=fs, freqmin=freqmin, freqmax=freqmax,
        min_baseline_seconds=min_baseline_seconds, num_cores=num_cores,
    )

    splits = ["train", "val", "test"]
    out_paths = {}
    for s in splits:
        d = Path(output_dir) / s
        d.mkdir(parents=True, exist_ok=True)
        out_paths[s] = d

    if num_cores is None:
        num_cores = max(1, multiprocessing.cpu_count() - 1)

    print("\n[PHASE 1] Scanning earthquake files...")
    eq_path = Path(eq_dir)
    if not eq_path.exists():
        print(f"[ERROR] Earthquake directory not found: {eq_path}")
        return
    mseed_files = list(eq_path.rglob("*.mseed"))
    scan_args = [(fp, fs, window_seconds, overlap) for fp in mseed_files]

    # (file, station) -> window count, plus the file's event id
    file_station_counts: Dict[Path, Dict[str, int]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as ex:
        for fpath, counts in ex.map(scan_single_mseed, scan_args):
            if counts:
                file_station_counts[fpath] = counts

    labelled = {f: c for f, c in file_station_counts.items()
                if parse_event_id(f.stem) in event_meta}
    n_unlabelled = len(file_station_counts) - len(labelled)
    total_windows = sum(sum(c.values()) for c in labelled.values())
    print(f"  -> {len(labelled)} labelled event files "
          f"({n_unlabelled} had no catalog magnitude and were dropped)")
    print(f"  -> {total_windows} extractable windows across "
          f"{len({s for c in labelled.values() for s in c})} stations")
    if not labelled:
        print("[ERROR] No labelled windows. Check that catalog EventIDs match "
              "the 'event_<EventID>_raw.mseed' filenames.")
        return

    # ---- group by the chosen split key -------------------------------------
    print(f"\n[PHASE 2] Allocating {split_by}-disjoint splits...")
    groups: Dict[str, List[Tuple[Path, str, int]]] = {}
    for fpath, counts in labelled.items():
        ev = parse_event_id(fpath.stem)
        for sta_key, w in counts.items():
            key = ev if split_by == "event" else sta_key
            groups.setdefault(key, []).append((fpath, sta_key, w))

    keys = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(keys)

    targets = {s: r * total_windows for s, r in zip(splits, split_ratios)}
    counts_by_split = {s: 0 for s in splits}
    key_split: Dict[str, str] = {}
    for key in keys:
        w_total = sum(w for _, _, w in groups[key])
        # largest relative deficit first, so ratios hold over the whole set
        best = max(splits, key=lambda s: (targets[s] - counts_by_split[s]) / max(targets[s], 1.0))
        key_split[key] = best
        counts_by_split[best] += w_total

    # per-(file, station) assignment, honouring the per-station cap
    per_station_files: Dict[str, List[Tuple[Path, int]]] = {}
    for key, entries in groups.items():
        for fpath, sta_key, w in entries:
            per_station_files.setdefault(sta_key, []).append((fpath, w))
    capped = _cap_station_windows(
        [(sta, sum(w for _, w in v), v) for sta, v in per_station_files.items()],
        max_windows_per_station, random.Random(seed + 1),
    )
    quota_lookup = {(sta, fp): q for sta, _tot, contribs in capped for fp, _w, q in contribs}
    kept_files = {(sta, fp) for sta, _t, contribs in capped for fp, _w, _q in contribs}

    file_assignments: Dict[Path, Dict[str, Tuple[str, Path, Optional[int]]]] = {}
    for key, entries in groups.items():
        split_name = key_split[key]
        for fpath, sta_key, _w in entries:
            if (sta_key, fpath) not in kept_files:
                continue
            file_assignments.setdefault(fpath, {})[sta_key] = (
                split_name, out_paths[split_name], quota_lookup.get((sta_key, fpath))
            )

    for s in splits:
        print(f"     {s:5s}: ~{counts_by_split[s]} windows (target {targets[s]:.0f})")

    print(f"\n[PHASE 3] Encoding on {num_cores} cores...")
    mp_context = (multiprocessing.get_context("spawn")
                  if getattr(encoder, "requires_spawn", False) else None)
    if mp_context:
        print("       (using 'spawn' workers -- required for torch-based encoders)")

    tasks = [(fp, asg) for fp, asg in file_assignments.items()]
    init_args = (station_baselines, encoder, fs, window_seconds, overlap,
                max_gap_fraction, freqmin, freqmax, event_meta, station_coords,
                per_component_aux)

    manifest: List[tuple] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_cores, mp_context=mp_context,
        initializer=_init_regression_worker, initargs=init_args,
    ) as ex:
        for res in ex.map(_process_regression_file, tasks, chunksize=16):
            if isinstance(res, str):
                print(res)
            elif res:
                manifest.extend(res)

    if not manifest:
        print("[ERROR] No windows were written.")
        return

    snr_cols = (["log_snr_0", "log_snr_1", "log_snr_2"] if per_component_aux else ["log_snr"])

    manifest_path = Path(output_dir) / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["split", "station_key", "event_id", "file_path", "filename",
                     "fs", "magnitude"] + snr_cols + ["distance_km"])
        wr.writerows(manifest)

    # ---- report label coverage and residual leakage ------------------------
    df = pd.DataFrame(manifest, columns=["split", "station_key", "event_id", "file_path",
                                         "filename", "fs", "magnitude"] + snr_cols + ["distance_km"])
    print(f"\n[PHASE 4] Wrote {len(df)} labelled windows to {manifest_path}")
    print("\n  Magnitude coverage per split:")
    for s in splits:
        sub = df[df.split == s]
        if len(sub):
            print(f"     {s:5s}: n={len(sub):5d}  M {sub.magnitude.min():.1f}-{sub.magnitude.max():.1f}  "
                  f"mean {sub.magnitude.mean():.2f}  events={sub.event_id.nunique()}  "
                  f"stations={sub.station_key.nunique()}")
        else:
            print(f"     {s:5s}: EMPTY -- adjust ratios or add data")

    other = "station_key" if split_by == "event" else "event_id"
    shared = 0
    for val, grp in df.groupby(other):
        if grp.split.nunique() > 1:
            shared += 1
    label = "stations" if split_by == "event" else "events"
    print(f"\n  [leakage] {shared}/{df[other].nunique()} {label} appear in more than one split.")
    if split_by == "event":
        print("            Events are disjoint (the per-event magnitude label cannot leak);"
              "\n            shared stations mean site response is seen across splits.")
    else:
        print("            Stations are disjoint, but shared events mean the SAME magnitude"
              "\n            label appears in train and test -- usually the worse leak for"
              "\n            regression. Consider --split-by event.")

    n_dist = int(df.distance_km.notna().sum())
    print(f"\n  distance_km present for {n_dist}/{len(df)} rows"
          + ("" if n_dist else "  (pass --station-catalog to enable it)"))
    for col in snr_cols:
        print(f"  {col} present for {int(df[col].notna().sum())}/{len(df)} rows")
    print("\n[COMPLETE] Regression dataset ready.")
