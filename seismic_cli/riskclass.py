"""
Three-class risk dataset: noise / low-risk (M < threshold) / high-risk
(M >= threshold), on encoded short windows.

This extends the {img, aux} pattern `regression.py` and `ram_aux.py` already
use, but `regression.py` is deliberately earthquake-only ("no noise class" --
see its docstring): it has no code path for a window with no catalog
magnitude at all. Folding a noise class into the SAME manifest as the two
earthquake risk bins needs that path added, plus a station-disjoint split
that spans all three classes together (regression.py's own `split_by`
options are earthquake-only) so a noise station and an earthquake station
never collide across train/val/test the way `core.run_balanced_preprocessing`
already guards against for the plain detection task.

**Splitting.** Stations from BOTH eq_dir and noise_dir are pooled and
assigned to splits together, by total window count, using the same
largest-relative-deficit algorithm `regression.py --split-by station` uses
(reused verbatim in spirit, generalized to two source directories). This is
a classification task (three discrete labels), not the continuous-magnitude
regression task `split_by="event"` exists to protect -- the same event
recorded at two stations in different splits is not a leak of a *continuous*
value the model could otherwise memorize, only of a coarse two-way bucket
already implied by the catalog everyone has access to. Event-level overlap
across splits is measured and printed as a diagnostic (mirroring
regression.py's own leakage report) rather than hard-enforced, since forcing
BOTH station- and event-disjointness simultaneously is a harder combinatorial
constraint (an event recorded at N stations would force all N into the same
split) not attempted here.

**Balancing.** `--balance-ratio` (default 4.0) caps `01_low_risk` and
`00_noise` at `ratio * count(02_high_risk)` PER SPLIT, applied at the file
level before encoding (not after), so windows that would be discarded are
never encoded in the first place. This is deliberately not strict 1:1:1
(which would throw away most of the abundant classes) and not full natural
imbalance (which would swamp the rare class during training).

**Dead-instrument rejection** (`--min-log-snr`, default -3.0). A window
whose RMS is below `exp(-3) ~ 5%` of its own station's long-term noise
baseline is rejected as an instrument fault, not kept as unusually quiet
data. Genuine ambient noise does not sit 20x below the station's own noise
floor; a recording that does is stuck, dead, or has had a gain change.

This was found empirically and the criterion is worth stating precisely,
because "drop the data the model gets wrong" would be cheating. Station
`6G.MADM` contributed 199 noise windows whose raw traces span ~58 counts
on a ~5.38-million-count DC offset with only ~50 unique values across
30,001 samples -- a stuck digitizer, verified by reading the MiniSEED
directly, not inferred from model error. Its RMS (~6 counts) against its
own baseline (~975-3118) gives log_snr ~ -6. The pooled noise log_snr
distribution has a clean gap here: 5th percentile -2.67, then an isolated
cluster at -6.0. The threshold is applied uniformly to every class and
every split, and is decided by the instrument physics plus that gap, not
by which windows any model misclassifies. It also removes 18 earthquake
windows, on the same reasoning -- a genuine event cannot be 20x quieter
than its own station's noise floor either.
"""

import concurrent.futures
import csv
import math
import multiprocessing
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from seismic_cli.core import (_cap_station_windows, clean_and_filter_1d,
                              compute_station_noise_baselines, scan_single_mseed,
                              select_components, window_array_indexed)
from seismic_cli.regression import (_station_coord, haversine_km, load_event_catalog,
                                    load_station_coords, parse_event_id)

RISK_CLASSES = ["00_noise", "01_low_risk", "02_high_risk"]
MANIFEST_COLUMNS = ["split", "risk_class", "station_key", "event_id", "file_path",
                    "filename", "fs", "magnitude", "log_snr", "distance_km"]


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def _process_earthquake_risk_file(args):
    (file_path, assignments, station_baselines, encoder, fs, window_seconds,
     overlap, max_gap_fraction, freqmin, freqmax, event_meta, station_coords,
     mag_threshold, min_log_snr) = args
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
            return []

        stations: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, float]]] = {}
        for tr in st:
            sta_key = f"{tr.stats.network}.{tr.stats.station}"
            if sta_key not in assignments:
                continue
            chan = tr.stats.channel[-1].upper()
            existing = stations.setdefault(sta_key, {}).get(chan)
            if existing is not None and len(existing[0]) >= tr.stats.npts:
                continue
            from seismic_cli.core import _masked_to_filled
            data, gap_mask = _masked_to_filled(tr.data)
            stations[sta_key][chan] = (data, gap_mask, tr.stats.sampling_rate)

        magnitude = meta["magnitude"]
        risk_class = "02_high_risk" if magnitude >= mag_threshold else "01_low_risk"

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

                snrs = []
                for i, comp in enumerate(selection):
                    _mu, sigma_noise = station_baselines.get((sta_key, comp), (None, None))
                    sigma_win = float(np.std(cleaned[:, i]))
                    if sigma_noise and sigma_noise > 0 and sigma_win > 0:
                        snrs.append(math.log(sigma_win / sigma_noise))
                log_snr = float(np.mean(snrs)) if snrs else float("nan")
                # Dead/stuck instrument -- see module docstring.
                if min_log_snr is not None and snrs and log_snr < min_log_snr:
                    continue

                stem = f"{file_path.stem}_{sta_key}_win{w_idx:03d}"
                filename = encoder(cleaned, fs_station, sta_key, selection,
                                   station_baselines, out_dir, stem)
                rows.append((split_name, risk_class, sta_key, str(event_id), str(file_path),
                             filename, fs_station, magnitude, log_snr, dist_km))
        return rows
    except Exception as e:
        return f"[WARN] Failed file {file_path.stem}: {e}"


def _process_noise_risk_file(args):
    (file_path, assignments, station_baselines, encoder, fs, window_seconds,
     overlap, max_gap_fraction, freqmin, freqmax, min_log_snr) = args
    from obspy import read
    from seismic_cli.core import _masked_to_filled
    try:
        st = read(str(file_path))
        try:
            st.merge(method=1)
        except Exception as e:
            return f"[WARN] merge failed for {file_path.name}: {e}"

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

            noise_data = np.column_stack([c[:min_len] for c in raw])
            gap_mask = np.column_stack([m[:min_len] for m in masks])
            windows, _ = window_array_indexed(
                noise_data, gap_mask, fs=fs_station, window_seconds=window_seconds,
                overlap=overlap, max_gap_fraction=max_gap_fraction,
            )

            split_name, out_dir, quota = assignments[sta_key]
            if quota is not None and len(windows) > quota:
                sel = np.linspace(0, len(windows) - 1, quota).round().astype(int)
                windows = [windows[i] for i in sorted(set(sel.tolist()))]

            for w_idx, win in windows:
                cleaned = np.zeros_like(win, dtype=np.float64)
                for i in range(win.shape[1]):
                    cleaned[:, i] = clean_and_filter_1d(win[:, i], fs_station, freqmin, freqmax)

                snrs = []
                for i, comp in enumerate(selection):
                    _mu, sigma_noise = station_baselines.get((sta_key, comp), (None, None))
                    sigma_win = float(np.std(cleaned[:, i]))
                    if sigma_noise and sigma_noise > 0 and sigma_win > 0:
                        snrs.append(math.log(sigma_win / sigma_noise))
                log_snr = float(np.mean(snrs)) if snrs else float("nan")
                # Dead/stuck instrument -- see module docstring.
                if min_log_snr is not None and snrs and log_snr < min_log_snr:
                    continue

                stem = f"{file_path.stem}_{sta_key}_win{w_idx:03d}"
                filename = encoder(cleaned, fs_station, sta_key, selection,
                                   station_baselines, out_dir, stem)
                rows.append((split_name, "00_noise", sta_key, "", str(file_path),
                             filename, fs_station, float("nan"), log_snr, float("nan")))
        return rows
    except Exception as e:
        return f"[WARN] Failed file {file_path.stem}: {e}"


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_riskclass_preprocessing(
    eq_dir: str,
    noise_dir: str,
    catalog_path: str,
    output_dir: str,
    encoder,
    station_catalog: Optional[str] = None,
    mag_threshold: float = 4.0,
    balance_ratio: float = 4.0,
    min_log_snr: Optional[float] = -3.0,
    split_ratios: tuple = (0.70, 0.15, 0.15),
    fs: float = 100.0,
    window_seconds: float = 3.0,
    overlap: float = 0.5,
    max_windows_per_station: Optional[int] = None,
    freqmin: float = 1.0,
    freqmax: float = 45.0,
    min_baseline_seconds: float = 60.0,
    max_gap_fraction: float = 0.05,
    num_cores: Optional[int] = None,
    seed: int = 42,
):
    print("=" * 60)
    print(f"RISK-CLASS DATASET  (noise / <M{mag_threshold} / >=M{mag_threshold}, "
          f"balance-ratio {balance_ratio})")
    if min_log_snr is not None:
        print(f"  dead-instrument rejection: dropping windows with log_snr < {min_log_snr} "
              f"(RMS below {100 * math.exp(min_log_snr):.1f}% of the station's own noise floor)")
    print("=" * 60)

    event_meta = load_event_catalog(catalog_path)
    station_coords = load_station_coords(station_catalog)
    station_baselines, _ = compute_station_noise_baselines(
        noise_dir, fs=fs, freqmin=freqmin, freqmax=freqmax,
        min_baseline_seconds=min_baseline_seconds,
    )

    splits = ["train", "val", "test"]
    out_paths = {s: Path(output_dir) / s for s in splits}
    for d in out_paths.values():
        d.mkdir(parents=True, exist_ok=True)

    if num_cores is None:
        num_cores = max(1, multiprocessing.cpu_count() - 1)

    # ---- PHASE 1: scan headers, classify earthquake files by risk bin ------
    print("\n[PHASE 1] Scanning earthquake and noise files...")
    eq_path, noise_path = Path(eq_dir), Path(noise_dir)
    eq_files = list(eq_path.rglob("*.mseed"))
    noise_files = list(noise_path.rglob("*.mseed"))

    scan_args_eq = [(fp, fs, window_seconds, overlap) for fp in eq_files]
    scan_args_no = [(fp, fs, window_seconds, overlap) for fp in noise_files]

    eq_counts: Dict[Path, Dict[str, int]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as ex:
        for fpath, counts in ex.map(scan_single_mseed, scan_args_eq):
            if counts:
                eq_counts[fpath] = counts
    no_counts: Dict[Path, Dict[str, int]] = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as ex:
        for fpath, counts in ex.map(scan_single_mseed, scan_args_no):
            if counts:
                no_counts[fpath] = counts

    # file -> risk_class, dropping earthquake files with no catalog magnitude
    eq_file_class: Dict[Path, str] = {}
    for fpath in eq_counts:
        ev = parse_event_id(fpath.stem)
        meta = event_meta.get(ev)
        if meta is None:
            continue
        eq_file_class[fpath] = ("02_high_risk" if meta["magnitude"] >= mag_threshold
                                else "01_low_risk")
    n_unlabelled = len(eq_counts) - len(eq_file_class)
    print(f"  -> earthquake: {len(eq_file_class)} labelled files "
          f"({n_unlabelled} had no catalog magnitude and were dropped)")
    print(f"  -> noise: {len(no_counts)} files")

    # ---- PHASE 2: unified station-disjoint split assignment ----------------
    print("\n[PHASE 2] Allocating station-disjoint splits (unified across all three classes)...")
    station_totals: Dict[str, int] = {}
    for fpath, counts in eq_counts.items():
        if fpath not in eq_file_class:
            continue
        for sta, w in counts.items():
            station_totals[sta] = station_totals.get(sta, 0) + w
    for fpath, counts in no_counts.items():
        for sta, w in counts.items():
            station_totals[sta] = station_totals.get(sta, 0) + w

    stations = sorted(station_totals)
    rng = random.Random(seed)
    rng.shuffle(stations)
    total_windows = sum(station_totals.values())
    targets = {s: r * total_windows for s, r in zip(splits, split_ratios)}
    counts_by_split = {s: 0 for s in splits}
    station_split: Dict[str, str] = {}
    for sta in stations:
        best = max(splits, key=lambda s: (targets[s] - counts_by_split[s]) / max(targets[s], 1.0))
        station_split[sta] = best
        counts_by_split[best] += station_totals[sta]
    for s in splits:
        print(f"     {s:5s}: ~{counts_by_split[s]} windows across all classes "
              f"(target {targets[s]:.0f})")

    # ---- PHASE 3: per-split, per-class window totals from header counts ----
    def _class_totals(file_class_map, counts_map, class_of_file):
        totals = {s: {c: 0 for c in RISK_CLASSES} for s in splits}
        for fpath, counts in counts_map.items():
            cls = class_of_file(fpath)
            if cls is None:
                continue
            for sta, w in counts.items():
                totals[station_split[sta]][cls] += w
        return totals

    per_split_class = {s: {c: 0 for c in RISK_CLASSES} for s in splits}
    for fpath, counts in eq_counts.items():
        cls = eq_file_class.get(fpath)
        if cls is None:
            continue
        for sta, w in counts.items():
            per_split_class[station_split[sta]][cls] += w
    for fpath, counts in no_counts.items():
        for sta, w in counts.items():
            per_split_class[station_split[sta]]["00_noise"] += w

    print("\n[PHASE 3] Natural class totals per split (before --balance-ratio cap):")
    for s in splits:
        row = per_split_class[s]
        print(f"     {s:5s}: noise={row['00_noise']:6d}  low_risk={row['01_low_risk']:6d}  "
              f"high_risk={row['02_high_risk']:6d}")

    # per-split cap for noise / low_risk = balance_ratio * high_risk count
    split_caps: Dict[str, Dict[str, Optional[int]]] = {}
    for s in splits:
        hi = per_split_class[s]["02_high_risk"]
        cap = None if hi == 0 else int(round(balance_ratio * hi))
        split_caps[s] = {"02_high_risk": None, "01_low_risk": cap, "00_noise": cap}

    # ---- PHASE 4: build per-file quotas honouring both max_windows_per_station
    #      and the balance-ratio cap, then encode -----------------------------
    def _build_assignments(file_class_map, counts_map, class_of_file):
        """Returns {file_path: {station: (split, out_dir, quota)}} for one source dir,
        applying the balance-ratio cap by randomly dropping whole files per (split, class)
        once that class's running total exceeds its cap -- cheaper than per-window
        subsampling and avoids encoding windows that will be discarded anyway."""
        by_split_class: Dict[Tuple[str, str], List[Tuple[Path, str, int]]] = {}
        for fpath, counts in counts_map.items():
            cls = class_of_file(fpath)
            if cls is None:
                continue
            for sta, w in counts.items():
                by_split_class.setdefault((station_split[sta], cls), []).append((fpath, sta, w))

        cap_rng = random.Random(seed + 7)
        kept_station_files: Dict[str, List[Tuple[Path, int]]] = {}
        for (split_name, cls), entries in by_split_class.items():
            cap = split_caps[split_name].get(cls)
            if cap is None:
                for fpath, sta, w in entries:
                    kept_station_files.setdefault(sta, []).append((fpath, w))
                continue
            shuffled = list(entries)
            cap_rng.shuffle(shuffled)
            running = 0
            for fpath, sta, w in shuffled:
                if running >= cap:
                    break
                kept_station_files.setdefault(sta, []).append((fpath, w))
                running += w

        per_station = [(sta, sum(w for _, w in v), v) for sta, v in kept_station_files.items()]
        capped = _cap_station_windows(per_station, max_windows_per_station, random.Random(seed + 11))
        quota_lookup = {(sta, fp): q for sta, _tot, contribs in capped for fp, _w, q in contribs}
        kept_files = {(sta, fp) for sta, _tot, contribs in capped for fp, _w, _q in contribs}

        assignments: Dict[Path, Dict[str, Tuple[str, Path, Optional[int]]]] = {}
        for (split_name, cls), entries in by_split_class.items():
            for fpath, sta, _w in entries:
                if (sta, fpath) not in kept_files:
                    continue
                assignments.setdefault(fpath, {})[sta] = (
                    split_name, out_paths[split_name], quota_lookup.get((sta, fpath))
                )
        return assignments

    eq_assignments = _build_assignments(eq_file_class, eq_counts, lambda fp: eq_file_class.get(fp))
    no_assignments = _build_assignments({}, no_counts, lambda fp: "00_noise")

    print(f"\n[PHASE 4] Encoding on {num_cores} cores...")
    mp_context = (multiprocessing.get_context("spawn")
                  if getattr(encoder, "requires_spawn", False) else None)
    if mp_context:
        print("       (using 'spawn' workers -- required for torch-based encoders)")

    eq_tasks = [(fp, asg, station_baselines, encoder, fs, window_seconds, overlap,
                max_gap_fraction, freqmin, freqmax, event_meta, station_coords,
                mag_threshold, min_log_snr)
               for fp, asg in eq_assignments.items()]
    no_tasks = [(fp, asg, station_baselines, encoder, fs, window_seconds, overlap,
                max_gap_fraction, freqmin, freqmax, min_log_snr)
               for fp, asg in no_assignments.items()]

    manifest: List[tuple] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores, mp_context=mp_context) as ex:
        for res in ex.map(_process_earthquake_risk_file, eq_tasks):
            if isinstance(res, str):
                print(res)
            elif res:
                manifest.extend(res)
        for res in ex.map(_process_noise_risk_file, no_tasks):
            if isinstance(res, str):
                print(res)
            elif res:
                manifest.extend(res)

    if not manifest:
        print("[ERROR] No windows were written.")
        return

    manifest_path = Path(output_dir) / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(MANIFEST_COLUMNS)
        wr.writerows(manifest)

    df = pd.DataFrame(manifest, columns=MANIFEST_COLUMNS)
    print(f"\n[PHASE 5] Wrote {len(df)} windows to {manifest_path}")
    print("\n  Final class counts per split:")
    for s in splits:
        sub = df[df.split == s]
        counts = sub.risk_class.value_counts()
        print(f"     {s:5s}: " + "  ".join(f"{c}={counts.get(c, 0)}" for c in RISK_CLASSES))

    eq_sub = df[df.risk_class != "00_noise"]
    shared_events = 0
    if len(eq_sub):
        for _ev, grp in eq_sub.groupby("event_id"):
            if grp.split.nunique() > 1:
                shared_events += 1
        print(f"\n  [diagnostic] {shared_events}/{eq_sub.event_id.nunique()} earthquake events "
              f"appear in more than one split (station-disjoint, not event-disjoint -- see "
              f"module docstring; the label at risk is a coarse two-way bucket, not a "
              f"continuous value).")

    n_dist = int(df.distance_km.notna().sum())
    n_eq_dist_eligible = len(eq_sub)
    print(f"\n  distance_km present for {n_dist}/{n_eq_dist_eligible} earthquake rows "
          f"(NaN for all noise rows by design)"
          + ("" if n_dist else "  (pass --station-catalog to enable it)"))
    print(f"  log_snr present for {int(df.log_snr.notna().sum())}/{len(df)} rows")
