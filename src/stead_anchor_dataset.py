"""
Builds a STEAD {seq, img} dataset that a model trained on the local Turkish
corpus can be evaluated on directly -- a genuine cross-corpus test, not a
re-training.

**Why STEAD anchoring is easy where the local data was hard.** The local
pipeline had to *infer* the P arrival, first with an STA/LTA pick (which
cannot fire before its LTA warm-up ends, so it never actually found the
arrival) and then from catalog geometry via TauP (~0.6 s MAD). STEAD ships
`p_arrival_sample` as an **analyst pick**, per trace, in the HDF5 attributes.
There is nothing to estimate: the arrival index is given. That makes this the
cleanest anchoring in the project, and it is why STEAD is worth the effort as
a validation corpus.

**The one thing that must not drift: the transform.** A cross-corpus number is
only meaningful if STEAD windows are turned into tensors by the *same* code
that produced the training tensors. So this imports `SpectrogramEncoder` and
`clean_and_filter_1d` from `seismic_cli` rather than reimplementing them --
`stead_data_process/main.py` keeps its own copy of `clean_and_filter_1d`, and
two copies drift. Everything here is a thin adapter around the real encoder.

**Component order.** STEAD stores (6000, 3) as **E, N, Z**. `seismic_cli`
orders components by role as **Z, N, E** (`_COMPONENT_ROLES`), so every trace
is reversed on load. Verified three ways: `stead_data_process/main.py` maps
index 0 -> blue and index 2 -> red against the pipeline's R=Z/G=N/B=E
convention; STEAD's own documentation states E, N, Z; and measured over 800
noise traces, index 2 carries the lowest power, as a vertical should.

**Amplitude handling.** Windows are standardized against each STEAD station's
own long-term noise level, computed from STEAD noise traces -- the equivalent
of `--baseline` on the local pipeline, and required for comparability, since
per-window standardization would delete the amplitude information the model
was trained to use. Stations with no usable noise profile are skipped rather
than silently falling back, because a fallback here would look like a quiet
station rather than a missing baseline.

Usage:
    python3 src/stead_anchor_dataset.py \\
        --noise-hdf5  /home/hogib/Projects/stead_data_process/raw/noise/chunk1.hdf5 \\
        --event-hdf5  /home/hogib/Projects/stead_data_process/raw/events/chunk2.hdf5 \\
        --output-dir  raw/data/dataset_stead_specdual_6s \\
        --window-seconds 6 --pre-arrival-seconds 2.0

`--event-hdf5` may be omitted while the earthquake archive is still
downloading; the script will then build the noise half only and say so.
"""

import argparse
import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py  # noqa: E402
from seismic_cli.core import clean_and_filter_1d, standardize  # noqa: E402
from seismic_cli.spectrogram import SpectrogramEncoder  # noqa: E402

# STEAD is (samples, 3) ordered E, N, Z; seismic_cli orders Z, N, E.
STEAD_TO_PIPELINE = [2, 1, 0]
COMPONENT_NAMES = ("Z", "N", "E")
STEAD_FS = 100.0
STEAD_TRACE_SAMPLES = 6000


def station_key_of(attrs) -> str:
    net = str(attrs.get("network_code", "") or "").strip()
    sta = str(attrs.get("receiver_code", "") or "").strip()
    return f"{net}.{sta}" if net else sta


def as_int(value) -> Optional[int]:
    """STEAD writes missing numeric attributes as an empty string, not NaN."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def as_float(value) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if math.isfinite(v) else None


def load_trace(dataset) -> Optional[np.ndarray]:
    """Reads one STEAD trace as (samples, 3) reordered to Z, N, E."""
    arr = dataset[:]
    if arr.ndim != 2:
        return None
    if arr.shape[0] < arr.shape[1]:
        arr = arr.T
    if arr.shape[1] != 3:
        return None
    return np.asarray(arr, dtype=np.float64)[:, STEAD_TO_PIPELINE]


def clean_window(win: np.ndarray, freqmin: float, freqmax: float) -> np.ndarray:
    """Per-component clean + bandpass, using seismic_cli's own function."""
    out = np.zeros_like(win, dtype=np.float64)
    for i in range(win.shape[1]):
        out[:, i] = clean_and_filter_1d(win[:, i].copy(), STEAD_FS, freqmin, freqmax)
    return out


def build_station_baselines(noise_path: Path, keys: List[str], freqmin: float,
                            freqmax: float, max_per_station: int,
                            encoder: SpectrogramEncoder, min_seconds: float = 60.0):
    """
    Builds BOTH per-station references in one pass over the noise:

      amplitude : (station, comp) -> (mean, std) of cleaned noise, the STEAD
                  analogue of `compute_station_noise_baselines`, used to
                  standardize `seq`.
      spectral  : (station, comp) -> median dB per frequency bin, the STEAD
                  analogue of `compute_station_spectral_baselines`, used by
                  `normalize_spec(normalize="station")` so STEAD images mean
                  "dB above this station's own noise floor" -- the same
                  quantity the model was trained on. Per-window z-scoring
                  here would hand the model a different distribution and any
                  drop in score would measure the mismatch, not generalization.

    Both must come from STEAD's own noise; carrying the Turkish profiles over
    would offset every value by an unrelated instrument's floor.
    """
    import torch

    torch_mod, spec_tf, db_tf = encoder._transforms()
    per_station: Dict[str, List[str]] = defaultdict(list)

    with h5py.File(noise_path, "r") as f:
        group = f["data"]
        for k in keys:
            per_station[station_key_of(group[k].attrs)].append(k)

        amp: Dict[Tuple[str, str], List[np.ndarray]] = defaultdict(list)
        frames: Dict[Tuple[str, str], List] = defaultdict(list)

        for sta, sta_keys in per_station.items():
            for k in sta_keys[:max_per_station]:
                arr = load_trace(group[k])
                if arr is None or arr.shape[0] < STEAD_TRACE_SAMPLES // 2:
                    continue
                cleaned = clean_window(arr, freqmin, freqmax)
                for i, comp in enumerate(COMPONENT_NAMES):
                    amp[(sta, comp)].append(cleaned[:, i])
                    t = torch_mod.from_numpy(
                        np.ascontiguousarray(cleaned[:, i])).float().unsqueeze(0)
                    frames[(sta, comp)].append(db_tf(spec_tf(t))[0])   # (freq, time)

    amplitude = {}
    for key, chunks in amp.items():
        stacked = np.concatenate(chunks)
        sd = float(np.std(stacked))
        if sd > 0 and math.isfinite(sd):
            amplitude[key] = (float(np.mean(stacked)), sd)

    min_frames = max(1, int(min_seconds * STEAD_FS / encoder.hop_length))
    spectral, rejected = {}, 0
    for key, chunks in frames.items():
        allf = torch.cat(chunks, dim=1)
        if allf.shape[1] < min_frames:
            rejected += 1
            continue
        spectral[key] = allf.median(dim=1).values.numpy()

    return amplitude, spectral, rejected


def encode_window(cleaned: np.ndarray, sta: str, encoder: SpectrogramEncoder,
                  baselines) -> Optional[dict]:
    """
    Turns one cleaned (samples, 3) window into the {seq, img} pair the
    classifier expects. Returns None unless this station has BOTH an amplitude
    baseline and a spectral profile for all three components -- `normalize_spec`
    falls back to per-window z-scoring when a profile is missing, which would
    quietly emit a tensor from a different distribution than the model was
    trained on rather than failing.
    """
    if any((sta, c) not in encoder.noise_profiles for c in COMPONENT_NAMES):
        return None
    seqs = []
    for i, comp in enumerate(COMPONENT_NAMES):
        mu_sigma = baselines.get((sta, comp))
        if mu_sigma is None:
            return None
        mu, sigma = mu_sigma
        seqs.append(standardize(cleaned[:, i], mu=mu, sigma=sigma))
    seq = np.stack(seqs, axis=-1).astype(np.float32)

    spec = encoder.normalize_spec(
        encoder.spec_db(cleaned, STEAD_FS), sta, list(COMPONENT_NAMES))
    return {"seq": torch.from_numpy(seq), "img": spec.contiguous().float()}


def collect_keys(path: Path) -> List[str]:
    with h5py.File(path, "r") as f:
        return list(f["data"].keys())


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--noise-hdf5", required=True, type=Path)
    p.add_argument("--event-hdf5", type=Path, default=None,
                   help="Omit while the earthquake archive is still downloading.")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--window-seconds", type=float, default=6.0)
    p.add_argument("--pre-arrival-seconds", type=float, default=2.0,
                   help="Must match the training set. The catalog-anchored local "
                        "dataset uses 2.0 s; the older STA/LTA-anchored one used "
                        "1.2 s (0.2 x 6 s).")
    p.add_argument("--freqmin", type=float, default=1.0)
    p.add_argument("--freqmax", type=float, default=45.0)
    p.add_argument("--n-fft", type=int, default=256)
    p.add_argument("--hop-length", type=int, default=None)
    p.add_argument("--top-db", type=float, default=80.0)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap traces per class (smoke test).")
    p.add_argument("--baseline-traces-per-station", type=int, default=8)
    p.add_argument("--min-snr-db", type=float, default=None,
                   help="Drop earthquake traces below this STEAD snr_db. Left off "
                        "by default -- filtering by SNR would rebuild exactly the "
                        "selection effect the local rebuild removed.")
    p.add_argument("--min-magnitude", type=float, default=None,
                   help="Keep only events at or above this magnitude. STEAD runs far "
                        "smaller than the Turkish corpus (median M1.09 vs M2.30, 44%% "
                        "below M1.0), so an unfiltered score mixes cross-corpus "
                        "generalization with a harder task. Pass 2.0 to match.")
    p.add_argument("--max-distance-km", type=float, default=None,
                   help="Keep only events within this epicentral distance. The Turkish "
                        "download radius caps at ~56 km while STEAD reaches 329 km, so "
                        "pass 56 alongside --min-magnitude 2.0 for a like-for-like set.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    target_samples = int(round(STEAD_FS * args.window_seconds))
    pre_samples = int(round(STEAD_FS * args.pre_arrival_seconds))
    rng = random.Random(args.seed)

    print("=" * 66)
    print("STEAD CROSS-CORPUS DATASET (analyst P picks, no trigger gate)")
    print("=" * 66)
    print(f"window {args.window_seconds}s = {target_samples} samples @ {STEAD_FS} Hz")
    print(f"pre-arrival {args.pre_arrival_seconds}s = {pre_samples} samples")
    print(f"components reordered E,N,Z -> Z,N,E via {STEAD_TO_PIPELINE}")

    print("\n[1/4] Indexing noise...")
    noise_keys = collect_keys(args.noise_hdf5)
    rng.shuffle(noise_keys)
    print(f"      {len(noise_keys)} noise traces")

    print("[2/4] Building per-station amplitude + spectral baselines from STEAD's own noise...")
    encoder = SpectrogramEncoder(
        n_fft=args.n_fft, hop_length=args.hop_length, top_db=args.top_db,
        nominal_fs=STEAD_FS, window_seconds=args.window_seconds,
        normalize="station", noise_profiles={})
    baselines, profiles, rejected = build_station_baselines(
        args.noise_hdf5, noise_keys, args.freqmin, args.freqmax,
        args.baseline_traces_per_station, encoder)
    encoder.noise_profiles = profiles
    stations = {k[0] for k in baselines}
    print(f"      amplitude: {len(baselines)} (station, component) pairs "
          f"across {len(stations)} stations")
    print(f"      spectral : {len(profiles)} profiles "
          f"({rejected} rejected for under 60s of noise)")
    if not baselines or not profiles:
        print("[ERROR] No usable noise baselines. Cannot continue.")
        return

    for cls in ("00_noise", "01_earthquake"):
        (args.output_dir / cls).mkdir(parents=True, exist_ok=True)

    manifest: List[dict] = []

    print("[3/4] Writing noise windows...")
    kept = skipped_no_baseline = skipped_short = 0
    limit = args.limit or len(noise_keys)
    with h5py.File(args.noise_hdf5, "r") as f:
        group = f["data"]
        for k in noise_keys:
            if kept >= limit:
                break
            arr = load_trace(group[k])
            if arr is None or arr.shape[0] < target_samples:
                skipped_short += 1
                continue
            sta = station_key_of(group[k].attrs)
            start = rng.randrange(0, arr.shape[0] - target_samples + 1)
            cleaned = clean_window(arr[start:start + target_samples], args.freqmin, args.freqmax)
            tensors = encode_window(cleaned, sta, encoder, baselines)
            if tensors is None:
                skipped_no_baseline += 1
                continue
            name = f"{k}_win000.pt"
            torch.save(tensors, args.output_dir / "00_noise" / name)
            manifest.append(dict(cls="00_noise", station_key=sta, trace_name=k,
                                 filename=name, start_sample=start,
                                 p_arrival_sample="", snr_db="", magnitude="",
                                 distance_km=""))
            kept += 1
    print(f"      kept {kept} | no baseline {skipped_no_baseline} | too short {skipped_short}")
    n_noise = kept

    n_events = 0
    if args.event_hdf5 and args.event_hdf5.exists():
        print("[4/4] Writing earthquake windows anchored on analyst P picks...")
        event_keys = collect_keys(args.event_hdf5)
        rng.shuffle(event_keys)
        kept = no_pick = no_baseline = out_of_range = low_snr = off_distribution = 0
        limit = args.limit or n_noise or len(event_keys)
        with h5py.File(args.event_hdf5, "r") as f:
            group = f["data"]
            for k in event_keys:
                if kept >= limit:
                    break
                attrs = group[k].attrs
                p_sample = as_int(attrs.get("p_arrival_sample"))
                if p_sample is None:
                    no_pick += 1
                    continue
                if args.min_magnitude is not None:
                    mag = as_float(attrs.get("source_magnitude"))
                    if mag is None or mag < args.min_magnitude:
                        off_distribution += 1
                        continue
                if args.max_distance_km is not None:
                    dist = as_float(attrs.get("source_distance_km"))
                    if dist is None or dist > args.max_distance_km:
                        off_distribution += 1
                        continue
                snr = attrs.get("snr_db")
                if args.min_snr_db is not None:
                    snr_val = as_float(str(snr).strip("[] ").split()[0]
                                       if str(snr).strip() else "")
                    if snr_val is None or snr_val < args.min_snr_db:
                        low_snr += 1
                        continue
                arr = load_trace(group[k])
                if arr is None:
                    out_of_range += 1
                    continue
                start = p_sample - pre_samples
                if start < 0 or start + target_samples > arr.shape[0]:
                    out_of_range += 1
                    continue
                sta = station_key_of(attrs)
                cleaned = clean_window(arr[start:start + target_samples],
                                       args.freqmin, args.freqmax)
                tensors = encode_window(cleaned, sta, encoder, baselines)
                if tensors is None:
                    no_baseline += 1
                    continue
                name = f"{k}_win000.pt"
                torch.save(tensors, args.output_dir / "01_earthquake" / name)
                manifest.append(dict(
                    cls="01_earthquake", station_key=sta, trace_name=k, filename=name,
                    start_sample=start, p_arrival_sample=p_sample,
                    snr_db=str(snr).replace("\n", " "),
                    magnitude=attrs.get("source_magnitude", ""),
                    distance_km=attrs.get("source_distance_km", "")))
                kept += 1
        n_events = kept
        print(f"      kept {kept} | no P pick {no_pick} | no baseline {no_baseline} "
              f"| window out of range {out_of_range} | below --min-snr-db {low_snr} "
              f"| outside magnitude/distance match {off_distribution}")
    else:
        print("[4/4] No --event-hdf5 given (or file missing) -- noise half only.")
        print("      Re-run with --event-hdf5 once the earthquake archive finishes.")

    if manifest:
        with open(args.output_dir / "manifest.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
            w.writeheader()
            w.writerows(manifest)

    print("\n" + "=" * 66)
    print(f"[done] {n_noise} noise + {n_events} earthquake windows -> {args.output_dir}")
    print(f"[done] manifest -> {args.output_dir / 'manifest.csv'}")
    if n_events and n_noise and abs(n_events - n_noise) / max(n_events, n_noise) > 0.02:
        print(f"[warn] classes are unbalanced ({n_noise} vs {n_events}); "
              f"report balanced accuracy / AUC rather than raw accuracy.")
    print("=" * 66)


if __name__ == "__main__":
    main()
