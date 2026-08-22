"""
Shared RAM-transform and dataset-generation logic used by the `generate-dataset`
CLI command. Baseline standardization is a flag, not a separate script --
`use_baseline_standardization=False` reproduces per-window self-standardization;
`True` standardizes each channel against that station's own long-term noise
statistics instead (falling back to self-standardization per-channel for any
station without enough noise data to build a reliable baseline).

Split allocation is UNIFIED across classes: each station is assigned to exactly
one of train/val/test, and both its earthquake and noise windows land in that
same split. (Allocating the two classes independently let the same station sit
in train for one class and test for the other -- with ~97% of earthquake
stations also having noise data, that quietly broke the station-disjoint
guarantee for nearly every station.)
"""

import concurrent.futures
import csv
import math
import multiprocessing
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.signal as signal
from obspy import read
from PIL import Image

# ALGORITHM FUNCTIONS

def standardize(x: np.ndarray, mu: Optional[float] = None, sigma: Optional[float] = None,
                 eps: float = 1e-12) -> np.ndarray:
    """
    Standardizes x using EITHER a provided (mu, sigma) -- e.g. a station's
    long-term noise baseline -- OR, if not provided, the window's own
    mean/std (plain per-window self-standardization).
    """
    x = np.asarray(x, dtype=np.float64)
    if mu is None:
        mu = np.mean(x)
    if sigma is None:
        sigma = np.std(x)
    if sigma < eps:
        sigma = eps
    return (x - mu) / sigma


def reshape_to_target_n(x: np.ndarray, target_n: int) -> Tuple[np.ndarray, int]:
    m = len(x)
    d = max(2, math.ceil(m / target_n))
    total_needed = d * target_n
    if total_needed > m:
        x = np.pad(x, (0, total_needed - m), mode='constant', constant_values=0)
    else:
        x = x[:total_needed]
    return x.reshape((target_n, d)).T, d


def ram_matrix(x: np.ndarray, target_n: int, mu: Optional[float] = None,
               sigma: Optional[float] = None, eps: float = 1e-12) -> Tuple[np.ndarray, int]:
    x_std = standardize(x, mu=mu, sigma=sigma, eps=eps)
    M, d = reshape_to_target_n(x_std, target_n=target_n)
    Xbar = np.mean(M, axis=1)
    norm_Xbar = np.linalg.norm(Xbar)
    if norm_Xbar < eps:
        norm_Xbar = eps
    n = M.shape[1]
    betas = np.empty(n, dtype=np.float64)
    for i in range(n):
        Xi = M[:, i]
        norm_Xi = np.linalg.norm(Xi)
        if norm_Xi < eps:
            norm_Xi = eps
        cos_val = np.dot(Xi, Xbar) / (norm_Xi * norm_Xbar)
        cos_val = np.clip(cos_val, -1.0, 1.0)
        betas[i] = np.arccos(cos_val)
    return betas[None, :] - betas[:, None], d


def to_uint8(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    mat_clipped = np.clip(mat, -np.pi, np.pi)
    out = (mat_clipped + np.pi) / (2 * np.pi)
    return (out * 255.0).round().astype(np.uint8)


def clean_and_filter_1d(x: np.ndarray, fs: float, freqmin: float, freqmax: float) -> np.ndarray:
    x = signal.detrend(x, type='linear')
    x = signal.detrend(x, type='constant')
    n = len(x)
    taper_len = int(n * 0.05)
    if taper_len > 0:
        window = signal.windows.hann(taper_len * 2)
        x[:taper_len] *= window[:taper_len]
        x[-taper_len:] *= window[-taper_len:]

    nyquist = fs / 2.0
    actual_freqmax = freqmax if nyquist > freqmax else nyquist - 1.0

    if actual_freqmax > freqmin:
        b, a = signal.butter(4, [freqmin, actual_freqmax], btype='bandpass', fs=fs)
        x = signal.filtfilt(b, a, x)

    return x


# CHANNEL SELECTION

# Preference order per component role. Sorting channel letters alphabetically
# and taking the first three could grab e.g. ['1','2','E'] at a station with
# mixed sensor codes -- two horizontals from one instrument plus one from
# another, and no vertical at all. Selecting by explicit role keeps the
# component->color mapping fixed (R=Z, G=N-ish, B=E-ish) for every station.
_COMPONENT_ROLES = (('Z',), ('N', '1'), ('E', '2'))


def select_components(available) -> Optional[Tuple[str, str, str]]:
    """
    Picks one channel letter per role (vertical, north-ish, east-ish) from the
    available component letters. Returns (z, n, e) or None if any role has no
    candidate -- a station without a usable vertical is skipped rather than
    silently fed a horizontal in the Z slot.
    """
    available = set(available)
    chosen = []
    for candidates in _COMPONENT_ROLES:
        for cand in candidates:
            if cand in available:
                chosen.append(cand)
                break
        else:
            return None
    return tuple(chosen)


# WINDOWING

def window_array(data: np.ndarray, fs: float = 100.0, window_seconds: float = 60.0, overlap: float = 0.5) -> List[np.ndarray]:
    """Kept for backward compatibility; generation uses window_array_indexed."""
    return [win for _, win in window_array_indexed(
        data, np.zeros(data.shape, dtype=bool), fs=fs,
        window_seconds=window_seconds, overlap=overlap, max_gap_fraction=1.0,
    )[0]]


def window_array_indexed(
    data: np.ndarray,
    gap_mask: np.ndarray,
    fs: float,
    window_seconds: float,
    overlap: float,
    max_gap_fraction: float = 0.05,
) -> Tuple[List[Tuple[int, np.ndarray]], int]:
    """
    Slides fixed-length windows over `data` (n_samples, n_channels), returning
    (original_window_index, window) pairs plus a count of windows rejected for
    excessive gap content. The original index is what goes into the output
    filename, so downstream reconstruction (eval-sta-lta) can always recover
    the exact sample range regardless of which windows were kept.

    gap_mask marks samples that were missing in the raw data and filled by
    interpolation during merging. A window whose worst channel exceeds
    `max_gap_fraction` of filled samples is rejected -- interpolated stretches
    are synthetic, and a mostly-synthetic "noise" window is not noise.
    """
    target_samples = int(fs * window_seconds)
    step_samples = int(target_samples * (1.0 - overlap))
    if step_samples < 1:
        raise ValueError("Overlap fraction too high; step size must be at least 1 sample.")

    n_samples = data.shape[0]
    windows: List[Tuple[int, np.ndarray]] = []
    n_gap_rejected = 0
    tolerance = int(target_samples * 0.05)

    if n_samples < (target_samples - tolerance):
        return windows, n_gap_rejected

    n_windows = math.ceil((n_samples - target_samples) / step_samples) + 1
    for i in range(n_windows):
        start_idx = i * step_samples
        end_idx = start_idx + target_samples
        win = data[start_idx:end_idx, :]

        if len(win) < (target_samples - tolerance):
            continue

        m = gap_mask[start_idx:end_idx, :]
        if m.size and float(m.mean(axis=0).max()) > max_gap_fraction:
            n_gap_rejected += 1
            continue

        if len(win) < target_samples:
            pad_length = target_samples - len(win)
            win = np.pad(win, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
        windows.append((i, win))

    return windows, n_gap_rejected


def _masked_to_filled(tr_data) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a (possibly masked) trace array into (filled_float64, gap_mask).
    Gaps are filled by linear interpolation so downstream filtering has
    contiguous data, but the mask records exactly which samples are synthetic
    so windowing can reject gap-heavy windows instead of training on them.
    """
    if isinstance(tr_data, np.ma.MaskedArray):
        mask = np.ma.getmaskarray(tr_data).copy()
        x = tr_data.astype(np.float64).filled(np.nan)
        if mask.any():
            idx = np.arange(len(x))
            good = ~mask
            if good.sum() >= 2:
                x[mask] = np.interp(idx[mask], idx[good], x[good])
            else:
                x[mask] = 0.0
        return x, mask
    x = np.asarray(tr_data, dtype=np.float64)
    return x, np.zeros(len(x), dtype=bool)


# --- STATION NOISE BASELINE COMPUTATION ---

def _accumulate_stats(existing: Optional[Tuple[float, float, int]], data: np.ndarray) -> Tuple[float, float, int]:
    s = float(np.sum(data))
    ss = float(np.sum(data.astype(np.float64) ** 2))
    n = len(data)
    if existing is None:
        return (s, ss, n)
    prev_s, prev_ss, prev_n = existing
    return (prev_s + s, prev_ss + ss, prev_n + n)


def _scan_noise_file(args: Tuple[Path, float, float]) -> Dict[Tuple[str, str], Tuple[float, float, int]]:
    """Per-file worker for `compute_station_noise_baselines`: returns this
    file's own (sum, sum-of-squares, n) contribution per (station, component).
    Module-level and picklable so it survives the ProcessPoolExecutor hand-off."""
    file_path, freqmin, freqmax = args
    out: Dict[Tuple[str, str], Tuple[float, float, int]] = {}
    try:
        st = read(str(file_path))
        st.merge(method=1, fill_value='interpolate')
    except Exception:
        return out

    for tr in st:
        sta_key = f"{tr.stats.network}.{tr.stats.station}"
        comp = tr.stats.channel[-1].upper()
        try:
            fs_actual = tr.stats.sampling_rate
            data = tr.data.astype(np.float64)
            if len(data) < int(fs_actual * 10):
                continue
            cleaned = clean_and_filter_1d(data, fs_actual, freqmin, freqmax)
        except Exception:
            continue
        out[(sta_key, comp)] = _accumulate_stats(out.get((sta_key, comp)), cleaned)
    return out


def compute_station_noise_baselines(
    noise_dir: str,
    fs: float,
    freqmin: float,
    freqmax: float,
    min_baseline_seconds: float = 60.0,
    num_cores: Optional[int] = None,
) -> Tuple[Dict[Tuple[str, str], Tuple[float, float]], int]:
    """
    Scans every noise mseed file, groups by (station_key, component), applies
    the SAME cleaning/filtering used on actual training windows, and
    accumulates running mean/std per (station, component). A station/component
    only gets a baseline if it accumulated at least `min_baseline_seconds`
    worth of usable noise data; otherwise it's left out and falls back to
    plain per-window self-standardization when used.

    Fans out one task per file across a `ProcessPoolExecutor` -- this used to
    be a single-threaded loop reading and filtering every noise file in the
    main process, which on a large noise corpus (thousands of files, each a
    full ObsPy read + merge + bandpass) dominated total dataset-generation
    time on one core while the rest of the pipeline used all of them. The
    per-file (sum, sum-of-squares, n) accumulation is associative, so merging
    partial results from workers gives bit-identical baselines to the
    sequential version.
    """
    noise_path = Path(noise_dir)
    if not noise_path.exists():
        print(f"[WARN] Noise directory not found for baseline computation: {noise_path}")
        return {}, 0

    print("\n[BASELINE] Scanning noise files to build per-station long-term baselines...")
    mseed_files = list(noise_path.rglob("*.mseed"))
    print(f"  -> Found {len(mseed_files)} noise files to scan.")

    if num_cores is None:
        num_cores = max(1, multiprocessing.cpu_count() - 1)

    accum: Dict[Tuple[str, str], Tuple[float, float, int]] = {}
    tasks = [(fp, freqmin, freqmax) for fp in mseed_files]
    done = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as ex:
        for partial in ex.map(_scan_noise_file, tasks, chunksize=16):
            done += 1
            if done % 200 == 0:
                print(f"  ...{done}/{len(mseed_files)} noise files processed")
            for key, (s1, ss1, n1) in partial.items():
                if key in accum:
                    s0, ss0, n0 = accum[key]
                    accum[key] = (s0 + s1, ss0 + ss1, n0 + n1)
                else:
                    accum[key] = (s1, ss1, n1)

    min_samples = int(min_baseline_seconds * fs)
    baselines: Dict[Tuple[str, str], Tuple[float, float]] = {}
    n_rejected = 0

    for key, (s, ss, n) in accum.items():
        if n < min_samples:
            n_rejected += 1
            continue
        mu = s / n
        variance = max(ss / n - mu ** 2, 0.0)
        sigma = math.sqrt(variance)
        if sigma < 1e-12:
            n_rejected += 1
            continue
        baselines[key] = (mu, sigma)

    n_stations_with_baseline = len({sta for sta, _ in baselines.keys()})
    print(f"  -> Built baselines for {len(baselines)} (station, component) pairs "
          f"across {n_stations_with_baseline} stations.")
    print(f"  -> Rejected {n_rejected} (station, component) pairs "
          f"(fewer than {min_baseline_seconds:.0f}s of usable noise data, or zero variance).")

    return baselines, n_stations_with_baseline


# PRE-SCAN LOGIC

def scan_single_mseed(args: Tuple[Path, float, float, float]) -> Tuple[Path, Dict[str, int]]:
    file_path, nominal_fs, window_seconds, overlap = args
    try:
        st = read(str(file_path), headonly=True)
    except Exception as e:
        print(f"\n[ERROR] Obspy failed to read {file_path.name}: {e}")
        return file_path, {}

    # comp -> (max npts across segments, sampling rate), grouped per station.
    # Each station's window count is computed with ITS OWN sampling rate --
    # a file can legitimately contain stations at different rates, and using
    # the first trace's rate for everyone mis-sizes every other station's
    # windows.
    stations: Dict[str, Dict[str, Tuple[int, float]]] = {}
    for tr in st:
        sta_key = f"{tr.stats.network}.{tr.stats.station}"
        chan = tr.stats.channel[-1].upper()
        if sta_key not in stations:
            stations[sta_key] = {}
        prev_npts, _ = stations[sta_key].get(chan, (0, tr.stats.sampling_rate))
        stations[sta_key][chan] = (max(prev_npts, tr.stats.npts), tr.stats.sampling_rate)

    station_window_counts = {}
    for sta_key, channels in stations.items():
        selection = select_components(channels.keys())
        if selection is None:
            continue

        rates = {channels[c][1] for c in selection}
        if len(rates) != 1:
            continue  # inconsistent sampling rates across components
        fs_station = rates.pop()

        target_samples = int(fs_station * window_seconds)
        tolerance_samples = int(target_samples * 0.05)
        step_samples = int(target_samples * (1.0 - overlap))
        if step_samples < 1:
            continue

        min_len = min(channels[c][0] for c in selection)
        if min_len >= (target_samples - tolerance_samples):
            n_win = ((min_len - target_samples + tolerance_samples) // step_samples) + 1
            if n_win > 0:
                station_window_counts[sta_key] = n_win

    return file_path, station_window_counts


# PROCESSING LOGIC

class RamImageEncoder:
    """
    Default per-window encoder: 3-channel RAM image (R=Z, G=N-ish, B=E-ish).

    Encoders are the pluggable step of the pipeline -- everything around them
    (station-disjoint splits, per-window caps, gap rejection, per-station
    sampling rates, manifest) is shared, so a new representation cannot drift
    away from those guarantees the way parallel copies of this pipeline did.
    """
    ext = ".png"

    def __init__(self, target_n: int = 64):
        self.target_n = target_n

    def __call__(self, cleaned_win, fs_station, sta_key, selection,
                 station_baselines, out_dir, stem):
        comp_z, comp_n, comp_e = selection
        mu_z, sigma_z = station_baselines.get((sta_key, comp_z), (None, None))
        mu_n, sigma_n = station_baselines.get((sta_key, comp_n), (None, None))
        mu_e, sigma_e = station_baselines.get((sta_key, comp_e), (None, None))

        # Columns follow `selection` order: 0=Z, 1=N-ish, 2=E-ish.
        ram_Z, _ = ram_matrix(cleaned_win[:, 0], target_n=self.target_n, mu=mu_z, sigma=sigma_z)
        ram_N, _ = ram_matrix(cleaned_win[:, 1], target_n=self.target_n, mu=mu_n, sigma=sigma_n)
        ram_E, _ = ram_matrix(cleaned_win[:, 2], target_n=self.target_n, mu=mu_e, sigma=sigma_e)

        rgb = np.stack([to_uint8(ram_Z), to_uint8(ram_N), to_uint8(ram_E)], axis=-1)
        filename = stem + self.ext
        Image.fromarray(rgb, mode="RGB").save(out_dir / filename)
        return filename


def _window_amplitude_scores(windows, selection, sta_key, station_baselines):
    """
    One amplitude score per window: the loudest component, expressed in units
    of that (station, component)'s long-term noise sigma when a baseline
    exists, and in raw counts otherwise.

    Deliberately computed on the RAW window -- no detrend, no bandpass. A
    standard deviation is already immune to the DC offset, and dividing by the
    station's own noise sigma puts every station on one comparable scale, which
    is what lets scores be ranked ACROSS stations in the global pass.

    Normalizing by the baseline is what makes global ranking meaningful. Ranked
    on raw counts, "loudest" would just mean "highest-gain instrument", and the
    mined negatives would all come from a handful of stations.
    """
    scores = np.empty(len(windows), dtype=np.float64)
    for j, (_w_idx, win) in enumerate(windows):
        best = 0.0
        for i, comp in enumerate(selection):
            v = float(np.std(win[:, i]))
            base = station_baselines.get((sta_key, comp))
            if base is not None and base[1] > 0:
                v /= base[1]
            if v > best:
                best = v
        scores[j] = best
    return scores


def _pick_amplitude_matched(scores, quota, target_scores, n_bins=40, rng=None):
    """
    Indices of `quota` noise windows whose amplitude distribution mirrors
    `target_scores` (the event windows), instead of a fixed percentile band.

    Why this exists. `_pick_from_band` keeps the 75th-99th percentile of the
    noise pool, which puts a hard floor under every negative's amplitude while
    the positives have none. On a 6 s window that is harmless -- events carry S
    and coda and are loud, so P(event | amplitude) rises monotonically. On a
    P-only window it is not: cutting at 1.4 s post-P removes the loud phases,
    35% of events end up quieter than the median mined noise, and below the
    band's lower edge there is almost no noise at all. P(event | amplitude)
    becomes U-shaped -- 0.67 in the quietest decile, 0.30 in the middle, 1.00
    in the loudest -- and a model can learn "very quiet -> event", which is an
    artifact of the mining rather than physics. In continuous data quiet
    windows are overwhelmingly noise.

    Matching the whole distribution rather than a central value is deliberate:
    drawing noise "about as loud as the average event" leaves the two classes
    with different amplitude *spreads*, so "extreme in either direction" stays
    learnable. Quantile bins equalise the shape, not just the location.

    The loud tail cannot be matched and should not be. Roughly a fifth of
    events are louder than any window the pool offers below its 99th
    percentile, and that cap is deliberate -- the loudest tail of screened
    noise is where a catalogue-missed earthquake hides. Where a bin cannot be
    filled, the shortfall is redistributed to the nearest bins that can, and
    the caller reports the residual rather than pretending amplitude was fully
    neutralised.

    Args:
        scores: Candidate noise amplitudes, in station-sigma units.
        quota: How many to select.
        target_scores: Event amplitudes, same units, defining the shape to match.
        n_bins: Quantile bins of the target distribution to match across.
        rng: Optional numpy Generator, for reproducibility.

    Returns:
        Array of indices into `scores`.
    """
    rng = rng or np.random.default_rng(0)
    n = len(scores)
    if quota >= n:
        return np.arange(n)

    edges = np.unique(np.quantile(target_scores, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return _pick_from_band(scores, quota, (0.0, 0.99))
    # Per-bin demand follows the target's own occupancy.
    demand = np.histogram(target_scores, bins=edges)[0].astype(float)
    demand = np.maximum(1, np.round(demand / demand.sum() * quota)).astype(int)

    bin_of = np.digitize(scores, edges) - 1
    pools = [np.flatnonzero(bin_of == b) for b in range(len(edges) - 1)]

    chosen, shortfall = [], 0
    for b, want in enumerate(demand):
        pool = pools[b]
        take = min(want, len(pool))
        if take:
            chosen.append(rng.choice(pool, size=take, replace=False))
        shortfall += want - take

    # Redistribute what the (mostly loud) empty bins could not supply, nearest
    # first, so the shortfall does not silently re-bias the result upward.
    if shortfall:
        used = set(np.concatenate(chosen).tolist()) if chosen else set()
        spare = np.array([i for i in range(n) if i not in used])
        if len(spare):
            order = np.argsort(np.abs(
                scores[spare] - np.median(target_scores)), kind="stable")
            chosen.append(spare[order[:shortfall]])

    out = np.concatenate(chosen) if chosen else np.array([], dtype=int)
    if len(out) > quota:
        out = rng.choice(out, size=quota, replace=False)
    return np.unique(out)


def _pick_from_band(scores, quota, band):
    """
    Indices of `quota` items drawn from the `band` percentile slice of `scores`,
    spread evenly ACROSS the slice rather than taken from its top.

    Why a band and not simply the loudest: the noisiest tail of a screened noise
    archive is where an earthquake missed by the catalog is most likely to be
    hiding, and mining that tail would inject positives into the negative class.
    The upper bound holds selection below it. Spreading across the band keeps a
    range of difficulty instead of collapsing onto one amplitude.

    Widens the band (downward first -- quieter, still harder than random) rather
    than under-filling the quota.
    """
    n = len(scores)
    if quota >= n:
        return np.arange(n)
    lo_q, hi_q = band
    order = np.argsort(scores, kind="stable")
    lo_i, hi_i = int(round(lo_q * n)), int(round(hi_q * n))
    lo_i = max(0, min(lo_i, n - 1))
    hi_i = max(lo_i + 1, min(hi_i, n))
    if hi_i - lo_i < quota:
        lo_i = max(0, hi_i - quota)
        if hi_i - lo_i < quota:
            hi_i = min(n, lo_i + quota)
    candidates = order[lo_i:hi_i]
    if len(candidates) <= quota:
        return candidates
    pick = np.linspace(0, len(candidates) - 1, quota).round().astype(int)
    return candidates[np.unique(pick)]


def _score_windows_task(args):
    """
    Scoring pre-pass worker: windows one file exactly as the generation worker
    will, but only measures amplitude -- no cleaning, no encoding, nothing
    written.

    Returns [(sta_key, w_idx, score)] for the stations assigned to the mined
    class. Window indices come from the same `window_array_indexed` call the
    generation pass uses, so an index selected here lands on the same samples
    there.
    """
    (file_path, station_assignments, station_baselines, fs, window_seconds,
     overlap, max_gap_fraction, target_class) = args
    try:
        st = read(str(file_path))
        try:
            st.merge(method=1)
        except Exception:
            return file_path, []

        stations: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, float]]] = {}
        for tr in st:
            sta_key = f"{tr.stats.network}.{tr.stats.station}"
            if sta_key not in station_assignments:
                continue
            if station_assignments[sta_key][1] != target_class:
                continue
            chan = tr.stats.channel[-1].upper()
            existing = stations.setdefault(sta_key, {}).get(chan)
            if existing is not None and len(existing[0]) >= tr.stats.npts:
                continue
            data, gap_mask = _masked_to_filled(tr.data)
            stations[sta_key][chan] = (data, gap_mask, tr.stats.sampling_rate)

        out = []
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
            raw_channels = [channels[c][0] for c in selection]
            raw_masks = [channels[c][1] for c in selection]
            min_len = min(len(ch) for ch in raw_channels)
            if min_len < (target_samples - tolerance):
                continue
            event_data = np.column_stack([ch[:min_len] for ch in raw_channels])
            gap_mask = np.column_stack([m[:min_len] for m in raw_masks])
            windows, _ = window_array_indexed(
                event_data, gap_mask, fs=fs_station,
                window_seconds=window_seconds, overlap=overlap,
                max_gap_fraction=max_gap_fraction,
            )
            if not windows:
                continue
            scores = _window_amplitude_scores(windows, selection, sta_key,
                                              station_baselines)
            for (w_idx, _win), sc in zip(windows, scores):
                out.append((sta_key, int(w_idx), float(sc)))
        return file_path, out
    except Exception:
        return file_path, []


def mseed_file_to_dataset(
    file_path: Path,
    station_assignments: Dict[str, Tuple[str, str, Path, Optional[int]]],
    station_baselines: Dict[Tuple[str, str], Tuple[float, float]],
    encoder,
    fs: float,
    window_seconds: float,
    overlap: float,
    max_gap_fraction: float = 0.05,
    freqmin: float = 1.0,
    freqmax: float = 45.0,
) -> List[Tuple[str, str, str, str, str, float]]:
    """
    Reads one mseed file ONCE and writes output only for the stations present
    in `station_assignments` (values: split, class, out_dir, window_quota).
    If `station_baselines` is empty (plain mode), every channel falls back to
    per-window self-standardization.

    window_quota, when not None, caps how many windows this (file, station)
    pair may emit; kept windows are chosen evenly spaced across the file but
    keep their ORIGINAL window index in the filename, so manifest-driven
    reconstruction still lands on the exact same samples.
    """
    st = read(str(file_path))
    try:
        st.merge(method=1)  # no fill_value: gaps stay masked so we can see them
    except Exception as e:
        print(f"[WARN] Failed to merge traces in {file_path.name}: {e}")
        return []

    # sta -> comp -> (filled_data, gap_mask, fs); keep the longest trace per
    # component so a stray duplicate (e.g. second location code) can't
    # silently replace the primary sensor.
    stations: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, float]]] = {}
    for tr in st:
        sta_key = f"{tr.stats.network}.{tr.stats.station}"
        if sta_key not in station_assignments:
            continue
        chan = tr.stats.channel[-1].upper()
        existing = stations.setdefault(sta_key, {}).get(chan)
        if existing is not None and len(existing[0]) >= tr.stats.npts:
            continue
        data, gap_mask = _masked_to_filled(tr.data)
        stations[sta_key][chan] = (data, gap_mask, tr.stats.sampling_rate)

    file_id = file_path.stem
    manifest_rows = []

    for sta_key, channels in stations.items():
        selection = select_components(channels.keys())
        if selection is None:
            continue

        rates = {channels[c][2] for c in selection}
        if len(rates) != 1:
            continue
        fs_station = rates.pop()

        target_samples = int(fs_station * window_seconds)
        tolerance_samples = int(target_samples * 0.05)

        raw_channels = [channels[c][0] for c in selection]
        raw_masks = [channels[c][1] for c in selection]
        min_len = min(len(ch) for ch in raw_channels)

        if min_len < (target_samples - tolerance_samples):
            continue

        event_data = np.column_stack([ch[:min_len] for ch in raw_channels])
        gap_mask = np.column_stack([m[:min_len] for m in raw_masks])

        windows, _ = window_array_indexed(
            event_data, gap_mask, fs=fs_station,
            window_seconds=window_seconds, overlap=overlap,
            max_gap_fraction=max_gap_fraction,
        )

        split_name, class_name, out_dir, window_quota = station_assignments[sta_key]

        if isinstance(window_quota, (set, frozenset)):
            # Hard-negative mining: the global pre-pass already decided exactly
            # which window indices survive, so honour that list verbatim. Any
            # local re-selection here would undo the cross-station ranking that
            # is the whole point of doing it globally.
            windows = [(w_idx, win) for w_idx, win in windows if w_idx in window_quota]
        elif window_quota is not None and len(windows) > window_quota:
            sel_idx = np.linspace(0, len(windows) - 1, window_quota).round().astype(int)
            windows = [windows[i] for i in sorted(set(sel_idx.tolist()))]

        for w_idx, win in windows:
            cleaned_win = np.zeros_like(win, dtype=np.float64)
            for i in range(win.shape[1]):
                cleaned_win[:, i] = clean_and_filter_1d(win[:, i], fs_station, freqmin, freqmax)

            stem = f"{file_id}_{sta_key}_win{w_idx:03d}"
            filename = encoder(cleaned_win, fs_station, sta_key, selection,
                               station_baselines, out_dir, stem)
            manifest_rows.append((split_name, class_name, sta_key, str(file_path), filename, fs_station))

    return manifest_rows


# Backwards-compatible alias for the RAM-specific entry point.
def mseed_file_to_ram_rgb(file_path, station_assignments, station_baselines,
                          target_n, fs, window_seconds, overlap, max_gap_fraction=0.05):
    return mseed_file_to_dataset(
        file_path, station_assignments, station_baselines, RamImageEncoder(target_n),
        fs, window_seconds, overlap, max_gap_fraction=max_gap_fraction,
    )


def _process_task(args):
    (file_path, station_assignments, station_baselines, encoder, fs,
     window_seconds, overlap, max_gap_fraction, freqmin, freqmax) = args
    try:
        return mseed_file_to_dataset(
            file_path, station_assignments, station_baselines, encoder, fs,
            window_seconds, overlap, max_gap_fraction=max_gap_fraction,
            freqmin=freqmin, freqmax=freqmax,
        )
    except Exception as e:
        return f"[WARN] Failed file {file_path.stem}: {e}"


# ORCHESTRATION

def _cap_station_windows(
    valid_source_info: List[Tuple[str, int, List[Tuple[Path, int]]]],
    max_windows_per_station: Optional[int],
    rng: random.Random,
) -> List[Tuple[str, int, List[Tuple[Path, int, Optional[int]]]]]:
    """
    Applies the per-station window cap by assigning each kept file a window
    QUOTA (None = keep all) instead of dropping whole files. The old
    file-granularity version couldn't cap below a single file's window count:
    one 300s noise file at 3s windows yields ~200 windows, so a cap of 20
    silently passed all ~200 through. Quotas are enforced at generation time
    by evenly subsampling each file's windows.
    """
    capped = []
    for station_key, total_windows, file_contribs in valid_source_info:
        if max_windows_per_station is None or total_windows <= max_windows_per_station:
            capped.append((station_key, total_windows,
                           [(fpath, w, None) for fpath, w in file_contribs]))
            continue

        shuffled = list(file_contribs)
        rng.shuffle(shuffled)

        kept = []
        remaining = max_windows_per_station
        for fpath, w_count in shuffled:
            if remaining <= 0:
                break
            take = min(w_count, remaining)
            kept.append((fpath, w_count, take))
            remaining -= take

        capped.append((station_key, max_windows_per_station - remaining, kept))

    return capped


def _write_split_manifest(manifest_path: Path, entries: List[Tuple[str, str, str, str, str, float]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'class_name', 'station_key', 'file_path', 'filename', 'fs'])
        writer.writerows(entries)


def _mine_hard_negatives_globally(file_to_assignments, station_baselines, fs,
                                  window_seconds, overlap, max_gap_fraction,
                                  target_class, band, num_cores, splits,
                                  match_amplitude=False,
                                  positive_class="01_earthquake"):
    """
    Replaces the mined class's per-file window QUOTAS with explicit per-file
    window WHITELISTS chosen by ranking every candidate window in a split
    against every other, across all stations.

    Why global and not per-file: a first version ranked windows inside each
    (file, station) group, which barely moved the benchmark -- the amplitude
    floor fell only 0.9535 -> 0.9312 where a global simulation had predicted
    ~0.86. The reason is that all ~99 windows of one 300 s noise file share a
    station and an hour, so they are nearly equally loud; almost all of the
    amplitude variance lives BETWEEN stations and times. Ranking per file can
    only exploit the small part. Scores are in units of each station's own
    noise sigma, so they are directly comparable across stations.

    Costs one extra read of the mined class's files. Splits are handled
    independently, so a train-set window can never displace a test-set one.
    """
    print(f"\n[HARD NEGATIVES] Scoring every candidate '{target_class}' window "
          f"for global ranking...")

    score_tasks, quota_by_key = [], {}
    for fpath, assignments in file_to_assignments.items():
        mined = {s: a for s, a in assignments.items() if a[1] == target_class}
        if not mined:
            continue
        score_tasks.append((fpath, assignments, station_baselines, fs,
                            window_seconds, overlap, max_gap_fraction, target_class))
        for sta_key, (split_name, _cls, _out, quota) in mined.items():
            quota_by_key[(fpath, sta_key)] = (split_name, quota)

    print(f"  -> {len(score_tasks)} files to score on {num_cores} cores.")
    candidates = {s: [] for s in splits}      # split -> [(fpath, sta, w_idx, score)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        for i, (fpath, rows) in enumerate(executor.map(_score_windows_task, score_tasks), 1):
            if i % 2000 == 0:
                print(f"  ...{i}/{len(score_tasks)} files scored")
            for sta_key, w_idx, score in rows:
                entry = quota_by_key.get((fpath, sta_key))
                if entry is None:
                    continue
                candidates[entry[0]].append((fpath, sta_key, w_idx, score))

    # When matching, the positives define the amplitude shape to mirror, so
    # they need scoring in the same station-sigma units. One extra read of the
    # event files; the noise pass above already paid the same cost.
    target_by_split = {s: [] for s in splits}
    if match_amplitude:
        print(f"[MATCH] Scoring '{positive_class}' windows to define the "
              f"target amplitude distribution...")
        pos_tasks, pos_split = [], {}
        for fpath, assignments in file_to_assignments.items():
            pos = {s: a for s, a in assignments.items() if a[1] == positive_class}
            if not pos:
                continue
            pos_tasks.append((fpath, assignments, station_baselines, fs,
                              window_seconds, overlap, max_gap_fraction,
                              positive_class))
            for sta_key, (split_name, _c, _o, _q) in pos.items():
                pos_split[(fpath, sta_key)] = split_name
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as ex:
            for i, (fpath, rows) in enumerate(ex.map(_score_windows_task, pos_tasks), 1):
                if i % 2000 == 0:
                    print(f"  ...{i}/{len(pos_tasks)} event files scored")
                for sta_key, _w, score in rows:
                    sp = pos_split.get((fpath, sta_key))
                    if sp is not None:
                        target_by_split[sp].append(score)

    whitelist: Dict[Tuple[Path, str], set] = {}
    for split_name in splits:
        cands = candidates[split_name]
        if not cands:
            continue
        # Reproduce the split's planned count: the sum of the quotas the
        # balancing stage assigned, so class balance is preserved exactly.
        target = sum(q for (f, s), (sp, q) in quota_by_key.items()
                     if sp == split_name and q is not None)
        if target <= 0:
            continue
        scores = np.array([c[3] for c in cands], dtype=np.float64)
        tgt = np.asarray(target_by_split.get(split_name, ()), dtype=np.float64)
        if match_amplitude and len(tgt):
            chosen = _pick_amplitude_matched(scores, target, tgt)
            unmatchable = float((tgt > scores.max()).mean())
            print(f"  -> {split_name}: amplitude-matched to {len(tgt)} events; "
                  f"{unmatchable:.1%} of events are louder than any candidate "
                  f"noise window and cannot be matched")
        else:
            chosen = _pick_from_band(scores, target, band)
        for idx in chosen:
            fpath, sta_key, w_idx, _sc = cands[idx]
            whitelist.setdefault((fpath, sta_key), set()).add(w_idx)
        sel = scores[chosen]
        print(f"  -> {split_name}: {len(chosen)} of {len(cands)} candidate windows "
              f"(target {target}) | score median {np.median(sel):.2f} "
              f"vs pool median {np.median(scores):.2f} "
              f"({np.median(sel) / max(np.median(scores), 1e-9):.2f}x louder)")

    rebuilt = {}
    for fpath, assignments in file_to_assignments.items():
        new_assign = {}
        for sta_key, (split_name, cls, out_dir, quota) in assignments.items():
            if cls == target_class:
                keep = whitelist.get((fpath, sta_key))
                # An empty whitelist is meaningful: the global ranking placed
                # none of this file's windows in the band. Pass the empty set
                # so the worker writes nothing, rather than falling back to a
                # quota and quietly re-adding random windows.
                new_assign[sta_key] = (split_name, cls, out_dir,
                                       frozenset(keep or ()))
            else:
                new_assign[sta_key] = (split_name, cls, out_dir, quota)
        rebuilt[fpath] = new_assign
    return rebuilt


def run_balanced_preprocessing(
    eq_dir: str,
    noise_dir: str,
    output_dir: str,
    split_ratios: tuple = (0.70, 0.15, 0.15),
    target_n: int = 64,
    fs: float = 100.0,
    window_seconds: float = 60.0,
    overlap: float = 0.5,
    limit_pictures: Optional[int] = None,
    max_windows_per_station: Optional[int] = None,
    use_baseline_standardization: bool = False,
    freqmin: float = 1.0,
    freqmax: float = 45.0,
    min_baseline_seconds: float = 60.0,
    num_cores: Optional[int] = None,
    max_gap_fraction: float = 0.05,
    generate_max: bool = False,
    encoder=None,
    hard_negatives: bool = False,
    hard_negative_band: Tuple[float, float] = (0.75, 0.99),
    match_negative_amplitude: bool = False,
):
    if encoder is None:
        encoder = RamImageEncoder(target_n)
    if generate_max and limit_pictures:
        raise ValueError("generate_max and limit_pictures are mutually exclusive: "
                         "--max means 'as many images as the data allows'.")

    print("=" * 60)
    mode_label = "baseline-standardized" if use_baseline_standardization else "plain per-window standardization"
    if generate_max:
        mode_label += ", MAX mode"
    if hard_negatives:
        mode_label += ", HARD NEGATIVES"
    print(f"STARTING DATASET GENERATION (station-disjoint splits, {mode_label})")
    print("=" * 60)
    hard_negative_class = "00_noise" if hard_negatives else None
    if hard_negatives:
        lo, hi = hard_negative_band
        print(f"[INFO] Hard-negative mining ON: noise windows are ranked by amplitude "
              f"relative to their station's noise floor, and the kept ones are drawn "
              f"from the {lo:.0%}-{hi:.0%} band rather than sampled evenly.")
        print(f"       The upper bound is deliberate -- the loudest tail of a screened "
              f"noise archive is where a catalog-missed earthquake would hide, and "
              f"mining it would put positives into the negative class.")
        print(f"[WARN] This makes the dataset intentionally UNREPRESENTATIVE of "
              f"deployment noise. Use it for training; keep a randomly-sampled test "
              f"set for any calibrated/absolute number.")
    if max_windows_per_station is not None:
        print(f"[INFO] Capping any single station's contribution to at most "
              f"{max_windows_per_station} windows across all its event files "
              f"(enforced per-window, not per-file).")

    if use_baseline_standardization:
        station_baselines, _ = compute_station_noise_baselines(
            noise_dir, fs=fs, freqmin=freqmin, freqmax=freqmax, min_baseline_seconds=min_baseline_seconds,
            num_cores=num_cores,
        )
    else:
        station_baselines = {}

    classes = [("01_earthquake", Path(eq_dir)), ("00_noise", Path(noise_dir))]
    class_names = [name for name, _ in classes]
    out_paths = {}

    for class_name, _ in classes:
        out_paths[class_name] = {}
        for split in ["train", "val", "test"]:
            split_dir = Path(output_dir) / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            out_paths[class_name][split] = split_dir

    if num_cores is None:
        num_cores = max(1, multiprocessing.cpu_count() - 1)
    cap_rng = random.Random(123)

    print("\n[PHASE 1] Scanning headers and grouping windows BY STATION...")
    class_data = {}
    station_sets = {}

    for class_name, source_path in classes:
        if not source_path.exists():
            print(f"[WARN] Input directory not found: {source_path}. Skipping.")
            class_data[class_name] = {"valid_sources": [], "total_windows": 0}
            station_sets[class_name] = set()
            continue

        mseed_files = list(source_path.rglob("*.mseed"))
        scan_args = [(fp, fs, window_seconds, overlap) for fp in mseed_files]

        station_groups: Dict[str, List[Tuple[Path, int]]] = {}

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            for file_path, station_window_counts in executor.map(scan_single_mseed, scan_args):
                for sta_key, w_count in station_window_counts.items():
                    station_groups.setdefault(sta_key, []).append((file_path, w_count))

        station_sets[class_name] = set(station_groups.keys())

        valid_source_info_raw = [
            (sta_key, sum(w for _, w in contribs), contribs)
            for sta_key, contribs in station_groups.items()
        ]

        valid_source_info = _cap_station_windows(valid_source_info_raw, max_windows_per_station, cap_rng)
        total_windows = sum(w for _, w, _ in valid_source_info)

        class_data[class_name] = {
            "valid_sources": valid_source_info,
            "total_windows": total_windows,
        }
        print(f"  -> {class_name.upper()}: {total_windows} extractable windows "
              f"across {len(valid_source_info)} unique STATIONS (post-cap).")

        if valid_source_info:
            counts = sorted((w for _, w, _ in valid_source_info), reverse=True)
            top = counts[0]
            print(f"     Largest single station now contributes {top} windows "
                  f"({top / total_windows * 100:.1f}% of this class's total).")

        if use_baseline_standardization and class_name == "01_earthquake":
            stations_with_no_baseline = [
                sta_key for sta_key, _, _ in valid_source_info
                if not any(k[0] == sta_key for k in station_baselines.keys())
            ]
            if stations_with_no_baseline:
                print(f"     [INFO] {len(stations_with_no_baseline)}/{len(valid_source_info)} "
                      f"earthquake stations have no usable noise baseline -- these will fall "
                      f"back to plain per-window self-standardization.")

    eq_stations = station_sets.get("01_earthquake", set())
    noise_stations = station_sets.get("00_noise", set())
    shared = eq_stations & noise_stations
    if eq_stations:
        print(f"\n[INFO] Station overlap across classes: {len(shared)}/{len(eq_stations)} "
              f"({len(shared) / len(eq_stations) * 100:.1f}%) of earthquake stations also have noise data.")

    print("\n[PHASE 2] Balancing classes...")
    eq_total = class_data["01_earthquake"]["total_windows"]
    noise_total = class_data["00_noise"]["total_windows"]

    if eq_total == 0 or noise_total == 0:
        print("[ERROR] One of the classes has 0 valid windows. Aborting this folder.")
        return

    if generate_max:
        print(f"  -> MAX mode: class totals are {eq_total} earthquake / {noise_total} noise windows.")
        print("  -> Every usable station will be assigned to a split; the surplus class is then "
              "trimmed per split (evenly-spaced window subsampling) to match the smaller class.")
        target_per_class = None
    else:
        bottleneck_size = min(eq_total, noise_total)

        if limit_pictures:
            target_per_class = min(bottleneck_size, limit_pictures // 2)
        else:
            target_per_class = bottleneck_size

        print(f"  -> Bottleneck dictates a maximum of {bottleneck_size} images per class.")
        print(f"  -> Final target set to {target_per_class} images per class (Total: {target_per_class * 2} images).")

    print("\n[PHASE 3] Allocating STATION-disjoint splits (unified across classes)...")

    # sta -> class -> (window_total, file_contribs)
    per_station: Dict[str, Dict[str, Tuple[int, List[Tuple[Path, int, Optional[int]]]]]] = {}
    for class_name in class_names:
        for sta_key, w_total, contribs in class_data[class_name]["valid_sources"]:
            per_station.setdefault(sta_key, {})[class_name] = (w_total, contribs)

    splits = ["train", "val", "test"]
    counts = {c: {s: 0 for s in splits} for c in class_names}
    n_stations = {c: {s: 0 for s in splits} for c in class_names}

    all_stations = sorted(per_station.keys())
    random.seed(42)
    random.shuffle(all_stations)

    file_to_assignments: Dict[Path, Dict[str, Tuple[str, str, Path, Optional[int]]]] = {}

    if generate_max:
        # --- MAX mode: use everything, then balance by trimming ---
        #
        # 1) Assign EVERY station to the split with the largest relative
        #    deficit against ratio-proportional targets, computed per class
        #    from that class's FULL window total. No station is dropped.
        class_totals = {c: class_data[c]["total_windows"] for c in class_names}
        targets = {c: {s: split_ratios[i] * class_totals[c] for i, s in enumerate(splits)}
                   for c in class_names}

        station_split: Dict[str, str] = {}
        for sta_key in all_stations:
            present_classes = list(per_station[sta_key].keys())
            best_split, best_need = None, None
            for cand in splits:
                need = sum((targets[c][cand] - counts[c][cand]) / max(targets[c][cand], 1.0)
                           for c in present_classes)
                if best_need is None or need > best_need:
                    best_split, best_need = cand, need
            station_split[sta_key] = best_split
            for class_name in present_classes:
                w_total, _ = per_station[sta_key][class_name]
                counts[class_name][best_split] += w_total
                n_stations[class_name][best_split] += 1

        # 2) Per split, trim the surplus class down to the smaller class by
        #    assigning per-(station, file) quotas via largest-remainder
        #    proportional rounding. The smaller class keeps its cap quotas
        #    untouched. Balance is on scan ESTIMATES; actual generated counts
        #    can differ slightly (gap rejection, header-vs-merged lengths).
        planned = {s: {c: counts[c][s] for c in class_names} for s in splits}
        trim_quota: Dict[Tuple[str, str], Dict[Path, int]] = {}  # (sta, class) -> fpath -> quota

        for split_name in splits:
            balanced = min(planned[split_name][c] for c in class_names)
            if balanced == 0:
                present = {c: planned[split_name][c] for c in class_names}
                print(f"  [WARN] Split '{split_name}' has a class with 0 windows ({present}) -- "
                      f"it will be empty after balancing. More stations are needed for this split.")
            for class_name in class_names:
                surplus_total = planned[split_name][class_name]
                if surplus_total <= balanced:
                    continue
                entries = []  # (sta_key, fpath, effective_window_count)
                for sta_key, split_of in station_split.items():
                    if split_of != split_name or class_name not in per_station[sta_key]:
                        continue
                    _w_total, contribs = per_station[sta_key][class_name]
                    for fpath, w_count, quota in contribs:
                        entries.append((sta_key, fpath, w_count if quota is None else quota))

                factor = balanced / surplus_total
                raw = [eff * factor for _, _, eff in entries]
                base = [int(r) for r in raw]
                short = balanced - sum(base)
                by_remainder = sorted(range(len(raw)), key=lambda i: raw[i] - base[i], reverse=True)
                for i in by_remainder[:short]:
                    base[i] += 1
                for (sta_key, fpath, _eff), q in zip(entries, base):
                    trim_quota.setdefault((sta_key, class_name), {})[fpath] = q
                planned[split_name][class_name] = balanced

        for sta_key, split_name in station_split.items():
            for class_name in per_station[sta_key].keys():
                _w_total, contribs = per_station[sta_key][class_name]
                out_dir = out_paths[class_name][split_name]
                file_quotas = trim_quota.get((sta_key, class_name))
                for fpath, _w_count, quota in contribs:
                    if file_quotas is not None:
                        quota = file_quotas.get(fpath, 0)
                    file_to_assignments.setdefault(fpath, {})[sta_key] = (split_name, class_name, out_dir, quota)

        for class_name in class_names:
            p = {s: planned[s][class_name] for s in splits}
            c = counts[class_name]
            ns = n_stations[class_name]
            print(f"  -> {class_name.upper()}:")
            print(f"     Assigned windows | Train: {c['train']:<6} | Val: {c['val']:<6} | Test: {c['test']:<6}")
            print(f"     After balancing  | Train: {p['train']:<6} | Val: {p['val']:<6} | Test: {p['test']:<6}")
            print(f"     Stations used    | Train: {ns['train']:<6} | Val: {ns['val']:<6} | Test: {ns['test']:<6}")
        total_planned = sum(planned[s][c] for s in splits for c in class_names)
        print(f"     Planned total: {total_planned} images (balanced per split; "
              f"actuals may differ slightly where windows are rejected at generation time).")
    else:
        targets = {}
        for class_name in class_names:
            t_train = int(target_per_class * split_ratios[0])
            t_val = int(target_per_class * split_ratios[1])
            targets[class_name] = {"train": t_train, "val": t_val,
                                    "test": target_per_class - t_train - t_val}

        for sta_key in all_stations:
            present_classes = list(per_station[sta_key].keys())

            # One split per STATION, shared by every class it appears in. This is
            # the actual station-disjoint guarantee: a station assigned to train
            # can never surface in val/test under either label.
            split_name = None
            for cand in splits:
                if any(counts[c][cand] < targets[c][cand] for c in present_classes):
                    split_name = cand
                    break
            if split_name is None:
                continue  # every split this station could help is already full

            for class_name in present_classes:
                w_total, contribs = per_station[sta_key][class_name]
                counts[class_name][split_name] += w_total
                n_stations[class_name][split_name] += 1
                out_dir = out_paths[class_name][split_name]
                for fpath, _w_count, quota in contribs:
                    file_to_assignments.setdefault(fpath, {})[sta_key] = (split_name, class_name, out_dir, quota)

            if all(counts[c][s] >= targets[c][s] for c in class_names for s in splits):
                break

        for class_name in class_names:
            t = targets[class_name]
            c = counts[class_name]
            ns = n_stations[class_name]
            print(f"  -> {class_name.upper()}:")
            print(f"     Target windows | Train: {t['train']:<6} | Val: {t['val']:<6} | Test: {t['test']:<6}")
            print(f"     Actual windows | Train: {c['train']:<6} | Val: {c['val']:<6} | Test: {c['test']:<6}")
            print(f"     Stations used  | Train: {ns['train']:<6} | Val: {ns['val']:<6} | Test: {ns['test']:<6}")
    print("     [INFO] Every station occupies exactly one split across BOTH classes.")

    if hard_negative_class is not None:
        file_to_assignments = _mine_hard_negatives_globally(
            file_to_assignments, station_baselines, fs, window_seconds, overlap,
            max_gap_fraction, hard_negative_class, hard_negative_band,
            num_cores, splits, match_amplitude=match_negative_amplitude,
        )

    tasks = [
        (fpath, assignments, station_baselines, encoder, fs, window_seconds,
         overlap, max_gap_fraction, freqmin, freqmax)
        for fpath, assignments in file_to_assignments.items()
    ]

    print(f"\n[PHASE 4] Processing {len(tasks)} file-level tasks "
          f"(each file read once, only assigned stations written) on {num_cores} cores...")

    # Encoders that pull in torch must run under 'spawn': torch's threading /
    # OpenMP state does not survive fork(), and the workers deadlock silently
    # (sleeping at 0% CPU, no output, forever) instead of erroring out.
    mp_context = None
    if getattr(encoder, "requires_spawn", False):
        mp_context = multiprocessing.get_context("spawn")
        print("       (using 'spawn' workers -- required for torch-based encoders)")

    full_manifest: List[Tuple[str, str, str, str, str, float]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores, mp_context=mp_context) as executor:
        for result in executor.map(_process_task, tasks):
            if isinstance(result, str):
                print(result)
            elif result:
                full_manifest.extend(result)

    print("\n[PHASE 5] Writing manifest...")
    manifest_path = Path(output_dir) / "manifest.csv"
    _write_split_manifest(manifest_path, full_manifest)
    print(f"  -> Wrote {len(full_manifest)} entries to {manifest_path}")

    print(f"\n[COMPLETE] Dataset generation finished successfully! ({mode_label})")
