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


def compute_station_noise_baselines(
    noise_dir: str,
    fs: float,
    freqmin: float,
    freqmax: float,
    min_baseline_seconds: float = 60.0,
) -> Tuple[Dict[Tuple[str, str], Tuple[float, float]], int]:
    """
    Scans every noise mseed file, groups by (station_key, component), applies
    the SAME cleaning/filtering used on actual training windows, and
    accumulates running mean/std per (station, component). A station/component
    only gets a baseline if it accumulated at least `min_baseline_seconds`
    worth of usable noise data; otherwise it's left out and falls back to
    plain per-window self-standardization when used.
    """
    noise_path = Path(noise_dir)
    if not noise_path.exists():
        print(f"[WARN] Noise directory not found for baseline computation: {noise_path}")
        return {}, 0

    print("\n[BASELINE] Scanning noise files to build per-station long-term baselines...")
    mseed_files = list(noise_path.rglob("*.mseed"))
    print(f"  -> Found {len(mseed_files)} noise files to scan.")

    accum: Dict[Tuple[str, str], Tuple[float, float, int]] = {}

    for i, file_path in enumerate(mseed_files, 1):
        if i % 200 == 0:
            print(f"  ...{i}/{len(mseed_files)} noise files processed")
        try:
            st = read(str(file_path))
            st.merge(method=1, fill_value='interpolate')
        except Exception:
            continue

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

            key = (sta_key, comp)
            accum[key] = _accumulate_stats(accum.get(key), cleaned)

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

def mseed_file_to_ram_rgb(
    file_path: Path,
    station_assignments: Dict[str, Tuple[str, str, Path, Optional[int]]],
    station_baselines: Dict[Tuple[str, str], Tuple[float, float]],
    target_n: int,
    fs: float,
    window_seconds: float,
    overlap: float,
    max_gap_fraction: float = 0.05,
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

        if window_quota is not None and len(windows) > window_quota:
            sel_idx = np.linspace(0, len(windows) - 1, window_quota).round().astype(int)
            windows = [windows[i] for i in sorted(set(sel_idx.tolist()))]

        comp_z, comp_n, comp_e = selection

        for w_idx, win in windows:
            cleaned_win = np.zeros_like(win, dtype=np.float64)
            for i in range(win.shape[1]):
                cleaned_win[:, i] = clean_and_filter_1d(win[:, i], fs_station, 1.0, 45.0)

            mu_z, sigma_z = station_baselines.get((sta_key, comp_z), (None, None))
            mu_n, sigma_n = station_baselines.get((sta_key, comp_n), (None, None))
            mu_e, sigma_e = station_baselines.get((sta_key, comp_e), (None, None))

            # Columns follow `selection` order: 0=Z, 1=N-ish, 2=E-ish.
            ram_Z_mat, _ = ram_matrix(cleaned_win[:, 0], target_n=target_n, mu=mu_z, sigma=sigma_z)
            ram_N_mat, _ = ram_matrix(cleaned_win[:, 1], target_n=target_n, mu=mu_n, sigma=sigma_n)
            ram_E_mat, _ = ram_matrix(cleaned_win[:, 2], target_n=target_n, mu=mu_e, sigma=sigma_e)

            # R=Z, G=N-ish, B=E-ish -- same mapping the previous sorted-channel
            # code produced for E/N/Z stations, now explicit and uniform for
            # 1/2-named stations too.
            rgb = np.stack([to_uint8(ram_Z_mat), to_uint8(ram_N_mat), to_uint8(ram_E_mat)], axis=-1)
            img = Image.fromarray(rgb, mode="RGB")

            filename = f"{file_id}_{sta_key}_win{w_idx:03d}.png"
            img.save(out_dir / filename)
            manifest_rows.append((split_name, class_name, sta_key, str(file_path), filename, fs_station))

    return manifest_rows


def _process_task(args):
    file_path, station_assignments, station_baselines, target_n, fs, window_seconds, overlap, max_gap_fraction = args
    try:
        return mseed_file_to_ram_rgb(
            file_path, station_assignments, station_baselines, target_n, fs,
            window_seconds, overlap, max_gap_fraction=max_gap_fraction,
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
):
    print("=" * 60)
    mode_label = "baseline-standardized" if use_baseline_standardization else "plain per-window standardization"
    print(f"STARTING DATASET GENERATION (station-disjoint splits, {mode_label})")
    print("=" * 60)
    if max_windows_per_station is not None:
        print(f"[INFO] Capping any single station's contribution to at most "
              f"{max_windows_per_station} windows across all its event files "
              f"(enforced per-window, not per-file).")

    if use_baseline_standardization:
        station_baselines, _ = compute_station_noise_baselines(
            noise_dir, fs=fs, freqmin=freqmin, freqmax=freqmax, min_baseline_seconds=min_baseline_seconds,
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
    targets = {}
    for class_name in class_names:
        t_train = int(target_per_class * split_ratios[0])
        t_val = int(target_per_class * split_ratios[1])
        targets[class_name] = {"train": t_train, "val": t_val,
                                "test": target_per_class - t_train - t_val}
    counts = {c: {s: 0 for s in splits} for c in class_names}
    n_stations = {c: {s: 0 for s in splits} for c in class_names}

    all_stations = sorted(per_station.keys())
    random.seed(42)
    random.shuffle(all_stations)

    file_to_assignments: Dict[Path, Dict[str, Tuple[str, str, Path, Optional[int]]]] = {}

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

    tasks = [
        (fpath, assignments, station_baselines, target_n, fs, window_seconds, overlap, max_gap_fraction)
        for fpath, assignments in file_to_assignments.items()
    ]

    print(f"\n[PHASE 4] Processing {len(tasks)} file-level tasks "
          f"(each file read once, only assigned stations written) on {num_cores} cores...")

    full_manifest: List[Tuple[str, str, str, str, str, float]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
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
