import concurrent.futures
import csv
import math
import multiprocessing
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.signal as signal
from obspy import read
from PIL import Image

# ALGORITHM FUNCTIONS

def standardize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = np.mean(x)
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


def ram_matrix(x: np.ndarray, target_n: int, eps: float = 1e-12) -> Tuple[np.ndarray, int]:
    x_std = standardize(x, eps=eps)
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


def window_array(data: np.ndarray, fs: float = 100.0, window_seconds: float = 60.0, overlap: float = 0.5) -> List[np.ndarray]:
    target_samples = int(fs * window_seconds)
    step_samples = int(target_samples * (1.0 - overlap))
    if step_samples < 1:
        raise ValueError("Overlap fraction too high; step size must be at least 1 sample.")

    n_samples = data.shape[0]
    windows = []
    tolerance = int(target_samples * 0.05)

    if n_samples < (target_samples - tolerance):
        return windows

    n_windows = math.ceil((n_samples - target_samples) / step_samples) + 1
    for i in range(n_windows):
        start_idx = i * step_samples
        end_idx = start_idx + target_samples
        win = data[start_idx:end_idx, :]

        if len(win) >= (target_samples - tolerance):
            if len(win) < target_samples:
                pad_length = target_samples - len(win)
                win = np.pad(win, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
            windows.append(win)

    return windows


# PRE-SCAN LOGIC

def scan_single_mseed(args: Tuple[Path, float, float, float]) -> Tuple[Path, Dict[str, int]]:
    """
    Returns (file_path, {station_key: window_count}) -- per-STATION counts,
    not an aggregate file total, since splitting now happens at the station
    level and one file can contain multiple stations.
    """
    file_path, nominal_fs, window_seconds, overlap = args
    try:
        st = read(str(file_path), headonly=True)
    except Exception as e:
        print(f"\n[ERROR] Obspy failed to read {file_path.name}: {e}")
        return file_path, {}

    actual_fs = st[0].stats.sampling_rate
    target_samples = int(actual_fs * window_seconds)
    tolerance_samples = int(target_samples * 0.05)
    step_samples = int(target_samples * (1.0 - overlap))

    stations = {}
    for tr in st:
        sta_key = f"{tr.stats.network}.{tr.stats.station}"
        chan = tr.stats.channel[-1].upper()
        if sta_key not in stations:
            stations[sta_key] = {}
        current_len = stations[sta_key].get(chan, 0)
        stations[sta_key][chan] = max(current_len, tr.stats.npts)

    station_window_counts = {}
    for sta_key, channels in stations.items():
        available_chans = sorted(channels.keys())
        if len(available_chans) >= 3:
            min_len = min(channels[ch] for ch in available_chans[:3])
            if min_len >= (target_samples - tolerance_samples):
                n_win = ((min_len - target_samples + tolerance_samples) // step_samples) + 1
                if n_win > 0:
                    station_window_counts[sta_key] = n_win

    return file_path, station_window_counts


# PROCESSING LOGIC

def mseed_file_to_ram_rgb(
    file_path: Path,
    station_assignments: Dict[str, Tuple[str, str, Path]],  # sta_key -> (split_name, class_name, out_dir)
    target_n: int,
    fs: float,
    window_seconds: float,
    overlap: float,
) -> List[Tuple[str, str, str, str, str]]:
    """
    Reads one mseed file ONCE and writes output only for the stations present
    in `station_assignments`, each to its assigned split's directory.

    Returns a list of (split_name, class_name, station_key, file_path, filename)
    manifest entries. class_name comes directly from the assignment made when
    this station was allocated to a split -- NOT reconstructed later from a
    station->class lookup, since the same physical station can legitimately
    belong to BOTH classes (it recorded a real earthquake AND provided a noise
    window elsewhere in time). A single global per-station label would silently
    collide for any station in both classes.
    """
    st = read(str(file_path))
    try:
        st.merge(method=1, fill_value='interpolate')
    except Exception as e:
        print(f"[WARN] Failed to merge traces in {file_path.name}: {e}")
        return []

    actual_fs = st[0].stats.sampling_rate

    stations = {}
    for tr in st:
        sta_key = f"{tr.stats.network}.{tr.stats.station}"
        if sta_key not in station_assignments:
            continue  # not assigned to any split for this run -- skip entirely
        if sta_key not in stations:
            stations[sta_key] = {}
        chan = tr.stats.channel[-1].upper()
        stations[sta_key][chan] = tr.data.astype(np.float64)

    target_samples = int(actual_fs * window_seconds)
    tolerance_samples = int(target_samples * 0.05)
    file_id = file_path.stem
    manifest_rows = []

    for sta_key, channels in stations.items():
        available_chans = sorted(channels.keys())
        if len(available_chans) < 3:
            continue

        raw_channels = [channels[ch] for ch in available_chans[:3]]
        min_len = min(len(ch) for ch in raw_channels)

        if min_len < (target_samples - tolerance_samples):
            continue

        trimmed_channels = [ch[:min_len] for ch in raw_channels]
        event_data = np.column_stack(trimmed_channels)

        windows = window_array(event_data, fs=actual_fs, window_seconds=window_seconds, overlap=overlap)

        split_name, class_name, out_dir = station_assignments[sta_key]

        for w_idx, win in enumerate(windows):
            cleaned_win = np.zeros_like(win, dtype=np.float64)
            for i in range(win.shape[1]):
                cleaned_win[:, i] = clean_and_filter_1d(win[:, i], actual_fs, 1.0, 45.0)

            ram_B_mat, _ = ram_matrix(cleaned_win[:, 0], target_n=target_n)
            ram_G_mat, _ = ram_matrix(cleaned_win[:, 1], target_n=target_n)
            ram_R_mat, _ = ram_matrix(cleaned_win[:, 2], target_n=target_n)

            ram_B = to_uint8(ram_B_mat)
            ram_G = to_uint8(ram_G_mat)
            ram_R = to_uint8(ram_R_mat)

            rgb = np.stack([ram_R, ram_G, ram_B], axis=-1)
            img = Image.fromarray(rgb, mode="RGB")

            filename = f"{file_id}_{sta_key}_win{w_idx:03d}.png"
            img.save(out_dir / filename)
            manifest_rows.append((split_name, class_name, sta_key, str(file_path), filename))

    return manifest_rows


def _process_task(args):
    file_path, station_assignments, target_n, fs, window_seconds, overlap = args
    try:
        return mseed_file_to_ram_rgb(file_path, station_assignments, target_n, fs, window_seconds, overlap)
    except Exception as e:
        return f"[WARN] Failed file {file_path.stem}: {e}"


# ORCHESTRATION

def _cap_station_windows(
    valid_source_info: List[Tuple[str, int, List[Tuple[Path, int]]]],
    max_windows_per_station: Optional[int],
    rng: random.Random,
) -> List[Tuple[str, int, List[Tuple[Path, int]]]]:
    """
    Caps how many windows a single station can contribute in total (summed
    across every event file it appears in). Without this, a station near
    many events could dominate a split the same way a single busy station
    did in the HDF5 pipeline.
    """
    if max_windows_per_station is None:
        return valid_source_info

    capped = []
    for station_key, total_windows, file_contribs in valid_source_info:
        if total_windows <= max_windows_per_station:
            capped.append((station_key, total_windows, file_contribs))
            continue

        shuffled = list(file_contribs)
        rng.shuffle(shuffled)

        kept = []
        running_total = 0
        for fpath, w_count in shuffled:
            if running_total + w_count > max_windows_per_station and kept:
                continue
            kept.append((fpath, w_count))
            running_total += w_count
            if running_total >= max_windows_per_station:
                break

        capped.append((station_key, running_total, kept))

    return capped


def _write_split_manifest(manifest_path: Path, entries: List[Tuple[str, str, str, str, str]]) -> None:
    """entries: (split, class_name, station_key, file_path, filename)"""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['split', 'class_name', 'station_key', 'file_path', 'filename'])
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
    limit_pictures: int = None,
    max_windows_per_station: Optional[int] = None,
):
    print("=" * 60)
    print("STARTING DATASET GENERATION (station-disjoint splits)")
    print("=" * 60)
    if max_windows_per_station is not None:
        print(f"[INFO] Capping any single station's contribution to at most "
              f"{max_windows_per_station} windows across all its event files.")

    classes = [("01_earthquake", Path(eq_dir)), ("00_noise", Path(noise_dir))]
    out_paths = {}

    for class_name, _ in classes:
        out_paths[class_name] = {}
        for split in ["train", "val", "test"]:
            split_dir = Path(output_dir) / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            out_paths[class_name][split] = split_dir

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

        # station_key -> list of (file_path, window_count) across every file it appears in
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

    print("\n[PHASE 3] Allocating STATION-disjoint splits...")

    # file_path -> {sta_key: (split_name, class_name, out_dir)}  -- grouped so each file is read once
    file_to_assignments: Dict[Path, Dict[str, Tuple[str, str, Path]]] = {}

    random.seed(42)

    for class_name, _ in classes:
        target_train = int(target_per_class * split_ratios[0])
        target_val = int(target_per_class * split_ratios[1])
        target_test = target_per_class - target_train - target_val

        stations = list(class_data[class_name]["valid_sources"])
        random.shuffle(stations)

        count_train = count_val = count_test = 0
        n_stations_train = n_stations_val = n_stations_test = 0

        for sta_key, w_count, file_contribs in stations:
            if target_train > 0 and count_train < target_train:
                split_name = "train"
                count_train += w_count
                n_stations_train += 1
            elif target_val > 0 and count_val < target_val:
                split_name = "val"
                count_val += w_count
                n_stations_val += 1
            elif target_test > 0 and count_test < target_test:
                split_name = "test"
                count_test += w_count
                n_stations_test += 1
            else:
                break

            out_dir = out_paths[class_name][split_name]

            for fpath, _ in file_contribs:
                # class_name is captured HERE, at the point the assignment is
                # actually made -- never reconstructed later from a per-station
                # lookup, since the same physical station can legitimately be
                # assigned once as earthquake (from an eq_dir file) and once as
                # noise (from a different noise_dir file). Keying file_to_assignments
                # by fpath (which differs between the two directories) means this
                # never collides -- each (file, station) pair keeps its own,
                # correct class_name.
                file_to_assignments.setdefault(fpath, {})[sta_key] = (split_name, class_name, out_dir)

        print(f"  -> {class_name.upper()}:")
        print(f"     Target windows | Train: {target_train:<6} | Val: {target_val:<6} | Test: {target_test:<6}")
        print(f"     Actual windows | Train: {count_train:<6} | Val: {count_val:<6} | Test: {count_test:<6}")
        print(f"     Stations used  | Train: {n_stations_train:<6} | Val: {n_stations_val:<6} | Test: {n_stations_test:<6}")

    tasks = [
        (fpath, assignments, target_n, fs, window_seconds, overlap)
        for fpath, assignments in file_to_assignments.items()
    ]

    print(f"\n[PHASE 4] Processing {len(tasks)} file-level tasks "
          f"(each file read once, only assigned stations written) on {num_cores} cores...")

    # (split, class_name, station_key, file_path, filename) -- class_name comes
    # straight from mseed_file_to_ram_rgb now, no reconstruction needed.
    full_manifest: List[Tuple[str, str, str, str, str]] = []
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

    print("\n[COMPLETE] Station-disjoint balanced dataset generation finished successfully!")


# MAIN WRAPPER

if __name__ == "__main__":

    BASE_WAVEFORMS_DIR = Path("data/batched_waveforms")
    NOISE_BATCH_DIR = "data/batched_noise_waveforms"

    target_directories = [
        "window_post_60s",
    ]

    TARGET_N = 64

    for dir_name in target_directories:
        match = re.search(r'(\d+)s$', dir_name)
        if not match:
            print(f"[WARN] Could not parse window seconds from {dir_name}. Skipping...")
            continue

        nominal_window_secs = float(match.group(1))
        eq_dir = BASE_WAVEFORMS_DIR / dir_name

        if not eq_dir.exists():
            print(f"[WARN] Directory {eq_dir} does not exist. Skipping...")
            continue

        output_dir = f"dataset_{dir_name}"

        print(f"\n{'#'*60}")
        print(f"RUNNING PIPELINE FOR: {dir_name} (Window Size: {nominal_window_secs}s)")
        print(f"{'#'*60}\n")

        # Short windows let a single long noise trace produce thousands of
        # overlapping slices via window_array -- easily more than an entire
        # split's target on its own, which collapses the noise side down to
        # 1-2 stations (this is exactly what happened on window_post_3s_anchored
        # before this cap was added: noise test drew from a single station).
        # Earthquake-side windows are unaffected by this since the *_anchored
        # files are already sliced to exactly the target length -- overlap
        # can't multiply a window out of a file with no room left to slide in.
        station_cap = 100 if nominal_window_secs <= 10 else None

        run_balanced_preprocessing(
            eq_dir=str(eq_dir),
            noise_dir=NOISE_BATCH_DIR,
            output_dir=output_dir,
            split_ratios=(0.70, 0.15, 0.15),
            target_n=TARGET_N,
            fs=100.0,
            window_seconds=nominal_window_secs,
            overlap=0.25,
            max_windows_per_station=station_cap,
        )
