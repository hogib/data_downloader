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
    """
    Reshape a 1D array into a (d, target_n) matrix, choosing d so that the
    output ALWAYS has exactly `target_n` columns -- regardless of how long
    the input window is.

    This replaces the old fixed-`d` approach. With a fixed `d`, short windows
    (e.g. a 3s window at 100 Hz = 300 samples) collapsed into tiny images
    (300 // 100 = 3x3 pixels), destroying almost all spatial information.
    By deriving `d` from the desired resolution instead, every window length
    produces a full-resolution image with no resizing/interpolation needed
    afterward -- every pixel is still a directly computed cosine-angle
    difference between real local feature vectors, never an interpolated one.
    """
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
    X_cols = M
    Xbar = np.mean(X_cols, axis=1)
    norm_Xbar = np.linalg.norm(Xbar)
    if norm_Xbar < eps:
        norm_Xbar = eps
    n = X_cols.shape[1]
    betas = np.empty(n, dtype=np.float64)
    for i in range(n):
        Xi = X_cols[:, i]
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

# CATALOG & METADATA LOGIC

def load_catalog(catalog_csv_path: Optional[str]) -> Dict[str, float]:
    catalog = {}
    if not catalog_csv_path or not Path(catalog_csv_path).exists():
        print(f"[INFO] No valid catalog CSV provided or file does not exist at '{catalog_csv_path}'. Skipping catalog mapping.")
        return catalog

    try:
        with open(catalog_csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                event_id = str(row.get('EventID', '')).strip()
                mag_val = row.get('Magnitude', None)
                if event_id and mag_val is not None:
                    try:
                        catalog[event_id] = float(mag_val)
                    except ValueError:
                        continue
        print(f"[INFO] Loaded {len(catalog)} earthquake events with magnitudes from catalog.")
    except Exception as e:
        print(f"[WARN] Failed to load catalog CSV: {e}")

    return catalog


def extract_magnitude(st, file_path: Path, catalog: Dict[str, float]) -> Optional[float]:
    # Regex to extract ID to prevent O(N*M) substring matches and false positives
    match_id = re.search(r'(\d{6,})', file_path.name)
    if match_id:
        event_id = match_id.group(1)
        if event_id in catalog:
            return catalog[event_id]

    # SAC fallback
    if hasattr(st[0].stats, 'sac') and 'mag' in st[0].stats.sac:
        return float(st[0].stats.sac.mag)

    # Filename fallback
    match_mag = re.search(r'(?:mag|M|m)[\s_]*(\d+(?:\.\d+)?)', file_path.name)
    if match_mag:
        return float(match_mag.group(1))

    return None

# PRE-SCAN LOGIC

def scan_single_mseed(args: Tuple[Path, float, float, float, dict]) -> Tuple[Path, int]:
    file_path, nominal_fs, window_seconds, overlap, catalog = args
    try:
        st = read(str(file_path), headonly=True)
    except Exception as e:
        print(f"\n[ERROR] Obspy failed to read {file_path.name}: {e}")
        return file_path, 0

    # Early Magnitude Filtering: Drop trace entirely if magnitude cannot be found
    mag = extract_magnitude(st, file_path, catalog)
    if mag is None or np.isnan(mag):
        return file_path, 0

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

    total_windows_in_file = 0
    for sta_key, channels in stations.items():
        available_chans = sorted(channels.keys())

        if len(available_chans) >= 3:
            min_len = min(channels[ch] for ch in available_chans[:3])

            if min_len >= (target_samples - tolerance_samples):
                n_win = ((min_len - target_samples + tolerance_samples) // step_samples) + 1
                if n_win > 0:
                    total_windows_in_file += n_win

    return file_path, total_windows_in_file

# PROCESSING LOGIC

def mseed_file_to_ram_rgb(
    file_path: Path,
    out_dir: Path,
    split: str,
    file_id: str,
    target_n: int,
    nominal_fs: float,
    window_seconds: float,
    overlap: float,
    catalog: Dict[str, float]
):
    st = read(str(file_path))
    try:
        st.merge(method=1, fill_value='interpolate')
    except Exception as e:
        return f"[WARN] Failed to merge traces in {file_path.name}: {e}"

    actual_fs = st[0].stats.sampling_rate
    magnitude = extract_magnitude(st, file_path, catalog)

    stations = {}
    for tr in st:
        sta_key = f"{tr.stats.network}.{tr.stats.station}"
        if sta_key not in stations:
            stations[sta_key] = {}
        chan = tr.stats.channel[-1].upper()
        stations[sta_key][chan] = tr.data.astype(np.float64)

    target_samples = int(actual_fs * window_seconds)
    tolerance_samples = int(target_samples * 0.05)
    results = []

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

        for w_idx, win in enumerate(windows):
            cleaned_win = np.zeros_like(win, dtype=np.float64)
            channel_sigmas = []
            for i in range(win.shape[1]):
                cleaned_win[:, i] = clean_and_filter_1d(win[:, i], actual_fs, 1.0, 45.0)
                # Capture amplitude info BEFORE the RAM method's own standardization
                # throws it away. Std-dev of the cleaned trace is a crude but real
                # proxy for signal amplitude/energy, which is what magnitude most
                # directly correlates with -- the angle matrix alone never sees this.
                channel_sigmas.append(np.std(cleaned_win[:, i]))

            ram_B_mat, d_used = ram_matrix(cleaned_win[:, 0], target_n=target_n)
            ram_G_mat, _ = ram_matrix(cleaned_win[:, 1], target_n=target_n)
            ram_R_mat, _ = ram_matrix(cleaned_win[:, 2], target_n=target_n)

            ram_B = to_uint8(ram_B_mat)
            ram_G = to_uint8(ram_G_mat)
            ram_R = to_uint8(ram_R_mat)

            rgb = np.stack([ram_R, ram_G, ram_B], axis=-1)
            img = Image.fromarray(rgb, mode="RGB")

            filename = f"{file_id}_{sta_key}_win{w_idx:03d}.png"
            img.save(out_dir / filename)

            log_sigmas = [float(np.log(s + 1e-12)) for s in channel_sigmas]

            results.append({
                'split': split,
                'filename': filename,
                'magnitude': magnitude,
                'log_sigma_ch0': log_sigmas[0],
                'log_sigma_ch1': log_sigmas[1],
                'log_sigma_ch2': log_sigmas[2],
            })

    return results


def _process_task(args):
    file_path, out_dir, split, file_id, target_n, fs, window_seconds, overlap, catalog = args
    try:
        return mseed_file_to_ram_rgb(file_path, out_dir, split, file_id, target_n, fs, window_seconds, overlap, catalog)
    except Exception as e:
        return f"[WARN] Failed file {file_id}: {e}"

# ORCHESTRATION

def allocate_files(valid_files_with_counts: list, target_total: int, split_ratios: tuple) -> tuple:
    random.seed(42)
    random.shuffle(valid_files_with_counts)

    target_train = int(target_total * split_ratios[0])
    target_val = int(target_total * split_ratios[1])
    target_test = target_total - target_train - target_val

    train_files, val_files, test_files = [], [], []
    c_train = c_val = c_test = 0

    for fpath, w_count in valid_files_with_counts:
        if target_train > 0 and c_train < target_train:
            train_files.append(fpath)
            c_train += w_count
        elif target_val > 0 and c_val < target_val:
            val_files.append(fpath)
            c_val += w_count
        elif target_test > 0 and c_test < target_test:
            test_files.append(fpath)
            c_test += w_count
        else:
            if c_train >= target_train and c_val >= target_val and c_test >= target_test:
                break

    return (train_files, val_files, test_files), (c_train, c_val, c_test), (target_train, target_val, target_test)


def run_regression_preprocessing(
    eq_dir: str,
    output_dir: str,
    catalog_csv_path: Optional[str] = None,
    split_ratios: tuple = (0.70, 0.15, 0.15),
    target_n: int = 64,
    fs: float = 100.0,
    window_seconds: float = 60.0,
    overlap: float = 0.5,
    limit_pictures: int = None
):
    print("="*60)
    print("STARTING DATASET GENERATION (REGRESSION)")
    print("="*60)

    catalog = load_catalog(catalog_csv_path)

    source_path = Path(eq_dir)
    splits = ["train", "val", "test"]

    out_paths = {}
    csv_paths = {s: Path(output_dir) / s / "labels.csv" for s in splits}

    for split in splits:
        split_dir = Path(output_dir) / split / "earthquakes"
        split_dir.mkdir(parents=True, exist_ok=True)
        out_paths[split] = split_dir

    csv_header = ['filename', 'magnitude', 'log_sigma_ch0', 'log_sigma_ch1', 'log_sigma_ch2']
    for split in splits:
        with open(csv_paths[split], 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)

    if not source_path.exists():
        print(f"[ERROR] Input directory not found: {source_path}. Aborting.")
        return

    num_cores = max(1, multiprocessing.cpu_count() - 1)

    print("\n[PHASE 1] Scanning headers to find available valid windows...")
    mseed_files = list(source_path.rglob("*.mseed"))
    scan_args = [(fp, fs, window_seconds, overlap, catalog) for fp in mseed_files]

    valid_files = []
    total_windows = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        for file_path, w_count in executor.map(scan_single_mseed, scan_args):
            if w_count > 0:
                valid_files.append((file_path, w_count))
                total_windows += w_count

    print(f"  -> Found {total_windows} extractable windows across {len(valid_files)} files with valid magnitudes.")

    if total_windows == 0:
        print("[ERROR] 0 valid windows found. Aborting.")
        return

    print("\n[PHASE 2] Allocating splits...")
    target_total = min(total_windows, limit_pictures) if limit_pictures else total_windows
    print(f"  -> Final target set to {target_total} images.")

    file_lists, actual_counts, target_counts = allocate_files(
        valid_files,
        target_total,
        split_ratios
    )

    train_f, val_f, test_f = file_lists
    print(f"     Target | Train: {target_counts[0]:<6} | Val: {target_counts[1]:<6} | Test: {target_counts[2]:<6}")
    print(f"     Actual | Train: {actual_counts[0]:<6} | Val: {actual_counts[1]:<6} | Test: {actual_counts[2]:<6}")

    tasks = []
    for fpath in train_f:
        tasks.append((fpath, out_paths["train"], "train", fpath.stem, target_n, fs, window_seconds, overlap, catalog))
    for fpath in val_f:
        tasks.append((fpath, out_paths["val"], "val", fpath.stem, target_n, fs, window_seconds, overlap, catalog))
    for fpath in test_f:
        tasks.append((fpath, out_paths["test"], "test", fpath.stem, target_n, fs, window_seconds, overlap, catalog))

    print(f"\n[PHASE 3] Processing {len(tasks)} file generation tasks on {num_cores} cores...")

    csv_buffer = {s: [] for s in splits}

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        for result in executor.map(_process_task, tasks):
            if isinstance(result, str):
                print(result)
            elif result:
                for item in result:
                    csv_buffer[item['split']].append([
                        item['filename'],
                        item['magnitude'],
                        item['log_sigma_ch0'],
                        item['log_sigma_ch1'],
                        item['log_sigma_ch2'],
                    ])

    print("\n[PHASE 4] Writing labels to CSV...")
    for split in splits:
        if csv_buffer[split]:
            with open(csv_paths[split], 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerows(csv_buffer[split])

    print("\n[COMPLETE] Regression dataset generation finished successfully!")


# MAIN WRAPPER

if __name__ == "__main__":

    BASE_WAVEFORMS_DIR = Path("data/batched_waveforms")
    CATALOG_CSV = "catalogs/extracted_earthquakes.csv"

    target_directories = [
        "window_post_3s",
        "window_post_6s",
        "window_post_60s",
        "window_post_100s",
        "window_post_120s",
        "window_post_200s",
    ]

    # Fixed target image resolution used for EVERY window length.
    # d is now derived per-window (see reshape_to_target_n) so a 3s window
    # and a 200s window both produce a full TARGET_N x TARGET_N image,
    # instead of the old fixed d=100 collapsing short windows to a few pixels.
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
        print(f"RUNNING REGRESSION PIPELINE FOR: {dir_name} (Window Size: {nominal_window_secs}s)")
        print(f"{'#'*60}\n")

        run_regression_preprocessing(
            eq_dir=str(eq_dir),
            catalog_csv_path=CATALOG_CSV,
            output_dir=output_dir,
            split_ratios=(0.70, 0.15, 0.15),
            target_n=TARGET_N,
            fs=100.0,
            window_seconds=nominal_window_secs,
            overlap=0.50,
        )
