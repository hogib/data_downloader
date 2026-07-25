import concurrent.futures
import math
import multiprocessing
import os
import random
from pathlib import Path
from typing import List, Tuple

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

def reshape_to_d_by_n(x: np.ndarray, d: int) -> np.ndarray:
    m = len(x)
    n = int(np.ceil(m / d))
    M = np.empty((d, n), dtype=np.float64)
    for col in range(n):
        base = col * d
        for row in range(d):
            idx = (base + row) % m
            M[row, col] = x[idx]
    return M

def ram_matrix(x: np.ndarray, d: int, eps: float = 1e-12) -> np.ndarray:
    x_std = standardize(x, eps=eps)
    M = reshape_to_d_by_n(x_std, d=d)
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
    return betas[None, :] - betas[:, None]

def to_uint8(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    mn, mx = np.min(mat), np.max(mat)
    if np.isclose(mx, mn):
        return np.zeros(mat.shape, dtype=np.uint8)
    out = (mat - mn) / (mx - mn)
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
    if fs / 2 > freqmax:
        b, a = signal.butter(4, [freqmin, freqmax], btype='bandpass', fs=fs)
        x = signal.filtfilt(b, a, x)
    return x

def window_array(data: np.ndarray, fs: float = 100.0, window_seconds: float = 60.0, overlap: float = 0.5) -> List[np.ndarray]:
    target_samples = int(fs * window_seconds)
    step_samples = int(target_samples * (1.0 - overlap))
    if step_samples < 1:
        raise ValueError("Overlap fraction too high; step size must be at least 1 sample.")
    n_samples = data.shape[0]
    windows = []
    if n_samples < target_samples:
        return windows
    n_windows = math.ceil((n_samples - target_samples) / step_samples) + 1
    for i in range(n_windows):
        start_idx = i * step_samples
        end_idx = start_idx + target_samples
        win = data[start_idx:end_idx, :]
        if len(win) == target_samples:
            windows.append(win)
    return windows


# PRE-SCAN

def scan_single_mseed(args: Tuple[Path, float, float, float]) -> Tuple[Path, int]:
    """Reads ONLY the headers of an mseed file to count extractable valid windows."""
    file_path, fs, window_seconds, overlap = args
    try:
        st = read(str(file_path), headonly=True)
    except Exception:
        return file_path, 0

    target_samples = int(fs * window_seconds)
    step_samples = int(target_samples * (1.0 - overlap))
    
    stations = {}
    for tr in st:
        sta_key = f"{tr.stats.network}.{tr.stats.station}"
        chan = tr.stats.channel[-1].upper()
        if sta_key not in stations:
            stations[sta_key] = {}
        stations[sta_key][chan] = tr.stats.npts
        
    total_windows_in_file = 0
    for sta_key, channels in stations.items():
        available_chans = sorted(channels.keys())
        if len(available_chans) >= 3:
            min_len = min(channels[ch] for ch in available_chans[:3])
            if min_len >= target_samples:
                n_win = ((min_len - target_samples) // step_samples) + 1
                total_windows_in_file += n_win
                
    return file_path, total_windows_in_file


# PROCESSING

def mseed_file_to_ram_rgb(file_path: Path, out_dir: Path, file_id: str, d: int, fs: float, window_seconds: float, overlap: float):
    st = read(str(file_path))
    stations = {}
    for tr in st:
        sta_key = f"{tr.stats.network}.{tr.stats.station}"
        if sta_key not in stations:
            stations[sta_key] = {}
        chan = tr.stats.channel[-1].upper()
        stations[sta_key][chan] = tr.data.astype(np.float64)

    for sta_key, channels in stations.items():
        available_chans = sorted(channels.keys())
        if len(available_chans) < 3:
            continue  
        
        raw_channels = [channels[ch] for ch in available_chans[:3]]
        min_len = min(len(ch) for ch in raw_channels)
        
        if min_len < int(fs * window_seconds):
            continue  
            
        trimmed_channels = [ch[:min_len] for ch in raw_channels]
        event_data = np.column_stack(trimmed_channels)

        cleaned_data = np.zeros_like(event_data, dtype=np.float64)
        for i in range(event_data.shape[1]):
            cleaned_data[:, i] = clean_and_filter_1d(event_data[:, i], fs, 1.0, 45.0)

        windows = window_array(cleaned_data, fs=fs, window_seconds=window_seconds, overlap=overlap)

        for w_idx, win in enumerate(windows):
            ram_B = to_uint8(ram_matrix(win[:, 0], d=d)) 
            ram_G = to_uint8(ram_matrix(win[:, 1], d=d)) 
            ram_R = to_uint8(ram_matrix(win[:, 2], d=d)) 
            
            rgb = np.stack([ram_R, ram_G, ram_B], axis=-1)
            img = Image.fromarray(rgb, mode="RGB")
            img.save(out_dir / f"{file_id}_{sta_key}_win{w_idx:03d}.png")

def _process_task(args):
    file_path, out_dir, file_id, d, fs, window_seconds, overlap = args
    try:
        mseed_file_to_ram_rgb(file_path, out_dir, file_id, d, fs, window_seconds, overlap)
        return None 
    except Exception as e:
        return f"[WARN] Failed file {file_id}: {e}"


def run_preprocessing_mseed(
    data_dir: str, 
    output_dir: str, 
    class_name: str, 
    split_ratios: tuple = (0.70, 0.15, 0.15),
    d: int = 64, 
    fs: float = 100.0,
    window_seconds: float = 60.0,
    overlap: float = 0.5,
    limit_pictures: int = 140000
):
    train_dir = Path(output_dir) / "train" / class_name
    val_dir = Path(output_dir) / "val" / class_name
    test_dir = Path(output_dir) / "test" / class_name

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    source_path = Path(data_dir)
    if not source_path.exists():
        print(f"[{class_name.upper()}] Directory not found: {source_path}. Skipping...")
        return

    mseed_files = list(source_path.rglob("*.mseed"))
    if not mseed_files:
        print(f"[{class_name.upper()}] No .mseed files found under {source_path}.")
        return

    num_cores = max(1, multiprocessing.cpu_count() - 1)
    
    # --- 1. SCAN PHASE ---
    print(f"[{class_name.upper()}] Phase 1: Scanning headers of {len(mseed_files)} files to count valid windows...")
    valid_files_with_counts = []
    
    scan_args = [(fp, fs, window_seconds, overlap) for fp in mseed_files]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        for file_path, w_count in executor.map(scan_single_mseed, scan_args):
            if w_count > 0:
                valid_files_with_counts.append((file_path, w_count))
                
    if not valid_files_with_counts:
        print(f"[{class_name.upper()}] No valid files with sufficient length/channels found.")
        return

    #  SPLIT PHASE
    print(f"[{class_name.upper()}] Phase 2: Allocating {len(valid_files_with_counts)} valid files...")
    
    random.seed(42)
    random.shuffle(valid_files_with_counts)

    # Determine the actual number of windows available in the dataset
    total_available_windows = sum(count for _, count in valid_files_with_counts)
    
    # Base our targets on the lesser of your hard limit OR the actual available data
    if limit_pictures and limit_pictures < total_available_windows:
        effective_total = limit_pictures
    else:
        effective_total = total_available_windows

    # Calculate exact targets based on the effective total and requested ratios
    target_train = int(effective_total * split_ratios[0])
    target_val = int(effective_total * split_ratios[1])
    target_test = effective_total - target_train - target_val  # absorbs any rounding errors

    train_files, val_files, test_files = [], [], []
    count_train = count_val = count_test = 0

    for fpath, w_count in valid_files_with_counts:
        # Check against target limits. If a split ratio is 0, its target is 0,
        # so this condition fails immediately and cleanly skips the block.
        if target_train > 0 and count_train < target_train:
            train_files.append(fpath)
            count_train += w_count
        elif target_val > 0 and count_val < target_val:
            val_files.append(fpath)
            count_val += w_count
        elif target_test > 0 and count_test < target_test:
            test_files.append(fpath)
            count_test += w_count
        else:
            # Breaks early if limit_pictures threshold is successfully reached
            break

    print(f"[{class_name.upper()}] Target Windows -> Train: {target_train} | Val: {target_val} | Test: {target_test}")
    print(f"[{class_name.upper()}] Actual Windows -> Train: {count_train} | Val: {count_val} | Test: {count_test}")
    print(f"[{class_name.upper()}] Files Used     -> Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}\n")

    # PROCESS PHASE
    print(f"[{class_name.upper()}] Phase 3: Extracting and converting {count_train + count_val + count_test} total windows...")
    
    def task_generator():
        for fpath in train_files:
            yield (fpath, train_dir, fpath.stem, d, fs, window_seconds, overlap)
        for fpath in val_files:
            yield (fpath, val_dir, fpath.stem, d, fs, window_seconds, overlap)
        for fpath in test_files:
            yield (fpath, test_dir, fpath.stem, d, fs, window_seconds, overlap)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        for error_msg in executor.map(_process_task, task_generator()):
            if error_msg:
                print(error_msg)
                
    print(f"[{class_name.upper()}] Processing complete!\n")

if __name__ == "__main__":
    datasets_to_process = [
        {"path": "data/batched_waveforms", "class_name": "01_earthquake"},
        {"path": "data/batched_noise_waveforms", "class_name": "00_noise"}
    ]

    for dataset in datasets_to_process:
        run_preprocessing_mseed(
            data_dir=dataset["path"],      
            output_dir="dataset",       
            class_name=dataset["class_name"],
            split_ratios=(0, 0, 1.0), 
            d=64,
            fs=100.0,                  
            window_seconds=60.0,       
            overlap=0.50,
        )
