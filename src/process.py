import concurrent.futures
import math
import multiprocessing
import os
import random
from pathlib import Path
from typing import List

import numpy as np
import scipy.signal as signal
from obspy import read
from PIL import Image


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

def mseed_file_to_ram_rgb(file_path: Path, out_dir: Path, file_id: str, d: int, fs: float, window_seconds: float, overlap: float):
    """Loads a multi-trace MiniSEED file, groups components, and generates RAM RGB images."""
    st = read(str(file_path))
    
    # Organize traces by station code to handle multi-station mseed files properly
    stations = {}
    for tr in st:
        sta_key = f"{tr.stats.network}.{tr.stats.station}"
        if sta_key not in stations:
            stations[sta_key] = {}
        # Identify channel component (e.g., HHN, HHE, HHZ -> 0, 1, 2)
        chan = tr.stats.channel[-1].upper()
        stations[sta_key][chan] = tr.data.astype(np.float64)

    # Process each station found in the stream
    for sta_key, channels in stations.items():
        # Ensure we have standard 3 components (Z, N/1, E/2) or fallback gracefully
        available_chans = sorted(channels.keys())
        if len(available_chans) < 3:
            continue  # Skip stations missing complete 3-component data
        
        # Stack channels into a (samples, 3) matrix
        event_data = np.column_stack([channels[ch] for ch in available_chans[:3]])

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
    limit_pictures: int = None
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

    # Gather all .mseed files recursively from batched subdirectories
    mseed_files = list(source_path.rglob("*.mseed"))
    if not mseed_files:
        print(f"[{class_name.upper()}] No .mseed files found under {source_path}.")
        return

    # Shuffle and split files into train/val/test
    random.seed(42)
    random.shuffle(mseed_files)

    if limit_pictures and len(mseed_files) > limit_pictures:
        mseed_files = mseed_files[:limit_pictures]

    n_total = len(mseed_files)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])

    train_files = mseed_files[:n_train]
    val_files = mseed_files[n_train:n_train + n_val]
    test_files = mseed_files[n_train + n_val:]

    print(f"[{class_name.upper()}] Total files found: {n_total}")
    print(f"[{class_name.upper()}] Split -> Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

    def task_generator():
        for fpath in train_files:
            yield (fpath, train_dir, fpath.stem, d, fs, window_seconds, overlap)
        for fpath in val_files:
            yield (fpath, val_dir, fpath.stem, d, fs, window_seconds, overlap)
        for fpath in test_files:
            yield (fpath, test_dir, fpath.stem, d, fs, window_seconds, overlap)

    num_cores = max(1, multiprocessing.cpu_count() - 1)
    print(f"[{class_name.upper()}] Processing on {num_cores} CPU cores...")
    
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
            split_ratios=(0.70, 0.15, 0.15), 
            d=64,
            fs=100.0,                  
            window_seconds=60.0,       
            overlap=0.50,
        )
