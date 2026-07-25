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


# PRE-SCAN LOGIC

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


# PROCESSING LOGIC

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


# ORCHESTRATION 

def allocate_files(valid_files_with_counts: list, target_total: int, split_ratios: tuple) -> tuple:
    """Helper function to cleanly split files into Train/Val/Test based on a strict total target."""
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


def run_balanced_preprocessing(
    eq_dir: str, 
    noise_dir: str, 
    output_dir: str, 
    split_ratios: tuple = (0.70, 0.15, 0.15),
    d: int = 64, 
    fs: float = 100.0,
    window_seconds: float = 60.0,
    overlap: float = 0.5,
    limit_pictures: int = None
):
    print("="*60)
    print("STARTING DATASET GENERATION")
    print("="*60)
    
    # Setup Directories
    classes = [("01_earthquake", Path(eq_dir)), ("00_noise", Path(noise_dir))]
    out_paths = {}
    
    for class_name, _ in classes:
        out_paths[class_name] = {}
        for split in ["train", "val", "test"]:
            split_dir = Path(output_dir) / split / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            out_paths[class_name][split] = split_dir

    num_cores = max(1, multiprocessing.cpu_count() - 1)
    
    # Scan and Count Everything
    print("\n[PHASE 1] Scanning headers to find available valid windows...")
    class_data = {}
    
    for class_name, source_path in classes:
        if not source_path.exists():
            raise FileNotFoundError(f"Input directory not found: {source_path}")
            
        mseed_files = list(source_path.rglob("*.mseed"))
        scan_args = [(fp, fs, window_seconds, overlap) for fp in mseed_files]
        
        valid_files = []
        total_windows = 0
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            for file_path, w_count in executor.map(scan_single_mseed, scan_args):
                if w_count > 0:
                    valid_files.append((file_path, w_count))
                    total_windows += w_count
                    
        class_data[class_name] = {
            "valid_files": valid_files,
            "total_windows": total_windows
        }
        print(f"  -> {class_name.upper()}: Found {total_windows} extractable windows across {len(valid_files)} files.")

    # Balance the Targets
    print("\n[PHASE 2] Balancing classes...")
    eq_total = class_data["01_earthquake"]["total_windows"]
    noise_total = class_data["00_noise"]["total_windows"]
    
    if eq_total == 0 or noise_total == 0:
        print("[ERROR] One of the classes has 0 valid windows. Aborting.")
        return

    # Find the maximum possible balanced dataset size (the bottleneck)
    bottleneck_size = min(eq_total, noise_total)
    
    # Apply limit_pictures if it exists (divided by 2 because it's the TOTAL dataset size desired)
    if limit_pictures:
        target_per_class = min(bottleneck_size, limit_pictures // 2)
    else:
        target_per_class = bottleneck_size
        
    print(f"  -> Bottleneck dictates a maximum of {bottleneck_size} images per class.")
    print(f"  -> Final target set to {target_per_class} images per class (Total: {target_per_class * 2} images).")

    # Allocate Splits Safely
    print("\n[PHASE 3] Allocating splits...")
    tasks = []
    
    for class_name, _ in classes:
        file_lists, actual_counts, target_counts = allocate_files(
            class_data[class_name]["valid_files"], 
            target_per_class, 
            split_ratios
        )
        
        train_f, val_f, test_f = file_lists
        print(f"  -> {class_name.upper()}:")
        print(f"     Target | Train: {target_counts[0]:<6} | Val: {target_counts[1]:<6} | Test: {target_counts[2]:<6}")
        print(f"     Actual | Train: {actual_counts[0]:<6} | Val: {actual_counts[1]:<6} | Test: {actual_counts[2]:<6}")

        # Build execution tasks
        for fpath in train_f:
            tasks.append((fpath, out_paths[class_name]["train"], fpath.stem, d, fs, window_seconds, overlap))
        for fpath in val_f:
            tasks.append((fpath, out_paths[class_name]["val"], fpath.stem, d, fs, window_seconds, overlap))
        for fpath in test_f:
            tasks.append((fpath, out_paths[class_name]["test"], fpath.stem, d, fs, window_seconds, overlap))

    # Generate Data
    print(f"\n[PHASE 4] Processing {len(tasks)} file generation tasks on {num_cores} cores...")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        for error_msg in executor.map(_process_task, tasks):
            if error_msg:
                print(error_msg)
                
    print("\n[COMPLETE] Balanced dataset generation finished successfully!")


if __name__ == "__main__":
    
    EQ_BATCH_DIR = "data/batched_waveforms/window_extended"
    NOISE_BATCH_DIR = "data/batched_noise_waveforms/noise_pre_3h"
    
    run_balanced_preprocessing(
        eq_dir=EQ_BATCH_DIR,      
        noise_dir=NOISE_BATCH_DIR,       
        output_dir="dataset",       
        split_ratios=(0.70, 0.15, 0.15), 
        d=32,
        fs=100.0,                  
        window_seconds=60.0,       
        overlap=0.50,
        limit_pictures=None, 
    )
