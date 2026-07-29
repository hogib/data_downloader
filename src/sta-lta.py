"""
STA/LTA baseline for the STATION-SPLIT mseed pipeline's manifest.csv
(generate_mseed_dataset_station_split.py). Reconstructs each window directly
from the raw mseed file the CNN's images were generated from -- same file,
same station, same window index -- so this is scored on the EXACT same data
the CNN's test-set evaluation used, no separate STEAD-style CSV needed.

Requires: obspy, pandas, numpy, scikit-learn
"""

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from obspy import read
from obspy.signal.trigger import classic_sta_lta
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)

MANIFEST_PATH = Path("dataset_window_post_60s/manifest.csv")
SPLIT_TO_EVALUATE = "test"

FS = 100.0
WINDOW_SECONDS = 60.0  # must match what the generator used for this dataset
OVERLAP = 0.50          # must match what the generator used for this dataset

STA_SECONDS = 1.0
LTA_SECONDS = 10.0  # must be comfortably shorter than WINDOW_SECONDS

FILENAME_RE = re.compile(r"_win(\d+)\.png$")


def extract_window_index(filename: str) -> Optional[int]:
    match = FILENAME_RE.search(filename)
    if not match:
        return None
    return int(match.group(1))


def get_window_from_mseed(
    file_path: str,
    station_key: str,
    window_index: int,
    fs: float,
    window_seconds: float,
    overlap: float,
) -> Optional[np.ndarray]:
    """
    Re-reads the raw mseed file and slices out the SAME window the RAM
    generator would have produced at this window_index, for this station.
    Mirrors window_array()'s indexing exactly: start = window_index * step.
    """
    try:
        st = read(file_path)
        st.merge(method=1, fill_value='interpolate')
    except Exception as e:
        print(f"  [SKIP] Could not open '{file_path}': {e}")
        return None

    channels = {}
    for tr in st:
        sta_key = f"{tr.stats.network}.{tr.stats.station}"
        if sta_key != station_key:
            continue
        chan = tr.stats.channel[-1].upper()
        channels[chan] = tr.data.astype(np.float64)

    available_chans = sorted(channels.keys())
    if len(available_chans) < 3:
        return None

    raw_channels = [channels[ch] for ch in available_chans[:3]]
    min_len = min(len(ch) for ch in raw_channels)
    event_data = np.column_stack([ch[:min_len] for ch in raw_channels])

    target_samples = int(fs * window_seconds)
    step_samples = int(target_samples * (1.0 - overlap))

    start_idx = window_index * step_samples
    end_idx = start_idx + target_samples

    if start_idx >= event_data.shape[0]:
        return None

    win = event_data[start_idx:end_idx, :]
    if len(win) < target_samples:
        pad_length = target_samples - len(win)
        win = np.pad(win, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)

    return win


def sta_lta_score(waveform: np.ndarray, fs: float, sta_sec: float, lta_sec: float) -> float:
    nsta = max(1, int(sta_sec * fs))
    nlta = max(nsta + 1, int(lta_sec * fs))

    max_ratio = 0.0
    for ch in range(waveform.shape[1]):
        trace = waveform[:, ch].astype(np.float64)
        if len(trace) <= nlta:
            continue
        try:
            cft = classic_sta_lta(trace, nsta, nlta)
            ratio = np.nanmax(cft) if len(cft) else 0.0
        except Exception:
            ratio = 0.0
        max_ratio = max(max_ratio, ratio)

    return max_ratio


def run_sta_lta_baseline():
    print(f"[load] Reading manifest: {MANIFEST_PATH}")
    df = pd.read_csv(MANIFEST_PATH)
    df = df[df["split"] == SPLIT_TO_EVALUATE].copy()
    df["label"] = (df["class_name"] == "01_earthquake").astype(int)
    print(f"[info] {len(df)} '{SPLIT_TO_EVALUATE}' entries "
          f"({(df['label'] == 1).sum()} earthquake, {(df['label'] == 0).sum()} noise)")

    # Quick upfront check: does the first file_path actually resolve from
    # wherever this script is being run? Catches path-mismatch issues
    # immediately instead of after silently skipping the whole split.
    if len(df) > 0:
        sample_path = Path(df.iloc[0]["file_path"])
        print(f"[check] Sample file_path from manifest: {sample_path}")
        print(f"[check] Current working directory:       {Path.cwd()}")
        print(f"[check] Resolves to:                     {sample_path.resolve()}")
        print(f"[check] Exists: {sample_path.exists()}")
        if not sample_path.exists():
            print("[WARN] That path does not exist from this working directory. "
                  "Run this script from the SAME directory you ran the dataset "
                  "generator from (the manifest stores paths as they were seen "
                  "at generation time -- if eq_dir/noise_dir were relative paths "
                  "like 'data/batched_waveforms/...', they only resolve correctly "
                  "when run from the same project root).")

    scores = []
    labels = []
    n_skipped = 0

    for _, row in df.iterrows():
        window_index = extract_window_index(row["filename"])
        if window_index is None:
            n_skipped += 1
            continue

        win = get_window_from_mseed(
            row["file_path"], row["station_key"], window_index,
            fs=FS, window_seconds=WINDOW_SECONDS, overlap=OVERLAP,
        )
        if win is None:
            n_skipped += 1
            continue

        score = sta_lta_score(win, fs=FS, sta_sec=STA_SECONDS, lta_sec=LTA_SECONDS)
        scores.append(score)
        labels.append(row["label"])

    print(f"[info] Skipped {n_skipped} entries (couldn't reconstruct window -- "
          f"check FS/WINDOW_SECONDS/OVERLAP match what the generator actually used).")

    scores = np.array(scores)
    labels = np.array(labels)

    if len(scores) == 0:
        print("[ERROR] No scores computed.")
        return

    auc = roc_auc_score(labels, scores)
    print(f"\nSTA/LTA AUC: {auc:.4f}")

    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    preds = (scores >= best_threshold).astype(int)

    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds)
    rec = recall_score(labels, preds)

    print(f"Best threshold (Youden's J): {best_threshold:.4f}")
    print(f"STA/LTA Accuracy:  {acc:.4f}")
    print(f"STA/LTA Precision: {prec:.4f}")
    print(f"STA/LTA Recall:    {rec:.4f}")

    print("\nComputed on the EXACT test windows (same file + station + window index) "
          "the CNN was evaluated on.")


if __name__ == "__main__":
    run_sta_lta_baseline()
