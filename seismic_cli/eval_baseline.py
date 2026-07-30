"""
STA/LTA baseline for any station-split manifest.csv produced by
generate-dataset. Reconstructs each window directly from the raw mseed file
the CNN's images were generated from -- same file, same station, same window
index -- so this is scored on the EXACT same data the CNN's test-set
evaluation used.
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


def run_eval_sta_lta(
    manifest_path: str,
    split: str = "test",
    fs: float = 100.0,
    window_seconds: float = 60.0,
    overlap: float = 0.5,
    sta_seconds: float = 1.0,
    lta_seconds: float = 10.0,
) -> None:
    manifest_path = Path(manifest_path)
    print(f"[load] Reading manifest: {manifest_path}")
    df = pd.read_csv(manifest_path)
    df = df[df["split"] == split].copy()
    df["label"] = (df["class_name"] == "01_earthquake").astype(int)
    print(f"[info] {len(df)} '{split}' entries "
          f"({(df['label'] == 1).sum()} earthquake, {(df['label'] == 0).sum()} noise)")

    if len(df) > 0:
        sample_path = Path(df.iloc[0]["file_path"])
        print(f"[check] Sample file_path from manifest: {sample_path}")
        print(f"[check] Current working directory:       {Path.cwd()}")
        print(f"[check] Resolves to:                     {sample_path.resolve()}")
        print(f"[check] Exists: {sample_path.exists()}")
        if not sample_path.exists():
            print("[WARN] That path does not exist from this working directory. "
                  "Run this from the SAME directory generate-dataset was run from.")

    if lta_seconds >= window_seconds:
        print(f"[WARN] lta_seconds ({lta_seconds}) >= window_seconds ({window_seconds}) -- "
              f"every trace will be too short for STA/LTA to produce a meaningful result. "
              f"Pick an lta_seconds comfortably smaller than window_seconds.")

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
            fs=fs, window_seconds=window_seconds, overlap=overlap,
        )
        if win is None:
            n_skipped += 1
            continue

        score = sta_lta_score(win, fs=fs, sta_sec=sta_seconds, lta_sec=lta_seconds)
        scores.append(score)
        labels.append(row["label"])

    print(f"[info] Skipped {n_skipped} entries (couldn't reconstruct window -- "
          f"check fs/window_seconds/overlap match what generate-dataset actually used).")

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
