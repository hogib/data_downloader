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
import scipy.signal
from obspy import read
from obspy.signal.trigger import classic_sta_lta
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from seismic_cli.core import select_components

FILENAME_RE = re.compile(r"_win(\d+)\.(?:png|pt)$")

# Floors/caps for the dynamically derived STA/LTA parameters.
MIN_STA_SECONDS = 0.05   # 5 samples at 100 Hz -- shorter is just noise
MAX_LTA_SECONDS = 10.0   # classic long-term average for full-length windows


def derive_sta_lta_params(window_seconds: float) -> tuple:
    """
    Derives (sta_seconds, lta_seconds) from the analysis window length.

    The classic 1s/10s defaults only make sense when the window is much
    longer than the LTA -- on a 3s or 6s window a 10s LTA cannot even be
    computed, and the whole baseline silently degenerates. Scaling rule:
    LTA = window/3 (capped at the classic 10s), STA = LTA/10 (floored at
    MIN_STA_SECONDS). At 60s this reproduces the original 1.0/10.0 exactly,
    so existing long-window results are unchanged; at 6s it yields 0.2/2.0
    and at 3s 0.1/1.0.

    **Known limitation, confirmed empirically (report.md, "STA/LTA Parameter
    Sensitivity on Anchored Windows").** `classic_sta_lta`'s characteristic
    function is exactly 0 for its first `nlta` samples (no long-term average
    exists yet). `anchor.py`'s default `pre_arrival_fraction=0.2` places the
    P-wave arrival at only 20% into an anchored window, but this formula's
    LTA is 33% of the window (window/3) -- so for ANY anchored window under
    ~50s, LTA/window > pre_arrival_fraction, meaning the arrival itself
    falls inside the forced-zero warm-up region and is invisible to the
    characteristic function. Measured on the 6s anchored dataset: this
    formula's defaults (STA=0.2s, LTA=2.0s) give AUC 0.51 (indistinguishable
    from random); a validation-selected LTA that respects the anchoring
    buffer (STA=0.03s, LTA=0.3s) gives AUC 0.82. This is NOT fixed here
    automatically, since 60s windows are unanchored (sliced from origin
    time, not arrival time) and this exact formula is relied on to reproduce
    their historical 1.0/10.0 parameters unchanged. For any ANCHORED window
    (anything generated via `anchor-windows`), pass `--sta-seconds`/
    `--lta-seconds` explicitly with LTA comfortably under
    `pre_arrival_fraction * window_seconds`, rather than trusting this
    auto-derivation.
    """
    lta_seconds = min(MAX_LTA_SECONDS, window_seconds / 3.0)
    sta_seconds = max(MIN_STA_SECONDS, lta_seconds / 10.0)
    return sta_seconds, lta_seconds


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

    # Keep the longest trace per component, mirroring generation, so a stray
    # duplicate channel can't shift which data gets reconstructed.
    channels = {}
    for tr in st:
        sta_key = f"{tr.stats.network}.{tr.stats.station}"
        if sta_key != station_key:
            continue
        chan = tr.stats.channel[-1].upper()
        existing = channels.get(chan)
        if existing is not None and len(existing) >= len(tr.data):
            continue
        channels[chan] = tr.data.astype(np.float64)

    selection = select_components(channels.keys())
    if selection is None:
        return None

    raw_channels = [channels[ch] for ch in selection]
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
            # Raw MiniSEED counts carry large DC offsets; classic STA/LTA
            # works on signal energy, so an un-detrended trace with a big
            # offset has its ratio pinned near 1 regardless of content --
            # silently crippling the baseline at exactly those stations.
            trace = scipy.signal.detrend(trace, type='linear')
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
    sta_seconds: Optional[float] = None,
    lta_seconds: Optional[float] = None,
) -> None:
    auto_sta, auto_lta = derive_sta_lta_params(window_seconds)
    derived = sta_seconds is None or lta_seconds is None
    if sta_seconds is None:
        sta_seconds = auto_sta
    if lta_seconds is None:
        lta_seconds = auto_lta
    source = "auto-derived from window length" if derived else "explicitly set"
    print(f"[params] window={window_seconds:g}s -> STA={sta_seconds:g}s, LTA={lta_seconds:g}s ({source})")
    if derived and lta_seconds > 0.15 * window_seconds:
        print(f"[WARN] LTA={lta_seconds:g}s is >15% of the window. If this manifest was built from "
              f"arrival-anchored windows (`anchor-windows`, default pre_arrival_fraction=0.2), the "
              f"arrival likely falls inside classic_sta_lta's forced-zero warm-up region, silently "
              f"crippling this baseline (measured: AUC 0.51 vs 0.82 with a properly bounded LTA on a "
              f"6s anchored dataset -- see derive_sta_lta_params docstring). Pass --sta-seconds/"
              f"--lta-seconds explicitly for anchored windows.")

    manifest_path = Path(manifest_path)
    print(f"[load] Reading manifest: {manifest_path}")
    df = pd.read_csv(manifest_path)
    df = df[df["split"] == split].copy()
    df["label"] = (df["class_name"] == "01_earthquake").astype(int)
    print(f"[info] {len(df)} '{split}' entries "
          f"({(df['label'] == 1).sum()} earthquake, {(df['label'] == 0).sum()} noise)")

    has_fs_column = "fs" in df.columns
    if has_fs_column:
        print("[info] Manifest records per-station sampling rates; using them for reconstruction.")
    else:
        print(f"[info] Older manifest without an 'fs' column; assuming {fs} Hz for every entry.")

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

        row_fs = fs
        if has_fs_column and pd.notna(row["fs"]):
            row_fs = float(row["fs"])

        win = get_window_from_mseed(
            row["file_path"], row["station_key"], window_index,
            fs=row_fs, window_seconds=window_seconds, overlap=overlap,
        )
        if win is None:
            n_skipped += 1
            continue

        score = sta_lta_score(win, fs=row_fs, sta_sec=sta_seconds, lta_sec=lta_seconds)
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
    print("[note] The threshold above is selected ON THIS SPLIT (oracle), which "
          "flatters STA/LTA's accuracy/precision/recall. Compare models via AUC; "
          "treat these thresholded numbers as STA/LTA's upper bound.")

    print("\nComputed on the EXACT test windows (same file + station + window index) "
          "the CNN was evaluated on.")
