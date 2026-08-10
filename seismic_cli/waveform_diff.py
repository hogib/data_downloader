"""
Waveform-diff probe: does the difference between two consecutive, non-
overlapping windows of raw seismic data change measurably closer to an
earthquake?

A recurrence-plot-style chaos-theory technique -- comparing a signal's state
at two points in time to detect a qualitative change -- applied here via two
parallel per-window-pair metrics:

    raw_diff : RMS of the elementwise difference between the two windows'
               cleaned, bandpass-filtered, but UNNORMALIZED samples.
               Preserves amplitude by construction.
    ram_diff : mean absolute difference between the two windows' RAM images
               (core.ram_matrix). RAM's core computation is a cosine
               similarity between column vectors, which divides out each
               vector's own norm regardless of preprocessing -- RAM(x) ==
               RAM(c*x) for any positive c (report.md 8.2). ram_diff is kept
               as a cheap, complementary shape-only signal, not the primary
               metric: it structurally cannot see an amplitude-only change.

Both come from `core.clean_and_filter_1d` output directly -- detrended and
bandpass-filtered only, no amplitude scaling -- the same array this
project's log_snr computation already uses, so amplitude genuinely survives
into raw_diff.
"""

from typing import Dict, List, Tuple

import numpy as np

from seismic_cli.core import clean_and_filter_1d, ram_matrix, window_array_indexed


def windowed_diffs(component_data: np.ndarray, gap_mask: np.ndarray, fs: float,
                   window_seconds: float, target_n: int = 64,
                   max_gap_fraction: float = 0.05) -> Dict[str, object]:
    """
    Non-overlapping windows over one already-filtered component; for each
    TRUE consecutive pair (gap-rejected windows break the chain -- a window
    two slots apart is not "the next window"), returns raw_diff and ram_diff.
    """
    data_2d = component_data[:, None]
    gap_2d = gap_mask[:, None]
    windows, _ = window_array_indexed(data_2d, gap_2d, fs=fs, window_seconds=window_seconds,
                                      overlap=0.0, max_gap_fraction=max_gap_fraction)

    raw_diffs: List[float] = []
    ram_diffs: List[float] = []
    diff_idx: List[int] = []  # idx of the SECOND window in each consecutive pair --
                              # the point in the recording each diff value refers to.
    prev_idx = None
    prev_seg = None
    prev_ram = None
    for idx, win in windows:
        seg = win[:, 0]
        R, _ = ram_matrix(seg, target_n=target_n)
        if prev_idx is not None and idx == prev_idx + 1:
            raw_diffs.append(float(np.sqrt(np.mean((seg - prev_seg) ** 2))))
            ram_diffs.append(float(np.mean(np.abs(R - prev_ram))))
            diff_idx.append(idx)
        prev_idx, prev_seg, prev_ram = idx, seg, R

    return {"raw_diff": raw_diffs, "ram_diff": ram_diffs, "diff_idx": diff_idx,
            "n_windows": len(windows)}


def process_component(raw_1d: np.ndarray, gap_mask: np.ndarray, fs: float,
                      window_seconds: float, freqmin: float, freqmax: float,
                      target_n: int = 64) -> Dict[str, object]:
    """clean_and_filter_1d then windowed_diffs, for one component's full trace."""
    cleaned = clean_and_filter_1d(raw_1d, fs, freqmin, freqmax)
    return windowed_diffs(cleaned, gap_mask, fs, window_seconds, target_n=target_n)
