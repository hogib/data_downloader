"""
Dual {seq, img} encoder for the CNN+LSTM (1D2D-EDL) architecture from
Wang & Zhao (2025, Applied Soft Computing 172:112889), applied to the RAM
waveform classification task this repo started with (earthquake vs. noise),
rather than the catalog forecasting task in `catalog.py`.

Per the paper's Fig. 7 and Sec. 3.3.1/3.3.3, the two channels do NOT share a
reshaped intermediate -- they are two independent views built straight from
the raw window:

    seq: the RAW standardized (m, 3) waveform (Z/N/E), one timestep per raw
         sample. "For the 1D channel, the 1D time series is first normalized
         and then input into the LSTM model" (Sec. 3.3.1) -- the LSTM+MSA
         branch never sees the RAM reshape at all.
    img: the (3, target_n, target_n) RAM difference image, built from the
         SAME standardized window via the RAM method (Sec. 3.2) -- exactly
         what `RamImageEncoder` renders to PNG for the CNN-only pipeline.

An earlier version of this encoder fed the LSTM branch the (target_n, d)
matrix the RAM image is computed from, on the theory that both channels
should see literally the same reshaped data. That was not what the paper
does (confirmed by reading it directly, not from memory) and is fixed here:
the "dual channel" idea in the paper is complementary feature EXTRACTORS
over the same raw signal, not a shared intermediate representation.

Plugs into `core.run_balanced_preprocessing` like `SpectrogramEncoder`, so it
inherits station-disjoint splits, per-window station caps, gap rejection,
per-station baselines, `--max` balancing, and the manifest.

**Uniform tensor shape.** Windows are resampled to a nominal rate before
standardizing, so every `seq` tensor has the same sample count `m` regardless
of a station's native sampling rate -- the same fix `SpectrogramEncoder`
needed, and necessary here because, unlike the image (`target_n` is fixed by
construction), the raw-signal `seq`'s length would otherwise vary with fs.

**Sequence length is the whole window, not a downsampled summary.** This
matches the paper (their bearing samples are 1000 raw points), but multi-head
self-attention is O(m^2) in memory. At 100 Hz a 6s window is m=600 (attention
matrix 600x600, trivial); a 60s window is m=6000 (36M entries per head --
noticeably heavier, may need a smaller batch size). This is an honest
consequence of following the paper's design, not a bug.
"""

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.signal as signal

from seismic_cli.core import ram_matrix, standardize


def _resample_to(x: np.ndarray, fs_from: float, fs_to: float) -> np.ndarray:
    """Polyphase resample so every window yields the same number of samples."""
    if abs(fs_from - fs_to) < 1e-9:
        return x
    g = math.gcd(int(round(fs_to)), int(round(fs_from)))
    up, down = int(round(fs_to)) // g, int(round(fs_from)) // g
    return signal.resample_poly(x, up, down, axis=0)


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """Force exactly n samples (resampling can be off by one)."""
    if len(x) == n:
        return x
    if len(x) > n:
        return x[:n]
    return np.pad(x, ((0, n - len(x)),) + ((0, 0),) * (x.ndim - 1), mode="constant")


class RamDualEncoder:
    ext = ".pt"
    requires_spawn = True

    def __init__(self, target_n: int = 64, nominal_fs: float = 100.0, window_seconds: float = 60.0):
        self.target_n = target_n
        self.nominal_fs = nominal_fs
        self.window_seconds = window_seconds

    def target_samples(self) -> int:
        return int(round(self.nominal_fs * self.window_seconds))

    def __call__(self, cleaned_win, fs_station, sta_key, selection,
                 station_baselines: Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]],
                 out_dir, stem):
        import torch

        comp_z, comp_n, comp_e = selection
        x = _fit_length(_resample_to(cleaned_win, fs_station, self.nominal_fs),
                        self.target_samples())

        seqs, imgs = [], []
        for i, comp in enumerate((comp_z, comp_n, comp_e)):
            mu, sigma = station_baselines.get((sta_key, comp), (None, None))
            # Both branches read the SAME (mu, sigma), so the paper's
            # "normalize once, feed both channels" step (Fig. 5/7) holds even
            # though seq and img are computed by two separate calls here.
            seqs.append(standardize(x[:, i], mu=mu, sigma=sigma))
            R, _ = ram_matrix(x[:, i], target_n=self.target_n, mu=mu, sigma=sigma)
            imgs.append((np.clip(R, -np.pi, np.pi) + np.pi) / (2 * np.pi))    # -> [0, 1]

        seq = np.stack(seqs, axis=-1).astype(np.float32)   # (m, 3) -- raw standardized Z/N/E
        img = np.stack(imgs, axis=0).astype(np.float32)    # (3, target_n, target_n) in [0, 1]

        filename = stem + self.ext
        torch.save({"seq": torch.from_numpy(seq), "img": torch.from_numpy(img)}, out_dir / filename)
        return filename
