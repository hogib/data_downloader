"""
Dual {seq, img} encoder for the CNN+LSTM (1D2D-EDL) architecture from
Wang & Zhao (2025), applied directly to the RAM waveform classification task
this repo started with (earthquake vs. noise), rather than the catalog
forecasting task in `catalog.py`.

Both branches see the SAME reshaped window:

    img: the (3, target_n, target_n) RAM difference image -- exactly what
         `RamImageEncoder` renders to PNG for the CNN-only pipeline.
    seq: the (target_n, 3*d) chunk matrix each image is computed FROM --
         `ram_matrix_and_chunks`'s (target_n, d) companion per component,
         concatenated across Z/N/E. Column i of the image and step i of the
         sequence describe the same d-sample chunk, so the two channels are
         genuinely two views of one window (the image is a pairwise-angle
         summary of the same chunks the sequence presents in raw form),
         matching the paper's dual-channel design rather than pairing two
         unrelated representations.

Plugs into `core.run_balanced_preprocessing` like `SpectrogramEncoder`, so it
inherits station-disjoint splits, per-window station caps, gap rejection,
per-station baselines, `--max` balancing, and the manifest.

**Uniform tensor shape.** `d = ceil(fs_station * window_seconds / target_n)`
depends on the station's sampling rate, so without correction two stations
recorded at different rates would produce different-length sequences and
break the default collate at batch time -- the same problem `SpectrogramEncoder`
solved for spectrograms. Every window is resampled to a nominal rate before
the RAM reshape, so `d` (and therefore every seq tensor's shape) is fixed
across the whole dataset. The image is unaffected either way: `target_n` is
fixed by construction, so `RamImageEncoder`'s plain image-only path never
needed this.
"""

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.signal as signal

from seismic_cli.core import ram_matrix_and_chunks


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
            R, M, _ = ram_matrix_and_chunks(x[:, i], target_n=self.target_n, mu=mu, sigma=sigma)
            imgs.append((np.clip(R, -np.pi, np.pi) + np.pi) / (2 * np.pi))    # -> [0, 1], matches to_uint8's map
            # M is already standardized inside ram_matrix_and_chunks against
            # the SAME (mu, sigma) as the image, so the 1D branch sees the
            # identical per-channel normalization as the 2D branch does.
            seqs.append(M)

        seq = np.concatenate(seqs, axis=1).astype(np.float32)   # (target_n, 3*d)
        img = np.stack(imgs, axis=0).astype(np.float32)         # (3, target_n, target_n) in [0, 1]

        filename = stem + self.ext
        torch.save({"seq": torch.from_numpy(seq), "img": torch.from_numpy(img)}, out_dir / filename)
        return filename
