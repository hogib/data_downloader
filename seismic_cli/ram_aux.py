"""
RAM image + amplitude-scalar encoder: the direct fix for RAM's diagnosed
blind spot (report.md 8.2).

RAM is EXACTLY scale-invariant: RAM(c*x) == RAM(x) for any positive c, so the
image cannot represent absolute amplitude or amplitude-above-noise -- exactly
the quantity STA/LTA and spectrogram-station-normalize use, and exactly why
those baselines have outperformed a RAM-only classifier on short windows.
`--baseline` standardization does not fix this (verified earlier in this
project): RAM's invariance makes it insensitive to WHICH (mu, sigma) the
image is computed with, so patching the image itself cannot work. The fix
instead has to bypass the image: compute the discarded scalar directly and
feed it to the classifier alongside the image, the same pattern already used
in `regression.py` for magnitude and in `catalog.py`'s aux features.

Two scalars per window, both cheap and already-established quantities in
this codebase:
    log_snr : log(window_rms / station_noise_rms), averaged over Z/N/E --
              identical definition to regression.py's log_snr. Requires a
              station noise baseline; defaults to 0.0 (dataset-average-ish)
              for any station without enough noise data to build one.
    log_rms : log(window_rms) alone, averaged over Z/N/E -- absolute level,
              independent of any station baseline being available at all.
"""

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from seismic_cli.core import ram_matrix, to_uint8

AUX_FEATURES = ["log_snr", "log_rms"]


class RamAuxEncoder:
    """
    Per-window encoder producing {img, aux} tensors: img is the standard
    (3, target_n, target_n) RAM image (plain per-window standardization --
    the established finding that baseline choice does not change RAM's
    content makes this choice immaterial to the image itself); aux is the
    (2,) [log_snr, log_rms] vector described above.

    `station_baselines` is a plain dict {(station_key, component): (mu, sigma)}
    baked in at construction (from `core.compute_station_noise_baselines`),
    independent of the `--baseline` flag's effect on the image -- this
    encoder always wants the noise baseline for log_snr regardless of
    whether the image itself uses it.
    """
    ext = ".pt"
    requires_spawn = True

    def __init__(self, target_n: int = 64,
                station_baselines: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None):
        self.target_n = target_n
        self.station_baselines = station_baselines or {}

    def __call__(self, cleaned_win, fs_station, sta_key, selection,
                 station_baselines, out_dir, stem):
        import torch

        imgs, snrs, rmss = [], [], []
        for i, comp in enumerate(selection):
            R, _ = ram_matrix(cleaned_win[:, i], target_n=self.target_n)
            imgs.append((np.clip(R, -np.pi, np.pi) + np.pi) / (2 * np.pi))

            sigma_win = float(np.std(cleaned_win[:, i]))
            if sigma_win > 0:
                rmss.append(math.log(sigma_win))
                _mu, sigma_noise = self.station_baselines.get((sta_key, comp), (None, None))
                if sigma_noise and sigma_noise > 0:
                    snrs.append(math.log(sigma_win / sigma_noise))

        img = np.stack(imgs, axis=0).astype(np.float32)
        log_snr = float(np.mean(snrs)) if snrs else 0.0
        log_rms = float(np.mean(rmss)) if rmss else 0.0
        aux = np.array([log_snr, log_rms], dtype=np.float32)

        filename = stem + self.ext
        torch.save({"img": torch.from_numpy(img), "aux": torch.from_numpy(aux)}, out_dir / filename)
        return filename
