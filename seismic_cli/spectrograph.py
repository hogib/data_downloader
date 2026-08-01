"""
mseed -> 3-channel log-power spectrogram tensors (.pt) for cnn_from_tensor.py.

This script used to carry its own copy of the dataset-generation logic and,
with it, the whole family of defects that were fixed in `seismic_cli`:

  * splits allocated by FILE, not by station -- the same instrument appeared in
    train and test, so the model could score on station identity (this is the
    leakage bug the RAM pipeline fixed long ago; it was never fixed here)
  * splits allocated independently per class, breaking station-disjointness
    again even once stations are grouped
  * no per-station window cap -- one long noise trace could dominate a split
  * `merge(fill_value='interpolate')` fabricating linear ramps across telemetry
    gaps, then training on them as real signal
  * the first trace's sampling rate applied to every station in a file
  * channels chosen alphabetically (`sorted(...)[:3]`), which can select two
    horizontals and no vertical, and swaps components between 1/2- and
    N/E-named stations
  * unseeded `random.shuffle`, so splits differed on every run
  * no manifest, so the STA/LTA baseline could not be scored on the same windows
  * the bandpass silently skipped entirely when nyquist <= freqmax

plus two that are specific to spectrograms:

  * tensors of different shapes: the time axis is 1 + n_samples // hop_length,
    so a 50 Hz station produced (3, 129, 47) where a 100 Hz one produced
    (3, 129, 94), and PyTorch's default collate throws when batching them
  * no amplitude normalization: dB is absolute, so an instrument with 1000x the
    gain shifts its whole spectrogram by +60 dB -- a station fingerprint sitting
    in the input, which is exactly what a leaky split lets the model exploit

It is now a thin wrapper over the shared, fixed pipeline. Prefer the CLI:

    seismic-cli generate-spectrogram-dataset \\
        --eq-dir data/batched_waveforms/window_post_60s \\
        --noise-dir data/batched_noise_waveforms \\
        --output-dir dataset_spectrogram_60s \\
        --window-seconds 60 --overlap 0.25 --max --normalize station
"""

import re
from pathlib import Path

from seismic_cli.core import run_balanced_preprocessing
from seismic_cli.spectrogram import (SpectrogramEncoder,
                                     compute_station_spectral_baselines)

# CONFIGURATION
BASE_WAVEFORMS_DIR = Path("data/batched_waveforms")
NOISE_BATCH_DIR = "data/batched_noise_waveforms"
TARGET_DIRECTORIES = ["window_post_60s"]

N_FFT = 256
TOP_DB = 80.0
NOMINAL_FS = 100.0
OVERLAP = 0.25
NORMALIZE = "station"      # "station" | "per_window" | "none" -- see spectrogram.py
GENERATE_MAX = True        # use every usable station, balanced per split
MAX_WINDOWS_PER_STATION = None   # e.g. 20 for short windows


def main():
    for dir_name in TARGET_DIRECTORIES:
        match = re.search(r"(\d+)s(?=_|$)", dir_name)
        if not match:
            print(f"[WARN] Could not parse window seconds from {dir_name}. Skipping...")
            continue

        window_seconds = float(match.group(1))
        eq_dir = BASE_WAVEFORMS_DIR / dir_name
        if not eq_dir.exists():
            print(f"[ERROR] Directory {eq_dir} does not exist. Skipping...")
            continue

        output_dir = f"dataset_spectrogram_{dir_name}"
        print(f"\n{'#' * 60}")
        print(f"RUNNING SPECTROGRAM PIPELINE FOR: {dir_name} ({window_seconds:g}s windows)")
        print(f"{'#' * 60}\n")

        profiles = {}
        if NORMALIZE == "station":
            profiles = compute_station_spectral_baselines(
                NOISE_BATCH_DIR, n_fft=N_FFT, top_db=TOP_DB, nominal_fs=NOMINAL_FS,
            )

        encoder = SpectrogramEncoder(
            n_fft=N_FFT, top_db=TOP_DB, nominal_fs=NOMINAL_FS,
            window_seconds=window_seconds, normalize=NORMALIZE, noise_profiles=profiles,
        )

        run_balanced_preprocessing(
            eq_dir=str(eq_dir),
            noise_dir=NOISE_BATCH_DIR,
            output_dir=output_dir,
            split_ratios=(0.70, 0.15, 0.15),
            fs=NOMINAL_FS,
            window_seconds=window_seconds,
            overlap=OVERLAP,
            max_windows_per_station=MAX_WINDOWS_PER_STATION,
            generate_max=GENERATE_MAX,
            encoder=encoder,
        )


if __name__ == "__main__":
    main()
