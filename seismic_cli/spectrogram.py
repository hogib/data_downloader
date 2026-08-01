"""
Spectrogram encoder for the shared dataset pipeline.

Produces 3-channel (Z / N-ish / E-ish) log-power spectrogram tensors saved as
.pt files, consumed by cnn_earthquake's `cnn_from_tensor.py`. It plugs into
`core.run_balanced_preprocessing`, so it inherits station-disjoint splits,
per-window station caps, gap rejection, per-station sampling rates, `--max`
balancing, and the manifest -- none of which the old standalone
`src/spectrograph.py` had.

Two properties matter specifically for spectrograms, and neither was handled
before:

**Uniform tensor shape.** The time axis length is
``1 + n_samples // hop_length``. Windows are sized in *seconds* using each
station's own sampling rate, so a 50 Hz station yields half the samples of a
100 Hz one and therefore a differently-shaped tensor -- which makes PyTorch's
default collate throw at batch time. Every window is resampled to the nominal
rate before transforming, so all tensors in a dataset share one shape.

**Amplitude normalization.** dB values are absolute, so an instrument with
1000x the gain shifts its whole spectrogram by +60 dB. Left alone, that is a
station fingerprint the model can memorize instead of learning seismology
(and, combined with the old split-by-file behavior, a direct leak). The
`normalize` modes trade off differently against the amplitude question:

- ``"station"`` (default): subtract that station's own median noise dB profile
  per frequency bin. The result is *dB above that station's noise floor* --
  instrument gain cancels, while genuine amplitude-above-background survives.
  This is the quantity STA/LTA triggers on and the one the RAM transform
  discards by construction (see report.md 8.2), so it is the reason to try
  spectrograms at all. Stations with no usable noise profile fall back to
  ``per_window``.
- ``"per_window"``: z-score each window over all bins. Removes instrument gain
  but *also* removes absolute amplitude -- the same blind spot RAM has.
- ``"none"``: raw dB. Keeps amplitude but leaks instrument gain; only sensible
  if you have already deconvolved the instrument response.
"""

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.signal as signal
from obspy import read

from seismic_cli.core import clean_and_filter_1d, select_components

NORMALIZE_MODES = ("station", "per_window", "none")


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


class SpectrogramEncoder:
    """
    Per-window encoder producing a (3, freq, time) float32 tensor.

    Picklable by design (plain attributes only) so it survives the
    ProcessPoolExecutor hand-off; torch/torchaudio are imported lazily inside
    the worker so the RAM-only path never needs them installed.

    `requires_spawn` tells the orchestrator to use 'spawn' workers. Under the
    default 'fork', torch's threading state does not survive the fork and the
    workers deadlock silently -- sleeping at 0% CPU with nothing written, which
    looks like a slow run rather than a hang. (The standalone script this
    replaces had the same fork+torch structure.)
    """
    ext = ".pt"
    requires_spawn = True

    def __init__(self, n_fft: int = 256, hop_length: Optional[int] = None,
                 top_db: float = 80.0, nominal_fs: float = 100.0,
                 window_seconds: float = 60.0, normalize: str = "station",
                 noise_profiles: Optional[Dict[Tuple[str, str], np.ndarray]] = None):
        if normalize not in NORMALIZE_MODES:
            raise ValueError(f"normalize must be one of {NORMALIZE_MODES}, got {normalize!r}")
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4
        self.top_db = top_db
        self.nominal_fs = nominal_fs
        self.window_seconds = window_seconds
        self.normalize = normalize
        self.noise_profiles = noise_profiles or {}
        self._tf = None  # lazily built per worker process

    # -- lazy torch setup -------------------------------------------------
    def _transforms(self):
        if self._tf is None:
            import torch
            import torchaudio.transforms as T
            # One thread per worker: the pool already provides parallelism, and
            # letting each worker spin up a full thread pool oversubscribes the
            # machine badly.
            torch.set_num_threads(1)
            self._tf = (
                torch,
                T.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, power=2.0),
                # top_db bounds the dynamic range; without it a near-silent bin
                # drags the tensor's floor arbitrarily low and destabilizes
                # whatever normalization runs afterwards.
                T.AmplitudeToDB(stype="power", top_db=self.top_db),
            )
        return self._tf

    def target_samples(self) -> int:
        return int(round(self.nominal_fs * self.window_seconds))

    def spec_db(self, cleaned_win: np.ndarray, fs_station: float):
        """(samples, 3) float array -> (3, freq, time) dB tensor at nominal fs."""
        torch, spec_tf, db_tf = self._transforms()
        x = _fit_length(_resample_to(cleaned_win, fs_station, self.nominal_fs),
                        self.target_samples())
        t = torch.from_numpy(np.ascontiguousarray(x.T)).float()
        return db_tf(spec_tf(t))

    # -- encoder protocol -------------------------------------------------
    def __call__(self, cleaned_win, fs_station, sta_key, selection,
                 station_baselines, out_dir, stem):
        torch, _, _ = self._transforms()
        s = self.spec_db(cleaned_win, fs_station)

        if self.normalize == "station":
            applied = False
            if self.noise_profiles:
                prof = [self.noise_profiles.get((sta_key, c)) for c in selection]
                if all(p is not None for p in prof):
                    ref = torch.from_numpy(np.stack(prof)).float().unsqueeze(-1)
                    s = s - ref            # dB above this station's noise floor
                    applied = True
            if not applied:               # no usable profile -> safe fallback
                s = (s - s.mean()) / (s.std() + 1e-6)
        elif self.normalize == "per_window":
            s = (s - s.mean()) / (s.std() + 1e-6)

        filename = stem + self.ext
        torch.save(s.contiguous(), out_dir / filename)
        return filename


def compute_station_spectral_baselines(
    noise_dir: str,
    n_fft: int = 256,
    hop_length: Optional[int] = None,
    top_db: float = 80.0,
    nominal_fs: float = 100.0,
    freqmin: float = 1.0,
    freqmax: float = 45.0,
    min_seconds: float = 60.0,
    max_files_per_station: int = 20,
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Median dB-per-frequency-bin noise profile for each (station, component).

    Uses the median over time frames, not the mean, so an event that slipped
    through the noise-window contamination filter cannot drag a station's
    floor upward. Requires `min_seconds` of usable data before a profile is
    trusted, mirroring the amplitude-baseline rule in core.py.
    """
    import torch
    import torchaudio.transforms as T

    hop_length = hop_length if hop_length is not None else n_fft // 4
    spec_tf = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=2.0)
    db_tf = T.AmplitudeToDB(stype="power", top_db=top_db)

    noise_path = Path(noise_dir)
    if not noise_path.exists():
        print(f"[WARN] Noise directory not found for spectral baselines: {noise_path}")
        return {}

    print("\n[SPECTRAL BASELINE] Building per-station noise dB profiles...")
    files = sorted(noise_path.rglob("*.mseed"))
    print(f"  -> Scanning {len(files)} noise files.")

    frames: Dict[Tuple[str, str], list] = {}
    seen: Dict[Tuple[str, str], int] = {}

    for i, fp in enumerate(files, 1):
        if i % 200 == 0:
            print(f"  ...{i}/{len(files)}")
        try:
            st = read(str(fp))
            st.merge(method=1, fill_value="interpolate")
        except Exception:
            continue
        for tr in st:
            sta = f"{tr.stats.network}.{tr.stats.station}"
            comp = tr.stats.channel[-1].upper()
            key = (sta, comp)
            if seen.get(key, 0) >= max_files_per_station:
                continue
            fs_actual = float(tr.stats.sampling_rate)
            data = np.asarray(tr.data, dtype=np.float64)
            if len(data) < int(fs_actual * 10):
                continue
            try:
                cleaned = clean_and_filter_1d(data, fs_actual, freqmin, freqmax)
                x = _resample_to(cleaned, fs_actual, nominal_fs)
                t = torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(0)
                s = db_tf(spec_tf(t))[0]           # (freq, time)
            except Exception:
                continue
            frames.setdefault(key, []).append(s)
            seen[key] = seen.get(key, 0) + 1

    min_frames = max(1, int(min_seconds * nominal_fs / hop_length))
    profiles: Dict[Tuple[str, str], np.ndarray] = {}
    rejected = 0
    for key, chunks in frames.items():
        allf = torch.cat(chunks, dim=1)
        if allf.shape[1] < min_frames:
            rejected += 1
            continue
        profiles[key] = allf.median(dim=1).values.numpy()

    n_sta = len({s for s, _ in profiles})
    print(f"  -> Built profiles for {len(profiles)} (station, component) pairs across {n_sta} stations.")
    print(f"  -> Rejected {rejected} pairs with under {min_seconds:.0f}s of usable noise.")
    return profiles
