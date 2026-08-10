"""
Seismic RAM/CNN pipeline CLI.

Usage examples:

  seismic-cli anchor-windows \\
      --source-dir data/batched_waveforms/window_post_60s \\
      --output-base-dir data/batched_waveforms \\
      --target-seconds 3 --target-seconds 6 --target-seconds 10

  seismic-cli generate-dataset \\
      --eq-dir data/batched_waveforms/window_post_60s \\
      --noise-dir data/batched_noise_waveforms \\
      --output-dir dataset_60s \\
      --window-seconds 60 --overlap 0.25 --baseline

  seismic-cli eval-sta-lta \\
      --manifest-path dataset_60s/manifest.csv \\
      --window-seconds 60 --overlap 0.25
"""

from typing import List, Optional

import typer

from seismic_cli import (anchor, catalog, eval_baseline, forecast, ram_aux,
                         ram_dual, regression, riskclass, spectrogram)
from seismic_cli.core import (RamImageEncoder, compute_station_noise_baselines,
                              run_balanced_preprocessing)

app = typer.Typer(help="Seismic RAM-image / CNN earthquake detection pipeline.")


@app.command("anchor-windows")
def anchor_windows_cmd(
    source_dir: str = typer.Option(..., help="Directory of already-downloaded long-window mseed files (e.g. 60s)."),
    output_base_dir: str = typer.Option(..., help="Base directory where anchored short-window subfolders get written."),
    target_seconds: List[float] = typer.Option(..., "--target-seconds", "-t",
        help="Target short window length(s) in seconds. Repeatable, e.g. -t 3 -t 6 -t 10."),
    pick_sta_seconds: float = typer.Option(1.0, help="STA window (seconds) used for the coarse arrival pick."),
    pick_lta_seconds: float = typer.Option(10.0, help="LTA window (seconds) used for the coarse arrival pick."),
    trigger_on: float = typer.Option(3.5, help="STA/LTA ratio to declare a trigger."),
    trigger_off: float = typer.Option(1.0, help="STA/LTA ratio to declare a trigger has ended."),
    pre_arrival_fraction: float = typer.Option(0.2, help="Fraction of the target window that sits BEFORE the arrival."),
    limit_files: Optional[int] = typer.Option(None, help="Only process the first N source files -- useful for a quick test run before committing to the full dataset."),
):
    """
    Derives arrival-anchored short windows from already-downloaded longer
    (e.g. 60s) mseed data, using a coarse STA/LTA pick -- no redownload needed.
    Fixes the origin-time-anchoring problem where short windows can miss the
    actual arrival entirely.
    """
    anchor.run_anchor_windows(
        source_dir=source_dir,
        output_base_dir=output_base_dir,
        target_seconds=target_seconds,
        pick_sta_seconds=pick_sta_seconds,
        pick_lta_seconds=pick_lta_seconds,
        trigger_on=trigger_on,
        trigger_off=trigger_off,
        pre_arrival_fraction=pre_arrival_fraction,
        limit_files=limit_files,
    )


@app.command("generate-dataset")
def generate_dataset_cmd(
    eq_dir: str = typer.Option(..., help="Directory of earthquake mseed files."),
    noise_dir: str = typer.Option(..., help="Directory of noise mseed files."),
    output_dir: str = typer.Option(..., help="Where to write the generated dataset (train/val/test + manifest.csv)."),
    window_seconds: float = typer.Option(60.0, help="Window length in seconds."),
    overlap: float = typer.Option(0.5, help="Overlap fraction for sliding windows (noise side only is affected in practice)."),
    target_n: int = typer.Option(64, help="RAM image resolution (n x n)."),
    train_ratio: float = typer.Option(0.70),
    val_ratio: float = typer.Option(0.15),
    test_ratio: float = typer.Option(0.15),
    limit_pictures: Optional[int] = typer.Option(None, help="Cap total dataset size (images across both classes). "
                                                             "Mutually exclusive with --max."),
    generate_max: bool = typer.Option(
        False, "--max",
        help="Generate the maximum possible dataset: every usable station is assigned to a "
             "split (ratios preserved), then the surplus class is trimmed per split via "
             "evenly-spaced window subsampling so classes stay balanced. Station caps and "
             "all other constraints still apply. Mutually exclusive with --limit-pictures."),
    max_windows_per_station: Optional[int] = typer.Option(
        None, help="Cap any single station's contribution -- important for short windows, "
                    "where noise stations can otherwise dominate a split."),
    baseline: bool = typer.Option(
        False, "--baseline/--no-baseline",
        help="Standardize each window against that station's long-term noise baseline "
             "instead of the window's own statistics (falls back to self-standardization "
             "per-channel for any station without enough noise data)."),
    freqmin: float = typer.Option(1.0, help="Bandpass low corner (Hz)."),
    freqmax: float = typer.Option(45.0, help="Bandpass high corner (Hz)."),
    min_baseline_seconds: float = typer.Option(
        60.0, help="Minimum seconds of usable noise data required before trusting a "
                    "station's baseline (only relevant with --baseline)."),
    num_cores: Optional[int] = typer.Option(None, help="Worker processes (default: cpu_count - 1)."),
):
    """
    Generates a station-disjoint, balanced RAM-image dataset (train/val/test)
    from earthquake and noise mseed directories, with a manifest.csv for
    downstream baseline comparisons. Use --baseline to standardize against
    each station's long-term noise statistics instead of per-window
    self-standardization; use --max to generate the largest balanced dataset
    the input data supports.
    """
    if generate_max and limit_pictures:
        raise typer.BadParameter("--max and --limit-pictures are mutually exclusive.")
    run_balanced_preprocessing(
        eq_dir=eq_dir,
        noise_dir=noise_dir,
        output_dir=output_dir,
        split_ratios=(train_ratio, val_ratio, test_ratio),
        target_n=target_n,
        fs=100.0,
        window_seconds=window_seconds,
        overlap=overlap,
        limit_pictures=limit_pictures,
        max_windows_per_station=max_windows_per_station,
        use_baseline_standardization=baseline,
        freqmin=freqmin,
        freqmax=freqmax,
        min_baseline_seconds=min_baseline_seconds,
        num_cores=num_cores,
        generate_max=generate_max,
    )


@app.command("generate-ram-aux-dataset")
def generate_ram_aux_dataset_cmd(
    eq_dir: str = typer.Option(..., help="Directory of earthquake mseed files."),
    noise_dir: str = typer.Option(..., help="Directory of noise mseed files (also used for the "
                                             "station noise baseline log_snr is measured against)."),
    output_dir: str = typer.Option(..., help="Where to write the dataset (train/val/test + manifest.csv)."),
    window_seconds: float = typer.Option(60.0, help="Window length in seconds."),
    overlap: float = typer.Option(0.5, help="Overlap fraction for sliding windows."),
    target_n: int = typer.Option(64, help="RAM image resolution (n x n)."),
    train_ratio: float = typer.Option(0.70),
    val_ratio: float = typer.Option(0.15),
    test_ratio: float = typer.Option(0.15),
    limit_pictures: Optional[int] = typer.Option(None, help="Cap total tensors. Mutually exclusive with --max."),
    generate_max: bool = typer.Option(False, "--max", help="Generate the maximum balanced dataset."),
    max_windows_per_station: Optional[int] = typer.Option(None, help="Per-window cap on any single station."),
    freqmin: float = typer.Option(1.0, help="Bandpass low corner (Hz)."),
    freqmax: float = typer.Option(45.0, help="Bandpass high corner (Hz)."),
    fs: float = typer.Option(100.0, help="Nominal sampling rate."),
    min_baseline_seconds: float = typer.Option(60.0, help="Minimum seconds of usable noise data "
                                                            "required before trusting a station's baseline "
                                                            "for log_snr (falls back to 0.0 otherwise)."),
    per_component_aux: bool = typer.Option(
        False, "--per-component-aux",
        help="Emit 6 per-component aux scalars [log_snr_Z,N,E, log_rms_Z,N,E] instead of "
             "the 2 Z/N/E-averaged scalars. Off by default so existing datasets reproduce "
             "byte-for-byte when regenerated."),
    num_cores: Optional[int] = typer.Option(None, help="Worker processes (default: cpu_count - 1)."),
):
    """
    Generates a station-disjoint, balanced dataset of paired {img, aux}
    tensors: the standard RAM image plus log_snr and log_rms -- the amplitude
    information RAM's exact scale-invariance structurally discards (see
    seismic_cli/ram_aux.py). Train with cnn_earthquake's cnn_ram_aux.py,
    which ablates the aux branch with --no-aux for a direct before/after
    comparison against the plain RAM-only classifier.
    """
    if generate_max and limit_pictures:
        raise typer.BadParameter("--max and --limit-pictures are mutually exclusive.")

    station_baselines, _ = compute_station_noise_baselines(
        noise_dir, fs=fs, freqmin=freqmin, freqmax=freqmax, min_baseline_seconds=min_baseline_seconds,
        num_cores=num_cores,
    )
    if not station_baselines:
        print("[WARN] No station noise baselines built; log_snr will default to 0.0 for every window.")

    encoder_cls = ram_aux.RamAuxEncoderV2 if per_component_aux else ram_aux.RamAuxEncoder
    encoder = encoder_cls(target_n=target_n, station_baselines=station_baselines)
    run_balanced_preprocessing(
        eq_dir=eq_dir, noise_dir=noise_dir, output_dir=output_dir,
        split_ratios=(train_ratio, val_ratio, test_ratio),
        target_n=target_n, fs=fs, window_seconds=window_seconds, overlap=overlap,
        limit_pictures=limit_pictures, max_windows_per_station=max_windows_per_station,
        freqmin=freqmin, freqmax=freqmax, num_cores=num_cores,
        generate_max=generate_max, encoder=encoder,
    )


@app.command("generate-spectrogram-dataset")
def generate_spectrogram_dataset_cmd(
    eq_dir: str = typer.Option(..., help="Directory of earthquake mseed files."),
    noise_dir: str = typer.Option(..., help="Directory of noise mseed files."),
    output_dir: str = typer.Option(..., help="Where to write the dataset (train/val/test + manifest.csv)."),
    window_seconds: float = typer.Option(60.0, help="Window length in seconds."),
    overlap: float = typer.Option(0.5, help="Overlap fraction for sliding windows."),
    n_fft: int = typer.Option(256, help="FFT size; frequency bins = n_fft//2 + 1."),
    hop_length: Optional[int] = typer.Option(None, help="STFT hop. Default n_fft//4."),
    top_db: float = typer.Option(80.0, help="Dynamic-range clamp for the dB conversion."),
    normalize: str = typer.Option(
        "station", help="station = subtract the station's median noise dB profile per frequency bin "
                        "(cancels instrument gain, KEEPS amplitude above noise -- the signal RAM discards); "
                        "per_window = z-score each window (drops absolute amplitude too); "
                        "none = raw dB (leaks instrument gain)."),
    train_ratio: float = typer.Option(0.70),
    val_ratio: float = typer.Option(0.15),
    test_ratio: float = typer.Option(0.15),
    limit_pictures: Optional[int] = typer.Option(None, help="Cap total tensors. Mutually exclusive with --max."),
    generate_max: bool = typer.Option(False, "--max", help="Generate the maximum balanced dataset."),
    max_windows_per_station: Optional[int] = typer.Option(None, help="Per-window cap on any single station."),
    freqmin: float = typer.Option(1.0, help="Bandpass low corner (Hz)."),
    freqmax: float = typer.Option(45.0, help="Bandpass high corner (Hz)."),
    fs: float = typer.Option(100.0, help="Nominal sampling rate; every window is resampled to this "
                                          "so all tensors share one shape."),
    num_cores: Optional[int] = typer.Option(None, help="Worker processes (default: cpu_count - 1)."),
):
    """
    Generates a station-disjoint, balanced dataset of 3-channel log-power
    spectrogram tensors (.pt), with the same guarantees as generate-dataset:
    unified station splits, per-window caps, gap rejection, per-station
    sampling rates, and a manifest for exact-window STA/LTA comparison.
    """
    if generate_max and limit_pictures:
        raise typer.BadParameter("--max and --limit-pictures are mutually exclusive.")
    if normalize not in spectrogram.NORMALIZE_MODES:
        raise typer.BadParameter(f"--normalize must be one of {spectrogram.NORMALIZE_MODES}")

    profiles = {}
    if normalize == "station":
        profiles = spectrogram.compute_station_spectral_baselines(
            noise_dir, n_fft=n_fft, hop_length=hop_length, top_db=top_db,
            nominal_fs=fs, freqmin=freqmin, freqmax=freqmax,
        )
        if not profiles:
            print("[WARN] No spectral baselines built; every window falls back to per-window "
                  "normalization (instrument gain is removed, absolute amplitude is not preserved).")

    encoder = spectrogram.SpectrogramEncoder(
        n_fft=n_fft, hop_length=hop_length, top_db=top_db, nominal_fs=fs,
        window_seconds=window_seconds, normalize=normalize, noise_profiles=profiles,
    )
    run_balanced_preprocessing(
        eq_dir=eq_dir, noise_dir=noise_dir, output_dir=output_dir,
        split_ratios=(train_ratio, val_ratio, test_ratio),
        fs=fs, window_seconds=window_seconds, overlap=overlap,
        limit_pictures=limit_pictures, max_windows_per_station=max_windows_per_station,
        freqmin=freqmin, freqmax=freqmax, num_cores=num_cores,
        generate_max=generate_max, encoder=encoder,
    )


@app.command("generate-spec-dual-dataset")
def generate_spec_dual_dataset_cmd(
    eq_dir: str = typer.Option(..., help="Directory of earthquake mseed files."),
    noise_dir: str = typer.Option(..., help="Directory of noise mseed files."),
    output_dir: str = typer.Option(..., help="Where to write the dataset (train/val/test + manifest.csv)."),
    window_seconds: float = typer.Option(60.0, help="Window length in seconds."),
    overlap: float = typer.Option(0.5, help="Overlap fraction for sliding windows."),
    n_fft: int = typer.Option(256, help="FFT size; frequency bins = n_fft//2 + 1."),
    hop_length: Optional[int] = typer.Option(None, help="STFT hop. Default n_fft//4."),
    top_db: float = typer.Option(80.0, help="Dynamic-range clamp for the dB conversion."),
    normalize: str = typer.Option(
        "station", help="Spectrogram (2D channel) normalization; see generate-spectrogram-dataset."),
    train_ratio: float = typer.Option(0.70),
    val_ratio: float = typer.Option(0.15),
    test_ratio: float = typer.Option(0.15),
    limit_pictures: Optional[int] = typer.Option(None, help="Cap total tensors. Mutually exclusive with --max."),
    generate_max: bool = typer.Option(False, "--max", help="Generate the maximum balanced dataset."),
    max_windows_per_station: Optional[int] = typer.Option(None, help="Per-window cap on any single station."),
    baseline: bool = typer.Option(
        False, "--baseline/--no-baseline",
        help="Standardize the 1D (raw-waveform) channel against that station's long-term "
             "noise baseline instead of the window's own statistics. Independent of "
             "--normalize, which controls the spectrogram (2D) channel only."),
    freqmin: float = typer.Option(1.0, help="Bandpass low corner (Hz)."),
    freqmax: float = typer.Option(45.0, help="Bandpass high corner (Hz)."),
    fs: float = typer.Option(100.0, help="Nominal sampling rate; every window is resampled to this "
                                          "so all tensors share one shape. Also sets the 1D branch's "
                                          "sequence length (fs * window_seconds) -- self-attention "
                                          "there is O(m^2), so long windows at high fs may need a "
                                          "smaller --batch-size when training."),
    min_baseline_seconds: float = typer.Option(60.0, help="Minimum seconds of usable noise data "
                                                            "required before trusting a station's baseline."),
    num_cores: Optional[int] = typer.Option(None, help="Worker processes (default: cpu_count - 1)."),
):
    """
    Generates a station-disjoint, balanced dataset of paired {seq, img}
    tensors for the dual-channel CNN+LSTM (1D2D-EDL) architecture, using a
    log-power SPECTROGRAM as the 2D channel instead of a RAM image -- see
    seismic_cli/spectrogram.py's SpectrogramDualEncoder for why this is worth
    testing (RAM is scale-invariant; spectrograms with --normalize station
    are not). seq (the 1D channel) is unchanged from generate-dual-dataset.
    Train with cnn_earthquake's cnn_lstm_classify.py.
    """
    if generate_max and limit_pictures:
        raise typer.BadParameter("--max and --limit-pictures are mutually exclusive.")
    if normalize not in spectrogram.NORMALIZE_MODES:
        raise typer.BadParameter(f"--normalize must be one of {spectrogram.NORMALIZE_MODES}")

    profiles = {}
    if normalize == "station":
        profiles = spectrogram.compute_station_spectral_baselines(
            noise_dir, n_fft=n_fft, hop_length=hop_length, top_db=top_db,
            nominal_fs=fs, freqmin=freqmin, freqmax=freqmax,
        )
        if not profiles:
            print("[WARN] No spectral baselines built; every window falls back to per-window "
                  "normalization for the 2D channel (instrument gain is removed, absolute "
                  "amplitude is not preserved).")

    spec_encoder = spectrogram.SpectrogramEncoder(
        n_fft=n_fft, hop_length=hop_length, top_db=top_db, nominal_fs=fs,
        window_seconds=window_seconds, normalize=normalize, noise_profiles=profiles,
    )
    encoder = spectrogram.SpectrogramDualEncoder(spec_encoder)
    run_balanced_preprocessing(
        eq_dir=eq_dir, noise_dir=noise_dir, output_dir=output_dir,
        split_ratios=(train_ratio, val_ratio, test_ratio),
        fs=fs, window_seconds=window_seconds, overlap=overlap,
        limit_pictures=limit_pictures, max_windows_per_station=max_windows_per_station,
        use_baseline_standardization=baseline, freqmin=freqmin, freqmax=freqmax,
        min_baseline_seconds=min_baseline_seconds, num_cores=num_cores,
        generate_max=generate_max, encoder=encoder,
    )


@app.command("generate-spec-dual-aux-dataset")
def generate_spec_dual_aux_dataset_cmd(
    eq_dir: str = typer.Option(..., help="Directory of earthquake mseed files."),
    noise_dir: str = typer.Option(..., help="Directory of noise mseed files (also used for the "
                                             "station noise baseline log_snr is measured against)."),
    output_dir: str = typer.Option(..., help="Where to write the dataset (train/val/test + manifest.csv)."),
    window_seconds: float = typer.Option(60.0, help="Window length in seconds."),
    overlap: float = typer.Option(0.5, help="Overlap fraction for sliding windows."),
    n_fft: int = typer.Option(256, help="FFT size; frequency bins = n_fft//2 + 1."),
    hop_length: Optional[int] = typer.Option(None, help="STFT hop. Default n_fft//4."),
    top_db: float = typer.Option(80.0, help="Dynamic-range clamp for the dB conversion."),
    normalize: str = typer.Option(
        "station", help="Spectrogram (2D channel) normalization; see generate-spectrogram-dataset."),
    train_ratio: float = typer.Option(0.70),
    val_ratio: float = typer.Option(0.15),
    test_ratio: float = typer.Option(0.15),
    limit_pictures: Optional[int] = typer.Option(None, help="Cap total tensors. Mutually exclusive with --max."),
    generate_max: bool = typer.Option(False, "--max", help="Generate the maximum balanced dataset."),
    max_windows_per_station: Optional[int] = typer.Option(None, help="Per-window cap on any single station."),
    baseline: bool = typer.Option(
        False, "--baseline/--no-baseline",
        help="Standardize the 1D (raw-waveform) channel against that station's long-term "
             "noise baseline instead of the window's own statistics. Independent of "
             "--normalize (2D channel) and of the aux branch's log_snr, which always "
             "needs a station baseline and is computed regardless of this flag."),
    freqmin: float = typer.Option(1.0, help="Bandpass low corner (Hz)."),
    freqmax: float = typer.Option(45.0, help="Bandpass high corner (Hz)."),
    fs: float = typer.Option(100.0, help="Nominal sampling rate; every window is resampled to this "
                                          "so all tensors share one shape. Also sets the 1D branch's "
                                          "sequence length (fs * window_seconds) -- self-attention "
                                          "there is O(m^2), so long windows at high fs may need a "
                                          "smaller --batch-size when training."),
    min_baseline_seconds: float = typer.Option(60.0, help="Minimum seconds of usable noise data "
                                                            "required before trusting a station's baseline."),
    per_component_aux: bool = typer.Option(
        False, "--per-component-aux",
        help="Emit 6 per-component aux scalars instead of 2 Z/N/E-averaged ones. Off by "
             "default so existing datasets reproduce byte-for-byte when regenerated."),
    num_cores: Optional[int] = typer.Option(None, help="Worker processes (default: cpu_count - 1)."),
):
    """
    generate-spec-dual-dataset plus a log_snr/log_rms aux vector -- tests
    whether the amplitude fix that helped the RAM classifiers (report.md
    10.5.3) also helps a 2D representation (a station-normalized spectrogram)
    that already preserves amplitude information, rather than one (RAM) that
    structurally cannot. See seismic_cli/spectrogram.py's
    SpectrogramDualAuxEncoder. Train with cnn_earthquake's
    cnn_lstm_classify_aux.py.
    """
    if generate_max and limit_pictures:
        raise typer.BadParameter("--max and --limit-pictures are mutually exclusive.")
    if normalize not in spectrogram.NORMALIZE_MODES:
        raise typer.BadParameter(f"--normalize must be one of {spectrogram.NORMALIZE_MODES}")

    profiles = {}
    if normalize == "station":
        profiles = spectrogram.compute_station_spectral_baselines(
            noise_dir, n_fft=n_fft, hop_length=hop_length, top_db=top_db,
            nominal_fs=fs, freqmin=freqmin, freqmax=freqmax,
        )
        if not profiles:
            print("[WARN] No spectral baselines built; every window falls back to per-window "
                  "normalization for the 2D channel (instrument gain is removed, absolute "
                  "amplitude is not preserved).")

    aux_baselines, _ = compute_station_noise_baselines(
        noise_dir, fs=fs, freqmin=freqmin, freqmax=freqmax, min_baseline_seconds=min_baseline_seconds,
        num_cores=num_cores,
    )
    if not aux_baselines:
        print("[WARN] No station noise baselines built; log_snr will default to 0.0 for every window.")

    spec_encoder = spectrogram.SpectrogramEncoder(
        n_fft=n_fft, hop_length=hop_length, top_db=top_db, nominal_fs=fs,
        window_seconds=window_seconds, normalize=normalize, noise_profiles=profiles,
    )
    encoder_cls = (spectrogram.SpectrogramDualAuxEncoderV2 if per_component_aux
                  else spectrogram.SpectrogramDualAuxEncoder)
    encoder = encoder_cls(spec_encoder, aux_baselines=aux_baselines)
    run_balanced_preprocessing(
        eq_dir=eq_dir, noise_dir=noise_dir, output_dir=output_dir,
        split_ratios=(train_ratio, val_ratio, test_ratio),
        fs=fs, window_seconds=window_seconds, overlap=overlap,
        limit_pictures=limit_pictures, max_windows_per_station=max_windows_per_station,
        use_baseline_standardization=baseline, freqmin=freqmin, freqmax=freqmax,
        min_baseline_seconds=min_baseline_seconds, num_cores=num_cores,
        generate_max=generate_max, encoder=encoder,
    )


@app.command("generate-dual-dataset")
def generate_dual_dataset_cmd(
    eq_dir: str = typer.Option(..., help="Directory of earthquake mseed files."),
    noise_dir: str = typer.Option(..., help="Directory of noise mseed files."),
    output_dir: str = typer.Option(..., help="Where to write the dataset (train/val/test + manifest.csv)."),
    window_seconds: float = typer.Option(60.0, help="Window length in seconds."),
    overlap: float = typer.Option(0.5, help="Overlap fraction for sliding windows."),
    target_n: int = typer.Option(64, help="RAM image resolution (n x n)."),
    fs: float = typer.Option(100.0, help="Nominal sampling rate; every window is resampled to this "
                                          "so all seq tensors share one shape. Also sets the 1D "
                                          "branch's sequence length (fs * window_seconds) -- self-"
                                          "attention there is O(m^2), so long windows at high fs "
                                          "may need a smaller --batch-size when training."),
    train_ratio: float = typer.Option(0.70),
    val_ratio: float = typer.Option(0.15),
    test_ratio: float = typer.Option(0.15),
    limit_pictures: Optional[int] = typer.Option(None, help="Cap total tensors. Mutually exclusive with --max."),
    generate_max: bool = typer.Option(False, "--max", help="Generate the maximum balanced dataset."),
    max_windows_per_station: Optional[int] = typer.Option(None, help="Per-window cap on any single station."),
    baseline: bool = typer.Option(
        False, "--baseline/--no-baseline",
        help="Standardize each channel against that station's long-term noise baseline "
             "instead of the window's own statistics, applied identically to both the "
             "seq and img branches (falls back to self-standardization per-channel for "
             "any station without enough noise data)."),
    freqmin: float = typer.Option(1.0, help="Bandpass low corner (Hz)."),
    freqmax: float = typer.Option(45.0, help="Bandpass high corner (Hz)."),
    min_baseline_seconds: float = typer.Option(60.0, help="Minimum seconds of usable noise data "
                                                            "required before trusting a station's baseline."),
    num_cores: Optional[int] = typer.Option(None, help="Worker processes (default: cpu_count - 1)."),
):
    """
    Generates a station-disjoint, balanced dataset of paired {seq, img} tensors
    for the dual-channel CNN+LSTM (1D2D-EDL) architecture from Wang & Zhao
    (2025), applied directly to earthquake/noise classification instead of the
    catalog forecasting task -- seq is the raw standardized (m, 3) Z/N/E
    waveform (the paper's 1D-channel LSTM input) and img is the RAM
    difference image built from the same window (see seismic_cli/ram_dual.py).
    Train with cnn_earthquake's cnn_lstm_classify.py.
    """
    if generate_max and limit_pictures:
        raise typer.BadParameter("--max and --limit-pictures are mutually exclusive.")
    encoder = ram_dual.RamDualEncoder(target_n=target_n, nominal_fs=fs, window_seconds=window_seconds)
    run_balanced_preprocessing(
        eq_dir=eq_dir, noise_dir=noise_dir, output_dir=output_dir,
        split_ratios=(train_ratio, val_ratio, test_ratio),
        target_n=target_n, fs=fs, window_seconds=window_seconds, overlap=overlap,
        limit_pictures=limit_pictures, max_windows_per_station=max_windows_per_station,
        use_baseline_standardization=baseline, freqmin=freqmin, freqmax=freqmax,
        min_baseline_seconds=min_baseline_seconds, num_cores=num_cores,
        generate_max=generate_max, encoder=encoder,
    )


@app.command("generate-dual-aux-dataset")
def generate_dual_aux_dataset_cmd(
    eq_dir: str = typer.Option(..., help="Directory of earthquake mseed files."),
    noise_dir: str = typer.Option(..., help="Directory of noise mseed files (also used for the "
                                             "station noise baseline log_snr is measured against)."),
    output_dir: str = typer.Option(..., help="Where to write the dataset (train/val/test + manifest.csv)."),
    window_seconds: float = typer.Option(60.0, help="Window length in seconds."),
    overlap: float = typer.Option(0.5, help="Overlap fraction for sliding windows."),
    target_n: int = typer.Option(64, help="RAM image resolution (n x n)."),
    fs: float = typer.Option(100.0, help="Nominal sampling rate; every window is resampled to this "
                                          "so all seq tensors share one shape. Also sets the 1D "
                                          "branch's sequence length (fs * window_seconds) -- self-"
                                          "attention there is O(m^2), so long windows at high fs "
                                          "may need a smaller --batch-size when training."),
    train_ratio: float = typer.Option(0.70),
    val_ratio: float = typer.Option(0.15),
    test_ratio: float = typer.Option(0.15),
    limit_pictures: Optional[int] = typer.Option(None, help="Cap total tensors. Mutually exclusive with --max."),
    generate_max: bool = typer.Option(False, "--max", help="Generate the maximum balanced dataset."),
    max_windows_per_station: Optional[int] = typer.Option(None, help="Per-window cap on any single station."),
    baseline: bool = typer.Option(
        False, "--baseline/--no-baseline",
        help="Standardize the seq/img branches against that station's long-term noise "
             "baseline instead of the window's own statistics. Independent of the aux "
             "branch's log_snr, which always needs a station baseline and is computed "
             "regardless of this flag."),
    freqmin: float = typer.Option(1.0, help="Bandpass low corner (Hz)."),
    freqmax: float = typer.Option(45.0, help="Bandpass high corner (Hz)."),
    min_baseline_seconds: float = typer.Option(60.0, help="Minimum seconds of usable noise data "
                                                            "required before trusting a station's baseline."),
    per_component_aux: bool = typer.Option(
        False, "--per-component-aux",
        help="Emit 6 per-component aux scalars instead of 2 Z/N/E-averaged ones. Off by "
             "default so existing datasets reproduce byte-for-byte when regenerated."),
    num_cores: Optional[int] = typer.Option(None, help="Worker processes (default: cpu_count - 1)."),
):
    """
    generate-dual-dataset plus a log_snr/log_rms aux vector -- tests whether
    the amplitude fix that helped the plain RAM classifier (test AUC 0.836 ->
    0.923, see cnn_ram_aux.py) also helps once the RAM image is one branch of
    the dual-channel model rather than the whole classifier. See
    seismic_cli/ram_dual.py's RamDualAuxEncoder. Train with cnn_earthquake's
    cnn_lstm_classify_aux.py.
    """
    if generate_max and limit_pictures:
        raise typer.BadParameter("--max and --limit-pictures are mutually exclusive.")

    aux_baselines, _ = compute_station_noise_baselines(
        noise_dir, fs=fs, freqmin=freqmin, freqmax=freqmax, min_baseline_seconds=min_baseline_seconds,
        num_cores=num_cores,
    )
    if not aux_baselines:
        print("[WARN] No station noise baselines built; log_snr will default to 0.0 for every window.")

    encoder_cls = ram_dual.RamDualAuxEncoderV2 if per_component_aux else ram_dual.RamDualAuxEncoder
    encoder = encoder_cls(target_n=target_n, nominal_fs=fs, window_seconds=window_seconds,
                          aux_baselines=aux_baselines)
    run_balanced_preprocessing(
        eq_dir=eq_dir, noise_dir=noise_dir, output_dir=output_dir,
        split_ratios=(train_ratio, val_ratio, test_ratio),
        target_n=target_n, fs=fs, window_seconds=window_seconds, overlap=overlap,
        limit_pictures=limit_pictures, max_windows_per_station=max_windows_per_station,
        use_baseline_standardization=baseline, freqmin=freqmin, freqmax=freqmax,
        min_baseline_seconds=min_baseline_seconds, num_cores=num_cores,
        generate_max=generate_max, encoder=encoder,
    )


@app.command("generate-regression-dataset")
def generate_regression_dataset_cmd(
    eq_dir: str = typer.Option(..., help="Directory of earthquake mseed files."),
    noise_dir: str = typer.Option(..., help="Noise mseed directory (used for the amplitude reference)."),
    catalog_path: str = typer.Option(..., help="Event catalog CSV with EventID + Magnitude (+ Latitude/Longitude)."),
    output_dir: str = typer.Option(..., help="Where to write tensors + manifest.csv."),
    station_catalog: Optional[str] = typer.Option(
        None, help="Optional station CSV (network/station/latitude/longitude) enabling distance_km."),
    encoding: str = typer.Option("spectrogram", help="spectrogram (.pt tensors) or ram (.png images)."),
    dual: bool = typer.Option(
        False, "--dual", help="Also write a raw-waveform 'seq' channel alongside the 2D image "
                              "('img'), for the dual-channel CNN+LSTM architecture "
                              "(cnn_earthquake's cnn_lstm_regression.py) instead of the "
                              "single-channel CNN (cnn_regression.py/cnn_magclass.py). Forces "
                              "--encoding's output to .pt regardless of choice, since RAM's "
                              "single-channel .png path has no seq slot."),
    window_seconds: float = typer.Option(60.0, help="Window length in seconds."),
    overlap: float = typer.Option(0.5, help="Sliding-window overlap fraction."),
    split_by: str = typer.Option(
        "event", help="event = keep whole events together (default; the magnitude label is per-event, "
                      "so a shared event leaks the target directly). station = station-disjoint instead."),
    n_fft: int = typer.Option(256, help="FFT size (spectrogram encoding)."),
    hop_length: Optional[int] = typer.Option(
        None, help="STFT hop (spectrogram encoding). Default n_fft//4 (=64, giving only ~5 time "
                    "frames from a 3s window); pass a smaller value (e.g. 16) for finer time "
                    "resolution at the same frequency resolution."),
    top_db: float = typer.Option(80.0, help="Dynamic-range clamp (spectrogram encoding)."),
    normalize: str = typer.Option("station", help="Spectrogram normalization; see generate-spectrogram-dataset."),
    target_n: int = typer.Option(64, help="RAM image resolution (ram encoding)."),
    train_ratio: float = typer.Option(0.70),
    val_ratio: float = typer.Option(0.15),
    test_ratio: float = typer.Option(0.15),
    max_windows_per_station: Optional[int] = typer.Option(None, help="Per-window cap on any single station."),
    freqmin: float = typer.Option(1.0, help="Bandpass low corner (Hz)."),
    freqmax: float = typer.Option(45.0, help="Bandpass high corner (Hz)."),
    fs: float = typer.Option(100.0, help="Nominal sampling rate."),
    seed: int = typer.Option(42),
    per_component_aux: bool = typer.Option(
        False, "--per-component-aux",
        help="Emit 3 per-component log_snr scalars (log_snr_0/1/2, one per Z/N/E) instead of "
             "one Z/N/E-averaged log_snr. Off by default so existing datasets reproduce "
             "byte-for-byte when regenerated."),
    num_cores: Optional[int] = typer.Option(None),
):
    """
    Builds a magnitude-labelled dataset from earthquake mseed: encoded windows
    plus, per window, the source magnitude and the two physical predictors it
    depends on (log SNR against the station's noise floor, and epicentral
    distance). Splits keep whole events together by default, since the label
    is per-event.

    --dual reuses the same {seq, img} encoders the detection pipeline already
    has (SpectrogramDualEncoder / RamDualEncoder) -- they implement the exact
    same per-window encoder protocol this orchestrator already calls, so no
    new dataset-generation code is needed, only the choice of encoder. Train
    the --dual output with cnn_earthquake's cnn_lstm_regression.py; the
    single-channel (non-dual) output still trains with
    cnn_regression.py/cnn_magclass.py as before.
    """
    if encoding == "spectrogram":
        profiles = {}
        if normalize == "station":
            profiles = spectrogram.compute_station_spectral_baselines(
                noise_dir, n_fft=n_fft, hop_length=hop_length, top_db=top_db, nominal_fs=fs,
                freqmin=freqmin, freqmax=freqmax,
            )
        spec_encoder = spectrogram.SpectrogramEncoder(
            n_fft=n_fft, hop_length=hop_length, top_db=top_db, nominal_fs=fs, window_seconds=window_seconds,
            normalize=normalize, noise_profiles=profiles,
        )
        encoder = spectrogram.SpectrogramDualEncoder(spec_encoder) if dual else spec_encoder
    elif encoding == "ram":
        encoder = (ram_dual.RamDualEncoder(target_n, nominal_fs=fs, window_seconds=window_seconds)
                  if dual else RamImageEncoder(target_n))
    else:
        raise typer.BadParameter("--encoding must be 'spectrogram' or 'ram'")

    regression.run_regression_preprocessing(
        eq_dir=eq_dir, noise_dir=noise_dir, catalog_path=catalog_path,
        output_dir=output_dir, encoder=encoder, station_catalog=station_catalog,
        split_ratios=(train_ratio, val_ratio, test_ratio), fs=fs,
        window_seconds=window_seconds, overlap=overlap,
        max_windows_per_station=max_windows_per_station, split_by=split_by,
        freqmin=freqmin, freqmax=freqmax, num_cores=num_cores, seed=seed,
        per_component_aux=per_component_aux,
    )


@app.command("generate-riskclass-dataset")
def generate_riskclass_dataset_cmd(
    eq_dir: str = typer.Option(..., help="Directory of earthquake mseed files."),
    noise_dir: str = typer.Option(..., help="Noise mseed directory (its own class here, not just an amplitude reference)."),
    catalog_path: str = typer.Option(..., help="Event catalog CSV with EventID + Magnitude (+ Latitude/Longitude)."),
    output_dir: str = typer.Option(..., help="Where to write tensors + manifest.csv."),
    station_catalog: Optional[str] = typer.Option(
        None, help="Optional station CSV (network/station/latitude/longitude) enabling distance_km."),
    encoding: str = typer.Option("spectrogram", help="spectrogram (.pt tensors) or ram (.png images)."),
    mag_threshold: float = typer.Option(4.0, help="Magnitude >= this is 02_high_risk, else 01_low_risk."),
    balance_ratio: float = typer.Option(4.0, help="Per split, caps 01_low_risk and 00_noise at "
                                        "ratio * count(02_high_risk); None-equivalent via a very large value."),
    min_log_snr: float = typer.Option(-3.0, help="Reject windows whose RMS is below exp(min_log_snr) of "
                                      "their station's own noise floor -- a stuck/dead instrument, not quiet "
                                      "data. Pass a very negative value (e.g. -99) to disable."),
    window_seconds: float = typer.Option(3.0, help="Window length in seconds."),
    overlap: float = typer.Option(0.5, help="Sliding-window overlap fraction."),
    n_fft: int = typer.Option(256, help="FFT size (spectrogram encoding)."),
    top_db: float = typer.Option(80.0, help="Dynamic-range clamp (spectrogram encoding)."),
    normalize: str = typer.Option("station", help="Spectrogram normalization; see generate-spectrogram-dataset."),
    target_n: int = typer.Option(64, help="RAM image resolution (ram encoding)."),
    train_ratio: float = typer.Option(0.70),
    val_ratio: float = typer.Option(0.15),
    test_ratio: float = typer.Option(0.15),
    max_windows_per_station: Optional[int] = typer.Option(None, help="Per-window cap on any single station."),
    freqmin: float = typer.Option(1.0, help="Bandpass low corner (Hz)."),
    freqmax: float = typer.Option(45.0, help="Bandpass high corner (Hz)."),
    fs: float = typer.Option(100.0, help="Nominal sampling rate."),
    seed: int = typer.Option(42),
    num_cores: Optional[int] = typer.Option(None),
):
    """
    Builds a three-class dataset (00_noise / 01_low_risk / 02_high_risk) from
    earthquake + noise mseed: encoded windows plus, per window, the source
    magnitude (NaN for noise) and the two physical predictors log SNR and
    epicentral distance. Station-disjoint splits span all three classes
    together (see riskclass.py's docstring for why this differs from
    generate-regression-dataset's event-disjoint default).
    """
    if encoding == "spectrogram":
        profiles = {}
        if normalize == "station":
            profiles = spectrogram.compute_station_spectral_baselines(
                noise_dir, n_fft=n_fft, top_db=top_db, nominal_fs=fs,
                freqmin=freqmin, freqmax=freqmax,
            )
        encoder = spectrogram.SpectrogramEncoder(
            n_fft=n_fft, top_db=top_db, nominal_fs=fs, window_seconds=window_seconds,
            normalize=normalize, noise_profiles=profiles,
        )
    elif encoding == "ram":
        encoder = RamImageEncoder(target_n)
    else:
        raise typer.BadParameter("--encoding must be 'spectrogram' or 'ram'")

    riskclass.run_riskclass_preprocessing(
        eq_dir=eq_dir, noise_dir=noise_dir, catalog_path=catalog_path,
        output_dir=output_dir, encoder=encoder, station_catalog=station_catalog,
        mag_threshold=mag_threshold, balance_ratio=balance_ratio,
        min_log_snr=min_log_snr,
        split_ratios=(train_ratio, val_ratio, test_ratio), fs=fs,
        window_seconds=window_seconds, overlap=overlap,
        max_windows_per_station=max_windows_per_station,
        freqmin=freqmin, freqmax=freqmax, num_cores=num_cores, seed=seed,
    )


@app.command("generate-catalog-dataset")
def generate_catalog_dataset_cmd(
    catalog_path: str = typer.Option(..., help="Earthquake catalog CSV (AFAD/Kandilli export)."),
    output_dir: str = typer.Option(..., help="Where to write window tensors + manifest.csv."),
    window_events: int = typer.Option(64, help="Events per sliding window (fixed length)."),
    stride_events: int = typer.Option(8, help="Events advanced between consecutive windows."),
    major_magnitude: float = typer.Option(6.0, help="Magnitude defining a 'major' earthquake."),
    min_magnitude: float = typer.Option(2.0, help="Drop catalog events below this magnitude."),
    target_n: int = typer.Option(32, help="RAM image resolution for the 2D channel."),
    lat_min: Optional[float] = typer.Option(None), lat_max: Optional[float] = typer.Option(None),
    lon_min: Optional[float] = typer.Option(None), lon_max: Optional[float] = typer.Option(None),
    center_lat: Optional[float] = typer.Option(None), center_lon: Optional[float] = typer.Option(None),
    radius_km: Optional[float] = typer.Option(None, help="Alternative to a bbox."),
    region: List[str] = typer.Option(
        [], "--region", "-r",
        help="Pool an additional fault zone: 'lat_min,lat_max,lon_min,lon_max'. Repeatable. "
             "Windows are built independently per region (so a window never mixes events from "
             "unrelated fault systems) and then pooled, raising the count of distinct target "
             "events beyond what any single zone supports. Overrides --lat-min/etc. when given; "
             "combine with --split-mode loeo to actually make use of the extra targets."),
    split_mode: str = typer.Option(
        "chronological", help="chronological (default; the honest choice for a single time-"
                              "ordered evaluation) | random (leaky; quantifies the leak only) | "
                              "loeo (leave-one-event-out CV -- writes a single flat 'all' split; "
                              "run cnn_lstm_loeo.py on the result. Preferable once pooling "
                              "--region gives you enough targets that a single chronological "
                              "cut is the bottleneck, since every event gets to be the test "
                              "fold once instead of only whichever ones land after the cut)."),
    embargo_days: Optional[float] = typer.Option(
        None, help="Gap between splits. Defaults to the full label horizon, so no training "
                   "window's label can reference an event inside the test period."),
    max_horizon_days: float = typer.Option(3650.0, help="Discard windows whose next major event is further out."),
    class_lo_days: Optional[float] = typer.Option(None, help="Lower risk-class boundary in days. Default: auto from train terciles."),
    class_hi_days: Optional[float] = typer.Option(None, help="Upper risk-class boundary in days. Default: auto from train terciles."),
    train_ratio: float = typer.Option(0.70), val_ratio: float = typer.Option(0.15),
    test_ratio: float = typer.Option(0.15),
    seed: int = typer.Option(42),
    decluster: bool = typer.Option(
        True, "--decluster/--no-decluster",
        help="Restrict prediction TARGETS to independent mainshocks via Gardner-Knopoff "
             "(1974) space-time windows, so one earthquake's own aftershock sequence isn't "
             "counted as multiple targets. Aftershocks stay in the window feature sequence "
             "either way. Disable only to reproduce the un-declustered behaviour."),
):
    """
    Builds sliding-window training data from an earthquake catalog for the
    dual-channel CNN+LSTM risk model: a per-event feature sequence (1D
    channel), a RAM image of it (2D channel), window-level physical scalars
    including b-value and largest Lyapunov exponent, and a time-to-next-major
    label. Splits are chronological with an embargo, because this is a
    forecasting task.
    """
    bbox = None
    if None not in (lat_min, lat_max, lon_min, lon_max):
        bbox = (lat_min, lat_max, lon_min, lon_max)
    center = (center_lat, center_lon) if None not in (center_lat, center_lon) else None

    regions = None
    if region:
        try:
            regions = [tuple(float(x) for x in r.split(",")) for r in region]
        except ValueError:
            raise typer.BadParameter("--region must be 'lat_min,lat_max,lon_min,lon_max'")
        if any(len(r) != 4 for r in regions):
            raise typer.BadParameter("--region must be 'lat_min,lat_max,lon_min,lon_max'")

    if split_mode not in ("chronological", "random", "loeo"):
        raise typer.BadParameter("--split-mode must be chronological, random, or loeo")

    catalog.run_catalog_dataset(
        catalog_path=catalog_path, output_dir=output_dir, window_events=window_events,
        stride_events=stride_events, major_magnitude=major_magnitude,
        min_magnitude=min_magnitude, target_n=target_n, bbox=bbox, center=center,
        radius_km=radius_km, regions=regions, split_mode=split_mode,
        ratios=(train_ratio, val_ratio, test_ratio), embargo_days=embargo_days,
        max_horizon_days=max_horizon_days, seed=seed,
        class_boundaries=((class_lo_days, class_hi_days)
                          if None not in (class_lo_days, class_hi_days) else None),
        decluster=decluster,
    )


@app.command("generate-catalog-forecast-dataset")
def generate_catalog_forecast_dataset_cmd(
    catalog_path: str = typer.Option(..., help="Earthquake catalog CSV (AFAD/Kandilli export)."),
    output_dir: str = typer.Option(..., help="Where to write window tensors + manifest.csv."),
    window_events: int = typer.Option(64, help="Events per sliding window (fixed length)."),
    stride_events: int = typer.Option(8, help="Events advanced between consecutive windows."),
    threshold: float = typer.Option(4.5, help="Magnitude defining a qualifying event."),
    horizon_days: float = typer.Option(30.0, help="Forecast horizon in days."),
    min_magnitude: float = typer.Option(2.0, help="Drop catalog events below this magnitude."),
    target_n: int = typer.Option(32, help="RAM image resolution for the 2D channel."),
    zone: List[str] = typer.Option(
        [], "--zone", "-z",
        help="Restrict to these fault zones (default: all of forecast.FAULT_ZONES: "
             "NAFZ, EAFZ, AEGEAN, CENTRAL). Repeatable."),
    train_ratio: float = typer.Option(0.70), val_ratio: float = typer.Option(0.15),
    test_ratio: float = typer.Option(0.15),
):
    """
    Builds the dual-channel {seq, img, aux} dataset for the DENSE per-zone
    forecasting target ("will M >= threshold occur in this zone within
    horizon_days?") instead of `generate-catalog-dataset`'s abandoned
    time-to-next-major tercile target (measured at chance, kappa -0.028; see
    report.md). This is the target forecast.py validated real signal for in
    2 of 4 zones under logistic regression -- never previously tried against
    the dual-channel network. Train the result with
    cnn_earthquake/src/cnn_lstm_forecast.py.
    """
    if zone:
        unknown = [z for z in zone if z not in forecast.FAULT_ZONES]
        if unknown:
            raise typer.BadParameter(
                f"Unknown zone(s) {unknown}; choices are {list(forecast.FAULT_ZONES)}")
        zones = {z: forecast.FAULT_ZONES[z] for z in zone}
    else:
        zones = forecast.FAULT_ZONES

    catalog.run_catalog_forecast_dataset(
        catalog_path=catalog_path, output_dir=output_dir, zones=zones,
        min_magnitude=min_magnitude, window_events=window_events,
        stride_events=stride_events, threshold=threshold, horizon_days=horizon_days,
        target_n=target_n, ratios=(train_ratio, val_ratio, test_ratio),
    )


@app.command("eval-sta-lta")
def eval_sta_lta_cmd(
    manifest_path: str = typer.Option(..., help="Path to the manifest.csv produced by generate-dataset."),
    split: str = typer.Option("test", help="Which split to evaluate (train/val/test)."),
    window_seconds: float = typer.Option(60.0, help="MUST match what generate-dataset used."),
    overlap: float = typer.Option(0.5, help="MUST match what generate-dataset used."),
    sta_seconds: Optional[float] = typer.Option(
        None, help="STA/LTA's short-term window (seconds). Default: auto-derived from "
                   "--window-seconds (LTA/10, floored at 0.05s) -- 1.0 at 60s, 0.2 at 6s, 0.1 at 3s."),
    lta_seconds: Optional[float] = typer.Option(
        None, help="STA/LTA's long-term window (seconds). Default: auto-derived from "
                   "--window-seconds (window/3, capped at 10s) -- 10.0 at 60s, 2.0 at 6s, 1.0 at 3s. "
                   "The old fixed 10s default couldn't run at all on 3s/6s windows."),
):
    """
    Evaluates the classic STA/LTA algorithm on the EXACT same test windows a
    CNN trained on this dataset would see (same file, station, window index),
    reconstructed directly from the raw mseed -- the fair baseline comparison.
    """
    eval_baseline.run_eval_sta_lta(
        manifest_path=manifest_path,
        split=split,
        fs=100.0,
        window_seconds=window_seconds,
        overlap=overlap,
        sta_seconds=sta_seconds,
        lta_seconds=lta_seconds,
    )




@app.command("generate-groundmotion-dataset")
def generate_groundmotion_dataset_cmd(
    eq_dir: str = typer.Option(..., help="Directory of 60s raw records (NOT the anchored 3s ones)."),
    catalog_path: str = typer.Option(..., help="Event catalog CSV with EventID + Magnitude (+ Latitude/Longitude)."),
    output_dir: str = typer.Option(..., help="Where to write tensors + manifest.csv."),
    cache_dir: str = typer.Option("data/station_inventory", help="StationXML cache; populated once, then offline."),
    station_catalog: Optional[str] = typer.Option(
        None, help="Station CSV (network/station/latitude/longitude) enabling distance_km."),
    target: str = typer.Option("vel", help="Stored input tensor: vel (cm/s, native) or acc (gal)."),
    label_seconds: float = typer.Option(
        25.0, help="Forward label duration after the input window closes. Fixed rather than "
                   "'to end of record' so the target does not depend on record length."),
    train_ratio: float = typer.Option(0.70),
    val_ratio: float = typer.Option(0.15),
    test_ratio: float = typer.Option(0.15),
    limit_files: Optional[int] = typer.Option(None, help="Process only the first N records (smoke test)."),
    seed: int = typer.Option(42),
    num_cores: Optional[int] = typer.Option(None),
):
    """
    Builds the peak-ground-motion dataset for the Nurtas replication:
    response-corrected (3, 300) input windows plus PGA/PGV labels.

    Two targets are emitted per window. `*_fwd` is the peak strictly AFTER the
    input window closes -- a genuine forecast. `*_full` is the peak over the
    whole record, which is the paper's quantity but overlaps the input, so a
    strong result on it is partly self-prediction. Roughly half this corpus has
    its peak at or before the input closes, so the two differ substantially;
    `peak_in_input` records which rows.

    Splits keep whole events together: one earthquake at twenty stations gives
    twenty correlated targets driven by the same source.
    """
    from seismic_cli.groundmotion import run_groundmotion_preprocessing

    run_groundmotion_preprocessing(
        eq_dir=eq_dir, catalog_path=catalog_path, output_dir=output_dir,
        cache_dir=cache_dir, station_catalog=station_catalog,
        split_ratios=(train_ratio, val_ratio, test_ratio), target=target,
        label_seconds=label_seconds, limit_files=limit_files,
        num_cores=num_cores, seed=seed,
    )


if __name__ == "__main__":
    app()
