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

from seismic_cli import anchor, eval_baseline
from seismic_cli.core import run_balanced_preprocessing

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
    limit_pictures: Optional[int] = typer.Option(None, help="Cap total dataset size (images across both classes)."),
    max_windows_per_station: Optional[int] = typer.Option(
        None, help="Cap any single station's contribution -- important for short windows, "
                    "where a single long noise trace can produce thousands of overlapping "
                    "slices and single-handedly dominate a split. If not set, this defaults "
                    "AUTOMATICALLY to 20 for window_seconds <= 10 (matching what's needed to "
                    "avoid station collapse at short windows), and no cap for longer windows. "
                    "Pass 0 to explicitly force NO cap even at short windows."),
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
    self-standardization.
    """
    if max_windows_per_station is None:
        resolved_cap = 20 if window_seconds <= 10 else None
        if resolved_cap is not None:
            print(f"[INFO] --max-windows-per-station not set; auto-applying a cap of "
                  f"{resolved_cap} (window_seconds <= 10) to prevent single-station "
                  f"domination. Pass --max-windows-per-station 0 to disable this.")
    elif max_windows_per_station == 0:
        resolved_cap = None
        print("[INFO] Station cap explicitly disabled (--max-windows-per-station 0).")
    else:
        resolved_cap = max_windows_per_station

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
        max_windows_per_station=resolved_cap,
        use_baseline_standardization=baseline,
        freqmin=freqmin,
        freqmax=freqmax,
        min_baseline_seconds=min_baseline_seconds,
        num_cores=num_cores,
    )


@app.command("eval-sta-lta")
def eval_sta_lta_cmd(
    manifest_path: str = typer.Option(..., help="Path to the manifest.csv produced by generate-dataset."),
    split: str = typer.Option("test", help="Which split to evaluate (train/val/test)."),
    window_seconds: float = typer.Option(60.0, help="MUST match what generate-dataset used."),
    overlap: float = typer.Option(0.5, help="MUST match what generate-dataset used."),
    sta_seconds: float = typer.Option(1.0, help="STA/LTA's short-term window (seconds)."),
    lta_seconds: float = typer.Option(10.0, help="STA/LTA's long-term window (seconds) -- must be comfortably shorter than window_seconds."),
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


if __name__ == "__main__":
    app()
