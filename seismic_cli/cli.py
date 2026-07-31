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


if __name__ == "__main__":
    app()
