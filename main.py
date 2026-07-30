import re
from pathlib import Path

# Import the core function from your existing script (assuming it's named preprocess.py)
from src.process import run_balanced_preprocessing


def main():
    base_waveforms_dir = Path("data/batched_waveforms")
    noise_batch_dir = "data/batched_noise_waveforms"

    target_directories = [
        "window_post_3s_anchored",
        "window_post_6s_anchored",
        # "window_post_100s",
        # "window_post_120s",
        # "window_post_200s",
        # "window_pre_100s",
        # "window_pre_200s"
    ]

    for dir_name in target_directories:
        # (?=_|$) lets this match "window_post_6s" (end of string) AND
        # "window_post_6s_anchored" (followed by an underscore) -- the old
        # r'(\d+)s$' required digits+s to be the very last thing in the
        # string, which silently skipped every *_anchored directory.
        match = re.search(r'(\d+)s(?=_|$)', dir_name)
        if not match:
            print(f"[WARN] Could not parse window seconds from {dir_name}. Skipping...")
            continue

        # Parse the nominal window size from the folder name
        nominal_window_secs = float(match.group(1))

        # Apply a 0.1 second buffer to account for trace clipping off-by-one errors.
        # Note: window_array() already has a built-in 5% length tolerance, so this
        # is a bit belt-and-suspenders -- harmless, just somewhat redundant.
        actual_window_secs = nominal_window_secs - 0.1
        eq_dir = str(base_waveforms_dir / dir_name)
        output_dir = f"dataset_{dir_name}"

        print(f"\n{'#'*60}")
        print(f"RUNNING PIPELINE FOR: {dir_name}")
        print(f"Targeting {actual_window_secs}s windows (Nominal {nominal_window_secs}s)")
        print(f"{'#'*60}\n")

        if not Path(eq_dir).exists():
            print(f"[ERROR] Directory {eq_dir} does not exist. Skipping...")
            continue

        # Short windows (with overlap) let a single long noise trace produce
        # thousands of overlapping slices -- easily more than an entire
        # split's target on its own, which collapses the noise side down to
        # 1-2 stations (see: window_post_3s_anchored producing a test set
        # drawn from a single noise station). Cap harder for shorter windows;
        # 60s+ windows don't produce nearly enough volume per station for
        # this to be an issue, so leave those uncapped.
        station_cap = 20 if nominal_window_secs <= 10 else None

        run_balanced_preprocessing(
            eq_dir=eq_dir,
            noise_dir=noise_batch_dir,
            output_dir=output_dir,
            split_ratios=(0.70, 0.15, 0.15),
            target_n=100,
            fs=100.0,
            window_seconds=actual_window_secs,
            overlap=0.25,
            max_windows_per_station=station_cap,
        )


if __name__ == "__main__":
    main()
