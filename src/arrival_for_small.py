"""
Fixes the short-window (3s/6s/10s) mislabeling problem WITHOUT redownloading
anything. The problem: window_post_6s etc. were sliced starting at event
ORIGIN time, but for stations near the edge of the search radius, the P-wave
arrival can land after the whole 6s window ends -- so a meaningful fraction
of "earthquake" short windows never actually contained the earthquake.

The fix: your window_post_60s (or longer) raw files, downloaded for the SAME
events, already contain the real arrival with plenty of pre-arrival buffer.
This script re-derives properly arrival-anchored short windows directly from
that existing data -- a coarse STA/LTA pick over the full 60s buffer locates
the approximate arrival sample (reliable here, since there's room for a real
LTA baseline), then slices a short window around it. No network calls.

Requires: obspy, numpy
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from obspy import Stream, read
from obspy.signal.trigger import classic_sta_lta, trigger_onset

SOURCE_DIR = Path("data/batched_waveforms/window_post_60s")  # already-downloaded, reliable data
SOURCE_WINDOW_SECONDS = 60.0

TARGET_WINDOWS = [
    ("window_post_3s_anchored", 3.0),
    ("window_post_6s_anchored", 6.0),
    ("window_post_10s_anchored", 10.0),
]
OUTPUT_BASE_DIR = Path("data/batched_waveforms")

FS_EXPECTED = 100.0
PICK_STA_SECONDS = 1.0
PICK_LTA_SECONDS = 10.0  # plenty of room in a 60s buffer, unlike at the short window lengths themselves
TRIGGER_ON = 3.5   # STA/LTA ratio to declare a trigger
TRIGGER_OFF = 1.0
PRE_ARRIVAL_FRACTION = 0.2  # arrival lands 20% into the sliced short window


def pick_arrival_sample(trace_data: np.ndarray, fs: float) -> Optional[int]:
    """
    Coarse arrival pick using classic STA/LTA + trigger_onset over the full
    (long) buffer. Returns the sample index of the first trigger onset, or
    None if nothing triggers (event too weak/noisy to pick reliably here --
    better to skip than to anchor on a bad guess).
    """
    nsta = max(1, int(PICK_STA_SECONDS * fs))
    nlta = max(nsta + 1, int(PICK_LTA_SECONDS * fs))

    if len(trace_data) <= nlta:
        return None

    try:
        cft = classic_sta_lta(trace_data, nsta, nlta)
        onsets = trigger_onset(cft, TRIGGER_ON, TRIGGER_OFF)
    except Exception:
        return None

    if len(onsets) == 0:
        return None

    return int(onsets[0][0])  # first trigger's onset sample


def slice_anchored_window(
    trace_data: np.ndarray,
    arrival_sample: int,
    fs: float,
    window_seconds: float,
    pre_arrival_fraction: float,
) -> Optional[np.ndarray]:
    target_samples = int(fs * window_seconds)
    n_samples = len(trace_data)

    start_idx = int(arrival_sample - pre_arrival_fraction * target_samples)

    if start_idx < 0:
        return None  # not enough pre-arrival buffer even in the long source file
    if start_idx + target_samples > n_samples:
        start_idx = n_samples - target_samples
        if start_idx < 0:
            return None

    return trace_data[start_idx:start_idx + target_samples]


def process_one_event_file(source_path: Path) -> List[Tuple[str, Path, Stream]]:
    """
    Returns a list of (target_dir_name, output_file_path, sliced_stream)
    for every target window size this event/station combination could be
    successfully anchored for.
    """
    try:
        st = read(str(source_path))
        st.merge(method=1, fill_value='interpolate')
    except Exception as e:
        print(f"[WARN] Failed to read {source_path.name}: {e}")
        return []

    outputs = []

    stations = {}
    for tr in st:
        sta_key = f"{tr.stats.network}.{tr.stats.station}"
        stations.setdefault(sta_key, []).append(tr)

    for sta_key, traces in stations.items():
        if len(traces) < 3:
            continue

        fs = traces[0].stats.sampling_rate
        # Use the vertical-ish/first-sorted-channel trace for picking -- P
        # arrivals are typically clearest on vertical, and any reasonably
        # energetic arrival will show up across all three anyway.
        pick_trace = sorted(traces, key=lambda t: t.stats.channel)[0]
        arrival_sample = pick_arrival_sample(pick_trace.data.astype(np.float64), fs)

        if arrival_sample is None:
            continue  # couldn't pick reliably -- skip rather than guess

        for target_dir_name, window_seconds in TARGET_WINDOWS:
            sliced_traces = []
            ok = True
            for tr in traces:
                sliced_data = slice_anchored_window(
                    tr.data.astype(np.float64), arrival_sample, fs,
                    window_seconds, PRE_ARRIVAL_FRACTION,
                )
                if sliced_data is None:
                    ok = False
                    break
                new_tr = tr.copy()
                # Downcast to float32 (seismic amplitude data doesn't need
                # float64 precision, and this halves file size) and clear the
                # stale mseed encoding metadata inherited from the original
                # int-encoded (STEIM1/STEIM2) source trace -- otherwise
                # obspy tries to write float data under integer encoding
                # metadata, which is either silently wrong or gets rejected.
                # Deleting it lets obspy pick the correct encoding for the
                # actual dtype at write time.
                new_tr.data = sliced_data.astype(np.float32)
                if hasattr(new_tr.stats, 'mseed'):
                    del new_tr.stats.mseed
                sliced_traces.append(new_tr)

            if not ok:
                continue

            out_dir = OUTPUT_BASE_DIR / target_dir_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / source_path.name  # same event filename, new directory
            outputs.append((target_dir_name, out_path, Stream(sliced_traces)))

    return outputs


def main():
    if not SOURCE_DIR.exists():
        print(f"[ERROR] Source directory not found: {SOURCE_DIR}")
        return

    source_files = list(SOURCE_DIR.rglob("*.mseed"))
    print(f"[info] Found {len(source_files)} source event files in {SOURCE_DIR}")

    counts = {name: 0 for name, _ in TARGET_WINDOWS}
    n_files_processed = 0
    n_files_no_pick = 0

    for i, source_path in enumerate(source_files, 1):
        if i % 200 == 0:
            print(f"  ...{i}/{len(source_files)} files processed")

        outputs = process_one_event_file(source_path)
        if not outputs:
            n_files_no_pick += 1
            continue

        n_files_processed += 1
        # Multiple stations in one file can each produce multiple target
        # windows; merge them per output file (per target dir) before writing.
        by_target: dict = {}
        for target_dir_name, out_path, sliced_stream in outputs:
            by_target.setdefault((target_dir_name, out_path), Stream()).extend(sliced_stream)

        for (target_dir_name, out_path), merged_stream in by_target.items():
            merged_stream.write(str(out_path), format="MSEED", encoding='FLOAT32')
            counts[target_dir_name] += 1

    print(f"\n[DONE] Processed {n_files_processed} event files with at least one successful pick "
          f"({n_files_no_pick} had no station with a reliable arrival pick).")
    for name, count in counts.items():
        print(f"  -> {name}: {count} anchored event files written to {OUTPUT_BASE_DIR / name}")

    print("\nPoint your dataset generator's eq_dir at these new *_anchored directories "
          "instead of the old origin-anchored window_post_3s/6s/10s ones.")


if __name__ == "__main__":
    main()
