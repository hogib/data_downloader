"""
Fixes the short-window mislabeling problem WITHOUT redownloading anything:
short windows sliced starting at event ORIGIN time can miss the actual
arrival entirely for stations near the edge of the search radius. This
re-derives properly arrival-anchored short windows directly from
already-downloaded longer (e.g. 60s) raw data, using a coarse STA/LTA pick
to locate the approximate arrival -- no network calls.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from obspy import Stream, read
from obspy.signal.trigger import classic_sta_lta, trigger_onset


def pick_arrival_sample(
    trace_data: np.ndarray,
    fs: float,
    pick_sta_seconds: float,
    pick_lta_seconds: float,
    trigger_on: float,
    trigger_off: float,
) -> Optional[int]:
    """
    Coarse arrival pick using classic STA/LTA + trigger_onset over the full
    (long) buffer. Returns None if nothing triggers -- better to skip than
    to anchor on a bad guess.
    """
    nsta = max(1, int(pick_sta_seconds * fs))
    nlta = max(nsta + 1, int(pick_lta_seconds * fs))

    if len(trace_data) <= nlta:
        return None

    try:
        cft = classic_sta_lta(trace_data, nsta, nlta)
        onsets = trigger_onset(cft, trigger_on, trigger_off)
    except Exception:
        return None

    if len(onsets) == 0:
        return None

    return int(onsets[0][0])


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
        return None
    if start_idx + target_samples > n_samples:
        start_idx = n_samples - target_samples
        if start_idx < 0:
            return None

    return trace_data[start_idx:start_idx + target_samples]


def process_one_event_file(
    source_path: Path,
    output_base_dir: Path,
    target_windows: List[Tuple[str, float]],
    pick_sta_seconds: float,
    pick_lta_seconds: float,
    trigger_on: float,
    trigger_off: float,
    pre_arrival_fraction: float,
) -> List[Tuple[str, Path, Stream]]:
    try:
        st = read(str(source_path))
        st.merge(method=1, fill_value='interpolate')
    except Exception as e:
        print(f"[WARN] Failed to read {source_path.name}: {e}")
        return []

    outputs = []

    stations: Dict[str, list] = {}
    for tr in st:
        sta_key = f"{tr.stats.network}.{tr.stats.station}"
        stations.setdefault(sta_key, []).append(tr)

    for sta_key, traces in stations.items():
        if len(traces) < 3:
            continue

        fs = traces[0].stats.sampling_rate
        pick_trace = sorted(traces, key=lambda t: t.stats.channel)[0]
        arrival_sample = pick_arrival_sample(
            pick_trace.data.astype(np.float64), fs,
            pick_sta_seconds, pick_lta_seconds, trigger_on, trigger_off,
        )

        if arrival_sample is None:
            continue

        for target_dir_name, window_seconds in target_windows:
            sliced_traces = []
            ok = True
            for tr in traces:
                sliced_data = slice_anchored_window(
                    tr.data.astype(np.float64), arrival_sample, fs,
                    window_seconds, pre_arrival_fraction,
                )
                if sliced_data is None:
                    ok = False
                    break
                new_tr = tr.copy()
                new_tr.data = sliced_data.astype(np.float32)
                if hasattr(new_tr.stats, 'mseed'):
                    del new_tr.stats.mseed
                sliced_traces.append(new_tr)

            if not ok:
                continue

            out_dir = output_base_dir / target_dir_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / source_path.name
            outputs.append((target_dir_name, out_path, Stream(sliced_traces)))

    return outputs


def run_anchor_windows(
    source_dir: str,
    output_base_dir: str,
    target_seconds: List[float],
    pick_sta_seconds: float = 1.0,
    pick_lta_seconds: float = 10.0,
    trigger_on: float = 3.5,
    trigger_off: float = 1.0,
    pre_arrival_fraction: float = 0.2,
    limit_files: Optional[int] = None,
) -> None:
    source_path = Path(source_dir)
    output_base = Path(output_base_dir)

    if not source_path.exists():
        print(f"[ERROR] Source directory not found: {source_path}")
        return

    target_windows = [
        (f"window_post_{int(s) if s == int(s) else s}s_anchored", float(s))
        for s in target_seconds
    ]

    source_files = list(source_path.rglob("*.mseed"))
    if limit_files is not None:
        source_files = source_files[:limit_files]
        print(f"[info] --limit-files set: processing only the first {limit_files} file(s).")
    print(f"[info] Found {len(source_files)} source event files in {source_path}")
    print(f"[info] Target windows: {[name for name, _ in target_windows]}")

    counts = {name: 0 for name, _ in target_windows}
    n_files_processed = 0
    n_files_no_pick = 0

    for i, src_path in enumerate(source_files, 1):
        if i % 200 == 0:
            print(f"  ...{i}/{len(source_files)} files processed")

        outputs = process_one_event_file(
            src_path, output_base, target_windows,
            pick_sta_seconds, pick_lta_seconds, trigger_on, trigger_off, pre_arrival_fraction,
        )
        if not outputs:
            n_files_no_pick += 1
            continue

        n_files_processed += 1
        by_target: dict = {}
        for target_dir_name, out_path, sliced_stream in outputs:
            by_target.setdefault((target_dir_name, out_path), Stream()).extend(sliced_stream)

        for (target_dir_name, out_path), merged_stream in by_target.items():
            merged_stream.write(str(out_path), format="MSEED", encoding='FLOAT32')
            counts[target_dir_name] += 1

    print(f"\n[DONE] Processed {n_files_processed} event files with at least one successful pick "
          f"({n_files_no_pick} had no station with a reliable arrival pick).")
    for name, count in counts.items():
        print(f"  -> {name}: {count} anchored event files written to {output_base / name}")
