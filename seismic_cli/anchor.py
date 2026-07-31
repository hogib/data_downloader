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
import scipy.signal
from obspy import Stream, read
from obspy.signal.trigger import classic_sta_lta, trigger_onset


def select_pick_traces(traces: list) -> list:
    """
    Priority-ordered candidates for the STA/LTA arrival pick: vertical (Z)
    first (P-wave onsets are cleanest there), then the remaining channels
    alphabetically, so a station with a dead/flat Z can still pick on a
    horizontal channel instead of being dropped entirely.
    """
    z_traces = [tr for tr in traces if tr.stats.channel[-1].upper() == 'Z']
    other_traces = sorted(
        (tr for tr in traces if tr.stats.channel[-1].upper() != 'Z'),
        key=lambda t: t.stats.channel,
    )
    return z_traces + other_traces


def pick_arrival_with_cft(
    trace_data: np.ndarray,
    fs: float,
    pick_sta_seconds: float,
    pick_lta_seconds: float,
    trigger_on: float,
    trigger_off: float,
) -> Tuple[Optional[int], float]:
    """
    Coarse arrival pick using classic STA/LTA + trigger_onset over the full
    (long) buffer. Returns (arrival_sample, max_cft): arrival_sample is None
    if nothing triggers, and max_cft is the highest STA/LTA ratio reached,
    so the caller can report how close failed stations came to triggering.
    """
    nsta = max(1, int(pick_sta_seconds * fs))
    nlta = max(nsta + 1, int(pick_lta_seconds * fs))

    if len(trace_data) <= nlta:
        return None, 0.0

    try:
        # Raw MiniSEED counts carry large DC offsets and drift; classic
        # STA/LTA works on signal energy, so a big offset pins the ratio
        # near 1 and nothing ever crosses trigger_on. Detrend a copy for
        # picking only -- the sliced output windows stay raw, since the
        # downstream pipeline does its own cleaning/filtering.
        x = scipy.signal.detrend(np.asarray(trace_data, dtype=np.float64), type='linear')
        cft = classic_sta_lta(x, nsta, nlta)
        onsets = trigger_onset(cft, trigger_on, trigger_off)
    except Exception:
        return None, 0.0

    max_cft = float(np.max(cft)) if len(cft) else 0.0

    if len(onsets) == 0:
        return None, max_cft

    return int(onsets[0][0]), max_cft


def pick_arrival_sample(
    trace_data: np.ndarray,
    fs: float,
    pick_sta_seconds: float,
    pick_lta_seconds: float,
    trigger_on: float,
    trigger_off: float,
) -> Optional[int]:
    """Backward-compatible wrapper around pick_arrival_with_cft."""
    return pick_arrival_with_cft(
        trace_data, fs, pick_sta_seconds, pick_lta_seconds, trigger_on, trigger_off,
    )[0]


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
    stats: Optional[dict] = None,
) -> List[Tuple[str, Path, Stream]]:
    def bump(key: str, amount: int = 1):
        if stats is not None:
            stats[key] = stats.get(key, 0) + amount

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
        bump('stations_seen')
        if len(traces) < 3:
            bump('stations_lt3_channels')
            continue

        fs = traces[0].stats.sampling_rate
        arrival_sample = None
        picked_component = None
        best_cft = 0.0
        for pick_trace in select_pick_traces(traces):
            arrival_sample, max_cft = pick_arrival_with_cft(
                pick_trace.data.astype(np.float64), fs,
                pick_sta_seconds, pick_lta_seconds, trigger_on, trigger_off,
            )
            best_cft = max(best_cft, max_cft)
            if arrival_sample is not None:
                picked_component = pick_trace.stats.channel[-1].upper()
                break

        if arrival_sample is None:
            bump('stations_no_pick')
            if stats is not None:
                stats.setdefault('failed_best_cfts', []).append(best_cft)
            continue

        bump('picked_on_z' if picked_component == 'Z' else 'picked_on_fallback')

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
    stats: dict = {}

    for i, src_path in enumerate(source_files, 1):
        if i % 200 == 0:
            print(f"  ...{i}/{len(source_files)} files processed")

        outputs = process_one_event_file(
            src_path, output_base, target_windows,
            pick_sta_seconds, pick_lta_seconds, trigger_on, trigger_off, pre_arrival_fraction,
            stats=stats,
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

    print("\n[PICK DIAGNOSTICS]")
    print(f"  stations seen:             {stats.get('stations_seen', 0)}")
    print(f"  skipped (<3 channels):     {stats.get('stations_lt3_channels', 0)}")
    print(f"  picked on Z:               {stats.get('picked_on_z', 0)}")
    print(f"  picked on fallback chan:   {stats.get('picked_on_fallback', 0)}")
    print(f"  no pick on any channel:    {stats.get('stations_no_pick', 0)}")
    failed_cfts = stats.get('failed_best_cfts', [])
    if failed_cfts:
        print(f"  failed stations' best STA/LTA ratio: "
              f"median {np.median(failed_cfts):.2f}, max {np.max(failed_cfts):.2f} "
              f"(trigger_on={trigger_on})")
        print(f"  -> if these cluster just below {trigger_on}, lower trigger_on; "
              f"if they sit near 1.0, the pick data is effectively flat.")
