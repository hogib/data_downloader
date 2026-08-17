"""
Catalog-anchored short windows -- the replacement for `arrival_for_small.py` /
`seismic_cli/anchor.py`, which anchor on an STA/LTA pick.

**Why this exists.** The STA/LTA anchor cannot work at these window lengths, for
a reason that is arithmetic rather than seismological. `classic_sta_lta` forces
its characteristic function to exactly 0 for the first `nlta` samples, because
no long-term average exists yet. With the pipeline's `PICK_LTA_SECONDS = 10.0`
at 100 Hz that is the first 1000 samples -- so no trigger can ever be declared
before t = 9.99 s. Measured on 250 sampled 60 s event files: of 290 picks,
**zero** fell before sample 999 and 48.3% fell at exactly sample 999.

Meanwhile the download geometry puts the real arrival much earlier. Events are
requested within `SEARCH_RADIUS_DEG = 0.5` (~55 km), so the P wave reaches the
station a median of ~7 s after origin time and essentially always inside 10 s
(measured: 99.4% below 10 s). The anchor therefore fires *after* the P arrival
in nearly every case, and the "post-P" window it cuts is in practice an
S-wave/coda window.

**What this does instead.** Origin time, hypocentre and depth come from the
event catalog; station coordinates come from a StationXML query cached to CSV.
The P arrival is predicted with TauP (iasp91) from epicentral distance and
depth, and the window is cut around that prediction. No trigger, no threshold,
so **no recording is dropped for being quiet** -- which is the entire point:
the STA/LTA gate silently discarded ~37% of event recordings, exactly the
low-SNR ones a detector should be judged on.

**Accuracy of the prediction.** Validated against an independently re-picked
STA/LTA that uses a short enough LTA to see the arrival (STA 0.2 s / LTA 1.0 s,
warm-up 1.0 s): median residual +0.84 s, MAD 0.63 s, 75.7% within +-2 s. The
positive median is expected -- a trigger lags a true onset, since energy has to
accumulate in the STA window before the ratio crosses. `PRE_ARRIVAL_SECONDS`
is set well above that spread so the arrival stays inside the window even when
the prediction runs early.

Every emitted window is logged to a sidecar CSV with its predicted arrival,
epicentral distance, depth and magnitude, so retention and any later drop can
be audited instead of inferred.

Requires: obspy, numpy, pandas
"""

import argparse
import csv
import multiprocessing
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from obspy import Stream, UTCDateTime, read
from obspy.geodetics import locations2degrees
from obspy.taup import TauPyModel

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = BASE_DIR / "raw/data/batched_waveforms/window_post_60s"
OUTPUT_BASE_DIR = BASE_DIR / "raw/data/batched_waveforms"
CATALOG_FILE = BASE_DIR / "catalogs/extracted_earthquakes.csv"
STATION_COORDS_FILE = BASE_DIR / "catalogs/station_coords.csv"

# Overridden by --window-seconds / --pre-arrival-seconds.
TARGET_WINDOWS = [("window_post_6s_catalog", 6.0)]

# Seconds of pre-arrival buffer inside each window. The predicted arrival has a
# ~0.6 s MAD against independent picks; 2.0 s keeps the onset inside the window
# even on the early tail, and leaves 4.0 s of post-arrival signal at 6 s.
#
# Shortening the window shortens this buffer, and the prediction spread does NOT
# shrink with it. At 3 s with a 1.0 s buffer the onset sits only ~1.6 MAD inside
# the window, so a prediction running early can push it out -- the retention
# figure printed at the end is the check on that, not an assumption.
PRE_ARRIVAL_SECONDS = 2.0

TAUP_MODEL = "iasp91"
PHASES = ["p", "P", "Pg", "Pn"]

# Rounding for the travel-time cache. 0.005 deg is ~0.55 km, which at a crustal
# 6 km/s is ~0.09 s of travel time -- far below the ~0.6 s prediction spread,
# so the cache costs nothing in accuracy and turns ~54k TauP calls into a few
# thousand.
DIST_ROUND = 3
DEPTH_ROUND = 0

EVENT_ID_RE = re.compile(r"event_(\d+)_raw")

_model: Optional[TauPyModel] = None
_tt_cache: Dict[Tuple[float, float], Optional[float]] = {}


def load_event_catalog(path: Path) -> Dict[str, Tuple[UTCDateTime, float, float, float, float]]:
    """EventID -> (origin_time, lat, lon, depth_km, magnitude)."""
    events = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                t = datetime.strptime(row["Date"].strip(), "%d/%m/%Y %H:%M:%S")
                events[row["EventID"].strip()] = (
                    UTCDateTime(t.replace(tzinfo=timezone.utc)),
                    float(row["Latitude"]), float(row["Longitude"]),
                    float(row["Depth"]), float(row["Magnitude"]),
                )
            except (ValueError, KeyError, TypeError):
                continue
    return events


def load_station_coords(path: Path) -> Dict[str, Tuple[float, float]]:
    """NET.STA -> (lat, lon), also keyed bare so NET mismatches still resolve."""
    coords = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                lat, lon = float(row["latitude"]), float(row["longitude"])
            except (ValueError, KeyError, TypeError):
                continue
            net, sta = row.get("network", "").strip(), row["station"].strip()
            if net:
                coords[f"{net}.{sta}"] = (lat, lon)
            coords.setdefault(sta, (lat, lon))
    return coords


def predicted_p_seconds(depth_km: float, distance_deg: float) -> Optional[float]:
    """
    Travel time of the first-arriving P phase, in seconds after origin time.
    Returns None if TauP finds no arrival for this geometry.
    """
    global _model
    key = (round(depth_km, DEPTH_ROUND), round(distance_deg, DIST_ROUND))
    if key in _tt_cache:
        return _tt_cache[key]
    if _model is None:
        _model = TauPyModel(TAUP_MODEL)
    try:
        arrivals = _model.get_travel_times(
            source_depth_in_km=max(0.0, key[0]),
            distance_in_degree=key[1],
            phase_list=PHASES,
        )
        value = min(a.time for a in arrivals) if arrivals else None
    except Exception:
        value = None
    _tt_cache[key] = value
    return value


def slice_around(trace_data: np.ndarray, arrival_sample: int, fs: float,
                 window_seconds: float) -> Optional[np.ndarray]:
    """
    Cuts `window_seconds` starting `PRE_ARRIVAL_SECONDS` before the arrival.
    Returns None when the buffer cannot supply a full window, rather than
    padding -- a short window would be a silent shape mismatch downstream.
    """
    target = int(round(fs * window_seconds))
    start = int(round(arrival_sample - PRE_ARRIVAL_SECONDS * fs))
    if start < 0 or start + target > len(trace_data):
        return None
    return trace_data[start:start + target]


def process_one_file(args):
    """
    Worker: one source event file -> (written_records, skip_reasons).

    Each record is (target_dir_name, out_path, Stream, metadata_row). Skips are
    counted by reason so the caller can report exactly what was lost and why.
    """
    src_path, events, station_coords = args
    skips = {}

    def skip(reason, n=1):
        skips[reason] = skips.get(reason, 0) + n

    match = EVENT_ID_RE.search(src_path.name)
    if not match:
        skip("filename_unparsed")
        return [], skips
    event_id = match.group(1)
    if event_id not in events:
        skip("event_not_in_catalog")
        return [], skips

    origin, elat, elon, edepth, emag = events[event_id]

    try:
        st = read(str(src_path))
        st.merge(method=1, fill_value="interpolate")
    except Exception:
        skip("unreadable_mseed")
        return [], skips

    by_station: Dict[str, list] = {}
    for tr in st:
        by_station.setdefault(f"{tr.stats.network}.{tr.stats.station}", []).append(tr)

    records = []
    for sta_key, traces in by_station.items():
        if len(traces) < 3:
            skip("fewer_than_3_channels")
            continue
        if sta_key not in station_coords:
            bare = sta_key.split(".")[-1]
            if bare not in station_coords:
                skip("station_coords_missing")
                continue
            slat, slon = station_coords[bare]
        else:
            slat, slon = station_coords[sta_key]

        dist_deg = locations2degrees(elat, elon, slat, slon)
        p_after_origin = predicted_p_seconds(edepth, dist_deg)
        if p_after_origin is None:
            skip("no_taup_arrival")
            continue

        fs = float(traces[0].stats.sampling_rate)
        # Arrival position measured in the trace's own time base, so a trace
        # that does not start exactly at origin time is still handled.
        offset = float(traces[0].stats.starttime - origin)
        arrival_sample = int(round((p_after_origin - offset) * fs))

        for dir_name, window_seconds in TARGET_WINDOWS:
            sliced, ok = [], True
            for tr in traces:
                data = slice_around(tr.data.astype(np.float64), arrival_sample,
                                    float(tr.stats.sampling_rate), window_seconds)
                if data is None:
                    ok = False
                    break
                new_tr = tr.copy()
                new_tr.data = data.astype(np.float32)
                if hasattr(new_tr.stats, "mseed"):
                    del new_tr.stats.mseed
                sliced.append(new_tr)
            if not ok:
                skip("window_outside_buffer")
                continue
            records.append((
                dir_name,
                OUTPUT_BASE_DIR / dir_name / src_path.name,
                Stream(sliced),
                dict(event_id=event_id, station_key=sta_key, magnitude=emag,
                     depth_km=edepth, distance_km=dist_deg * 111.195,
                     predicted_p_s=round(p_after_origin, 3), fs=fs),
            ))
    return records, skips


def main():
    global TARGET_WINDOWS, PRE_ARRIVAL_SECONDS
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window-seconds", type=float, default=6.0)
    ap.add_argument("--pre-arrival-seconds", type=float, default=None,
                    help="Default: one third of --window-seconds, matching the "
                         "6 s / 2.0 s geometry.")
    ap.add_argument("--out-name", default=None,
                    help="Output subdirectory. Default window_post_<N>s_catalog.")
    ap.add_argument("--limit", type=int, default=None, help="Process only N files.")
    a = ap.parse_args()

    ws = a.window_seconds
    PRE_ARRIVAL_SECONDS = a.pre_arrival_seconds if a.pre_arrival_seconds is not None else ws / 3.0
    name = a.out_name or f"window_post_{int(ws) if ws == int(ws) else ws}s_catalog"
    TARGET_WINDOWS = [(name, ws)]
    limit = a.limit

    print("=" * 64)
    print("CATALOG-ANCHORED WINDOW GENERATION (TauP predicted P, no trigger gate)")
    print("=" * 64)

    events = load_event_catalog(CATALOG_FILE)
    coords = load_station_coords(STATION_COORDS_FILE)
    print(f"[load] {len(events)} catalog events | {len(coords)} station coordinate keys")

    files = sorted(SOURCE_DIR.glob("*.mseed"))
    if limit:
        files = files[:limit]
        print(f"[info] limited to first {limit} files")
    print(f"[info] {len(files)} source files in {SOURCE_DIR.name}")
    print(f"[info] window {TARGET_WINDOWS[0][1]}s, {PRE_ARRIVAL_SECONDS}s pre-arrival buffer")

    for dir_name, _ in TARGET_WINDOWS:
        (OUTPUT_BASE_DIR / dir_name).mkdir(parents=True, exist_ok=True)

    n_cores = max(1, multiprocessing.cpu_count() - 4)
    print(f"[info] {n_cores} worker processes\n")

    written, all_skips, meta_rows = 0, {}, []
    grouped: Dict[Path, list] = {}

    with multiprocessing.Pool(n_cores) as pool:
        tasks = ((f, events, coords) for f in files)
        for i, (records, skips) in enumerate(pool.imap_unordered(process_one_file, tasks, chunksize=16), 1):
            for k, v in skips.items():
                all_skips[k] = all_skips.get(k, 0) + v
            for dir_name, out_path, stream, meta in records:
                grouped.setdefault(out_path, []).append((stream, meta))
            if i % 2000 == 0:
                print(f"  ...{i}/{len(files)} files | {len(grouped)} output files staged")

    print(f"\n[write] merging per-event streams and writing {len(grouped)} files...")
    for out_path, entries in grouped.items():
        merged = Stream()
        for stream, meta in entries:
            merged += stream
            meta_rows.append(dict(meta, filename=out_path.name))
        merged.write(str(out_path), format="MSEED")
        written += 1

    meta_path = OUTPUT_BASE_DIR / TARGET_WINDOWS[0][0] / "window_metadata.csv"
    if meta_rows:
        with open(meta_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
            w.writeheader()
            w.writerows(meta_rows)

    print("\n" + "=" * 64)
    print(f"[done] {written} event files written, {len(meta_rows)} station recordings kept")
    print(f"[done] metadata -> {meta_path}")
    if all_skips:
        print("\nSkipped (by reason):")
        for reason, n in sorted(all_skips.items(), key=lambda kv: -kv[1]):
            print(f"  {reason:28s} {n}")
    total_attempted = len(meta_rows) + sum(
        n for r, n in all_skips.items() if r in ("fewer_than_3_channels", "station_coords_missing",
                                                 "no_taup_arrival", "window_outside_buffer"))
    if total_attempted:
        print(f"\nStation-recording retention: {len(meta_rows)}/{total_attempted} "
              f"= {len(meta_rows) / total_attempted * 100:.1f}%   "
              f"(STA/LTA anchoring retained ~62.6%)")
    print("=" * 64)


if __name__ == "__main__":
    main()
