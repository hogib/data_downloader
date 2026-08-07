"""
Response-corrected peak ground motion labels (PGA / PGV) per event-station.

Built for a replication of Nurtas et al. (ACDSA 2025), which forecasts peak
ground acceleration from the first 3 s of three-component waveform and reports
validation MAE 2.61 gal / R2 0.714 for a CNN-BiLSTM+attention model. Their input
tensor is (300, 3) -- the same shape as our existing `window_post_3s_anchored`
windows -- so the input side needs nothing new. The LABEL is what is missing,
and it cannot be taken from raw counts.

**Why raw counts are not enough.** Counts are proportional to ground motion only
within one instrument. Sensitivities differ station to station (KOERI's HH*
channels run ~2.5e9 counts/(m/s)), so a model trained on raw peaks would partly
learn which station recorded the event. Removing the instrument response
converts to physical units and makes the target comparable across the network.

**Our sensors are the wrong class for a like-for-like replication, and this is
stated rather than papered over.** 100 % of our channels are HH* -- high-gain
broadband VELOCITY seismometers. K-NET, the paper's source, is a strong-motion
ACCELEROMETER network. Getting acceleration therefore requires differentiating
velocity, which amplifies high-frequency noise exactly where broadband data is
weakest. So `pgv_cms` is computed as the physically native quantity (and is a
standard early-warning intensity measure in its own right), and `pga_gal`
alongside it for numerical comparability with the paper. Expect PGV to be the
better-behaved target.

**Three failure modes are recorded as explicit columns rather than silently
absorbed**, because on this project the characteristic defect is a plausible
wrong number, not a crash:

  * `n_masked_frac` -- fraction of samples that were gaps filled by
    interpolation. Measured directly: two M2.3 events produced apparent peaks of
    10,350 gal and 7,176 gal from raw sample values of exactly 2**31, which are
    fill sentinels. `core._masked_to_filled` handles them; a naive
    `read().merge().data` does not.
  * `clipped` -- any raw sample within 1 % of the 24-bit digitizer rail. A M7.7
    record measured 42.96 gal against a M6.2's 71.06 gal, which is inverted and
    consistent with saturation -- a known failure mode of high-gain sensors in
    strong motion.
  * `response_ok` -- whether a real StationXML response was applied.

Downstream code can then exclude suspect records by an explicit, label-
independent rule instead of discovering the problem as a strange R2.

**The target window matches the paper**: the peak is taken over t0 + 3 s to the
end of the record, i.e. "peak AFTER the first 3 seconds", which is the quantity
a 3-second input could legitimately forecast. Note this is NOT the record's true
PGA -- for near-source stations the real peak can fall inside the excluded first
3 s, and for those the label understates the hazard.
"""

import math
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from seismic_cli.core import _masked_to_filled, select_components

# 24-bit digitizer full scale. Samples within CLIP_TOL of the rail are treated
# as saturated.
DIGITIZER_RAIL = 2 ** 23
CLIP_TOL = 0.01

M_S2_TO_GAL = 100.0     # 1 m/s^2 = 100 gal
M_S_TO_CMS = 100.0      # 1 m/s   = 100 cm/s


def inventory_path(cache_dir: Path, network: str, station: str) -> Path:
    return Path(cache_dir) / f"{network}.{station}.xml"


def get_inventory(network: str, station: str, cache_dir: Path,
                  client_name: str = "KOERI", timeout: int = 60):
    """
    StationXML for one station, cached on disk.

    The ~150-180 station fetch is a one-time cost; every later run is offline.
    A station that genuinely has no response is cached as a failure marker too,
    so a permanently-missing station is not re-requested on every run.
    """
    from obspy import read_inventory

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = inventory_path(cache_dir, network, station)
    miss = path.with_suffix(".missing")

    if path.exists():
        try:
            return read_inventory(str(path))
        except Exception:
            path.unlink(missing_ok=True)
    if miss.exists():
        return None

    try:
        from obspy.clients.fdsn import Client
        inv = Client(client_name, timeout=timeout).get_stations(
            network=network, station=station, level="response")
        inv.write(str(path), format="STATIONXML")
        return inv
    except Exception as e:
        miss.write_text(f"{type(e).__name__}: {e}\n")
        return None


def _peak_after(x: np.ndarray, fs: float, skip_seconds: float) -> float:
    """Peak of a 1-D series after the first `skip_seconds`."""
    i0 = int(round(skip_seconds * fs))
    if i0 >= len(x):
        return float("nan")
    return float(np.max(np.abs(x[i0:])))


def _vector_peak_after(comps, fs: float, skip_seconds: float) -> float:
    """
    Peak of the vector magnitude sqrt(N^2 + E^2 + Z^2) after `skip_seconds`.

    Vector magnitude rather than the max of per-component peaks: the components
    peak at slightly different instants, so taking their individual maxima
    overestimates the true ground motion vector.
    """
    n = min(len(c) for c in comps)
    i0 = int(round(skip_seconds * fs))
    if i0 >= n:
        return float("nan")
    stack = np.column_stack([c[:n] for c in comps])[i0:]
    return float(np.max(np.sqrt(np.sum(stack ** 2, axis=1))))


def ground_motion_for_station(traces: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
                              network: str, station: str, cache_dir: Path,
                              skip_seconds: float = 3.0,
                              starttime=None,
                              pre_filt=(0.05, 0.1, 40.0, 45.0)) -> Optional[dict]:
    """
    PGA/PGV for one (event, station), plus the quality flags.

    `traces` maps component letter -> (data, gap_mask, sampling_rate), the same
    structure `regression._process_regression_file` builds.

    `starttime` is REQUIRED for a correct response removal, not cosmetic:
    StationXML responses carry validity epochs, so a trace stamped with a
    placeholder time falls outside every epoch and `remove_response` rejects
    the station. Passing UTCDateTime(0) here silently zeroed the entire dataset
    on the first run -- every record came back response_ok=False.

    Returns None only when the station cannot be assembled at all (missing
    components, mismatched sampling rates). A record with a bad response or
    heavy gaps is RETURNED with flags set, so the caller decides -- dropping it
    here would hide the failure rate.
    """
    from obspy import Stream, Trace, UTCDateTime

    if starttime is None:
        raise ValueError("starttime is required: responses are epoch-bounded")
    sel = select_components(traces.keys())
    if sel is None:
        return None
    rates = {traces[c][2] for c in sel}
    if len(rates) != 1:
        return None
    fs = rates.pop()
    if fs <= 0:
        return None

    raw = [traces[c][0] for c in sel]
    masks = [traces[c][1] for c in sel]
    n = min(len(c) for c in raw)
    if n < int(fs * (skip_seconds + 1.0)):
        return None
    raw = [c[:n] for c in raw]
    masks = [m[:n] for m in masks]

    masked_frac = float(np.mean(np.column_stack(masks)))
    peak_counts = float(max(np.max(np.abs(c)) for c in raw))
    clipped = bool(peak_counts >= DIGITIZER_RAIL * (1.0 - CLIP_TOL))

    inv = get_inventory(network, station, cache_dir)
    out = dict(network=network, station=station, sampling_rate=float(fs),
               n_samples=int(n), n_masked_frac=masked_frac,
               peak_counts=peak_counts, clipped=clipped,
               response_ok=inv is not None,
               pga_gal=float("nan"), pgv_cms=float("nan"))
    if inv is None:
        return out

    def _corrected(output):
        st = Stream()
        for comp, data in zip(sel, raw):
            tr = Trace(data=np.asarray(data, dtype=np.float64))
            tr.stats.network, tr.stats.station = network, station
            tr.stats.channel = f"HH{comp}"
            tr.stats.sampling_rate = fs
            tr.stats.starttime = UTCDateTime(starttime)
            st += tr
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st.remove_response(inventory=inv, output=output, pre_filt=pre_filt,
                               water_level=60, taper=True, taper_fraction=0.05)
        return [t.data for t in st]

    try:
        vel = _corrected("VEL")
        acc = _corrected("ACC")
    except Exception:
        out["response_ok"] = False
        return out

    out["pgv_cms"] = _vector_peak_after(vel, fs, skip_seconds) * M_S_TO_CMS
    out["pga_gal"] = _vector_peak_after(acc, fs, skip_seconds) * M_S2_TO_GAL
    return out


def extract_event_file(file_path: Path, cache_dir: Path,
                       skip_seconds: float = 3.0) -> list:
    """
    All station labels from one `event_<EventID>_raw.mseed`.

    Mirrors `regression._process_regression_file`'s trace-grouping so the two
    stay consistent about component selection and gap handling.
    """
    from obspy import read

    from seismic_cli.regression import parse_event_id

    try:
        st = read(str(file_path))
        try:
            st.merge(method=1)
        except Exception:
            pass
    except Exception:
        return []

    event_id = parse_event_id(file_path.stem)
    if event_id is None:
        return []

    stations: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, float]]] = {}
    starts: Dict[str, object] = {}
    for tr in st:
        key = f"{tr.stats.network}.{tr.stats.station}"
        chan = tr.stats.channel[-1].upper()
        existing = stations.setdefault(key, {}).get(chan)
        if existing is not None and len(existing[0]) >= tr.stats.npts:
            continue
        data, gap_mask = _masked_to_filled(tr.data)
        stations[key][chan] = (data, gap_mask, tr.stats.sampling_rate)
        starts[key] = tr.stats.starttime

    rows = []
    for key, chans in stations.items():
        net, sta = key.split(".", 1)
        r = ground_motion_for_station(chans, net, sta, cache_dir, skip_seconds,
                                      starttime=starts.get(key))
        if r is None:
            continue
        r["event_id"] = event_id
        r["station_key"] = key
        r["log_pga"] = math.log10(r["pga_gal"]) if r["pga_gal"] > 0 else float("nan")
        r["log_pgv"] = math.log10(r["pgv_cms"]) if r["pgv_cms"] > 0 else float("nan")
        rows.append(r)
    return rows
