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
stated rather than papered over.** Effectively all our channels are HH* --
high-gain broadband VELOCITY seismometers. K-NET, the paper's source, is a
strong-motion ACCELEROMETER network. Getting acceleration therefore requires
differentiating velocity, which amplifies high-frequency noise exactly where
broadband data is weakest. So `pgv_cms` is computed as the physically native
quantity (and is a standard early-warning intensity measure in its own right),
and `pga_gal` alongside it for numerical comparability with the paper. Expect
PGV to be the better-behaved target.

--------------------------------------------------------------------------
Where the label window opens, and why there are two of them
--------------------------------------------------------------------------

The first version of this module took the peak over `[record_start + 3 s, end]`.
That was wrong in a way that would not have crashed: the input window sits at
`[arrival - 0.6 s, arrival + 2.4 s]` and the arrival lands ~10-12 s into the
60 s record, so the label window **entirely contained the input window**. A
model could have read its own target off its own input.

The arrival is not recoverable from the anchored files -- `anchor.py` slices the
data but never shifts `stats.starttime`, so an anchored window carries its
parent record's start time (see defect note below). It IS recoverable by
replaying the same deterministic STA/LTA pick against the 60 s record, which
reproduces the stored anchored windows bit-exactly (verified: max abs diff 0.0).
That is what `arrival_sample_for_station` does, and the pick parameters here
must stay identical to the ones that wrote the corpus.

Given the arrival, two different quantities are worth having, so BOTH are
emitted rather than one being chosen silently:

  * `*_fwd`  -- peak over `[arrival + 2.4 s, +LABEL_SECONDS]`, i.e. strictly
    after everything the model saw. Zero overlap with the input. This is a
    genuine forecast target and is the one to headline.
  * `*_full` -- peak over the whole record. This is the paper's quantity
    ("the PGA of the record"), retained for comparability, but it OVERLAPS the
    input window and a strong result on it is partly self-prediction.

**The measurement that forced this.** Over 576 station-events, the record's peak
falls at a median of `arrival + 2.21 s`, while the input window closes at
`arrival + 2.4 s`. So:

    33.0 %  of records have their peak INSIDE the 3 s input window
    18.1 %  peak BEFORE the input window opens (late picks, triggered on S/coda)
    ~51 %   have their peak at or before the input closes

At a median 44 km and with 86 % of the catalog at M2-3, peak ground motion
simply arrives within ~2 s of the P pick. There is very little "future" left to
forecast, and `*_fwd` is therefore substantially a measurement of coda decay for
half the corpus. That is a property of this data, not a bug, but it is the
single most important caveat on any number produced downstream -- and it is
something the paper never checks, because it never separates the two windows.
`peak_rel_arrival_s` and `peak_in_input` are emitted per row so the split can be
reported rather than assumed.

--------------------------------------------------------------------------
Quality flags -- recorded as columns, never silently absorbed
--------------------------------------------------------------------------

On this project the characteristic defect is a plausible wrong number, not a
crash, so each known failure mode gets an explicit, label-independent column:

  * `n_masked_frac` -- fraction of samples that were gaps filled by
    interpolation. Measured directly: two M2.3 events produced apparent peaks of
    10,350 gal and 7,176 gal from raw sample values of exactly 2**31, which are
    fill sentinels. `core._masked_to_filled` handles them; a naive
    `read().merge().data` does not.
  * `clipped` -- any raw sample within 1 % of the 24-bit digitizer rail. A M7.7
    record measured 42.96 gal against a M6.2's 71.06 gal, which is inverted and
    consistent with saturation -- a known failure mode of high-gain sensors in
    strong motion.
  * `response_ok` -- whether a real StationXML response was applied. Note this
    is TIME-dependent, not station-dependent: 6G channel epochs open 2013-01-01,
    so the same station corrects fine on a later event and fails on an earlier
    one.
  * `sens_mismatch` -- relative disagreement between the response's reported
    overall sensitivity and the product of its stage gains. Across 828 cached
    channel-epochs this is cleanly bimodal: 97.1 % agree to within 0.01 %, and
    2.9 % disagree by a factor of ~690,000. The latter are all 6G stations
    (ATIM, BOZM, BUYM, GBZM, IGDM, KMRM, MADM, YNKM). A station whose stage
    gains disagree with its reported sensitivity by six orders of magnitude
    cannot produce a trustworthy amplitude, so this is surfaced for filtering
    (`SENS_MISMATCH_TOL`) instead of being left to appear as a strange R2.
  * `pick_at_floor` -- the STA/LTA pick returned exactly `nlta - 1`, the first
    computable sample, meaning the arrival is at or before 10.0 s and was not
    localized. 42 % of stations. Verified not to be an algorithm artifact:
    on pure noise the CFT is zero before `nlta` and never triggers.
  * `max_cft` -- the highest STA/LTA ratio reached, so pick marginality stays
    visible downstream.

--------------------------------------------------------------------------
Defect found in existing code (not fixed here, routed around)
--------------------------------------------------------------------------

`anchor.py:178-182` builds each anchored window with `tr.copy()` and replaces
`.data`, but never advances `stats.starttime` to match the slice. Every anchored
window in `window_post_{3,6,10}s_anchored` therefore carries its parent 60 s
record's start time. Nothing downstream noticed because the classifiers never
used absolute time; it matters here because it destroys the arrival time.
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

# These MUST match the values `anchor.py` was run with to write
# `window_post_3s_anchored`, or the re-derived arrival will not correspond to
# the stored input window and every label will be silently misaligned.
PICK_STA_SECONDS = 1.0
PICK_LTA_SECONDS = 10.0
TRIGGER_ON = 3.5
TRIGGER_OFF = 1.0
PRE_ARRIVAL_FRACTION = 0.2
INPUT_SECONDS = 3.0

# Forward label duration, fixed rather than "to end of record" so the target is
# not a function of how long the record happens to be. 25 s keeps 88.2 % of
# station-events and spans the S-P interval out to ~200 km.
LABEL_SECONDS = 25.0

# Above this relative disagreement between reported sensitivity and the product
# of stage gains, the amplitude is not trustworthy. The observed distribution is
# bimodal with nothing between 0.001 and 6.9e5, so the exact cut is not delicate.
SENS_MISMATCH_TOL = 0.05


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


def sensitivity_mismatch(inv) -> float:
    """
    Worst relative gap between reported overall sensitivity and the product of
    stage gains, over every channel epoch in the inventory.

    obspy's `remove_response` warns about this ("computed and reported
    sensitivities differ by more than 5 percent") and then continues, which is
    exactly the silent-wrong-number pattern this project keeps hitting. Turning
    the warning into a number makes it filterable.
    """
    worst = 0.0
    for net in inv:
        for sta in net:
            for ch in sta:
                r = getattr(ch, "response", None)
                if r is None or r.instrument_sensitivity is None:
                    continue
                reported = r.instrument_sensitivity.value
                if not reported:
                    continue
                prod = 1.0
                for stage in r.response_stages:
                    if stage.stage_gain is None:
                        prod = None
                        break
                    prod *= stage.stage_gain
                if prod is None:
                    continue
                worst = max(worst, abs(prod - reported) / abs(reported))
    return float(worst)


def arrival_sample_for_station(traces, fs: float) -> Tuple[Optional[int], float]:
    """
    Replay `anchor.py`'s STA/LTA pick to recover the arrival sample.

    `traces` is the list of obspy Traces for one station. Returns
    (arrival_sample, max_cft); arrival_sample is None when nothing triggers,
    which is the same condition under which `anchor.py` wrote no anchored
    window -- so those stations are absent from the input corpus anyway.
    """
    from seismic_cli.anchor import pick_arrival_with_cft, select_pick_traces

    arrival, best = None, 0.0
    for tr in select_pick_traces(traces):
        arrival, cft = pick_arrival_with_cft(
            np.asarray(tr.data, dtype=np.float64), fs,
            PICK_STA_SECONDS, PICK_LTA_SECONDS, TRIGGER_ON, TRIGGER_OFF)
        best = max(best, cft)
        if arrival is not None:
            return arrival, best
    return None, best


def _vector_magnitude(comps) -> np.ndarray:
    """
    sqrt(N^2 + E^2 + Z^2) sample by sample.

    Vector magnitude rather than the max of per-component peaks: the components
    peak at slightly different instants, so taking their individual maxima
    overestimates the true ground motion vector.
    """
    n = min(len(c) for c in comps)
    stack = np.column_stack([c[:n] for c in comps])
    return np.sqrt(np.sum(stack ** 2, axis=1))


def _peak_in(mag: np.ndarray, i0: int, i1: Optional[int] = None) -> float:
    i0 = max(0, i0)
    i1 = len(mag) if i1 is None else min(i1, len(mag))
    if i0 >= i1:
        return float("nan")
    return float(np.max(mag[i0:i1]))


def ground_motion_for_station(traces: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
                              network: str, station: str, cache_dir: Path,
                              arrival_sample: int,
                              starttime=None,
                              label_seconds: float = LABEL_SECONDS,
                              pre_filt=(0.05, 0.1, 40.0, 45.0)) -> Optional[dict]:
    """
    PGA/PGV for one (event, station), plus the quality flags.

    `traces` maps component letter -> (data, gap_mask, sampling_rate), the same
    structure `regression._process_regression_file` builds.

    `arrival_sample` is the STA/LTA pick from the 60 s record; both label
    windows are defined relative to it, so it must come from
    `arrival_sample_for_station` rather than being assumed.

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
    raw = [c[:n] for c in raw]
    masks = [m[:n] for m in masks]

    # Input window is [arrival - 0.2*3s, arrival + 0.8*3s]; the forward label
    # opens where it closes.
    win = int(round(INPUT_SECONDS * fs))
    in_start = int(arrival_sample - PRE_ARRIVAL_FRACTION * win)
    in_end = in_start + win
    fwd_end = in_end + int(round(label_seconds * fs))
    if in_start < 0 or in_end >= n:
        return None

    masked_frac = float(np.mean(np.column_stack(masks)))
    peak_counts = float(max(np.max(np.abs(c)) for c in raw))
    clipped = bool(peak_counts >= DIGITIZER_RAIL * (1.0 - CLIP_TOL))

    inv = get_inventory(network, station, cache_dir)
    avail = max(0, min(fwd_end, n) - in_end)
    out = dict(network=network, station=station, sampling_rate=float(fs),
               n_samples=int(n), n_masked_frac=masked_frac,
               peak_counts=peak_counts, clipped=clipped,
               response_ok=inv is not None,
               sens_mismatch=float("nan"),
               arrival_s=arrival_sample / fs,
               label_seconds=avail / fs,
               label_truncated=bool(fwd_end > n),
               peak_rel_arrival_s=float("nan"), peak_in_input=False,
               pga_gal_fwd=float("nan"), pgv_cms_fwd=float("nan"),
               pga_gal_full=float("nan"), pgv_cms_full=float("nan"))
    if inv is None:
        return out
    out["sens_mismatch"] = sensitivity_mismatch(inv)

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
        vel_mag = _vector_magnitude(_corrected("VEL"))
        acc_mag = _vector_magnitude(_corrected("ACC"))
    except Exception:
        out["response_ok"] = False
        return out

    out["pgv_cms_fwd"] = _peak_in(vel_mag, in_end, fwd_end) * M_S_TO_CMS
    out["pga_gal_fwd"] = _peak_in(acc_mag, in_end, fwd_end) * M_S2_TO_GAL
    out["pgv_cms_full"] = _peak_in(vel_mag, 0) * M_S_TO_CMS
    out["pga_gal_full"] = _peak_in(acc_mag, 0) * M_S2_TO_GAL

    # Where the record's true peak sits relative to what the model sees. This
    # is the diagnostic that showed half the corpus peaks inside the input.
    peak_idx = int(np.argmax(vel_mag))
    out["peak_rel_arrival_s"] = (peak_idx - arrival_sample) / fs
    out["peak_in_input"] = bool(in_start <= peak_idx < in_end)
    return out


def extract_event_file(file_path: Path, cache_dir: Path,
                       label_seconds: float = LABEL_SECONDS) -> list:
    """
    All station labels from one `event_<EventID>_raw.mseed` (a 60 s record).

    Mirrors `regression._process_regression_file`'s trace-grouping so the two
    stay consistent about component selection and gap handling, and reuses
    `anchor.py`'s pick so the label window lines up with the anchored input
    window the model is actually given.
    """
    from obspy import read

    from seismic_cli.regression import parse_event_id

    try:
        st = read(str(file_path))
        try:
            st.merge(method=1, fill_value="interpolate")
        except Exception:
            pass
    except Exception:
        return []

    event_id = parse_event_id(file_path.stem)
    if event_id is None:
        return []

    by_station: Dict[str, list] = {}
    for tr in st:
        by_station.setdefault(f"{tr.stats.network}.{tr.stats.station}", []).append(tr)

    rows = []
    for key, trs in by_station.items():
        if len(trs) < 3:
            continue
        fs = trs[0].stats.sampling_rate
        arrival, max_cft = arrival_sample_for_station(trs, fs)
        if arrival is None:
            continue          # anchor.py wrote no input window for this station

        chans: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}
        for tr in trs:
            comp = tr.stats.channel[-1].upper()
            existing = chans.get(comp)
            if existing is not None and len(existing[0]) >= tr.stats.npts:
                continue
            data, gap_mask = _masked_to_filled(tr.data)
            chans[comp] = (data, gap_mask, tr.stats.sampling_rate)

        net, sta = key.split(".", 1)
        r = ground_motion_for_station(chans, net, sta, cache_dir, arrival,
                                      starttime=trs[0].stats.starttime,
                                      label_seconds=label_seconds)
        if r is None:
            continue
        r["event_id"] = event_id
        r["station_key"] = key
        r["max_cft"] = max_cft
        r["pick_at_floor"] = bool(arrival == int(PICK_LTA_SECONDS * fs) - 1)
        for src, dst in (("pga_gal_fwd", "log_pga_fwd"), ("pgv_cms_fwd", "log_pgv_fwd"),
                         ("pga_gal_full", "log_pga_full"), ("pgv_cms_full", "log_pgv_full")):
            v = r[src]
            r[dst] = math.log10(v) if v == v and v > 0 else float("nan")
        rows.append(r)
    return rows
