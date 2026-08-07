"""
Regional earthquake forecasting: will a M >= threshold event occur in this
fault zone within the next N days?

**Why this replaces the time-to-next-mainshock formulation.** `catalog.py`
labels each window with the time until the next independent mainshock, binned
into terciles. Measured under 264-event leave-one-event-out CV, a gradient
boosted model over its nine seismicity indicators scores 31.49 % accuracy
against a 33.33 % chance floor, with kappa -0.028 -- at chance, with a
near-uniform confusion matrix (see `cnn_earthquake/catalog_report.md`).

There is a structural reason to expect that. Gardner-Knopoff declustering
removes aftershocks *for target selection*, and aftershock sequences are the
most predictable part of seismicity (Omori decay). What remains is mainshock
timing, which on this catalog is close to memoryless -- measured coefficient
of variation of inter-mainshock gaps is 0.67-1.17 across regions, and CV = 1
is exactly Poisson. For a memoryless process P(wait | history) = P(wait), so
no model can beat chance. The old target was close to unlearnable by
construction.

This module changes the target rather than the model, which is what
`catalog.report_major_events`'s own remediation advice suggests ("switch
target definition: 'max magnitude in the next N days' is a dense regression
problem rather than a rare-event one"). The reformulated target is dense --
every window has one, no declustering is applied to it, and clustered
seismicity now *helps* instead of being defined away.

**Two fixes to the feature set, both measured.**

  * `n_events` was in the old aux vector and is *constant* -- it equals
    `window_events` by construction. One of nine features was dead weight.
    Dropped.
  * `days_since_prev_major` was absent, despite being the single most
    informative quantity for a renewal process and the first seismicity
    indicator in the comparable published work. Measured Spearman correlation
    with the old target: +0.129 (p = 2.6e-32), on par with the best feature
    that *was* included. Added.

Rate- and energy-acceleration features are added on the same reasoning: a
burst of activity is the physical precursor signal this task is supposed to
detect, and the old feature set only measured the window's average rate.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from seismic_cli.catalog import b_value_aki, energy_joules, load_catalog

# Committed so they are never lost again: the previous pooled dataset's four
# regions were never recorded anywhere and could not be reconstructed.
# Seismotectonic provinces of Turkey, chosen to be individually well populated
# (each holds >= 16k M>=2 events and >= 54 M>=4.5 events on this catalog) and
# to cover 92.8 % of it.
FAULT_ZONES: Dict[str, Tuple[float, float, float, float]] = {
    # name: (lat_min, lat_max, lon_min, lon_max)
    "NAFZ":    (39.5, 42.0, 26.0, 42.0),   # North Anatolian Fault Zone
    "EAFZ":    (36.5, 39.5, 35.0, 42.0),   # East Anatolian Fault Zone
    "AEGEAN":  (36.0, 40.0, 25.0, 30.0),   # Aegean / western Anatolian extension
    "CENTRAL": (34.0, 37.5, 28.0, 36.0),   # Central Anatolia / Cyprus arc
}

FEATURES: List[str] = [
    "log_duration_days",     # how long the 64 events took -- burst vs. quiet
    "log_rate",              # events per day over the whole window
    "log_rate_recent",       # events per day over the window's last quarter
    "rate_accel",            # log_rate_recent - log_rate; activity accelerating?
    "mean_mag",
    "max_mag",
    "mag_std",
    "log_total_energy",
    "log_energy_recent_frac",  # share of window energy released in the last quarter
    "b_value",
    "days_since_prev_major",   # renewal-process feature the old set lacked
]


def _spearman_safe(a, b):
    from scipy.stats import spearmanr
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def build_region_windows(
    df: pd.DataFrame,
    region: str,
    bbox: Tuple[float, float, float, float],
    window_events: int = 64,
    stride_events: int = 8,
    threshold: float = 4.5,
    horizon_days: float = 30.0,
) -> pd.DataFrame:
    """
    Slides a fixed-event-count window through one region's catalog and labels
    each window with whether a M >= `threshold` event follows within
    `horizon_days` of the window's last event.

    Fixed event count (rather than fixed duration) keeps the window's
    statistical basis constant; its DURATION becomes a feature, and is
    informative in its own right -- 64 events in 3 days is a very different
    state from 64 over 3 years.

    Windows whose horizon extends past the end of the catalog are dropped: we
    cannot know whether they were positive, and keeping them would silently
    label "no record" as "no earthquake".
    """
    la0, la1, lo0, lo1 = bbox
    s = df[df.lat.between(la0, la1) & df.lon.between(lo0, lo1)].sort_values("time")
    s = s.reset_index(drop=True)
    if len(s) < window_events * 2:
        return pd.DataFrame()

    t = s.time.to_numpy()
    mag = s.magnitude.to_numpy(dtype=np.float64)
    energy = energy_joules(mag)
    major_times = t[mag >= threshold]
    catalog_end = t[-1]
    horizon = np.timedelta64(int(horizon_days), "D")

    rows = []
    for start in range(0, len(s) - window_events + 1, stride_events):
        sl = slice(start, start + window_events)
        wt, wm, we = t[sl], mag[sl], energy[sl]
        end = wt[-1]

        # Unknowable label: the horizon runs past the catalog.
        if end + horizon > catalog_end:
            continue

        dur = max(float((wt[-1] - wt[0]) / np.timedelta64(1, "D")), 1e-3)
        q = max(window_events // 4, 2)
        dur_recent = max(float((wt[-1] - wt[-q]) / np.timedelta64(1, "D")), 1e-3)

        prev = major_times[major_times < end]
        days_since = (float((end - prev[-1]) / np.timedelta64(1, "D"))
                      if len(prev) else np.nan)

        e_tot = max(float(np.sum(we)), 1e-12)
        e_recent = max(float(np.sum(we[-q:])), 1e-12)

        log_rate = math.log10(window_events / dur)
        log_rate_recent = math.log10(q / dur_recent)

        fut = major_times[(major_times > end) & (major_times <= end + horizon)]
        rows.append({
            "region": region,
            "end_time": pd.Timestamp(end),
            "start_time": pd.Timestamp(wt[0]),
            "log_duration_days": math.log10(dur),
            "log_rate": log_rate,
            "log_rate_recent": log_rate_recent,
            "rate_accel": log_rate_recent - log_rate,
            "mean_mag": float(np.mean(wm)),
            "max_mag": float(np.max(wm)),
            "mag_std": float(np.std(wm)),
            "log_total_energy": math.log10(e_tot),
            "log_energy_recent_frac": math.log10(e_recent / e_tot),
            "b_value": b_value_aki(wm),
            "days_since_prev_major": days_since,
            "label": int(len(fut) > 0),
        })

    return pd.DataFrame(rows)


def build_dataset(
    catalog_path: str,
    zones: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    min_magnitude: float = 2.0,
    window_events: int = 64,
    stride_events: int = 8,
    threshold: float = 4.5,
    horizon_days: float = 30.0,
) -> pd.DataFrame:
    zones = zones or FAULT_ZONES
    df = load_catalog(catalog_path, min_magnitude=min_magnitude)
    parts = []
    print(f"\n[build] target: M >= {threshold} within {horizon_days:.0f} days, "
          f"{window_events}-event windows, stride {stride_events}")
    for name, bbox in zones.items():
        w = build_region_windows(df, name, bbox, window_events, stride_events,
                                 threshold, horizon_days)
        if w.empty:
            print(f"  {name:9s} SKIPPED (too few events)")
            continue
        print(f"  {name:9s} windows={len(w):6d}  positive rate {w.label.mean():.3f}  "
              f"{w.end_time.min().date()} -> {w.end_time.max().date()}")
        parts.append(w)
    out = pd.concat(parts, ignore_index=True).sort_values("end_time").reset_index(drop=True)
    print(f"  {'TOTAL':9s} windows={len(out):6d}  positive rate {out.label.mean():.3f}")
    return out


def chronological_split(d: pd.DataFrame, horizon_days: float = 30.0,
                        ratios: Sequence[float] = (0.70, 0.15, 0.15)):
    """
    Time-ordered split with a horizon embargo.

    The label looks FORWARD `horizon_days`, so a window sitting just before a
    boundary is labelled by events on the far side of it. Dropping one horizon
    of windows at each boundary is exactly enough to close that, and unlike a
    blanket embargo it costs only a few days of data.
    """
    d = d.sort_values("end_time").reset_index(drop=True)
    t0, t1 = d.end_time.iloc[0], d.end_time.iloc[-1]
    span = (t1 - t0).total_seconds() / 86400.0
    cut_tr = t0 + pd.Timedelta(days=span * ratios[0])
    cut_va = t0 + pd.Timedelta(days=span * (ratios[0] + ratios[1]))
    emb = pd.Timedelta(days=horizon_days)

    train = d[d.end_time <= cut_tr - emb]
    val = d[(d.end_time > cut_tr) & (d.end_time <= cut_va - emb)]
    test = d[d.end_time > cut_va]
    dropped = len(d) - len(train) - len(val) - len(test)
    print(f"\n[split] chronological with a {horizon_days:.0f}-day horizon embargo")
    print(f"        train <= {cut_tr.date()} | val <= {cut_va.date()} | test > {cut_va.date()}")
    for nm, part in (("train", train), ("val", val), ("test", test)):
        print(f"        {nm:5s} n={len(part):6d}  positive rate {part.label.mean():.3f}")
    print(f"        {dropped} windows dropped inside embargo bands")
    return train.copy(), val.copy(), test.copy()


def zone_major_times(catalog_path: str, zone: str, threshold: float = 4.5) -> np.ndarray:
    """Origin times of qualifying events inside one zone's bbox, sorted."""
    la0, la1, lo0, lo1 = FAULT_ZONES[zone]
    cat = load_catalog(catalog_path, min_magnitude=threshold)
    sel = cat[cat.lat.between(la0, la1) & cat.lon.between(lo0, lo1)]
    return np.sort(sel.time.to_numpy().astype("datetime64[ns]"))


def build_blocks(d: pd.DataFrame, zone: str, horizon_days: float,
                 catalog_end, major_times) -> pd.DataFrame:
    """
    Disjoint consecutive `horizon_days` blocks for one zone: one forecast and
    one outcome each.

    A block [t, t+H) is positive iff a qualifying event's ORIGIN TIME falls
    inside it. Its forecast comes from the last window ending STRICTLY BEFORE t
    -- the information a forecaster would actually hold when the block opens.

    Outcomes come from `major_times` (the catalog) rather than from window
    labels. Inheriting labels would be wrong: a window ending at
    `block_start + 25d` carries a horizon reaching `block_start + 55d`, so
    aggregating window labels marks a block positive for events happening up to
    a full horizon AFTER it closes. That inflated every base rate by ~30 % in a
    first version, and since the base rate is the reference for Brier skill and
    information gain it moved the goalposts rather than the scores.

    Shared by `cnn_earthquake/src/forecast_blocks.py` (evaluation) and
    `forecast_now.py` (operational), so the two cannot drift apart.
    """
    g = d[d.region == zone].sort_values("end_time").reset_index(drop=True)
    if g.empty:
        return pd.DataFrame()

    H = pd.Timedelta(days=horizon_days)
    edges, t = [], g.end_time.min()
    while t + H <= min(g.end_time.max(), catalog_end):
        edges.append(t)
        t = t + H

    ends = g.end_time.to_numpy()
    mt = np.sort(np.asarray(major_times, dtype="datetime64[ns]"))
    rows = []
    for lo in edges:
        hi = lo + H
        prior = np.searchsorted(ends, np.datetime64(lo), side="left") - 1
        if prior < 0:
            continue
        i0 = np.searchsorted(mt, np.datetime64(lo), side="left")
        i1 = np.searchsorted(mt, np.datetime64(hi), side="left")
        rows.append({"region": zone, "block_start": lo, "block_end": hi,
                     "fc_index": int(prior), "fc_time": g.end_time.iloc[prior],
                     "label": int(i1 > i0), "n_major": int(i1 - i0)})
    return pd.DataFrame(rows)
