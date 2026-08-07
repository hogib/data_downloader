"""
Catalog-derived sliding-window datasets for earthquake time-to-event modelling.

This is the data side of the dual-channel (CNN + LSTM) model. Unlike the
detection pipeline, which reads continuous waveforms, this module works on an
earthquake CATALOG -- a sequence of discrete events (time, lat, lon, depth,
magnitude) -- and turns it into fixed-length sliding windows carrying:

    seq  (T, F)   per-event feature sequence  -> the 1D (LSTM+attention) channel
    img  (3, n, n) RAM image of three of those series -> the 2D (CNN) channel
    aux  (A,)     window-level scalars       -> absolute scale, b-value, Lyapunov
    label         time until the next major earthquake, as class and as days

**Why `aux` exists.** The RAM transform is exactly scale-invariant
(cnn_earthquake/report.md 8.2): RAM(c*x) == RAM(x) to machine precision. For a
magnitude series the absolute level *is* signal -- energy release rate is the
whole point -- so the image alone would discard it. Window-level scalars carry
it explicitly, the same fix used for magnitude regression.

**Why the split is chronological.** This is a forecasting task, so a random
split is not merely optimistic, it is invalid: overlapping sliding windows
share events outright, and training on windows that postdate the test period
lets the model see the future. Splits are therefore strictly by time, with an
embargo gap so that no training window's LABEL horizon can reach into the test
period. `--split-mode random` exists only to demonstrate how large that
leak is; it is never the honest choice.
"""

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Feature channels making up the 1D sequence, in order.
SEQ_FEATURES = ["magnitude", "log_dt", "depth", "log_energy", "cum_energy_frac", "dist_km"]
# The three series rendered as the RGB RAM image (must be a subset of the above).
IMAGE_FEATURES = ["magnitude", "log_dt", "log_energy"]
AUX_FEATURES = ["n_events", "log_duration_days", "log_rate", "mean_mag", "max_mag",
                "log_total_energy", "b_value", "lyapunov", "mag_std"]
RISK_CLASSES = ["lt_1y", "1_5y", "gt_5y"]


def _fmt_days(d: float) -> str:
    """Compact, honest label for a boundary: days under a year, else years."""
    if d < 365.0:
        return f"{d:.0f}d"
    y = d / 365.0
    return f"{y:.0f}y" if abs(y - round(y)) < 0.05 else f"{y:.1f}y"


def class_names_for(lo: float, hi: float) -> Tuple[str, str, str]:
    """
    Derives class names from the boundaries actually in force.

    `RISK_CLASSES` above is only accurate when the boundaries really are 1 and
    5 years. `assign_risk_classes` derives TERCILES by default, and on a
    catalog whose mainshock recurrence is measured in weeks those terciles are
    nowhere near a year -- on the pooled 4-region dataset they come out at 26 d
    and 71 d, which made every window labelled `gt_5y` actually 71-817 days
    out. Nothing crashed; the numbers were simply reported under names wrong by
    more than an order of magnitude. Names are therefore generated from the
    cut points instead of assumed.
    """
    return (f"lt_{_fmt_days(lo)}", f"{_fmt_days(lo)}_{_fmt_days(hi)}", f"gt_{_fmt_days(hi)}")


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------

def _pick(df: pd.DataFrame, candidates) -> Optional[str]:
    lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def load_catalog(path: str, min_magnitude: Optional[float] = None) -> pd.DataFrame:
    """
    Reads an AFAD / Kandilli style catalog export into a normalized frame with
    columns: time (datetime64), lat, lon, depth, magnitude. Column names are
    auto-detected, and a split Date + Time pair is joined if present.
    """
    df = pd.read_csv(path)
    c_dt = _pick(df, ["datetime", "date_time", "origin_time", "time", "tarih", "olus zamani",
                      "oluş zamanı", "date", "tarih_saat"])
    c_date = _pick(df, ["date", "tarih"])
    c_time = _pick(df, ["time", "saat"])
    c_lat = _pick(df, ["latitude", "lat", "enlem"])
    c_lon = _pick(df, ["longitude", "lon", "long", "boylam"])
    c_dep = _pick(df, ["depth", "derinlik", "depth_km"])
    c_mag = _pick(df, ["magnitude", "mag", "ml", "mw", "buyukluk", "büyüklük"])

    if c_mag is None:
        raise ValueError(f"No magnitude column in {path}; saw {list(df.columns)}")

    if c_date and c_time and c_date != c_time:
        ts = pd.to_datetime(df[c_date].astype(str).str.strip() + " " +
                            df[c_time].astype(str).str.strip(),
                            errors="coerce", dayfirst=True)
    elif c_dt:
        ts = pd.to_datetime(df[c_dt], errors="coerce", dayfirst=True)
    else:
        raise ValueError(f"No usable date/time column in {path}; saw {list(df.columns)}")

    out = pd.DataFrame({
        "time": ts,
        "lat": pd.to_numeric(df[c_lat], errors="coerce") if c_lat else np.nan,
        "lon": pd.to_numeric(df[c_lon], errors="coerce") if c_lon else np.nan,
        "depth": pd.to_numeric(df[c_dep], errors="coerce") if c_dep else np.nan,
        "magnitude": pd.to_numeric(df[c_mag], errors="coerce"),
    })
    n0 = len(out)
    out = out.dropna(subset=["time", "magnitude"]).sort_values("time").reset_index(drop=True)
    if min_magnitude is not None:
        out = out[out.magnitude >= min_magnitude].reset_index(drop=True)
    out["depth"] = out["depth"].fillna(out["depth"].median() if out["depth"].notna().any() else 10.0)

    print(f"[catalog] {path}: {len(out)}/{n0} usable events, "
          f"{out.time.min().date()} to {out.time.max().date()}, "
          f"M {out.magnitude.min():.1f}-{out.magnitude.max():.1f}")
    return out


def filter_region(df: pd.DataFrame, bbox=None, center=None, radius_km=None) -> pd.DataFrame:
    """bbox = (lat_min, lat_max, lon_min, lon_max); or center=(lat,lon) + radius_km."""
    if bbox:
        la0, la1, lo0, lo1 = bbox
        m = df.lat.between(la0, la1) & df.lon.between(lo0, lo1)
        out = df[m].reset_index(drop=True)
        print(f"[region] bbox {bbox}: {len(out)}/{len(df)} events")
        return out
    if center and radius_km:
        d = haversine_km(df.lat.to_numpy(), df.lon.to_numpy(), center[0], center[1])
        out = df[d <= radius_km].reset_index(drop=True)
        print(f"[region] {radius_km} km around {center}: {len(out)}/{len(df)} events")
        return out
    return df


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(np.asarray(lat2) - np.asarray(lat1))
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ---------------------------------------------------------------------------
# Physical / chaos metrics (the quantities IP3 is built around)
# ---------------------------------------------------------------------------

def b_value_aki(mags: np.ndarray, mc: Optional[float] = None) -> float:
    """
    Aki (1965) maximum-likelihood b-value: b = log10(e) / (mean(M) - Mc).
    A falling b-value is a classic precursor claim, so it belongs in the
    feature set rather than being left for the model to rediscover.
    """
    m = np.asarray(mags, dtype=np.float64)
    if len(m) < 10:
        return float("nan")
    mc = float(np.min(m)) if mc is None else mc
    denom = float(np.mean(m) - mc)
    if denom <= 1e-6:
        return float("nan")
    return float(math.log10(math.e) / denom)


def max_lyapunov_rosenstein(x: np.ndarray, emb_dim: int = 4, delay: int = 1,
                            mean_period: int = 3, max_iter: Optional[int] = None) -> float:
    """
    Largest Lyapunov exponent by Rosenstein et al. (1993).

    Rosenstein is used rather than Wolf because it is the standard choice for
    SHORT, noisy series -- which is exactly what a sliding catalog window is.
    Returns NaN when the window is too short to embed, so callers can treat it
    as missing rather than silently receiving a fabricated number.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    m = n - (emb_dim - 1) * delay
    if m < 10:
        return float("nan")
    # Delay embedding
    emb = np.empty((m, emb_dim))
    for i in range(emb_dim):
        emb[:, i] = x[i * delay: i * delay + m]

    # Nearest neighbour excluding temporally close points (Theiler window)
    d2 = ((emb[:, None, :] - emb[None, :, :]) ** 2).sum(-1)
    idx = np.arange(m)
    d2[np.abs(idx[:, None] - idx[None, :]) <= mean_period] = np.inf
    nn = np.argmin(d2, axis=1)
    if not np.isfinite(d2[idx, nn]).any():
        return float("nan")

    steps = max_iter if max_iter is not None else min(m // 4, 20)
    if steps < 3:
        return float("nan")
    div = []
    for k in range(steps):
        ok = (idx + k < m) & (nn + k < m)
        if ok.sum() < 3:
            break
        d = np.linalg.norm(emb[idx[ok] + k] - emb[nn[ok] + k], axis=1)
        d = d[d > 0]
        if len(d) < 3:
            break
        div.append(np.mean(np.log(d)))
    if len(div) < 3:
        return float("nan")
    # Slope of the initial linear growth region = largest Lyapunov exponent
    y = np.asarray(div)
    t = np.arange(len(y))
    return float(np.polyfit(t, y, 1)[0])


def energy_joules(mag: np.ndarray) -> np.ndarray:
    """Gutenberg-Richter energy: log10 E = 1.5 M + 4.8 (E in joules)."""
    return np.power(10.0, 1.5 * np.asarray(mag, dtype=np.float64) + 4.8)


# ---------------------------------------------------------------------------
# Declustering (mainshock/aftershock separation for TARGET selection)
# ---------------------------------------------------------------------------

def gardner_knopoff_windows(mag: float) -> Tuple[float, float]:
    """
    Gardner & Knopoff (1974) space-time windows for a mainshock of this
    magnitude: (L_km, T_days). An event inside another, larger event's window
    is its aftershock, not an independent earthquake.
    """
    l_km = 10.0 ** (0.1238 * mag + 0.983)
    if mag >= 6.5:
        t_days = 10.0 ** (0.032 * mag + 2.7389)
    else:
        t_days = 10.0 ** (0.5409 * mag - 0.547)
    return float(l_km), float(t_days)


def decluster_gardner_knopoff(df: pd.DataFrame) -> np.ndarray:
    """
    Flags independent mainshocks, largest magnitude first: any smaller event
    falling inside a claimed mainshock's Gardner-Knopoff space-time window --
    on EITHER side of it in time -- is marked dependent (aftershock if after,
    foreshock if before) and excluded from future claims.

    Returns a boolean mask over `df` (True = independent mainshock).

    This matters only for TARGET selection -- dependent events stay in the
    catalog as window FEATURES (they are real seismicity). Without it, a
    single mainshock's aftershock sequence masquerades as many independent
    "targets", which both inflates the apparent sample size and collapses the
    label horizon onto Omori-decay timescales (days-weeks) instead of
    tectonic recurrence (years) -- exactly what happened on a real run where
    8 of 31 "targets" turned out to be one M6.2's own aftershocks dated the
    same day. The window is applied symmetrically in time (not just after,
    per the strict 1974 formulation) so that a foreshock hours or days ahead
    of a mainshock -- e.g. an M4.3 the same day as a nearby M4.8 -- is also
    absorbed rather than counted as its own independent target.
    """
    n = len(df)
    is_main = np.ones(n, dtype=bool)
    order = np.argsort(-df.magnitude.to_numpy(dtype=np.float64))  # largest first
    t = df.time.to_numpy()
    lat = df.lat.to_numpy(dtype=np.float64)
    lon = df.lon.to_numpy(dtype=np.float64)
    mag = df.magnitude.to_numpy(dtype=np.float64)

    for i in order:
        if not is_main[i]:
            continue                        # already claimed as someone else's dependent event
        l_km, t_days = gardner_knopoff_windows(mag[i])
        dt_days = (t - t[i]) / np.timedelta64(1, "D")
        d_km = haversine_km(lat, lon, lat[i], lon[i])
        # abs(dt_days) so foreshocks (dt < 0) are claimed too, not just
        # aftershocks (dt > 0); dt == 0 (same-timestamp/same-day ties in a
        # date-only catalog) must also be included or those escape entirely.
        # mag <= mag[i] stops a smaller event -- before OR after the larger
        # one -- from claiming the larger event as ITS dependent; caught on
        # real data where an M3.9 36 minutes before the M6.2 Marmara/Silivri
        # mainshock was demoting the M6.2 itself.
        cand = is_main & (np.abs(dt_days) <= t_days) & (d_km <= l_km) & (mag <= mag[i])
        cand[i] = False                     # never claim the mainshock itself
        is_main[cand] = False

    n_removed = n - int(is_main.sum())
    print(f"[decluster] Gardner-Knopoff: {int(is_main.sum())}/{n} events are independent "
          f"mainshocks ({n_removed} flagged as aftershocks of a larger nearby event).")
    return is_main


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def report_major_events(df: pd.DataFrame, major_magnitude: float,
                        max_horizon_days: float = 3650.0,
                        target_mask: Optional[np.ndarray] = None) -> int:
    """
    Lists the events that will serve as prediction targets, before any windowing.

    This is the single most important number in the whole setup and it used to
    be invisible: the label is "time until the next M >= threshold", so the
    number of such events IS the sample size. A region with one qualifying
    event cannot support train/val/test at all, no matter how many thousands of
    windows the slider produces from it -- every window is labelled by the same
    earthquake. Surfacing this up front turns a confusing empty split into an
    obvious, actionable diagnosis.

    `target_mask`, when given, additionally restricts targets to independent
    mainshocks (see `decluster_gardner_knopoff`) -- otherwise an aftershock
    sequence counts as many "targets" for the same earthquake.
    """
    eligible = df.magnitude >= major_magnitude
    if target_mask is not None:
        eligible = eligible & pd.Series(target_mask, index=df.index)
    majors = df[eligible].sort_values("time")
    n = len(majors)
    print(f"\n[targets] {n} event(s) with M >= {major_magnitude} in this region "
          f"-- these are the prediction targets, and their count is the real sample size.")
    if n:
        for _, r in majors.iterrows():
            print(f"            {pd.Timestamp(r.time).date()}  M{r.magnitude:.1f}  "
                  f"({r.lat:.2f}, {r.lon:.2f})")
        gaps = majors.time.diff().dt.days.dropna()
        if len(gaps):
            print(f"          inter-event gaps: median {gaps.median():.0f} d, "
                  f"min {gaps.min():.0f} d, max {gaps.max():.0f} d")
            if gaps.max() > max_horizon_days:
                print(f"          [note] the largest gap exceeds --max-horizon-days "
                      f"({max_horizon_days:.0f} d), so windows in that stretch are discarded.")

    if n < 4:
        print(f"\n  [!] {n} target event(s) is not enough to build a usable dataset.")
        print("      A chronological split needs targets in EVERY period, and each split")
        print("      needs several distinct events before its metrics mean anything.")
        print("      Expect empty train/val splits below. Options, most effective first:")
        print(f"        * lower --major-magnitude (M>={major_magnitude} is rare; M>=4.5 or 5.0")
        print("          is far denser and still a meaningful target)")
        print("        * widen the region, or drop the bbox entirely to pool fault zones")
        print("        * extend the catalog further back in time (more years = more targets)")
        print("        * switch target definition: 'max magnitude in the next N days' is a")
        print("          dense regression problem rather than a rare-event one")
    return n


def build_windows(df: pd.DataFrame, window_events: int, stride_events: int,
                  major_magnitude: float, max_horizon_days: float = 3650.0,
                  target_mask: Optional[np.ndarray] = None, region_label: str = "default"
                  ) -> List[dict]:
    """
    Slides a fixed-EVENT-COUNT window along the catalog.

    Fixed event count (rather than fixed duration) keeps the sequence length
    constant, which both the LSTM and the RAM reshape need; the window's
    duration becomes a feature instead, and is informative in its own right
    (a burst of N events in 3 days is a very different state from N events
    over 3 years).

    A window is labelled by the time from its LAST event to the next TARGET
    event of magnitude >= major_magnitude (restricted to `target_mask` when
    given, i.e. independent mainshocks only -- see `decluster_gardner_knopoff`).
    Windows containing a raw M >= major_magnitude event (target or not) are
    still dropped: they describe the aftermath, not a precursor state, and
    keeping them would let the model read the answer off its own input. Every
    other event, including aftershocks excluded from `target_mask`, remains in
    the window FEATURE sequence -- that seismicity is real and informative.
    """
    t = df.time.to_numpy()
    mag = df.magnitude.to_numpy(dtype=np.float64)
    lat, lon = df.lat.to_numpy(dtype=np.float64), df.lon.to_numpy(dtype=np.float64)
    depth = df.depth.to_numpy(dtype=np.float64)
    energy = energy_joules(mag)

    target_eligible = mag >= major_magnitude
    if target_mask is not None:
        target_eligible = target_eligible & np.asarray(target_mask, dtype=bool)
    major_idx = np.flatnonzero(target_eligible)
    if len(major_idx) == 0:
        raise ValueError(f"No target events with M >= {major_magnitude} in this region "
                         f"(after declustering, if enabled); lower --major-magnitude, "
                         f"widen the region, or pass --no-decluster.")
    major_times = t[major_idx]

    out = []
    n = len(df)
    for start in range(0, n - window_events + 1, stride_events):
        sl = slice(start, start + window_events)
        wmag = mag[sl]
        if float(np.max(wmag)) >= major_magnitude:
            continue                        # contains the event it would predict
        wt = t[sl]
        end_time = wt[-1]

        nxt = major_times[major_times > end_time]
        if len(nxt) == 0:
            continue                        # no future major event on record
        days = float((nxt[0] - end_time) / np.timedelta64(1, "D"))
        if days > max_horizon_days:
            continue                        # beyond the catalog's reliable horizon

        dt_days = np.diff(wt) / np.timedelta64(1, "D")
        dt_days = np.concatenate([[np.median(dt_days) if len(dt_days) else 1.0], dt_days])
        dt_days = np.clip(dt_days, 1e-4, None)

        wlat, wlon = lat[sl], lon[sl]
        clat = float(np.nanmean(wlat)) if np.isfinite(wlat).any() else 0.0
        clon = float(np.nanmean(wlon)) if np.isfinite(wlon).any() else 0.0
        dist = haversine_km(wlat, wlon, clat, clon)
        dist = np.nan_to_num(dist, nan=0.0)

        we = energy[sl]
        seq = {
            "magnitude": wmag,
            "log_dt": np.log10(dt_days),
            "depth": np.nan_to_num(depth[sl], nan=10.0),
            "log_energy": np.log10(we),
            "cum_energy_frac": np.cumsum(we) / max(float(np.sum(we)), 1e-12),
            "dist_km": dist,
        }
        duration = float((wt[-1] - wt[0]) / np.timedelta64(1, "D"))
        aux = {
            "n_events": float(window_events),
            "log_duration_days": math.log10(max(duration, 1e-3)),
            "log_rate": math.log10(window_events / max(duration, 1e-3)),
            "mean_mag": float(np.mean(wmag)),
            "max_mag": float(np.max(wmag)),
            "log_total_energy": math.log10(max(float(np.sum(we)), 1e-12)),
            "b_value": b_value_aki(wmag),
            "lyapunov": max_lyapunov_rosenstein(wmag),
            "mag_std": float(np.std(wmag)),
        }
        out.append({
            "start_idx": start,
            "end_time": pd.Timestamp(end_time),
            "start_time": pd.Timestamp(wt[0]),
            "target_time": pd.Timestamp(nxt[0]),   # the major event this window is labelled by
            "days_to_major": days,
            "region": region_label,
            "seq": seq,
            "aux": aux,
        })

    print(f"[windows] {len(out)} windows of {window_events} events (stride {stride_events})")
    return out


# ---------------------------------------------------------------------------
# Multi-region pooling
# ---------------------------------------------------------------------------

def pool_regions(df: pd.DataFrame, regions: Sequence[Tuple[float, float, float, float]],
                 window_events: int, stride_events: int, major_magnitude: float,
                 max_horizon_days: float = 3650.0, decluster: bool = True,
                 region_names: Optional[Sequence[str]] = None) -> List[dict]:
    """
    Builds windows independently per region and pools them.

    Windows are built PER REGION, not on the union of events, because a
    sliding window is a local sequence -- pooling raw events first and then
    sliding chronologically across the union would interleave, in one
    window, events from unrelated fault systems on opposite ends of the
    country that only appear adjacent because of a shared timestamp
    ordering. Declustering is also run separately per region for the same
    reason: a Gardner-Knopoff window is only ever tens to a couple hundred
    km wide, so regions far enough apart never interact anyway, but running
    it per region keeps the reported target counts attributable to a place.

    This is the fix for the sample-size ceiling found on a single-region
    (Marmara-only) run: one fault zone has too few independent M>=4 events
    for a chronological split or any real evaluation to be stable. Pooling
    several zones raises the number of distinct target EVENTS (the real
    sample size), not just the number of windows.
    """
    names = list(region_names) if region_names else [f"region{i+1}" for i in range(len(regions))]
    all_windows: List[dict] = []
    print(f"\n[pool] {len(regions)} region(s) requested")
    for name, bbox in zip(names, regions):
        print(f"\n--- {name}: bbox {bbox} ---")
        rdf = filter_region(df, bbox=bbox)
        if len(rdf) < window_events * 2:
            print(f"[SKIP] {name}: only {len(rdf)} events after filtering, need at least "
                  f"{window_events * 2}.")
            continue
        target_mask = decluster_gardner_knopoff(rdf) if decluster else None
        n_targets = report_major_events(rdf, major_magnitude, max_horizon_days, target_mask=target_mask)
        if n_targets == 0:
            print(f"[SKIP] {name}: no target events, nothing to slide windows toward.")
            continue
        rw = build_windows(rdf, window_events, stride_events, major_magnitude,
                           max_horizon_days=max_horizon_days, target_mask=target_mask,
                           region_label=name)
        all_windows.extend(rw)

    print(f"\n[pool] {len(all_windows)} total windows across {len(regions)} region(s), "
          f"{len({(w['region'], w['target_time']) for w in all_windows})} distinct target events")
    return all_windows


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def chronological_split(windows: List[dict], ratios=(0.70, 0.15, 0.15),
                        embargo_days: Optional[float] = None,
                        max_horizon_days: float = 3650.0) -> Dict[str, List[dict]]:
    """
    Time-ordered split with a LABEL-AWARE embargo.

    Two distinct leaks have to be closed:
      1. Overlapping windows share events, so any shuffled split puts nearly
         identical inputs in train and test.
      2. A window's LABEL looks forward to the next major earthquake, so a
         window near a boundary can be labelled by an event lying inside a
         later split -- the future leaking backwards through the target.

    (2) is handled by dropping exactly those windows whose target event falls
    beyond their own split's boundary, rather than by a blanket time gap. A
    fixed embargo wide enough to be safe (the full label horizon) discards
    most of a catalog and can empty a split outright; the label-aware rule
    removes only the windows that actually leak. `embargo_days` remains
    available as an ADDITIONAL hard gap and defaults to none.
    """
    ws = sorted(windows, key=lambda w: w["end_time"])
    times = pd.Series([w["end_time"] for w in ws])
    t0, t1 = times.iloc[0], times.iloc[-1]
    span = (t1 - t0).total_seconds() / 86400.0
    cut_train = t0 + pd.Timedelta(days=span * ratios[0])
    cut_val = t0 + pd.Timedelta(days=span * (ratios[0] + ratios[1]))
    emb = pd.Timedelta(days=embargo_days or 0.0)

    parts = {"train": [], "val": [], "test": [], "_dropped": []}
    n_label_leak = 0
    for w in ws:
        e, tgt = w["end_time"], w["target_time"]
        if e <= cut_train - emb:
            split, boundary = "train", cut_train
        elif cut_train < e <= cut_val - emb:
            split, boundary = "val", cut_val
        elif e > cut_val:
            split, boundary = "test", None
        else:
            parts["_dropped"].append(w)
            continue
        if boundary is not None and tgt > boundary:
            n_label_leak += 1          # labelled by an event in a later split
            parts["_dropped"].append(w)
            continue
        parts[split].append(w)

    print(f"[split] chronological, label-aware embargo"
          + (f" + {embargo_days:.0f} d hard gap" if embargo_days else ""))
    print(f"        train <= {cut_train.date()}  |  val <= {cut_val.date()}  |  "
          f"test > {cut_val.date()}")
    print(f"        {len(parts['train'])} / {len(parts['val'])} / {len(parts['test'])} windows "
          f"({len(parts['_dropped'])} dropped; {n_label_leak} of those labelled by a "
          f"later split's event)")

    # Effective sample size is the number of distinct target events, not windows.
    print("        distinct target (major) events per split -- the real sample size:")
    for s in ("train", "val", "test"):
        n_ev = len({(w["region"], w["target_time"]) for w in parts[s]})
        print(f"          {s:5s}: {n_ev} event(s) across {len(parts[s])} windows")
        if 0 < n_ev < 3:
            print(f"                 [!] {n_ev} distinct event(s) -- every window in this split "
                  f"describes\n                     essentially the same episode. Treat its "
                  f"metrics as anecdote.")
    return parts


def random_split(windows: List[dict], ratios=(0.70, 0.15, 0.15), seed: int = 42):
    """Deliberately leaky; provided only as a contrast to quantify the leak."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(windows))
    n_tr = int(len(windows) * ratios[0])
    n_va = int(len(windows) * ratios[1])
    parts = {"train": [windows[i] for i in idx[:n_tr]],
             "val": [windows[i] for i in idx[n_tr:n_tr + n_va]],
             "test": [windows[i] for i in idx[n_tr + n_va:]],
             "_dropped": []}
    print("[split] RANDOM -- overlapping windows and future labels both leak. "
          "Use only to measure how inflated the leaky number is.")
    return parts


# ---------------------------------------------------------------------------
# Encoding + writing
# ---------------------------------------------------------------------------

def assign_risk_classes(parts: Dict[str, List[dict]],
                        boundaries: Optional[Sequence[float]] = None,
                        boundary_split: str = "train") -> Sequence[float]:
    """
    Turns `days_to_major` into the 3 risk classes.

    The fixed 1-year / 5-year cut points only make sense for the multi-year
    recurrence of M>=6 events. Lower the target threshold -- often necessary,
    since a single region rarely has enough M>=6 events to train on -- and
    recurrence collapses to months, putting every window in `lt_1y` and making
    the task vacuous. `boundaries=None` therefore derives cut points from the
    `boundary_split` split's tercile days (normally "train"; for a flat/LOEO
    dataset there is no train/val/test distinction at write time, so the
    caller passes "all" instead -- documented there as an approximation,
    since strict train-only derivation isn't well-defined until the CV folds
    are formed at training time).
    """
    if boundaries is None:
        days = np.array([w["days_to_major"] for w in parts[boundary_split]], dtype=np.float64)
        if len(days) < 3:
            boundaries = (365.0, 1825.0)
        else:
            boundaries = tuple(np.quantile(days, [1 / 3, 2 / 3]))
        print(f"[classes] boundaries auto-derived from '{boundary_split}' terciles: "
              f"< {boundaries[0]:.0f} d  |  {boundaries[0]:.0f}-{boundaries[1]:.0f} d  |  "
              f"> {boundaries[1]:.0f} d")
    else:
        print(f"[classes] fixed boundaries: < {boundaries[0]:.0f} d  |  "
              f"{boundaries[0]:.0f}-{boundaries[1]:.0f} d  |  > {boundaries[1]:.0f} d")

    lo, hi = float(boundaries[0]), float(boundaries[1])
    names = class_names_for(lo, hi)
    if list(names) != list(RISK_CLASSES):
        print(f"[classes] labels: {names[0]} / {names[1]} / {names[2]}"
              f"   (not the fixed {RISK_CLASSES[0]}/{RISK_CLASSES[1]}/{RISK_CLASSES[2]} names -- "
              f"these boundaries are not at 1 and 5 years)")
    for split in parts:
        if split.startswith("_"):
            continue
        for w in parts[split]:
            d = w["days_to_major"]
            w["risk_class"] = names[0] if d < lo else (
                names[1] if d < hi else names[2])
    return (lo, hi)


def encode_and_write(parts: Dict[str, List[dict]], output_dir: str, target_n: int = 32,
                     seq_features: Sequence[str] = SEQ_FEATURES,
                     image_features: Sequence[str] = IMAGE_FEATURES) -> None:
    import torch

    from seismic_cli.core import ram_matrix, to_uint8

    root = Path(output_dir)
    split_names = [s for s in parts if not s.startswith("_")]
    rows = []
    for split in split_names:
        d = root / split
        d.mkdir(parents=True, exist_ok=True)
        for i, w in enumerate(parts[split]):
            seq = np.stack([w["seq"][f] for f in seq_features], axis=-1).astype(np.float32)

            # 2D channel: one RAM image per chosen series, stacked as RGB --
            # the direct analogue of the Z/N/E stacking in the waveform pipeline.
            chans = []
            for f in image_features:
                R, _ = ram_matrix(w["seq"][f].astype(np.float64), target_n=target_n)
                chans.append(to_uint8(R))
            img = np.stack(chans, axis=0).astype(np.float32) / 255.0

            aux = np.array([w["aux"][k] for k in AUX_FEATURES], dtype=np.float32)

            name = f"win{i:06d}.pt"
            torch.save({"seq": torch.from_numpy(seq),
                        "img": torch.from_numpy(img),
                        "aux": torch.from_numpy(aux)}, d / name)
            rows.append((split, name, w.get("region", "default"), w["target_time"],
                         w["start_time"], w["end_time"],
                         w["days_to_major"], w["risk_class"],
                         *[w["aux"][k] for k in AUX_FEATURES]))

    mpath = root / "manifest.csv"
    with open(mpath, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["split", "filename", "region", "target_time", "start_time", "end_time",
                     "days_to_major", "risk_class", *AUX_FEATURES])
        wr.writerows(rows)

    df = pd.DataFrame(rows, columns=["split", "filename", "region", "target_time",
                                     "start_time", "end_time", "days_to_major", "risk_class",
                                     *AUX_FEATURES])
    print(f"\n[write] {len(df)} windows -> {mpath}")
    print(f"        seq {seq.shape}  img {img.shape}  aux ({len(AUX_FEATURES)},)")
    print("\n  Class balance per split (this is what the baseline must beat):")
    for split in split_names:
        sub = df[df.split == split]
        if sub.empty:
            print(f"     {split:5s}: EMPTY")
            continue
        counts = sub.risk_class.value_counts()
        major = counts.max() / len(sub)
        dist = "  ".join(f"{c}={counts.get(c,0)}" for c in RISK_CLASSES)
        n_ev = sub[["region", "target_time"]].drop_duplicates().shape[0]
        print(f"     {split:5s}: n={len(sub):5d}  {dist}   majority={major:.3f}   "
              f"({n_ev} distinct target events)")
    print("\n  [!] A model must beat the TEST majority-class rate above to mean anything."
          "\n      IP4's 70% target is reachable by predicting one class if that rate is high.")
    n_lyap = int(df.lyapunov.notna().sum())
    print(f"  lyapunov computed for {n_lyap}/{len(df)} windows; "
          f"b_value for {int(df.b_value.notna().sum())}/{len(df)}")


def run_catalog_dataset(catalog_path: str, output_dir: str, window_events: int = 64,
                        stride_events: int = 8, major_magnitude: float = 6.0,
                        min_magnitude: Optional[float] = 2.0, target_n: int = 32,
                        bbox=None, center=None, radius_km=None,
                        regions: Optional[Sequence[Tuple[float, float, float, float]]] = None,
                        region_names: Optional[Sequence[str]] = None,
                        split_mode: str = "chronological", ratios=(0.70, 0.15, 0.15),
                        embargo_days: Optional[float] = None,
                        max_horizon_days: float = 3650.0, seed: int = 42,
                        class_boundaries: Optional[Sequence[float]] = None,
                        decluster: bool = True) -> None:
    """
    `regions`, when given, pools independently-windowed sliding windows from
    several fault zones (see `pool_regions`) instead of a single bbox/center.
    `split_mode="loeo"` skips the chronological split entirely and writes
    every window into a single "all" split; use `cnn_lstm_loeo.py` on the
    result to run leave-one-event-out cross-validation, which is the more
    honest evaluation once there are enough pooled target events that a
    single chronological cut is no longer the bottleneck.
    """
    print("=" * 64)
    print("CATALOG SLIDING-WINDOW DATASET (time-to-major-earthquake)")
    print("=" * 64)
    df = load_catalog(catalog_path, min_magnitude=min_magnitude)

    if regions:
        windows = pool_regions(df, regions, window_events, stride_events, major_magnitude,
                               max_horizon_days=max_horizon_days, decluster=decluster,
                               region_names=region_names)
    else:
        rdf = filter_region(df, bbox=bbox, center=center, radius_km=radius_km)
        if len(rdf) < window_events * 2:
            print(f"[ERROR] Only {len(rdf)} events after filtering; need at least "
                  f"{window_events * 2} for a usable dataset.")
            return
        target_mask = None
        if decluster:
            target_mask = decluster_gardner_knopoff(rdf)
        else:
            print("[decluster] disabled (--no-decluster) -- aftershock sequences may appear "
                  "as multiple independent targets.")
        report_major_events(rdf, major_magnitude, max_horizon_days, target_mask=target_mask)
        windows = build_windows(rdf, window_events, stride_events, major_magnitude,
                                max_horizon_days=max_horizon_days, target_mask=target_mask)

    if not windows:
        print("[ERROR] No usable windows.")
        return

    n_targets = len({(w["region"], w["target_time"]) for w in windows})

    if split_mode == "loeo":
        # No train/val/test split here -- every window goes into one "all"
        # bucket, and `cnn_lstm_loeo.py` forms the folds (one per distinct
        # target event) at training time. Boundaries come from the whole
        # pool rather than a "train" split, since that split doesn't exist
        # yet; see the docstring on `assign_risk_classes`.
        if n_targets < 5:
            print(f"\n[ERROR] Only {n_targets} distinct target event(s) -- leave-one-event-out "
                  f"CV needs several folds to say anything. Pool more regions, lower "
                  f"--major-magnitude, or extend the catalog. Nothing was written.")
            return
        parts = {"all": windows}
        assign_risk_classes(parts, class_boundaries, boundary_split="all")
        encode_and_write(parts, output_dir, target_n=target_n)
        print(f"\n[COMPLETE] Flat dataset ready for leave-one-event-out CV "
              f"({n_targets} target events -> up to {n_targets} folds).")
        return

    parts = (chronological_split(windows, ratios, embargo_days, max_horizon_days)
             if split_mode == "chronological" else random_split(windows, ratios, seed))

    if not parts["train"] or not parts["test"]:
        empty = [s for s in ("train", "val", "test") if not parts[s]]
        print(f"\n[ERROR] Split(s) {empty} came out empty, so no model can be trained.")
        print(f"        The {len(windows)} windows point at only {n_targets} distinct target "
              f"event(s).")
        print("        With too few targets, every early window is labelled by an event in a")
        print("        later split and is correctly dropped as leakage -- which empties train.")
        print("        This is a property of the region/threshold, not a fixable split rule:")
        print("        lower --major-magnitude, widen the region, extend the catalog, or use")
        print("        --split-mode loeo, which does not require a single chronological cut.")
        print("        Nothing was written.")
        return

    assign_risk_classes(parts, class_boundaries)
    encode_and_write(parts, output_dir, target_n=target_n)
    print("\n[COMPLETE] Catalog window dataset ready.")
