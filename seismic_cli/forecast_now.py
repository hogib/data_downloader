"""
Operational forecast: probability of a M >= threshold event in each fault zone
over the next `horizon_days`, from the catalog as it stands today.

This is the only part of the project that produces an output someone could act
on, so it is deliberately conservative about what it claims.

**What was measured** (`cnn_earthquake/catalog_report.md` 4.7-4.9, over ~190
disjoint 30-day blocks per zone, which is the honest sample size once 11-46x
window overlap is removed):

    zone     block AUC   95% CI            calibrated BSS   status
    EAFZ       0.6209    [0.529, 0.706]        +0.032       usable
    AEGEAN     0.5987    [0.522, 0.674]        +0.005       discriminates, no value
    NAFZ       0.4519    [0.367, 0.540]        -0.019       not forecastable
    CENTRAL    0.4778    [0.378, 0.578]        -0.015       not forecastable

Only EAFZ produces probabilities that beat climatology by a usable margin.
AEGEAN ranks better than chance yet its calibrated probabilities sit within
0.005 Brier skill of simply quoting the base rate. NAFZ and CENTRAL are
indistinguishable from chance and are reported as such rather than being
silently dropped -- a forecaster that quietly omits the zones it cannot handle
is worse than one that says so.

**Calibration is not optional here.** The raw model ranks above chance while
emitting probabilities WORSE than climatology -- it predicted 0.089 where 0.410
was observed. Raw scores are therefore never surfaced as probabilities; a Platt
calibrator fitted on that zone's historical blocks is always applied. Because
every block used for fitting is strictly in the past, this uses no future
information.

**What this cannot do.** It gives no location beyond the zone, no magnitude
beyond "at or above the threshold", and no time beyond "somewhere in the next
`horizon_days`". A zone-scale 30-day probability slightly better than
climatology is the entire claim.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from seismic_cli.forecast import (FAULT_ZONES, FEATURES, build_blocks, build_dataset,
                                  zone_major_times)

# Empirical status per zone, measured in catalog_report.md 4.7-4.8. Recorded as
# a constant so the operational path states what was actually established
# rather than implying every zone is equally trustworthy.
ZONE_STATUS: Dict[str, Dict[str, object]] = {
    "EAFZ":    {"auc": 0.6209, "ci": (0.529, 0.706), "bss": +0.032, "usable": True},
    "AEGEAN":  {"auc": 0.5987, "ci": (0.522, 0.674), "bss": +0.005, "usable": False},
    "NAFZ":    {"auc": 0.4519, "ci": (0.367, 0.540), "bss": -0.019, "usable": False},
    "CENTRAL": {"auc": 0.4778, "ci": (0.378, 0.578), "bss": -0.015, "usable": False},
}

MIN_TRAIN_WINDOWS = 400
MIN_CAL_BLOCKS = 40


def _fit_logistic(X, y, Xq):
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    if len(np.unique(y)) < 2:
        return np.full(len(Xq), float(y.mean()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m = LogisticRegression(max_iter=2000).fit(np.nan_to_num(X, nan=0.0), y)
    return m.predict_proba(np.nan_to_num(Xq, nan=0.0))[:, 1]


def forecast_zone(d: pd.DataFrame, catalog_path: str, zone: str,
                  threshold: float, horizon_days: float) -> Optional[dict]:
    """
    One zone's live forecast.

    Returns None (rather than a fabricated number) when the zone lacks enough
    history to train or to calibrate.
    """
    g = d[d.region == zone].sort_values("end_time").reset_index(drop=True)
    if len(g) < MIN_TRAIN_WINDOWS or g.label.nunique() < 2:
        return None

    major = zone_major_times(catalog_path, zone, threshold)
    blocks = build_blocks(d, zone, horizon_days, d.end_time.max(), major)
    if len(blocks) < MIN_CAL_BLOCKS or blocks.label.nunique() < 2:
        return None

    # Historical block scores, each trained only on windows fully resolved
    # before that block opened -- same construction as the evaluation.
    emb = pd.Timedelta(days=horizon_days)
    hist_scores, hist_y = [], []
    for _, row in blocks.iterrows():
        tr = g[g.end_time <= row.block_start - emb]
        if len(tr) < MIN_TRAIN_WINDOWS or tr.label.nunique() < 2:
            continue
        s = _fit_logistic(tr[FEATURES].to_numpy(float), tr.label.to_numpy(),
                          g.loc[[row.fc_index], FEATURES].to_numpy(float))[0]
        hist_scores.append(s)
        hist_y.append(int(row.label))

    if len(hist_scores) < MIN_CAL_BLOCKS or len(set(hist_y)) < 2:
        return None

    # Live score: train on everything, predict from the most recent window.
    latest = g.iloc[[-1]]
    raw = _fit_logistic(g[FEATURES].to_numpy(float), g.label.to_numpy(),
                        latest[FEATURES].to_numpy(float))[0]

    # Platt calibration on the zone's own history. Never surface `raw`.
    from sklearn.linear_model import LogisticRegression
    cal = LogisticRegression(max_iter=1000).fit(
        np.asarray(hist_scores).reshape(-1, 1), np.asarray(hist_y))
    p = float(cal.predict_proba(np.array([[raw]]))[0, 1])

    climatology = float(np.mean(hist_y))
    return {
        "zone": zone,
        "as_of": latest.end_time.iloc[0],
        "probability": p,
        "climatology": climatology,
        "lift": p / climatology if climatology > 0 else float("nan"),
        "n_blocks_calibration": len(hist_scores),
        "days_since_prev_major": float(latest.days_since_prev_major.iloc[0]),
    }


def run_forecast_now(catalog_path: str, threshold: float = 4.5,
                     horizon_days: float = 30.0, window_events: int = 64,
                     stride_events: int = 8,
                     zones: Optional[List[str]] = None) -> pd.DataFrame:
    d = build_dataset(catalog_path, FAULT_ZONES, window_events=window_events,
                      stride_events=stride_events, threshold=threshold,
                      horizon_days=horizon_days)
    names = zones or [z for z in FAULT_ZONES if (d.region == z).any()]

    rows = []
    for z in names:
        r = forecast_zone(d, catalog_path, z, threshold, horizon_days)
        if r is None:
            print(f"  {z:9s} SKIPPED -- insufficient history to train or calibrate")
            continue
        rows.append(r)
    out = pd.DataFrame(rows)

    cat_end = d.end_time.max()
    print(f"\n{'='*84}")
    print(f"FORECAST: P(M >= {threshold} within {horizon_days:.0f} days) per fault zone")
    print(f"catalog through {cat_end:%Y-%m-%d}")
    print(f"{'='*84}")
    print(f"{'zone':9s} {'forecast':>9s} {'climatology':>12s} {'lift':>6s} | "
          f"{'block AUC':>9s} {'95% CI':>15s} | status")
    print("-" * 84)

    for _, r in out.iterrows():
        st = ZONE_STATUS.get(r.zone, {})
        ci = st.get("ci", (float("nan"),) * 2)
        status = ("USABLE" if st.get("usable") else
                  "not established" if not np.isfinite(st.get("auc", np.nan))
                  or ci[0] <= 0.5 else "ranks > chance, no usable skill")
        print(f"{r.zone:9s} {r.probability:9.3f} {r.climatology:12.3f} "
              f"{r.lift:6.2f} | {st.get('auc', float('nan')):9.4f} "
              f"[{ci[0]:.3f}, {ci[1]:.3f}] | {status}")

    print("-" * 84)
    usable = [r.zone for _, r in out.iterrows() if ZONE_STATUS.get(r.zone, {}).get("usable")]
    if usable:
        print(f"  Act on: {', '.join(usable)}.")
    print("  Every other zone's number is printed for completeness and should NOT be")
    print("  acted on: its 95 % CI includes chance, or its calibrated Brier skill is")
    print("  at or below climatology. See cnn_earthquake/catalog_report.md 4.7-4.8.")
    print("\n  Scope: zone-scale only. No location within the zone, no magnitude above")
    print(f"  the {threshold} threshold, no timing within the {horizon_days:.0f}-day window.")
    return out
