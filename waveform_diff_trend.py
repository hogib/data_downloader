"""
Dense-trend follow-up to waveform_diff_probe.py, now that
data/batched_waveforms/day_before_24h gives real continuous pre-event data
(15 pilot events, see catalogs/pilot_continuous_sample.csv) instead of two
sparse 5-minute snapshots. For each (event, station), computes the full
raw_diff/ram_diff sequence at 50s resolution across the ~24h before the
event, then tests whether diff magnitude correlates with time-to-event
(Spearman) -- i.e. does the signal get more "different from itself" as the
event approaches. Aggregates that correlation across all (event, station)
series to see if there's a consistent direction, not just per-event noise.

Same caveat as the probe this replaces: 15 events is not a large sample, and
there is still no non-precursor control (a day picked at random, not
followed by an event) to rule out "this is just what continuous background
noise generally does over a day." Treat a positive result here as grounds
for that controlled follow-up, not a finding.
"""

import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from obspy import UTCDateTime, read
from scipy.stats import spearmanr, wilcoxon

from seismic_cli.core import _masked_to_filled, select_components
from seismic_cli.waveform_diff import process_component

WINDOW_SECONDS = 50.0
FREQMIN, FREQMAX = 1.0, 45.0
DAY_BEFORE_DIR = Path("data/batched_waveforms/day_before_24h")
SAMPLE_CATALOG = "catalogs/pilot_continuous_sample.csv"


def _parse_date(date_str: str) -> Optional[UTCDateTime]:
    try:
        return UTCDateTime(pd.to_datetime(date_str, format="%d/%m/%Y %H:%M:%S"))
    except Exception:
        try:
            return UTCDateTime(date_str)
        except Exception:
            return None


def _process_event_file(args) -> List[dict]:
    file_path, event_id, origin_time, magnitude = args
    try:
        st = read(str(file_path))
        try:
            st.merge(method=1)
        except Exception:
            return []

        stations: Dict[str, Dict[str, tuple]] = {}
        for tr in st:
            sta_key = f"{tr.stats.network}.{tr.stats.station}"
            chan = tr.stats.channel[-1].upper()
            existing = stations.setdefault(sta_key, {}).get(chan)
            if existing is not None and len(existing[0]) >= tr.stats.npts:
                continue
            data, gap_mask = _masked_to_filled(tr.data)
            stations[sta_key][chan] = (data, gap_mask, tr.stats.sampling_rate, tr.stats.starttime)

        rows = []
        for sta_key, channels in stations.items():
            selection = select_components(channels.keys())
            if selection is None:
                continue
            rates = {channels[c][2] for c in selection}
            if len(rates) != 1:
                continue
            fs = rates.pop()
            step_samples = int(fs * WINDOW_SECONDS)

            hours_list, raw_list, ram_list = [], [], []
            for comp in selection:
                data, gap_mask, _, start_t = channels[comp]
                result = process_component(data, gap_mask, fs, WINDOW_SECONDS, FREQMIN, FREQMAX)
                for idx, r, m in zip(result["diff_idx"], result["raw_diff"], result["ram_diff"]):
                    window_start_abs = start_t + idx * step_samples / fs
                    hours_before = (origin_time - window_start_abs) / 3600.0
                    hours_list.append(hours_before)
                    raw_list.append(r)
                    ram_list.append(m)

            if len(hours_list) < 20:
                continue

            hours_arr = np.array(hours_list)
            raw_arr = np.array(raw_list)
            ram_arr = np.array(ram_list)
            rho_raw, p_raw = spearmanr(hours_arr, raw_arr)
            rho_ram, p_ram = spearmanr(hours_arr, ram_arr)
            rows.append({
                "event_id": event_id, "station_key": sta_key, "magnitude": magnitude,
                "n_diffs": len(hours_list), "rho_raw": rho_raw, "p_raw": p_raw,
                "rho_ram": rho_ram, "p_ram": p_ram,
            })
        return rows
    except Exception as e:
        return [{"event_id": event_id, "error": str(e)}]


def main():
    cat = pd.read_csv(SAMPLE_CATALOG)
    cat["EventID"] = cat["EventID"].astype(str)

    tasks = []
    for _, row in cat.iterrows():
        origin = _parse_date(row["Date"])
        if origin is None:
            continue
        fp = DAY_BEFORE_DIR / f"event_{row['EventID']}_raw.mseed"
        if fp.exists():
            tasks.append((fp, row["EventID"], origin, row["Magnitude"]))

    print(f"[scan] {len(tasks)}/{len(cat)} events have a day_before_24h file")

    all_rows: List[dict] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as ex:
        for rows in ex.map(_process_event_file, tasks):
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    errors = df[df.get("error").notna()] if "error" in df.columns else pd.DataFrame()
    if len(errors):
        print(f"[warn] {len(errors)} event(s) failed:")
        for _, r in errors.iterrows():
            print(f"  {r['event_id']}: {r['error']}")
    df = df[df.get("error").isna()] if "error" in df.columns else df

    out_path = Path("waveform_diff_trend_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\n[write] {out_path} ({len(df)} (event, station) series)")
    print(df[["event_id", "station_key", "magnitude", "n_diffs", "rho_raw", "p_raw", "rho_ram", "p_ram"]]
          .to_string(index=False))

    print("\n" + "=" * 70)
    for metric, rho_col, p_col in (("raw_diff", "rho_raw", "p_raw"), ("ram_diff", "rho_ram", "p_ram")):
        rhos = df[rho_col].dropna().to_numpy()
        n_sig = int((df[p_col].dropna() < 0.05).sum())
        print(f"\n--- {metric}: per-series correlation with hours-before-event ---")
        print(f"  n series = {len(rhos)}   mean(rho) = {rhos.mean():+.3f}   "
              f"median(rho) = {np.median(rhos):+.3f}")
        print(f"  {n_sig}/{len(rhos)} series individually significant at p<0.05 "
              f"(expect ~{0.05*len(rhos):.1f} by chance alone)")
        print(f"  sign: {int((rhos < 0).sum())} negative (diff grows approaching event), "
              f"{int((rhos > 0).sum())} positive (diff shrinks approaching event)")
        if len(rhos) >= 5:
            stat, p = wilcoxon(rhos)
            print(f"  one-sample Wilcoxon (rho vs 0): statistic={stat:.1f}  p={p:.4g}")
            print(f"  verdict: {'SYSTEMATIC across events (p<0.05)' if p < 0.05 else 'no consistent cross-event trend'}")

    print("\n" + "=" * 70)
    print("CAVEAT: n=15 events, no non-precursor control (a day NOT followed by")
    print("an event). A positive result here is grounds for that controlled")
    print("follow-up, not a finding on its own.")


if __name__ == "__main__":
    main()
