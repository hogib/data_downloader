"""
Waveform-diff probe (see MAGNITUDE_CNN_CHEATSHEET.md's neighboring
conversation / plan file for the full motivation): does the difference
between two consecutive, non-overlapping windows of raw pre-event seismic
data differ measurably at 3 hours vs. 6 hours before the event?

Data constraint (this is why the comparison is 3h-vs-6h and not a dense
trend): `data/batched_noise_waveforms/noise_pre_3h`/`noise_pre_6h` are each a
single 5-minute snapshot at a fixed offset before the event (src/download.py
NOISE_BATCHES), not continuous multi-hour recordings. There is also no
non-precursor control in this data -- these files exist only because an
event followed. This script is a plausibility probe, not a forecasting
feature: a result here shows (at most) "closer-to-event windows look
different from further-out windows," not "this predicts earthquakes."

Usage:
    python waveform_diff_probe.py
"""

import concurrent.futures
import multiprocessing
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from seismic_cli.core import _masked_to_filled, select_components
from seismic_cli.regression import load_event_catalog, parse_event_id
from seismic_cli.waveform_diff import process_component

WINDOW_LENGTHS = (50.0, 145.0)
FREQMIN, FREQMAX = 1.0, 45.0
CATALOG_PATH = "catalogs/deprem_katalog_utc.csv"
NOISE_3H = Path("data/batched_noise_waveforms/noise_pre_3h")
NOISE_6H = Path("data/batched_noise_waveforms/noise_pre_6h")

# event_meta (~22MB, ~0.2s to pickle) is sent to each worker ONCE via
# initializer= rather than per-task -- ProcessPoolExecutor.map pickles every
# task argument for IPC, so including it directly in each of ~11k tasks
# would cost ~2300s of pure pickling alone (measured; the exact bug already
# hit and fixed once this session in regression.py).
_worker: Dict[str, object] = {}


def _init_worker(event_meta):
    _worker["event_meta"] = event_meta


def _process_offset_file(args) -> List[dict]:
    file_path, offset_label = args
    event_meta = _worker["event_meta"]
    from obspy import read
    try:
        st = read(str(file_path))
        try:
            st.merge(method=1)
        except Exception:
            return []

        event_id = parse_event_id(file_path.stem)
        meta = event_meta.get(event_id)
        if meta is None:
            return []

        stations: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, float]]] = {}
        for tr in st:
            sta_key = f"{tr.stats.network}.{tr.stats.station}"
            chan = tr.stats.channel[-1].upper()
            existing = stations.setdefault(sta_key, {}).get(chan)
            if existing is not None and len(existing[0]) >= tr.stats.npts:
                continue
            data, gap_mask = _masked_to_filled(tr.data)
            stations[sta_key][chan] = (data, gap_mask, tr.stats.sampling_rate)

        rows = []
        for sta_key, channels in stations.items():
            selection = select_components(channels.keys())
            if selection is None:
                continue
            rates = {channels[c][2] for c in selection}
            if len(rates) != 1:
                continue
            fs = rates.pop()

            row = {"event_id": str(event_id), "station_key": sta_key,
                   "magnitude": meta["magnitude"], "offset": offset_label}
            ok_any = False
            for w in WINDOW_LENGTHS:
                pooled_raw, pooled_ram, total_windows = [], [], 0
                for comp in selection:
                    data, gap_mask, _ = channels[comp]
                    result = process_component(data, gap_mask, fs, w, FREQMIN, FREQMAX)
                    pooled_raw.extend(result["raw_diff"])
                    pooled_ram.extend(result["ram_diff"])
                    total_windows += result["n_windows"]
                suffix = f"w{int(w)}"
                if total_windows >= 3 and pooled_raw:
                    row[f"raw_diff_{suffix}"] = float(np.mean(pooled_raw))
                    row[f"ram_diff_{suffix}"] = float(np.mean(pooled_ram))
                    row[f"n_windows_{suffix}"] = total_windows
                    ok_any = True
                else:
                    row[f"raw_diff_{suffix}"] = np.nan
                    row[f"ram_diff_{suffix}"] = np.nan
                    row[f"n_windows_{suffix}"] = total_windows
            if ok_any:
                rows.append(row)
        return rows
    except Exception:
        return []


def run_probe(num_cores: Optional[int] = None) -> pd.DataFrame:
    event_meta = load_event_catalog(CATALOG_PATH)

    files_3h = {p.name: p for p in NOISE_3H.glob("*.mseed")}
    files_6h = {p.name: p for p in NOISE_6H.glob("*.mseed")}
    common = sorted(set(files_3h) & set(files_6h))
    print(f"[scan] {len(common)} filenames present in both noise_pre_3h and noise_pre_6h")

    tasks = ([(files_3h[n], "3h") for n in common] +
             [(files_6h[n], "6h") for n in common])

    if num_cores is None:
        num_cores = max(1, multiprocessing.cpu_count() - 1)

    all_rows: List[dict] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=num_cores, initializer=_init_worker, initargs=(event_meta,),
    ) as ex:
        for rows in ex.map(_process_offset_file, tasks, chunksize=16):
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    print(f"[process] {len(df)} (event, station, offset) rows with at least one usable window length")

    id_cols = ["event_id", "station_key", "magnitude"]
    df_3h = df[df.offset == "3h"].drop(columns="offset")
    df_6h = df[df.offset == "6h"].drop(columns="offset")
    paired = df_3h.merge(df_6h, on=id_cols, suffixes=("_3h_off", "_6h_off"))
    print(f"[pair] {len(paired)} (event, station) pairs with both offsets present")
    return paired


def report_comparison(paired: pd.DataFrame) -> None:
    print("\n" + "=" * 70)
    for metric in ("raw_diff", "ram_diff"):
        for w in WINDOW_LENGTHS:
            suffix = f"w{int(w)}"
            col_3h = f"{metric}_{suffix}_3h_off"
            col_6h = f"{metric}_{suffix}_6h_off"
            sub = paired[[col_3h, col_6h, "magnitude"]].dropna()
            n = len(sub)
            print(f"\n--- {metric}, window={w:.0f}s (n={n} paired events) ---")
            if n < 10:
                print("  too few paired, clean observations to test")
                continue
            a, b = sub[col_3h].to_numpy(), sub[col_6h].to_numpy()
            diff = a - b
            stat, p = wilcoxon(a, b)
            print(f"  mean(3h)={a.mean():.5g}  mean(6h)={b.mean():.5g}  "
                  f"mean(3h-6h)={diff.mean():+.5g}  median(3h-6h)={np.median(diff):+.5g}")
            print(f"  Wilcoxon signed-rank: statistic={stat:.1f}  p={p:.4g}")
            verdict = "SYSTEMATIC DIFFERENCE (p<0.01)" if p < 0.01 else "no clear difference"
            print(f"  verdict: {verdict}")

            terciles = pd.qcut(sub["magnitude"], 3, labels=["small", "medium", "large"], duplicates="drop")
            for label in terciles.cat.categories if hasattr(terciles, "cat") else []:
                mask = terciles == label
                if mask.sum() < 10:
                    continue
                _, p_t = wilcoxon(a[mask.to_numpy()], b[mask.to_numpy()])
                print(f"    [{label:6s} mag, n={mask.sum():4d}] mean(3h-6h)={diff[mask.to_numpy()].mean():+.5g}  p={p_t:.4g}")

    print("\n" + "=" * 70)
    print("CAVEAT: this shows (at most) 'closer-to-event windows look different")
    print("from further-out windows,' not 'this predicts earthquakes.' There is")
    print("no non-precursor control in this data -- an observed difference could")
    print("be a general property of continuous background noise over a few")
    print("hours, unrelated to the impending event. Treat any positive result")
    print("here as grounds for a properly controlled follow-up, not a finding.")


def main():
    paired = run_probe()
    out_path = Path("waveform_diff_probe_results.csv")
    paired.to_csv(out_path, index=False)
    print(f"\n[write] {out_path} ({len(paired)} rows)")
    report_comparison(paired)


if __name__ == "__main__":
    main()
