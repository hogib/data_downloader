# Seismic Waveform Batch Downloader

A multithreaded script to bulk-download earthquake and ambient noise waveforms (MiniSEED) from FDSN servers via ObsPy.

Designed for machine learning pipelines: it caches station lookups spatially to reduce network calls and strictly filters ambient noise windows to ensure they don't overlap with other seismic events in your catalog.

## Requirements

```bash
pip install pandas obspy
```

or if you use uv:

```bash
python3 -m venv .venv
source .venv/bin/activate
uv sync
```

## Input Data

Place your event catalog at `catalogs/extracted_earthquakes.csv`. The script expects the following columns:

- `EventID` (optional, script falls back to row index if missing)
- `Latitude`
- `Longitude`
- `Date` (Format: `DD/MM/YYYY HH:MM:SS` or standard ISO string)

## Configuration

Open the script and adjust the global variables at the top to fit your needs:

- `EARTHQUAKE_BATCHES` / `NOISE_BATCHES`: Define your target extraction windows in seconds relative to the event time `(folder_name, start_offset, end_offset)`.
- `FDSN_CLIENT`: Set your target FDSN node (e.g., `"KOERI"`, `"IRIS"`, `"GFZ"`).
- `SEARCH_RADIUS_DEG`: Radius around the epicenter to pull stations from (default is `0.5` degrees).
- `MAX_WORKERS`: Number of concurrent download threads. Keep this reasonable to avoid getting IP-banned by the FDSN server.
- `FILE_LIMIT`: Set a number for testing small batches, or `None` to run the full catalog.

## Usage

Run the script directly:

```bash
python src/download.py
```

Downloaded `.mseed` files will be automatically sorted into `data/batched_waveforms/<window_name>/` and `data/batched_noise_waveforms/<window_name>/`.

## Please don't use preprocessors. they have many bugs I've yet to fix
