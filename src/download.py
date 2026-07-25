import functools
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

# Earthquake waveform windows: (window_name, start_offset_sec, end_offset_sec)
EARTHQUAKE_BATCHES = [
    ("window_pre_200s", -200, 0),
    ("window_post_60s", 0, 60),
    ("window_extended", -30, 300),
    ("window_post_200s", 0, 200),
]

# Noise waveform windows: (window_name, start_offset_sec, end_offset_sec)
NOISE_BATCHES = [
    ("noise_pre_3h", -11100, -10800),
    ("noise_pre_6h", -21900, -21600),
]

# General Settings
CATALOG_FILE = Path("catalogs/extracted_earthquakes.csv")
BASE_OUTPUT_DIR = Path("data")
FILE_LIMIT = 10000  # Set to None to process the full catalog
SEARCH_RADIUS_DEG = 0.5  # ~55 km radius around event
FDSN_CLIENT = "KOERI"
MAX_WORKERS = 6  # Number of concurrent threads


@functools.lru_cache(maxsize=2048)
def fetch_station_queries(lat_round: float, lon_round: float, radius: float, client_name: str) -> tuple:
  """
  Caches station metadata. If events occur within ~1.1km of each other 
  (rounded to 2 decimal places), it instantly reuses the station list 
  instead of making a slow network request.
  """
  client = Client(client_name)
  try:
    inventory = client.get_stations(
        latitude=lat_round, longitude=lon_round, maxradius=radius, channel="HH*"
    )
    queries = [(net.code, sta.code, "*", "HH*") for net in inventory for sta in net]
    return tuple(queries)
  except Exception:
    return ()


def process_event_and_noise(
    row, index, df_full, eq_batches, noise_batches, output_base_dir, client
):
  """Worker function to process both earthquake and noise windows for a single event."""
  event_id = row.get("EventID", f"idx_{index}")
  event_time = row.get("ParsedUTC")

  if event_time is None:
    return f"[error] Could not parse date for EventID {event_id}."

  try:
    lat = float(row["Latitude"])
    lon = float(row["Longitude"])

    # OPTIMIZATION 1: Use cached station queries
    lat_round, lon_round = round(lat, 2), round(lon, 2)
    station_queries = fetch_station_queries(lat_round, lon_round, SEARCH_RADIUS_DEG, FDSN_CLIENT)

    if not station_queries:
      return f"[warning] No stations found for EventID location {event_id}"

    results = []
    bulk_query = []
    tasks_to_slice = []  # Tracks which windows to extract from the master stream

    # --- PREPARE EARTHQUAKE BATCHES ---
    eq_output_base = output_base_dir / "batched_waveforms"
    for batch_name, start_offset, end_offset in eq_batches:
      batch_dir = eq_output_base / batch_name
      batch_dir.mkdir(parents=True, exist_ok=True)
      output_file = batch_dir / f"event_{event_id}_raw.mseed"

      if not output_file.exists():
        starttime = event_time + start_offset
        endtime = event_time + end_offset
        bulk_query.extend([
            (net, sta, loc, chan, starttime, endtime)
            for net, sta, loc, chan in station_queries
        ])
        tasks_to_slice.append((batch_name, starttime, endtime, output_file))

    # --- PREPARE NOISE BATCHES ---
    noise_output_base = output_base_dir / "batched_noise_waveforms"
    for batch_name, start_offset, end_offset in noise_batches:
      batch_dir = noise_output_base / batch_name
      batch_dir.mkdir(parents=True, exist_ok=True)
      output_file = batch_dir / f"noise_event_{event_id}_raw.mseed"

      if not output_file.exists():
        starttime = event_time + start_offset
        endtime = event_time + end_offset

        # Contamination check
        buffer = 60
        overlapping_events = df_full[
            (df_full["ParsedUTC"] >= (starttime - buffer))
            & (df_full["ParsedUTC"] <= (endtime + buffer))
        ]

        if not overlapping_events.empty:
          results.append(
              f"[skipped] [Noise-{batch_name}] Event {event_id} overlaps with "
              f"{len(overlapping_events)} other event(s)."
          )
          continue

        bulk_query.extend([
            (net, sta, loc, chan, starttime, endtime)
            for net, sta, loc, chan in station_queries
        ])
        tasks_to_slice.append((batch_name, starttime, endtime, output_file))

    # If all files already exist, exit early
    if not bulk_query:
      return f"[skip] All files already exist for {event_id}"

    # OPTIMIZATION 2: Single consolidated network request for ALL windows
    try:
        st_master = client.get_waveforms_bulk(bulk_query)
    except Exception as e:
        return f"[warning] No data returned for {event_id}: {e}"

    # SLICE AND SAVE IN RAM
    for batch_name, starttime, endtime, output_file in tasks_to_slice:
        # ObsPy safely extracts just the exact timeframe needed for this batch
        st_sliced = st_master.slice(starttime, endtime)
        if len(st_sliced) > 0:
            st_sliced.write(str(output_file), format="MSEED")
            results.append(f"[success] [{batch_name}] Saved {event_id} ({len(st_sliced)} traces)")
        else:
            results.append(f"[warning] [{batch_name}] No traces matching window for {event_id}")

    return "\n".join(results)

  except Exception as e:
    return f"[error] Failed processing EventID {event_id}: {e}"


def download_all_concurrent(
    catalog_path: Path,
    output_base_dir: Path,
    eq_batches: list[tuple[str, int, int]],
    noise_batches: list[tuple[str, int, int]],
    file_limit: int | None = None,
    max_workers: int = 4,
) -> None:
  """Download both earthquake waveforms and safe ambient noise concurrently."""
  catalog_path = Path(catalog_path)
  output_base_dir = Path(output_base_dir)

  if not catalog_path.exists():
    raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

  print(f"[init] Loading catalog: {catalog_path.resolve()}")
  df = pd.read_csv(catalog_path)

  # Pre-parse timestamps for collision detection
  catalog_times = []
  for _, row in df.iterrows():
    try:
      parsed_dt = pd.to_datetime(row["Date"], format="%d/%m/%Y %H:%M:%S")
      catalog_times.append(UTCDateTime(parsed_dt))
    except Exception:
      try:
        catalog_times.append(UTCDateTime(row["Date"]))
      except Exception:
        catalog_times.append(None)
  df["ParsedUTC"] = catalog_times

  if file_limit is not None and file_limit > 0:
    df_process = df.head(file_limit)
    print(
        f"[limit] Processing limited to first {len(df_process)} events using"
        f" {max_workers} threads."
    )
  else:
    df_process = df

  # Notice: The client instantiation here is passed into the worker.
  client = Client(FDSN_CLIENT)

  # Execute downloads concurrently per event
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {
        executor.submit(
            process_event_and_noise,
            row,
            index,
            df,
            eq_batches,
            noise_batches,
            output_base_dir,
            client,
        ): index
        for index, row in df_process.iterrows()
    }

    for future in as_completed(futures):
      res = future.result()
      if res:
        print(res)

  print("\n" + "=" * 50)
  print(
      "[complete] All earthquake and noise batches saved under:"
      f" {output_base_dir.resolve()}"
  )
  print("=" * 50)


if __name__ == "__main__":
  download_all_concurrent(
      catalog_path=CATALOG_FILE,
      output_base_dir=BASE_OUTPUT_DIR,
      eq_batches=EARTHQUAKE_BATCHES,
      noise_batches=NOISE_BATCHES,
      file_limit=FILE_LIMIT,
      max_workers=MAX_WORKERS,
  )
