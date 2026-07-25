from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

# Each tuple defines a noise window configuration: (window_name, start_offset_sec, end_offset_sec)
NOISE_BATCHES = [
    ("noise_pre_3h", -11100, -10800),
    ("noise_pre_6h", -21900, -21600),
]

# General Settings
CATALOG_FILE = Path("extracted_earthquakes.csv")
BASE_OUTPUT_DIR = Path("data/batched_noise_waveforms")
FILE_LIMIT = 1000  # Set to None to process the full catalog
SEARCH_RADIUS_DEG = 0.5
FDSN_CLIENT = "KOERI"
MAX_WORKERS = 4  # Number of concurrent download threads (adjust based on server limits)


def process_single_event(
    row, index, df_full, batches, output_base_dir, client
):
  """Helper function to process and download noise for a single event."""
  event_id = row.get("EventID", f"idx_{index}")
  event_time = row.get("ParsedUTC")

  if event_time is None:
    return f"[error] Could not parse date for EventID {event_id}."

  try:
    lat = float(row["Latitude"])
    lon = float(row["Longitude"])

    # Query nearby stations once per event location
    inventory = client.get_stations(
        latitude=lat, longitude=lon, maxradius=SEARCH_RADIUS_DEG, channel="HH*"
    )

    if not inventory:
      return f"[warning] No stations found for EventID location {event_id}"

    station_queries = [
        (net.code, sta.code, "*", "HH*") for net in inventory for sta in net
    ]

    results = []
    for batch_name, start_offset, end_offset in batches:
      batch_dir = output_base_dir / batch_name
      batch_dir.mkdir(parents=True, exist_ok=True)

      output_file = batch_dir / f"noise_event_{event_id}_raw.mseed"

      if output_file.exists():
        continue

      starttime = event_time + start_offset
      endtime = event_time + end_offset

      # Contamination check against catalog
      buffer = 60
      overlapping_events = df_full[
          (df_full["ParsedUTC"] >= (starttime - buffer))
          & (df_full["ParsedUTC"] <= (endtime + buffer))
      ]

      if not overlapping_events.empty:
        results.append(
            f"[skipped] [{batch_name}] Event {event_id} overlaps with"
            f" {len(overlapping_events)} other event(s)."
        )
        continue

      bulk_query = [
          (net, sta, loc, chan, starttime, endtime)
          for net, sta, loc, chan in station_queries
      ]

      st = client.get_waveforms_bulk(bulk_query)
      st.write(str(output_file), format="MSEED")
      results.append(
          f"[success] [{batch_name}] Saved clean noise for event location"
          f" {event_id} ({len(st)} traces)"
      )

    return "\n".join(results)

  except Exception as e:
    return f"[error] Failed processing noise for EventID {event_id}: {e}"


def download_batched_noise_parallel(
    catalog_path: Path,
    output_base_dir: Path,
    batches: list[tuple[str, int, int]],
    file_limit: int | None = None,
    max_workers: int = 4,
) -> None:
  """Download ambient noise waveforms concurrently using a ThreadPoolExecutor."""
  catalog_path = Path(catalog_path)
  output_base_dir = Path(output_base_dir)

  if not catalog_path.exists():
    raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

  print(f"[init] Loading catalog: {catalog_path.resolve()}")
  df = pd.read_csv(catalog_path)

  # Pre-parse catalog timestamps
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

  # Note: FDSN Client objects are generally thread-safe for reads,
  # but instantiating a client per thread or using a shared one works well.
  client = Client(FDSN_CLIENT)

  # Execute downloads in parallel pools
  with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = {
        executor.submit(
            process_single_event,
            row,
            index,
            df,
            batches,
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
      "[complete] Parallel clean noise batch directories saved under:"
      f" {output_base_dir.resolve()}"
  )
  print("=" * 50)


if __name__ == "__main__":
  download_batched_noise_parallel(
      catalog_path=CATALOG_FILE,
      output_base_dir=BASE_OUTPUT_DIR,
      batches=NOISE_BATCHES,
      file_limit=FILE_LIMIT,
      max_workers=MAX_WORKERS,
  )
