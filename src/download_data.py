from pathlib import Path

import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

# Each tuple defines a window configuration: (window_name, start_offset_sec, end_offset_sec)
# e.g., ("window_pre_200s", -200, 0) means 200 seconds before event to origin time.
# e.g., ("window_post_60s", 0, 60) means origin time to 60 seconds after event.
WINDOW_BATCHES = [
    ("window_pre_200s", -200, 0),
    ("window_post_60s", 0, 60),
    ("window_extended", -30, 300),  # 30s before to 5m after (standard)
]

# General Settings
CATALOG_FILE = Path("extracted_earthquakes.csv")
BASE_OUTPUT_DIR = Path("data/batched_waveforms")
FILE_LIMIT = 1000  # Set to None to process the full catalog
SEARCH_RADIUS_DEG = 0.5  # ~55 km radius around event
FDSN_CLIENT = "KOERI"



def download_batched_waveforms(
    catalog_path: Path,
    output_base_dir: Path,
    batches: list[tuple[str, int, int]],
    file_limit: int | None = None,
) -> None:
  """Download seismic waveforms for multiple time-window batches per event."""
  catalog_path = Path(catalog_path)
  output_base_dir = Path(output_base_dir)

  if not catalog_path.exists():
    raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

  # Load catalog
  print(f"[init] Loading catalog: {catalog_path.resolve()}")
  df = pd.read_csv(catalog_path)

  if file_limit is not None and file_limit > 0:
    df = df.head(file_limit)
    print(f"[limit] Processing limited to first {len(df)} events.")

  # Initialize client
  client = Client(FDSN_CLIENT)

  # Process each event row
  for index, row in df.iterrows():
    event_id = row.get("EventID", f"idx_{index}")

    try:
      # Parse event date/time
      try:
        parsed_dt = pd.to_datetime(row["Date"], format="%d/%m/%Y %H:%M:%S")
        event_time = UTCDateTime(parsed_dt)
      except Exception:
        event_time = UTCDateTime(row["Date"])

      lat = float(row["Latitude"])
      lon = float(row["Longitude"])

      # Query nearby stations once per event
      inventory = client.get_stations(
          latitude=lat, longitude=lon, maxradius=SEARCH_RADIUS_DEG, channel="HH*"
      )

      if not inventory:
        print(f"[warning] No stations found for EventID {event_id}")
        continue

      # Build station query list
      station_queries = [
          (net.code, sta.code, "*", "HH*") for net in inventory for sta in net
      ]

      # Loop through each defined window batch
      for batch_name, start_offset, end_offset in batches:
        # Create dedicated directory for this batch window
        batch_dir = output_base_dir / batch_name
        batch_dir.mkdir(parents=True, exist_ok=True)

        output_file = batch_dir / f"event_{event_id}_raw.mseed"

        # Skip if already downloaded
        if output_file.exists():
          continue

        # Calculate specific window boundaries relative to event time
        starttime = event_time + start_offset
        endtime = event_time + end_offset

        # Build bulk request for this time window
        bulk_query = [
            (net, sta, loc, chan, starttime, endtime)
            for net, sta, loc, chan in station_queries
        ]

        # Fetch and save
        st = client.get_waveforms_bulk(bulk_query)
        st.write(str(output_file), format="MSEED")
        print(
            f"[success] [{batch_name}] Saved event {event_id} ({len(st)}"
            " traces)"
        )

    except Exception as e:
      print(f"[error] Failed processing EventID {event_id}: {e}")

  print("\n" + "=" * 50)
  print(f"[complete] All batch directories saved under: {output_base_dir.resolve()}")
  print("=" * 50)


if __name__ == "__main__":
  download_batched_waveforms(
      catalog_path=CATALOG_FILE,
      output_base_dir=BASE_OUTPUT_DIR,
      batches=WINDOW_BATCHES,
      file_limit=FILE_LIMIT,
  )
