import functools
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

# Earthquake waveform windows: (window_name, start_offset_sec, end_offset_sec)
EARTHQUAKE_BATCHES = [
    ("window_post_60s", 0, 60),
    ("window_post_120s", 0, 120),
    ("window_post_200s", 0, 200),
    ("window_post_100s", 0, 100),
    ("window_post_3s", 0, 3),
    ("window_post_6s", 0, 6),
    ("window_post_10s", 0, 10),
]

# Noise waveform windows: (window_name, start_offset_sec, end_offset_sec)
NOISE_BATCHES = [
    ("noise_pre_3h", -11100, -10800),
    ("noise_pre_6h", -21900, -21600),
]

# General Settings
CATALOG_FILE = Path("catalogs/extracted_earthquakes.csv")  # filtered: which events to actually download
FULL_CATALOG_FILE = Path("catalogs/deprem_katalog_utc.csv")  # unfiltered: the complete raw catalog, used
                                                              # only to check noise windows for contamination
                                                              # by ANY cataloged event (including ones that
                                                              # were filtered out of the download list, e.g.
                                                              # small quakes below your magnitude threshold)
BASE_OUTPUT_DIR = Path("data")
FILE_LIMIT = 10000  # Set to None to process the full catalog
SEARCH_RADIUS_DEG = 0.5  # ~55 km radius around event
FDSN_CLIENT = "KOERI"
MAX_WORKERS = 15  # Number of concurrent threads

# How much padding (seconds) around any other cataloged event to treat as
# potentially contaminated when carving out "noise" windows. Widened from the
# original 60s: larger events can have coda/aftershock energy ringing on for
# several minutes, so a tight buffer risks labeling real seismic signal as noise.
NOISE_CONTAMINATION_BUFFER_SEC = 300


@functools.lru_cache(maxsize=2048)
def fetch_station_queries(lat_round: float, lon_round: float, radius: float, client_name: str) -> tuple:
    """Fetches and caches station queries within a specific radius.

    Reduces redundant network calls by caching station lists for coordinate
    pairs. Rounding inputs to two decimal places allows events within ~1.1 km
    of each other to share the same station metadata.

    Args:
        lat_round (float): Rounded latitude of the event.
        lon_round (float): Rounded longitude of the event.
        radius (float): Search radius in degrees.
        client_name (str): Name of the FDSN client to query (e.g., "KOERI").

    Returns:
        tuple: A sequence of tuples containing (network, station, location, channel)
            strings for bulk waveform requests. Returns an empty tuple if the
            FDSN query fails.
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
    row: pd.Series,
    index: int,
    df_collision_check: pd.DataFrame,
    eq_batches: list,
    noise_batches: list,
    output_base_dir: Path,
    client: Client,
    contamination_buffer_sec: int = NOISE_CONTAMINATION_BUFFER_SEC,
) -> str:
    """Processes earthquake and noise waveform batches for a single catalog event.

    Resolves station metadata, performs collision detection to prevent noise
    contamination from overlapping events, queries the FDSN client for a bulk
    waveform stream, and slices the stream into the target MiniSEED windows.

    Args:
        row (pd.Series): A single row from the catalog DataFrame containing event metadata.
        index (int): The index of the row, used as a fallback event ID.
        df_collision_check (pd.DataFrame): The FULL, unfiltered catalog (with
            ParsedUTC already populated), used for noise-window collision
            detection. This must NOT be the magnitude/type-filtered catalog --
            using the filtered one would let events that were filtered out
            (e.g. small quakes) silently slip into "noise" windows undetected.
        eq_batches (list): List of tuples specifying earthquake windows
            (batch_name, start_offset, end_offset).
        noise_batches (list): List of tuples specifying noise windows
            (batch_name, start_offset, end_offset).
        output_base_dir (Path): The root directory where output folders will be created.
        client (obspy.clients.fdsn.Client): The initialized FDSN client object.
        contamination_buffer_sec (int): Seconds of padding around any other
            cataloged event's origin time to treat as potentially contaminated.

    Returns:
        str: A formatted string containing the execution logs and status of the downloads.
    """
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

                # Contamination check -- run against the FULL unfiltered catalog,
                # so events excluded from the download list (small quakes, unrated
                # types, etc.) still count as potential contamination.
                buffer = contamination_buffer_sec
                overlapping_events = df_collision_check[
                    (df_collision_check["ParsedUTC"] >= (starttime - buffer))
                    & (df_collision_check["ParsedUTC"] <= (endtime + buffer))
                ]

                if not overlapping_events.empty:
                    results.append(
                        f"[skipped] [Noise-{batch_name}] Event {event_id} overlaps with "
                        f"{len(overlapping_events)} other cataloged event(s) "
                        f"(checked against full catalog)."
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


def _parse_catalog_times(df: pd.DataFrame) -> pd.DataFrame:
    """Parses the 'Date' column into a 'ParsedUTC' column of UTCDateTime objects."""
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
    df = df.copy()
    df["ParsedUTC"] = catalog_times
    return df


def download_all_concurrent(
    catalog_path: Path,
    output_base_dir: Path,
    eq_batches: list[tuple[str, int, int]],
    noise_batches: list[tuple[str, int, int]],
    full_catalog_path: Path | None = None,
    file_limit: int | None = None,
    max_workers: int = 4,
    contamination_buffer_sec: int = NOISE_CONTAMINATION_BUFFER_SEC,
) -> None:
    """Executes concurrent downloads for earthquake and noise waveforms.

    Parses the event catalog to datetime objects, subsets the dataframe if a
    limit is provided, and distributes the extraction process across a thread pool.

    Args:
        catalog_path (Path): File path to the (typically filtered) CSV catalog
            that determines WHICH events get downloaded.
        output_base_dir (Path): Root directory for saving generated MiniSEED files.
        eq_batches (list): Earthquake time windows to extract (name, start, end).
        noise_batches (list): Ambient noise time windows to extract (name, start, end).
        full_catalog_path (Path | None, optional): Path to the FULL, unfiltered
            catalog CSV, used only for noise-window collision detection. If not
            provided, falls back to using `catalog_path` for collision checks
            too (same behavior as before -- not recommended, since filtered-out
            small events would then be invisible to the contamination check).
        file_limit (int | None, optional): Maximum number of events to process.
            Defaults to None (processes the entire catalog).
        max_workers (int, optional): Number of concurrent threads to use. Defaults to 4.
        contamination_buffer_sec (int, optional): Seconds of padding around any
            other cataloged event to treat as potentially contaminated.

    Raises:
        FileNotFoundError: If the specified catalog_path does not exist.
    """
    catalog_path = Path(catalog_path)
    output_base_dir = Path(output_base_dir)

    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog file not found: {catalog_path}")

    print(f"[init] Loading download catalog: {catalog_path.resolve()}")
    df = pd.read_csv(catalog_path)
    df = _parse_catalog_times(df)

    if full_catalog_path is not None:
        full_catalog_path = Path(full_catalog_path)
        if not full_catalog_path.exists():
            raise FileNotFoundError(f"Full catalog file not found: {full_catalog_path}")
        print(f"[init] Loading full catalog for collision detection: {full_catalog_path.resolve()}")
        df_collision_check = pd.read_csv(full_catalog_path)
        df_collision_check = _parse_catalog_times(df_collision_check)
    else:
        print(
            "[warning] No full_catalog_path provided -- collision detection will "
            "only check against the (filtered) download catalog. Events filtered "
            "out of that catalog (e.g. small quakes) can silently contaminate "
            "noise windows."
        )
        df_collision_check = df

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
                df_collision_check,
                eq_batches,
                noise_batches,
                output_base_dir,
                client,
                contamination_buffer_sec,
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
        full_catalog_path=FULL_CATALOG_FILE,
        file_limit=FILE_LIMIT,
        max_workers=MAX_WORKERS,
    )
