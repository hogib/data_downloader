from pathlib import Path
from typing import Optional

import pandas as pd


def extract_earthquakes(
    catalog_path: str | Path,
    output_path: str | Path = "filtered_earthquakes.csv",
    min_magnitude: Optional[float] = None,
    magnitude_column: str = "Magnitude",
) -> Path:
    """Load catalog CSV and extract only valid, sufficiently large earthquake
    events (filtering out non-tectonic events like blasts or quarry shocks
    based on 'Type', AND filtering out small events based on magnitude).

    Note: the 'Type' column (ML, MW, MS, Mb, ...) indicates which magnitude
    SCALE was used to rate an event -- it says nothing about how large the
    event actually was. A ML 1.2 and a MW 6.5 both have valid 'Type' values.
    Filtering on Type alone (as the original version of this function did)
    does NOT restrict the catalog to "larger events" -- it only drops rows
    with missing/unrated magnitude scales. The `min_magnitude` filter below
    is what actually does that job.
    """
    catalog_path = Path(catalog_path)
    output_path = Path(output_path)

    # 1. Load the catalog CSV
    print(f"[load] Reading catalog from: {catalog_path}")
    df = pd.read_csv(catalog_path)

    # 2. Filter by event Type (drops unrated/non-tectonic-scale rows)
    if "Type" in df.columns:
        print(f"[info] Event types found in catalog: {df['Type'].unique().tolist()}")
        valid_types = ["ML", "MW", "MS", "Mb", "MD", "mw", "ml", "ms", "mb", "md"]
        eq_df = df[df["Type"].isin(valid_types)].copy()
        print(f"[info] Type filter kept {len(eq_df)}/{len(df)} events.")
    else:
        print("[warning] 'Type' column not found. Skipping type filter.")
        eq_df = df.copy()

    # 3. Filter by magnitude VALUE -- this is the step that actually
    #    restricts the catalog to "larger events".
    if min_magnitude is not None:
        if magnitude_column in eq_df.columns:
            before = len(eq_df)
            eq_df[magnitude_column] = pd.to_numeric(eq_df[magnitude_column], errors="coerce")
            dropped_unparseable = eq_df[magnitude_column].isna().sum()
            eq_df = eq_df[eq_df[magnitude_column] >= min_magnitude].copy()
            print(
                f"[info] Magnitude filter (>= {min_magnitude}) kept "
                f"{len(eq_df)}/{before} events "
                f"({dropped_unparseable} had unparseable magnitude values and were dropped)."
            )
        else:
            print(
                f"[warning] magnitude_column='{magnitude_column}' not found in catalog "
                f"columns {df.columns.tolist()}. Skipping magnitude filter -- "
                f"double check the column name!"
            )
    else:
        print("[warning] No min_magnitude provided. Catalog is NOT filtered by event size.")

    # 4. Save the filtered earthquake catalog
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eq_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[success] Extracted {len(eq_df)} earthquakes out of {len(df)} total events.")
    print(f"[saved] Filtered catalog saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    # Replace 'catalog.csv' with your actual input file path
    input_csv = "catalogs/deprem_katalog_utc.csv"

    # Set this to whatever size threshold defines "larger events" for your use case.
    MIN_MAGNITUDE = 2.0

    extract_earthquakes(
        input_csv,
        "extracted_earthquakes.csv",
        min_magnitude=MIN_MAGNITUDE,
    )
