from pathlib import Path

import pandas as pd


def extract_earthquakes(
    catalog_path: str | Path,
    output_path: str | Path = "filtered_earthquakes.csv",
) -> Path:
  """Load catalog CSV and extract only valid earthquake events (filtering out

  non-tectonic events like blasts, quarry shocks, or noise based on 'Type').
  """
  catalog_path = Path(catalog_path)
  output_path = Path(output_path)

  # 1. Load the catalog CSV
  print(f"[load] Reading catalog from: {catalog_path}")
  df = pd.read_csv(catalog_path)

  # 2. Inspect available types (optional diagnostic)
  if "Type" in df.columns:
    print(f"[info] Event types found in catalog: {df['Type'].unique().tolist()}")

    # Filter out non-earthquake types if applicable (e.g., keeping standard magnitude types)
    # Typically, valid earthquake types are magnitude indicators like 'ML', 'MW', 'MS', 'Mb'
    valid_types = ["ML", "MW", "MS", "Mb", "MD", "mw", "ml", "ms", "mb", "md"]
    eq_df = df[df["Type"].isin(valid_types)].copy()
  else:
    print("[warning] 'Type' column not found. Returning full dataframe.")
    eq_df = df.copy()

  # 3. Save the filtered earthquake catalog
  output_path.parent.mkdir(parents=True, exist_ok=True)
  eq_df.to_csv(output_path, index=False, encoding="utf-8-sig")

  print(f"[success] Extracted {len(eq_df)} earthquakes out of {len(df)} total events.")
  print(f"[saved] Filtered catalog saved to: {output_path}")

  return output_path


if __name__ == "__main__":
  # Example usage:
  # Replace 'catalog.csv' with your actual input file path
  input_csv = "deprem_katalog_utc.csv"
  extract_earthquakes(input_csv, "extracted_earthquakes.csv")
