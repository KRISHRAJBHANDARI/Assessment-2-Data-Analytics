# =============================================================================
# 01_load_data.py
# PRT564 Data Analytics and Visualisation — Assessment 2
#
# Purpose:
#   ETL Step 1 — Extract and integrate both data sources:
#     1. Opal Trip Counts CSV (data.gov.au)
#     2. ABS NSW Population Excel (3101051.xlsx)
#   Merges the two heterogeneous datasets on year and saves the result
#   to outputs/merged_opal_population.csv for use by downstream scripts.
#
#   Heterogeneous data justification:
#     The Opal dataset is a transactional CSV (tap-on counts, monthly).
#     The ABS dataset is a government statistical release in Excel format
#     with a complex multi-sheet, multi-row-header structure.
#     Combining them requires format-specific extraction logic for each source,
#     demonstrating integration of genuinely heterogeneous data.
#
# Author: Nasla Maharjan (S398425)
# =============================================================================

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path so config.py is always importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def load_opal_data():
    """
    Load and perform initial standardisation on the raw Opal trip CSV.

    Steps:
      - Read CSV from config path
      - Strip whitespace and apply title-case to Travel_Mode column
        (raw data has inconsistent casing e.g. 'train' vs 'Train')
      - Parse Year_Month string ('Jul-2016') into a proper datetime column

    Returns:
        pd.DataFrame: Cleaned raw Opal data with a 'Date' column added.
    """
    print("Loading Opal trip data...")
    opal_raw = pd.read_csv(config.OPAL_CSV)

    print(f"  Records loaded:   {opal_raw.shape[0]:,}")
    print(f"  Columns:          {list(opal_raw.columns)}")
    print(f"  Date range:       {opal_raw['Year_Month'].min()} to {opal_raw['Year_Month'].max()}")
    print(f"  Raw travel modes: {sorted(opal_raw['Travel_Mode'].unique())}")

    # Standardise mode names — remove leading/trailing whitespace and apply
    # consistent title case so 'train', 'Train', 'TRAIN' all become 'Train'.
    # This is essential before groupby aggregation.
    opal_raw["Travel_Mode"] = opal_raw["Travel_Mode"].str.strip().str.title()
    print(f"  Standardised modes: {sorted(opal_raw['Travel_Mode'].unique())}")

    # Parse date string format 'Jul-2016' into datetime
    opal_raw["Date"] = pd.to_datetime(opal_raw["Year_Month"], format="%b-%Y")

    return opal_raw


def aggregate_and_pivot(opal_raw):
    """
    Aggregate trip counts and pivot into a wide monthly time-series.

    Steps:
      - Sum trip counts across all card types (Adult, Concession, etc.)
        per month per transport mode
      - Pivot so each mode becomes a column
      - Add Total_Trips column as row sum
      - Fill NaN with 0 (month-mode combinations with no trips)

    Args:
        opal_raw (pd.DataFrame): Standardised raw Opal data.

    Returns:
        pd.DataFrame: Wide-format monthly data with one row per month.
    """
    print("\nAggregating trips by month and mode...")

    # Sum across all card types — we care about total trips per mode per month,
    # not the breakdown by card type at this stage.
    opal_monthly = (
        opal_raw.groupby(["Date", "Travel_Mode"])["Trip"]
        .sum()
        .reset_index()
        .sort_values("Date")
        .reset_index(drop=True)
    )
    print(f"  Month-mode combinations: {len(opal_monthly)}")

    # Pivot: Date as index, each mode as a column
    opal_pivot = (
        opal_monthly.pivot(index="Date", columns="Travel_Mode", values="Trip")
        .fillna(0)                    # Months with no trips for a mode → 0
        .reset_index()
        .sort_values("Date")
    )

    # Total trips across all modes
    mode_cols = [c for c in opal_pivot.columns if c in config.TRANSPORT_MODES + ["Unallocated"]]
    opal_pivot["Total_Trips"] = opal_pivot[mode_cols].sum(axis=1)
    opal_pivot["Year"] = opal_pivot["Date"].dt.year

    print(f"  Monthly records:         {len(opal_pivot)}")
    print(f"  Columns: {list(opal_pivot.columns)}")

    return opal_pivot


def load_population_data():
    """
    Extract NSW total population from ABS 3101051.xlsx.

    The ABS Excel file has a complex structure:
      - Sheet Data1: male population by age group, rows 1-10 are headers,
        data starts at row 11 (0-indexed: row 10).
      - Sheet Data2: female population by age group, same structure.
      - Column 0 of each sheet contains the date.
      - Remaining columns contain age-group-specific counts.

    We sum all age groups for males and females separately, then add them
    to get total NSW population per year.

    Returns:
        pd.DataFrame: Annual population data with columns ['Date', 'Total_Population', 'Year']
    """
    print("\nLoading ABS population data...")

    data1 = pd.read_excel(
        config.POPULATION_XLSX,
        sheet_name=config.ABS_SHEET_MALE,
        header=None
    )
    data2 = pd.read_excel(
        config.POPULATION_XLSX,
        sheet_name=config.ABS_SHEET_FEMALE,
        header=None
    )
    print(f"  Data1 (male) shape:   {data1.shape}")
    print(f"  Data2 (female) shape: {data2.shape}")

    # Extract dates from column 0, starting from row ABS_DATA_START_ROW
    dates = pd.to_datetime(
        data1.iloc[config.ABS_DATA_START_ROW:, 0].reset_index(drop=True)
    )

    # Sum all age group columns for each sex
    male_data   = data1.iloc[config.ABS_DATA_START_ROW:, 1:].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    female_data = data2.iloc[config.ABS_DATA_START_ROW:, 1:].apply(pd.to_numeric, errors="coerce").sum(axis=1)
    pop_total   = male_data + female_data

    population_data = pd.DataFrame({
        "Date":             dates.values,
        "Total_Population": pop_total.values,
    }).sort_values("Date").reset_index(drop=True)

    population_data["Year"] = population_data["Date"].dt.year

    print(f"  Population records: {len(population_data)}")
    print(f"  Date range: {population_data['Date'].min().date()} to {population_data['Date'].max().date()}")
    print(f"  Population range: {population_data['Total_Population'].min():,.0f} "
          f"to {population_data['Total_Population'].max():,.0f}")

    return population_data


def merge_datasets(opal_pivot, population_data):
    """
    Merge monthly Opal data with annual ABS population data.

    Merge strategy: left join on Year.
    Limitation acknowledged: population data is annual, so each monthly Opal
    row receives the same population value for its calendar year. This assumes
    population is stable within each year — a known simplification.
    Quarterly ABS data would reduce this approximation in future work.

    Args:
        opal_pivot (pd.DataFrame): Wide-format monthly Opal data.
        population_data (pd.DataFrame): Annual population data.

    Returns:
        pd.DataFrame: Merged dataset with both Opal and population columns.
    """
    print("\nMerging Opal and population datasets...")

    # Use annual population values only (one row per year)
    pop_by_year = (
        population_data[["Year", "Total_Population"]]
        .drop_duplicates(subset="Year")
    )

    merged = (
        opal_pivot
        .merge(pop_by_year, on="Year", how="left")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Remove rows where population could not be matched
    # (years outside ABS data range — typically partial years at boundaries)
    merged_clean = merged[merged["Total_Population"].notna()].copy()

    missing = merged_clean.isnull().sum().sum()
    print(f"  Merged rows:     {len(merged_clean)}")
    print(f"  Date range:      {merged_clean['Date'].min().date()} to {merged_clean['Date'].max().date()}")
    print(f"  Time span:       {(merged_clean['Date'].max() - merged_clean['Date'].min()).days / 365:.1f} years")
    print(f"  Missing values:  {missing}")

    if missing > 0:
        print(f"  WARNING: {missing} missing values detected — check ABS date coverage.")

    return merged_clean


def main():
    """
    Run the full ETL pipeline:
      1. Load Opal CSV
      2. Aggregate and pivot
      3. Load ABS population Excel
      4. Merge on year
      5. Save merged dataset to outputs/

    This function is called directly when running:
        python 01_load_data.py
    It is also imported by run_all.py and 02_preprocess.py.
    """
    print("=" * 60)
    print("STEP 1: DATA LOADING & INTEGRATION (ETL)")
    print("=" * 60)

    opal_raw       = load_opal_data()
    opal_pivot     = aggregate_and_pivot(opal_raw)
    population_data = load_population_data()
    merged_clean   = merge_datasets(opal_pivot, population_data)

    # Save merged dataset for use by 02_preprocess.py
    merged_clean.to_csv(config.MERGED_CSV, index=False)
    print(f"\n  Saved: {config.MERGED_CSV}")
    print("\nStep 1 complete.\n")

    return merged_clean


if __name__ == "__main__":
    main()
