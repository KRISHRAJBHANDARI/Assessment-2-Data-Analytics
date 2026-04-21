# =============================================================================
# 02_preprocess.py
# PRT564 Data Analytics and Visualisation — Assessment 2
#
# Purpose:
#   ETL Step 2 — Feature engineering and data preparation for regression.
#   Takes the merged dataset from 01_load_data.py and creates all derived
#   features needed for EDA and modelling.
#
#   Features created and their justifications:
#     Per_Capita_Trips       Trips / Population × 1000
#                            Normalises demand for fair comparison over time.
#                            Essential for detecting the inverse relationship.
#
#     Log_Total_Trips        ln(Total_Trips)
#     Log_Population         ln(Population)
#                            Log-log specification enables elasticity modelling.
#                            β₁ in log-log regression = % change in trips per
#                            1% change in population (standard in economics).
#
#     Population_YoY_Growth  % change over 12-month lag
#     Trips_YoY_Growth       % change over 12-month lag
#                            Annual growth rates used in EDA comparisons and
#                            as a predictor in Model 2/3.
#
#     Time_Index             0, 1, 2, ... n-1
#                            Linear time trend. Captures cumulative effects of
#                            infrastructure investment and service expansion
#                            that population alone cannot explain.
#
#     Month, Quarter, Season Temporal dummies extracted from Date.
#                            Month is used in Model 3 to capture seasonality
#                            confirmed by the seasonal heatmap in EDA.
#
#     COVID_Flag             1 = Mar 2020 – Dec 2021, else 0
#                            Binary dummy for COVID period.
#                            Without this flag, the 47% demand collapse is
#                            misattributed to population or time trend,
#                            corrupting all other coefficients.
#
#     Post_COVID_Recovery    1 = Jan 2022 onwards, else 0
#                            Captures the permanently lower post-COVID demand
#                            baseline. Without this flag, Time_Index becomes
#                            negative (later dates = lower trips, which is
#                            spurious). This flag separates three structural
#                            demand regimes: pre-COVID / COVID / recovery.
#
# Author: Nasla Maharjan (S398425)
# =============================================================================

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def load_merged_data():
    """
    Load the merged dataset produced by 01_load_data.py.
    Parses the Date column back to datetime (it is stored as string in CSV).

    Returns:
        pd.DataFrame: Merged Opal + population dataset.
    """
    print("Loading merged dataset from Step 1...")
    df = pd.read_csv(config.MERGED_CSV, parse_dates=["Date"])
    print(f"  Loaded {len(df)} rows, {df.shape[1]} columns.")
    return df


def engineer_features(df):
    """
    Create all derived features required for EDA and regression.

    This function adds columns to the dataframe in-place and returns
    the enriched dataframe. All feature rationale is documented in the
    module docstring above.

    Args:
        df (pd.DataFrame): Merged Opal + population data.

    Returns:
        pd.DataFrame: Feature-enriched dataframe.
    """
    print("\nEngineering features...")

    # ── PER-CAPITA DEMAND ────────────────────────────────────────────────────
    # Normalise total trips by population to detect structural demand changes
    # that raw counts would obscure (e.g. trips declining per person even as
    # total trips look stable because more people are moving to NSW).
    df["Per_Capita_Trips"] = (
        df["Total_Trips"] / df["Total_Population"] * 1000
    )

    # ── GROWTH RATES ─────────────────────────────────────────────────────────
    # Year-over-year (12-month lag) removes seasonal effects and captures
    # structural demand changes. Used in Model 2/3 as a predictor.
    df["Population_YoY_Growth"] = df["Total_Population"].pct_change(12) * 100
    df["Trips_YoY_Growth"]      = df["Total_Trips"].pct_change(12) * 100

    # Month-over-month changes — used in EDA time-series plots
    df["Trips_MoM_Change"]     = df["Total_Trips"].diff()
    df["Trips_MoM_Pct_Change"] = df["Total_Trips"].pct_change() * 100

    # ── LOG TRANSFORMATIONS ──────────────────────────────────────────────────
    # Log-log regression: ln(Trips) ~ β₀ + β₁·ln(Population)
    # Coefficient β₁ = demand elasticity with respect to population.
    # Log transform also stabilises variance in count data.
    df["Log_Total_Trips"] = np.log(df["Total_Trips"])
    df["Log_Population"]  = np.log(df["Total_Population"])

    # ── MODE PROPORTIONS AND PER-CAPITA BY MODE ───────────────────────────────
    # Used in EDA mode share charts to show how the transport mix has shifted.
    mode_cols = [c for c in df.columns if c in config.TRANSPORT_MODES]
    for mode in mode_cols:
        df[f"{mode}_Proportion"] = (df[mode] / df["Total_Trips"] * 100)
        df[f"{mode}_Per_Capita"] = (df[mode] / df["Total_Population"] * 1000)

    # ── TEMPORAL VARIABLES ───────────────────────────────────────────────────
    # Time_Index: sequential integer 0, 1, 2, ... n-1
    # Captures the linear time trend caused by cumulative infrastructure
    # investment, route expansion, and service frequency improvements.
    df["Time_Index"] = range(len(df))
    df["Month"]      = df["Date"].dt.month
    df["Quarter"]    = df["Date"].dt.quarter
    df["Season"]     = df["Date"].dt.month % 12 // 3 + 1

    # ── COVID STRUCTURAL PERIOD FLAGS ────────────────────────────────────────
    # Three structurally distinct demand periods exist in the data:
    #
    #   Pre-COVID  (before Mar 2020) — baseline growing demand
    #   COVID      (Mar 2020 – Dec 2021) — 47% demand collapse
    #   Recovery   (Jan 2022 onwards) — partial recovery, new lower baseline
    #
    # Using only one dummy treats Recovery the same as Pre-COVID.
    # That is statistically incorrect — post-COVID per-capita trips remain
    # permanently lower. Two dummies correctly capture all three regimes.

    df["COVID_Flag"] = (
        (df["Date"] >= config.COVID_START) &
        (df["Date"] <= config.COVID_END)
    ).astype(int)

    df["Post_COVID_Recovery"] = (
        df["Date"] > config.COVID_END
    ).astype(int)

    print(f"  Features created. Total columns: {df.shape[1]}")

    # Verify COVID flag counts
    pre   = df[(df["COVID_Flag"] == 0) & (df["Post_COVID_Recovery"] == 0)]
    covid = df[df["COVID_Flag"] == 1]
    post  = df[df["Post_COVID_Recovery"] == 1]
    print(f"\n  Structural periods:")
    print(f"    Pre-COVID:   {len(pre):3d} months  avg {pre['Total_Trips'].mean()/1e6:.1f}M trips/month")
    print(f"    COVID:       {len(covid):3d} months  avg {covid['Total_Trips'].mean()/1e6:.1f}M trips/month")
    print(f"    Recovery:    {len(post):3d} months  avg {post['Total_Trips'].mean()/1e6:.1f}M trips/month")

    pct_drop = (pre["Total_Trips"].mean() - covid["Total_Trips"].mean()) / pre["Total_Trips"].mean() * 100
    gap      = (pre["Total_Trips"].mean() - post["Total_Trips"].mean()) / 1e6
    print(f"\n    COVID demand reduction:  {pct_drop:.1f}%")
    print(f"    Recovery gap remaining:  {gap:.1f}M trips/month")

    return df


def build_regression_dataset(df):
    """
    Build the clean 102-observation regression dataset used by all three models.

    All models (M1, M2, M3) are trained on the SAME observations — this is
    essential for fair statistical comparison via paired tests in 05_evaluate.py.

    The dataset starts from July 2017 because the first 12 months are dropped
    by dropna() — YoY growth rates require a 12-month lag, so the first year
    of monthly data produces NaN values for those columns.

    Args:
        df (pd.DataFrame): Feature-enriched merged dataset.

    Returns:
        pd.DataFrame: Regression-ready dataset (102 observations).
    """
    print("\nBuilding regression dataset...")

    regression_cols = [
        "Date", "Log_Total_Trips", "Log_Population",
        "Time_Index", "Population_YoY_Growth",
        "COVID_Flag", "Post_COVID_Recovery", "Month"
    ]

    regression_data = (
        df[regression_cols]
        .dropna()
        .reset_index(drop=True)
    )

    print(f"  Observations: {len(regression_data)}")
    print(f"  Date range:   {regression_data['Date'].min().date()} to {regression_data['Date'].max().date()}")
    print(f"  Missing values: {regression_data.isnull().sum().sum()}")

    return regression_data


def print_summary_statistics(df):
    """
    Print key descriptive statistics to the console.
    Used as a data quality checkpoint before proceeding to EDA.

    Args:
        df (pd.DataFrame): Feature-enriched merged dataset.
    """
    print("\nKEY DESCRIPTIVE STATISTICS:")
    print(f"  Total trips range:      {df['Total_Trips'].min()/1e6:.2f}M – {df['Total_Trips'].max()/1e6:.2f}M")
    print(f"  Mean monthly trips:     {df['Total_Trips'].mean()/1e6:.2f}M")
    print(f"  Population range:       {df['Total_Population'].min():,.0f} – {df['Total_Population'].max():,.0f}")
    print(f"  Population growth:      {(df['Total_Population'].max() - df['Total_Population'].min()) / df['Total_Population'].min() * 100:.1f}%")

    pc_pre  = df.loc[df["Date"] < config.COVID_START, "Per_Capita_Trips"].mean()
    pc_post = df.loc[df["Date"] > config.COVID_END,   "Per_Capita_Trips"].mean()
    print(f"  Per-capita (pre-COVID): {pc_pre:.1f} trips/1000 residents/month")
    print(f"  Per-capita (post-COVID):{pc_post:.1f} trips/1000 residents/month")


def main():
    """
    Run the full preprocessing pipeline:
      1. Load merged data from Step 1
      2. Engineer all features
      3. Build regression dataset
      4. Print summary statistics
      5. Save both datasets to outputs/

    Called directly via: python 02_preprocess.py
    Also imported by run_all.py and downstream scripts.
    """
    print("=" * 60)
    print("STEP 2: PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 60)

    df              = load_merged_data()
    df              = engineer_features(df)
    regression_data = build_regression_dataset(df)

    print_summary_statistics(df)

    # Save enriched full dataset (used by EDA)
    df.to_csv(config.MERGED_CSV, index=False)
    print(f"\n  Updated: {config.MERGED_CSV}")

    # Save regression-ready dataset (used by models + evaluation)
    regression_data.to_csv(config.REGRESSION_CSV, index=False)
    print(f"  Saved:   {config.REGRESSION_CSV}")

    print("\nStep 2 complete.\n")

    return df, regression_data


if __name__ == "__main__":
    main()
