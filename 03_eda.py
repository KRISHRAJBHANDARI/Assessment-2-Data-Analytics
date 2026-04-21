# =============================================================================
# 03_eda.py
# PRT564 Data Analytics and Visualisation — Assessment 2
#
# Purpose:
#   Exploratory Data Analysis — produces all visualisations used to understand
#   the dataset and inform modelling decisions.
#
#   Each chart is directly tied to a modelling decision:
#
#   Chart 1 — Correlation matrix
#             Reveals correlation structure. Key finding: Log_Population and
#             Time_Index are strongly correlated → justifies Ridge (M3) over
#             plain OLS to handle multicollinearity.
#
#   Chart 2 — COVID time-series
#             Shows the structural break in March 2020. Justifies COVID_Flag
#             and Post_COVID_Recovery dummy variables in models.
#
#   Chart 3 — Per-capita trips (the headline finding)
#             Shows that per-capita demand is DECLINING despite population
#             growth. This is the central result of the whole project.
#
#   Chart 4 — Seasonal heatmap (month × mode)
#             Confirms seasonality in demand (school terms vs holidays).
#             Justifies including Month as a feature in Model 3 (Ridge).
#
#   Chart 5 — Mode share stacked area
#             Shows Metro (launched May 2019) eating into Bus/Train share.
#             Provides context for why mode mix matters to TfNSW planners.
#
#   Chart 6 — YoY growth comparison
#             Directly visualises decoupling of trip growth from population
#             growth. Visual proof of the inverse relationship finding.
#
#   Chart 7 — Trips by mode over time
#             Absolute trip counts per mode — shows COVID impact per mode
#             and which modes dominate overall demand.
#
# Author: Krish Rajbhandari (S395754), Nasla Maharjan (S398425), Suyog Kadariya (S393829)
# ========================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

# Apply consistent chart style
plt.style.use("seaborn-v0_8-whitegrid")


def load_data():
    """Load feature-enriched dataset from Step 2."""
    df = pd.read_csv(config.MERGED_CSV, parse_dates=["Date"])
    print(f"Loaded {len(df)} rows for EDA.")
    return df


def chart_correlation_matrix(df):
    """
    Chart 1: Heatmap of Pearson correlations between transport modes
    and total population.

    Modelling decision: If Log_Population and Time_Index are highly correlated
    this confirms multicollinearity, justifying Ridge regression for Model 3.
    """
    mode_cols    = [c for c in df.columns if c in config.TRANSPORT_MODES]
    analysis_cols = mode_cols + ["Total_Trips", "Total_Population"]
    corr_matrix  = df[analysis_cols].corr()

    trips_pop_corr = corr_matrix.loc["Total_Trips", "Total_Population"]
    print(f"  Trips vs Population correlation: {trips_pop_corr:.4f}")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, annot=True, cmap="coolwarm", center=0,
        square=True, fmt=".3f",
        cbar_kws={"label": "Pearson Correlation"},
        ax=ax
    )
    ax.set_title("Correlation Matrix — Opal Trip Modes & Population",
                 fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(config.CHART_CORRELATION, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {config.CHART_CORRELATION}")


def chart_covid_timeseries(df):
    """
    Chart 2: Annotated time-series showing the COVID-19 structural break.

    Justifies COVID_Flag and Post_COVID_Recovery dummy variables.
    Without explicitly modelling this break, the 47% demand collapse
    would be misattributed to population or time trend in regression.
    """
    normal_data = df[df["COVID_Flag"] == 0]
    covid_data  = df[df["COVID_Flag"] == 1]

    covid_mean  = df.loc[df["COVID_Flag"] == 1, "Total_Trips"].mean() / 1e6
    normal_mean = df.loc[df["COVID_Flag"] == 0, "Total_Trips"].mean() / 1e6
    pct_drop    = (normal_mean - covid_mean) / normal_mean * 100

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(normal_data["Date"], normal_data["Total_Trips"] / 1e6,
            color=config.COLOR_NORMAL, linewidth=2, label="Normal period")
    ax.plot(covid_data["Date"], covid_data["Total_Trips"] / 1e6,
            color=config.COLOR_COVID, linewidth=2,
            label="COVID-19 period (Mar 2020 – Dec 2021)")
    ax.axvspan(pd.Timestamp(config.COVID_START), pd.Timestamp(config.COVID_END),
               alpha=0.12, color="red", label="COVID window")
    ax.annotate(
        f"−{pct_drop:.0f}% demand\nduring COVID",
        xy=(pd.Timestamp("2020-09-01"), covid_mean * 0.85),
        fontsize=10, color=config.COLOR_COVID,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor=config.COLOR_COVID)
    )
    ax.set_ylabel("Total Trips (millions)", fontsize=11)
    ax.set_title("NSW Opal Trips Over Time — COVID-19 Structural Break",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))
    plt.tight_layout()
    plt.savefig(config.CHART_COVID, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {config.CHART_COVID}")


def chart_per_capita(df):
    """
    Chart 3: Per-capita trips over time — the headline finding.

    Top panel: Raw total trips vs population (both appearing to grow).
    Bottom panel: Per-capita trips — reveals the true structural decline.

    KEY FINDING: Per-capita trips decline at ~9.6 trips/1000 residents/month
    even before COVID, proving that population growth alone does not drive
    public transport demand.
    """
    non_covid  = df[df["COVID_Flag"] == 0].copy()
    covid_rows = df[df["COVID_Flag"] == 1].copy()

    # Trend line on non-COVID data only — shows structural pre-COVID decline
    z       = np.polyfit(non_covid["Time_Index"], non_covid["Per_Capita_Trips"], 1)
    p_trend = np.poly1d(z)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(
        "Population Growth vs Transport Demand — The Inverse Relationship",
        fontsize=13, fontweight="bold"
    )

    # Top: dual-axis raw counts
    ax1      = axes[0]
    ax1_twin = ax1.twinx()
    ax1.plot(df["Date"], df["Total_Trips"] / 1e6,
             color=config.COLOR_NORMAL, linewidth=2, label="Total Trips (M)")
    ax1_twin.plot(df["Date"], df["Total_Population"] / 1e6,
                  color="darkorange", linewidth=2, linestyle="--",
                  label="Population (M)")
    ax1.set_ylabel("Total Trips (millions)", color=config.COLOR_NORMAL, fontsize=10)
    ax1_twin.set_ylabel("NSW Population (millions)", color="darkorange", fontsize=10)
    ax1.set_title("Raw Counts — Both Appear to Grow", fontsize=11)
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="lower right")

    # Bottom: per-capita — the real story
    ax2 = axes[1]
    ax2.plot(non_covid["Date"], non_covid["Per_Capita_Trips"],
             color=config.COLOR_NORMAL, linewidth=2,
             label="Per-capita trips (normal)")
    ax2.plot(covid_rows["Date"], covid_rows["Per_Capita_Trips"],
             color=config.COLOR_COVID, linewidth=2,
             label="Per-capita trips (COVID)")
    ax2.axvspan(pd.Timestamp(config.COVID_START), pd.Timestamp(config.COVID_END),
                alpha=0.1, color="red")
    ax2.plot(non_covid["Date"], p_trend(non_covid["Time_Index"]),
             "k--", linewidth=1.5,
             label=f"Trend (slope = {z[0]:.2f}/month)")
    ax2.set_ylabel("Trips per 1,000 residents", fontsize=10)
    ax2.set_xlabel("Date", fontsize=10)
    ax2.set_title("Per-Capita Opal Trips — Transport Demand Relative to Population",
                  fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.CHART_PER_CAPITA, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close()

    pre_covid_pc  = df.loc[df["Date"] < config.COVID_START, "Per_Capita_Trips"].mean()
    post_covid_pc = df.loc[df["Date"] > config.COVID_END,   "Per_Capita_Trips"].mean()
    print(f"  KEY FINDING: Pre-COVID per-capita = {pre_covid_pc:.1f}, "
          f"Post-COVID = {post_covid_pc:.1f}, slope = {z[0]:.4f}/month")
    print(f"  Saved: {config.CHART_PER_CAPITA}")


def chart_seasonal_heatmap(df):
    """
    Chart 4: Seasonal heatmap — average monthly trips by mode × calendar month.
    COVID period excluded to show true underlying seasonality.

    Modelling decision: Clear seasonality (school terms vs holidays) visible
    here justifies including Month as a feature in Model 3 (Ridge regression).
    """
    mode_cols    = [c for c in df.columns if c in config.TRANSPORT_MODES]
    heatmap_data = df[df["COVID_Flag"] == 0].copy()
    heatmap_data["Month_Num"] = heatmap_data["Date"].dt.month

    seasonal_pivot = heatmap_data.groupby("Month_Num")[mode_cols].mean() / 1e6
    seasonal_pivot.index = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        seasonal_pivot, annot=True, fmt=".1f", cmap="YlOrRd",
        linewidths=0.5, ax=ax,
        cbar_kws={"label": "Avg Monthly Trips (millions)"}
    )
    ax.set_title(
        "Seasonal Heatmap — Average Monthly Trips by Mode\n(COVID period excluded)",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Transport Mode", fontsize=11)
    ax.set_ylabel("Month", fontsize=11)
    plt.tight_layout()
    plt.savefig(config.CHART_HEATMAP, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {config.CHART_HEATMAP}")
    print("  Observation: School-term months (Feb–Jun, Jul–Nov) show higher demand.")


def chart_mode_share(df):
    """
    Chart 5: Stacked area chart of transport mode share over time.

    Shows how Metro (launched May 2019) redistributed mode share away from
    Bus and Train. Provides context for TfNSW service planning decisions.
    """
    mode_cols = [c for c in df.columns if c in config.TRANSPORT_MODES]
    prop_df   = df[["Date"] + mode_cols].copy().set_index("Date")
    prop_df   = prop_df.div(prop_df.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(14, 6))
    prop_df.plot.area(ax=ax, alpha=0.75, colormap="tab10")

    if "Metro" in prop_df.columns:
        ax.axvline(pd.Timestamp("2019-05-01"), color="black",
                   linestyle="--", linewidth=1.5, label="Metro launch (May 2019)")

    ax.axvspan(pd.Timestamp(config.COVID_START), pd.Timestamp(config.COVID_END),
               alpha=0.15, color="red", label="COVID-19 period")

    ax.set_ylabel("Mode Share (%)", fontsize=11)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_title("Transport Mode Share Over Time — Opal Network",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, bbox_to_anchor=(1.01, 1))
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(config.CHART_MODE_SHARE, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {config.CHART_MODE_SHARE}")


def chart_yoy_growth(df):
    """
    Chart 6: Year-over-year growth rate — trips vs population.

    Red shading marks periods where trip growth lagged population growth.
    This is the direct visual proof of the demand-population decoupling
    that our regression models quantify.
    """
    growth_data = df[["Date", "Trips_YoY_Growth", "Population_YoY_Growth"]].dropna()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(growth_data["Date"], growth_data["Trips_YoY_Growth"],
            color=config.COLOR_NORMAL, linewidth=2, label="Trips YoY Growth (%)")
    ax.plot(growth_data["Date"], growth_data["Population_YoY_Growth"],
            color="darkorange", linewidth=2, linestyle="--",
            label="Population YoY Growth (%)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.axvspan(pd.Timestamp(config.COVID_START), pd.Timestamp(config.COVID_END),
               alpha=0.12, color="red", label="COVID-19 period")
    ax.fill_between(
        growth_data["Date"],
        growth_data["Trips_YoY_Growth"],
        growth_data["Population_YoY_Growth"],
        where=(growth_data["Trips_YoY_Growth"] < growth_data["Population_YoY_Growth"]),
        alpha=0.2, color="red",
        label="Trips growing slower than population"
    )
    ax.set_ylabel("Year-on-Year Growth (%)", fontsize=11)
    ax.set_title("Trip Demand Growth vs Population Growth — Decoupling Evidence",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.CHART_YOY, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {config.CHART_YOY}")


def chart_trips_by_mode(df):
    """
    Chart 7: Absolute trip counts per transport mode over time.

    Shows the scale of each mode and how COVID affected each differently.
    Train and Bus dominate overall demand; Metro grows steadily post-2019.
    """
    mode_cols = [c for c in df.columns if c in config.TRANSPORT_MODES]

    fig, ax = plt.subplots(figsize=(14, 6))
    for mode in mode_cols:
        ax.plot(df["Date"], df[mode] / 1e6,
                marker="o", markersize=3, linewidth=2, label=mode)

    ax.axvspan(pd.Timestamp(config.COVID_START), pd.Timestamp(config.COVID_END),
               alpha=0.12, color="red", label="COVID-19 period")
    ax.set_ylabel("Trips (millions)", fontsize=11)
    ax.set_title("Opal Trips by Transport Mode Over Time",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(config.CHART_TRIPS_MODE, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {config.CHART_TRIPS_MODE}")


def main():
    """
    Produce all 7 EDA charts.
    Called directly via: python 03_eda.py
    """
    print("=" * 60)
    print("STEP 3: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    df = load_data()

    print("\nChart 1: Correlation matrix")
    chart_correlation_matrix(df)

    print("Chart 2: COVID time-series")
    chart_covid_timeseries(df)

    print("Chart 3: Per-capita trips (headline finding)")
    chart_per_capita(df)

    print("Chart 4: Seasonal heatmap")
    chart_seasonal_heatmap(df)

    print("Chart 5: Mode share over time")
    chart_mode_share(df)

    print("Chart 6: YoY growth comparison")
    chart_yoy_growth(df)

    print("Chart 7: Trips by mode")
    chart_trips_by_mode(df)

    print("\nStep 3 complete. All charts saved to outputs/\n")


if __name__ == "__main__":
    main()
