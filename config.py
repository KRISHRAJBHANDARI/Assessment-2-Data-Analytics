# =============================================================================
# config.py
# PRT564 Data Analytics and Visualisation — Assessment 2
# Project: Predictive Modelling of NSW Opal Trip Demand
#
# Purpose:
#   Central configuration file. All file paths, constants, and shared
#   settings are defined here so no script contains hardcoded values.
#   Change paths here once if you move data files — nothing else needs editing.
#
# Authors:
#   Nasla Maharjan    (S398425) — ETL, preprocessing
#   Krish Rajbhandari (S395754) — modelling, evaluation
#   Suyog Kadariya    (S393829) — EDA, findings, presentation
#
# Unit:    PRT564 | Campus: Sydney | Year: 2026
# GitHub:  https://github.com/KRISHRAJBHANDARI/Assessment-2---PRT-564
# =============================================================================

import os

# ── BASE DIRECTORIES ──────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# ── INPUT FILE PATHS ──────────────────────────────────────────────────────────
# Place both source files in the data/ folder before running.
OPAL_CSV        = os.path.join(DATA_DIR, "opal_trips.csv")
POPULATION_XLSX = os.path.join(DATA_DIR, "3101051.xlsx")

# ── OUTPUT FILE PATHS ─────────────────────────────────────────────────────────
# Intermediate datasets (written by 01/02, read by later scripts)
MERGED_CSV          = os.path.join(OUTPUTS_DIR, "merged_opal_population.csv")
REGRESSION_CSV      = os.path.join(OUTPUTS_DIR, "regression_full_dataset.csv")
MODEL_RESULTS_CSV   = os.path.join(OUTPUTS_DIR, "model_comparison_results.csv")

# Chart output paths
CHART_CORRELATION   = os.path.join(OUTPUTS_DIR, "correlation_matrix.png")
CHART_COVID         = os.path.join(OUTPUTS_DIR, "covid_timeseries.png")
CHART_PER_CAPITA    = os.path.join(OUTPUTS_DIR, "per_capita_trips.png")
CHART_HEATMAP       = os.path.join(OUTPUTS_DIR, "seasonal_heatmap.png")
CHART_MODE_SHARE    = os.path.join(OUTPUTS_DIR, "mode_share.png")
CHART_YOY           = os.path.join(OUTPUTS_DIR, "yoy_growth_comparison.png")
CHART_TRIPS_MODE    = os.path.join(OUTPUTS_DIR, "trips_by_mode.png")
CHART_MODEL_COMPARE = os.path.join(OUTPUTS_DIR, "model_comparison.png")
CHART_RESIDUALS     = os.path.join(OUTPUTS_DIR, "residual_diagnostics_all.png")

# ── ABS EXCEL SHEET SETTINGS ──────────────────────────────────────────────────
# ABS 3101051.xlsx has male age data in Data1, female in Data2.
# Data rows start at Excel row 11 (0-indexed row 10).
ABS_SHEET_MALE    = "Data1"
ABS_SHEET_FEMALE  = "Data2"
ABS_DATA_START_ROW = 10   # iloc row index where actual data begins

# ── COVID PERIOD DEFINITIONS ─────────────────────────────────────────────────
# Based on NSW public health orders and mobility restriction periods.
COVID_START    = "2020-03-01"
COVID_END      = "2021-12-31"
RECOVERY_START = "2022-01-01"

# ── TRANSPORT MODES OF INTEREST ───────────────────────────────────────────────
TRANSPORT_MODES = ["Bus", "Train", "Light Rail", "Ferry", "Metro"]

# ── REGRESSION FEATURE SETS ───────────────────────────────────────────────────
# Features used for each model — defined centrally so 04_models.py and
# 05_evaluate.py always use identical feature sets.
FEATURES_M1 = ["Log_Population"]

FEATURES_M2 = ["Log_Population", "Time_Index",
                "Population_YoY_Growth", "COVID_Flag"]

FEATURES_M3 = ["Log_Population", "Time_Index",
                "Population_YoY_Growth", "COVID_Flag",
                "Post_COVID_Recovery", "Month"]

TARGET_COL = "Log_Total_Trips"

# ── RIDGE REGRESSION SETTINGS ─────────────────────────────────────────────────
RIDGE_ALPHAS    = (1e-3, 1e3, 100)   # (min, max, n_values) for np.logspace
RIDGE_CV_FOLDS  = 5
RIDGE_SCORING   = "r2"

# ── CHART STYLE ───────────────────────────────────────────────────────────────
CHART_DPI       = 150
CHART_STYLE     = "seaborn-v0_8-whitegrid"
COLOR_NORMAL    = "steelblue"
COLOR_COVID     = "crimson"
COLOR_GOLD      = "#E8A020"

# ── STATISTICAL TEST SETTINGS ─────────────────────────────────────────────────
ALPHA           = 0.05      # significance level for all hypothesis tests

# ── ENSURE OUTPUT DIRECTORY EXISTS ────────────────────────────────────────────
os.makedirs(OUTPUTS_DIR, exist_ok=True)
