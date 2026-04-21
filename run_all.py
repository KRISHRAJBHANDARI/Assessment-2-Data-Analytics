# =============================================================================
# run_all.py
# PRT564 Data Analytics and Visualisation — Assessment 2
#
# Purpose:
#   Master pipeline script. Runs all six analysis steps in order.
#   This single script reproduces the entire analysis from raw data
#   to final findings, satisfying the HD reproducibility requirement.
#
#   Execution order:
#     Step 1 — 01_load_data.py    Load and merge Opal + ABS data
#     Step 2 — 02_preprocess.py   Feature engineering, COVID flags
#     Step 3 — 03_eda.py          Produce all 7 EDA charts
#     Step 4 — 04_models.py       Fit M1, M2, M3 regression models
#     Step 5 — 05_evaluate.py     Statistical tests + evaluation charts
#     Step 6 — 06_findings.py     Print findings and non-technical summary
#
# Usage:
#   python run_all.py
#
# Prerequisites:
#   1. Install dependencies:  pip install -r requirements.txt
#   2. Place data files in data/ folder:
#        data/opal_trips.csv
#        data/3101051.xlsx
#
# Output:
#   All charts and CSVs are saved to outputs/ automatically.
#
# Authors:
#   Nasla Maharjan    (S398425) — Steps 1, 2
#   Suyog Kadariya    (S393829) — Steps 3, 6
#   Krish Rajbhandari (S395754) — Steps 4, 5
# =============================================================================

import sys
import os
import time

# Ensure all scripts can import config from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def check_prerequisites():
    """
    Verify that required input files exist before running the pipeline.
    Raises a clear error message if either file is missing.
    """
    missing = []
    if not os.path.exists(config.OPAL_CSV):
        missing.append(f"  ✗ {config.OPAL_CSV}")
    if not os.path.exists(config.POPULATION_XLSX):
        missing.append(f"  ✗ {config.POPULATION_XLSX}")

    if missing:
        print("ERROR: Required data files not found:")
        for f in missing:
            print(f)
        print("\nPlease place both files in the data/ folder and try again.")
        print("  - opal_trips.csv    (from data.gov.au)")
        print("  - 3101051.xlsx      (from Australian Bureau of Statistics)")
        sys.exit(1)

    print("✓ Data files found.")


def run_step(step_num, description, module_name):
    """
    Import and execute a pipeline step module.

    Args:
        step_num     (int): Step number for display.
        description  (str): Human-readable step description.
        module_name  (str): Module filename without .py extension.

    Returns:
        Any: Whatever the step's main() function returns.
    """
    print(f"\n{'='*60}")
    print(f"RUNNING STEP {step_num}: {description}")
    print(f"{'='*60}")

    start = time.time()

    # Dynamically import the module and run its main() function
    import importlib
    mod    = importlib.import_module(module_name)
    result = mod.main()

    elapsed = time.time() - start
    print(f"✓ Step {step_num} completed in {elapsed:.1f}s")

    return result


def main():
    """
    Run the full analysis pipeline end-to-end.
    """
    total_start = time.time()

    print("=" * 60)
    print("PRT564 ASSESSMENT 2 — OPAL TRIP DEMAND ANALYSIS")
    print("Full reproducible pipeline")
    print("=" * 60)
    print(f"\nProject directory: {config.BASE_DIR}")
    print(f"Outputs will be saved to: {config.OUTPUTS_DIR}\n")

    # Check data files exist before starting
    check_prerequisites()

    # ── STEP 1: ETL — Load and integrate data sources ────────────────────────
    merged_clean = run_step(
        1, "Data Loading & Integration (ETL)",
        "01_load_data"
    )

    # ── STEP 2: Preprocessing & Feature Engineering ──────────────────────────
    merged_enriched, regression_data = run_step(
        2, "Preprocessing & Feature Engineering",
        "02_preprocess"
    )

    # ── STEP 3: Exploratory Data Analysis ────────────────────────────────────
    run_step(
        3, "Exploratory Data Analysis (7 charts)",
        "03_eda"
    )

    # ── STEP 4: Regression Modelling ─────────────────────────────────────────
    df, y, m1, m2, m3 = run_step(
        4, "Regression Modelling (M1, M2, M3)",
        "04_models"
    )

    # ── STEP 5: Statistical Evaluation ───────────────────────────────────────
    # Run tests directly using results from step 4
    print(f"\n{'='*60}")
    print("RUNNING STEP 5: Statistical Evaluation")
    print(f"{'='*60}")
    start = time.time()

    import importlib
    eval_mod = importlib.import_module("05_evaluate")

    eval_mod.test1_coefficient_ttest(m1, y)
    eval_mod.test2_ftest(m1, y)
    eval_mod.test3_shapiro_wilk(m1, m2, m3)
    eval_mod.test4_paired_ttest(m1, m2, m3)
    eval_mod.test5_wilcoxon(m1, m2, m3)
    eval_mod.test_durbin_watson(m1, m2, m3)
    eval_mod.plot_model_comparison(m1, m2, m3)
    eval_mod.plot_residual_diagnostics(df, y, m1, m2, m3)
    eval_mod.print_final_comparison_table(m1, m2, m3)

    print(f"✓ Step 5 completed in {time.time() - start:.1f}s")

    # ── STEP 6: Findings & Interpretation ────────────────────────────────────
    run_step(
        6, "Findings & Non-Technical Interpretation",
        "06_findings"
    )

    # ── PIPELINE COMPLETE ─────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"PIPELINE COMPLETE — {total_elapsed:.1f}s total")
    print("=" * 60)
    print("\nOutputs saved to outputs/:")
    output_files = [
        ("Data",   "merged_opal_population.csv"),
        ("Data",   "regression_full_dataset.csv"),
        ("Data",   "model_comparison_results.csv"),
        ("Chart",  "correlation_matrix.png"),
        ("Chart",  "covid_timeseries.png"),
        ("Chart",  "per_capita_trips.png"),
        ("Chart",  "seasonal_heatmap.png"),
        ("Chart",  "mode_share.png"),
        ("Chart",  "yoy_growth_comparison.png"),
        ("Chart",  "trips_by_mode.png"),
        ("Chart",  "model_comparison.png"),
        ("Chart",  "residual_diagnostics_all.png"),
    ]
    for ftype, fname in output_files:
        full_path = os.path.join(config.OUTPUTS_DIR, fname)
        status    = "✓" if os.path.exists(full_path) else "✗ MISSING"
        print(f"  [{ftype:5s}] {status}  {fname}")

    print("\nPush the outputs/ folder to GitHub along with this project.")
    print("Include the GitHub link in your submission .txt file.")


if __name__ == "__main__":
    main()
