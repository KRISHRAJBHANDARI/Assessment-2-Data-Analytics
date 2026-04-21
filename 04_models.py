# =============================================================================
# 04_models.py
# PRT564 Data Analytics and Visualisation — Assessment 2
#
# Purpose:
#   Train all three regression models and return fitted model objects
#   and predictions for use by 05_evaluate.py.
#
#   Model 1 — Simple Log-Log OLS
#     Equation: ln(Trips) = β₀ + β₁ · ln(Population)
#     Purpose:  Baseline. Tests whether population alone predicts demand.
#     Expected: Very low R² — population alone is insufficient.
#
#   Model 2 — Multiple OLS with COVID Flag
#     Equation: ln(T) = β₀ + β₁·ln(Pop) + β₂·Time + β₃·PopGrowth + β₄·COVID
#     Purpose:  Tests whether service/time trend + COVID structural break
#               dramatically improves fit over population alone.
#     Expected: Large R² improvement over M1.
#
#   Model 3 — Ridge Regression (L2 Regularised, CV-tuned)
#     Equation: ln(T) ~ all M2 features + Post_COVID_Recovery + Month  [L2]
#     Purpose:  Handles multicollinearity between Log_Population and
#               Time_Index (both trend upward → correlated at r≈0.95).
#               OLS inflates coefficient variance in this case.
#               Ridge adds L2 penalty: minimises (RSS + λ·Σβ²),
#               shrinking unstable correlated coefficients toward zero.
#     Additional features:
#       Post_COVID_Recovery — separates the permanently lower demand
#         baseline post-2022 from the pre-COVID growth regime. Without
#         this, Time_Index goes spuriously negative.
#       Month — captures seasonal demand variation confirmed by EDA
#         heatmap (school terms vs holiday periods).
#     λ selection: 5-fold cross-validation over 100 candidate values.
#
#   All models are trained on the SAME 102-observation dataset so that
#   paired statistical tests in 05_evaluate.py are valid comparisons.
#
# Author: Suyog Kadariya (S393829)
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def load_regression_data():
    """
    Load the 102-observation regression dataset built by 02_preprocess.py.

    Returns:
        pd.DataFrame: Regression-ready dataset.
        np.ndarray:   Target variable y (Log_Total_Trips).
    """
    df = pd.read_csv(config.REGRESSION_CSV, parse_dates=["Date"])
    y  = df[config.TARGET_COL].values
    print(f"Regression dataset loaded: {len(df)} observations")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    return df, y


def fit_model1(df, y):
    """
    Model 1: Simple Log-Log OLS Regression.

    Equation: ln(Trips) = β₀ + β₁ · ln(Population)

    This is the baseline model. The elasticity interpretation of β₁ is:
    "A 1% increase in population corresponds to β₁% change in Opal trips."
    If β₁ is near zero or negative, population alone is an inadequate
    predictor — which is our central finding.

    Args:
        df (pd.DataFrame): Regression dataset.
        y  (np.ndarray):   Target variable.

    Returns:
        dict: Model object, predictions, residuals, and metrics.
    """
    X = df[config.FEATURES_M1].values
    m = LinearRegression().fit(X, y)
    y_pred    = m.predict(X)
    residuals = y - y_pred

    metrics = {
        "r2":   r2_score(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "mae":  mean_absolute_error(y, y_pred),
    }

    print(f"\nMODEL 1 — Simple Log-Log OLS")
    print(f"  Equation: ln(Trips) = {m.intercept_:.4f} + {m.coef_[0]:.4f} × ln(Pop)")
    print(f"  Elasticity: 1% pop growth → {m.coef_[0]:.4f}% trip change")
    print(f"  R²   = {metrics['r2']:.4f}  ({metrics['r2']*100:.2f}% variance explained)")
    print(f"  RMSE = {metrics['rmse']:.6f}")
    print(f"  MAE  = {metrics['mae']:.6f}")
    print(f"  → Population explains only {metrics['r2']*100:.2f}% of variance.")
    if m.coef_[0] < 0:
        print(f"  → Negative elasticity confirms the inverse relationship.")

    return {
        "name":      "M1: Simple Log-Log OLS",
        "model":     m,
        "X":         X,
        "y_pred":    y_pred,
        "residuals": residuals,
        "features":  config.FEATURES_M1,
        **metrics,
    }


def fit_model2(df, y):
    """
    Model 2: Multiple OLS with COVID Dummy.

    Equation: ln(T) = β₀ + β₁·ln(Pop) + β₂·Time + β₃·PopGrowth + β₄·COVID

    Adding Time_Index captures the cumulative effect of infrastructure
    investment and service expansion — the real driver of demand.
    Adding COVID_Flag is critical: without it, the 47% demand collapse during
    COVID is misattributed to population or time trend, biasing all other
    coefficients. COVID_Flag isolates the pandemic as a structural break.

    The COVID effect is interpreted via exponentiation of the coefficient:
      e^β₄ = multiplier on trips during COVID relative to non-COVID.
      (1 - e^β₄) × 100 = % reduction in trips during COVID (model estimate).
    The model estimate is lower than the raw 47% because Time_Index partially
    absorbs some of the COVID effect — both acknowledged in findings.

    Args:
        df (pd.DataFrame): Regression dataset.
        y  (np.ndarray):   Target variable.

    Returns:
        dict: Model object, predictions, residuals, and metrics.
    """
    X = df[config.FEATURES_M2].values
    m = LinearRegression().fit(X, y)
    y_pred    = m.predict(X)
    residuals = y - y_pred

    metrics = {
        "r2":   r2_score(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "mae":  mean_absolute_error(y, y_pred),
    }

    # COVID coefficient interpretation
    covid_coef      = m.coef_[3]  # index 3 = COVID_Flag
    covid_multiplier = np.exp(covid_coef)
    covid_pct_drop  = (1 - covid_multiplier) * 100

    print(f"\nMODEL 2 — Multiple OLS + COVID Flag")
    print(f"  Intercept: {m.intercept_:.4f}")
    for fname, coef in zip(config.FEATURES_M2, m.coef_):
        print(f"  {fname:30s}: {coef:+.6f}")
    print(f"  R²   = {metrics['r2']:.4f}  (+{metrics['r2'] - 0:.4f} vs M1 baseline)")
    print(f"  RMSE = {metrics['rmse']:.6f}")
    print(f"  MAE  = {metrics['mae']:.6f}")
    print(f"  COVID coefficient: {covid_coef:.4f} log-units → e^β₄ = {covid_multiplier:.3f}")
    print(f"  → Model estimates {covid_pct_drop:.1f}% fewer trips during COVID")
    print(f"  (Raw data shows 47%; model estimate is lower because Time_Index")
    print(f"  partially absorbs the COVID effect — see 05_evaluate.py findings.)")

    return {
        "name":      "M2: Multiple OLS + COVID",
        "model":     m,
        "X":         X,
        "y_pred":    y_pred,
        "residuals": residuals,
        "features":  config.FEATURES_M2,
        **metrics,
    }


def fit_model3(df, y):
    """
    Model 3: Ridge Regression (L2 regularised, CV-tuned).

    Why Ridge over OLS for this model:
      Log_Population and Time_Index are highly correlated (both trend upward
      over time, r≈0.95). This multicollinearity inflates OLS coefficient
      variance, making individual coefficients unstable and unreliable.
      Ridge adds an L2 penalty to the loss function:
        Minimise: RSS + λ · Σβ²
      This shrinks correlated coefficients toward zero, producing more stable
      and interpretable estimates.

    Why Post_COVID_Recovery is added:
      Without this flag, Time_Index goes negative in the model — it sees
      "later dates = lower trips" because post-COVID months have permanently
      lower demand. The Post_COVID_Recovery dummy absorbs this new baseline,
      allowing Time_Index to correctly capture the pre-COVID growth trend.

    Why Month is added:
      EDA seasonal heatmap confirmed that school-term months (Feb–Jun, Jul–Nov)
      show systematically higher demand than holiday months. Including Month
      as a feature captures this seasonality in the model.

    Standardisation:
      Features are standardised (zero mean, unit variance) before Ridge.
      OLS is scale-invariant; Ridge is NOT — larger-scaled features receive
      a disproportionately large penalty without standardisation.
      Standardised coefficients are interpretable as relative feature importance.

    λ selection:
      RidgeCV tests 100 candidate λ values (log-spaced 0.001 to 1000) and
      selects the one that maximises R² in 5-fold cross-validation.
      The selected λ is printed and recorded — not manually tuned.

    Args:
        df (pd.DataFrame): Regression dataset.
        y  (np.ndarray):   Target variable.

    Returns:
        dict: Model object, scaler, predictions, residuals, and metrics.
    """
    X      = df[config.FEATURES_M3].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Generate 100 candidate λ values log-spaced between 0.001 and 1000
    alphas   = np.logspace(*config.RIDGE_ALPHAS)
    ridge_cv = RidgeCV(
        alphas=alphas,
        scoring=config.RIDGE_SCORING,
        cv=config.RIDGE_CV_FOLDS
    )
    ridge_cv.fit(X_scaled, y)

    y_pred    = ridge_cv.predict(X_scaled)
    residuals = y - y_pred

    metrics = {
        "r2":   r2_score(y, y_pred),
        "rmse": np.sqrt(mean_squared_error(y, y_pred)),
        "mae":  mean_absolute_error(y, y_pred),
    }

    print(f"\nMODEL 3 — Ridge Regression (L2, CV-tuned)")
    print(f"  Optimal λ (selected by {config.RIDGE_CV_FOLDS}-fold CV): {ridge_cv.alpha_:.4f}")
    print(f"  R²   = {metrics['r2']:.4f}")
    print(f"  RMSE = {metrics['rmse']:.6f}")
    print(f"  MAE  = {metrics['mae']:.6f}")
    print(f"\n  Standardised coefficients (relative feature importance):")
    for fname, coef in zip(config.FEATURES_M3, ridge_cv.coef_):
        bar   = "█" * int(abs(coef) * 50)
        sign  = "+" if coef > 0 else "-"
        print(f"  {fname:28s}: {coef:+.4f}  {sign}{bar}")

    time_coef = ridge_cv.coef_[1]   # index 1 = Time_Index
    print(f"\n  Time_Index coefficient: {time_coef:+.4f}")
    if time_coef > 0:
        print("  ✓ POSITIVE — correctly captures pre-COVID growth trend")
    else:
        print("  ⚠ Negative — Post_COVID_Recovery flag may need adjustment")

    return {
        "name":      "M3: Ridge (CV-tuned)",
        "model":     ridge_cv,
        "scaler":    scaler,
        "X":         X_scaled,
        "y_pred":    y_pred,
        "residuals": residuals,
        "features":  config.FEATURES_M3,
        "lambda":    ridge_cv.alpha_,
        **metrics,
    }


def print_model_progression(m1, m2, m3):
    """Print the R² progression across all three models."""
    print("\nMODEL PROGRESSION SUMMARY:")
    print(f"  M1 (population only):              R² = {m1['r2']:.4f}  ({m1['r2']*100:.2f}%)")
    print(f"  M2 (+ time + COVID):               R² = {m2['r2']:.4f}  (+{m2['r2']-m1['r2']:.4f})")
    print(f"  M3 (+ recovery + month, Ridge):    R² = {m3['r2']:.4f}  (+{m3['r2']-m2['r2']:.4f})")
    print(f"\n  Best model: {max([m1, m2, m3], key=lambda m: m['r2'])['name']}")


def main():
    """
    Fit all three models and save a comparison table.
    Called directly via: python 04_models.py
    Also called by run_all.py and 05_evaluate.py.

    Returns:
        tuple: (df, y, m1_results, m2_results, m3_results)
    """
    print("=" * 60)
    print("STEP 4: REGRESSION MODELLING")
    print("=" * 60)

    df, y = load_regression_data()
    m1    = fit_model1(df, y)
    m2    = fit_model2(df, y)
    m3    = fit_model3(df, y)

    print_model_progression(m1, m2, m3)

    # Save comparison table
    comparison = pd.DataFrame([
        {
            "Model":       m["name"],
            "R2":          round(m["r2"], 4),
            "RMSE":        round(m["rmse"], 6),
            "MAE":         round(m["mae"], 6),
            "Lambda_Ridge": m.get("lambda", "—"),
        }
        for m in [m1, m2, m3]
    ])
    comparison.to_csv(config.MODEL_RESULTS_CSV, index=False)
    print(f"\n  Saved comparison table: {config.MODEL_RESULTS_CSV}")
    print("\nStep 4 complete.\n")

    return df, y, m1, m2, m3


if __name__ == "__main__":
    main()
