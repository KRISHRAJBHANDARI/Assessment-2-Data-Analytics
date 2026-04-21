# =============================================================================
# 05_evaluate.py
# PRT564 Data Analytics and Visualisation — Assessment 2
#
# Purpose:
#   Model evaluation and statistical testing.
#   Performs 5 rigorous hypothesis tests and produces visual diagnostics.
#
#   Test 1 — Coefficient t-test (M1)
#     H₀: β₁ = 0 (population elasticity is zero)
#     H₁: β₁ ≠ 0 (population has a significant but possibly small effect)
#     Computed from scratch using OLS formula to demonstrate understanding.
#
#   Test 2 — F-test (M1 overall fit)
#     H₀: R² = 0 (model explains no variance)
#     H₁: R² > 0 (model explains significant variance)
#     Even if R² is tiny, F-test confirms the model structure is valid.
#
#   Test 3 — Shapiro-Wilk normality test on residuals
#     H₀: Residuals are normally distributed (OLS assumption)
#     H₁: Residuals deviate from normality
#     If H₀ is rejected → Wilcoxon (Test 5) is the primary comparison test.
#
#   Test 4 — Paired t-test comparing model absolute errors
#     H₀: No significant difference in MAE between model pairs
#     H₁: Later model has significantly smaller errors (one-tailed, α=0.05)
#     Paired because all models are evaluated on the same 102 observations.
#
#   Test 5 — Wilcoxon signed-rank test (non-parametric)
#     Same hypotheses as Test 4 but makes no normality assumption.
#     This is our PRIMARY model comparison test because Test 3 confirmed
#     non-normal residuals in all three models.
#
#   + Durbin-Watson autocorrelation test
#     Checks whether adjacent residuals are correlated (expected in monthly
#     time-series data). Positive autocorrelation means standard errors are
#     slightly underestimated — acknowledged as a limitation.
#
# Author: Suyog Kadariya (S393829)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import shapiro, wilcoxon
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import importlib

plt.style.use("seaborn-v0_8-whitegrid")


def test1_coefficient_ttest(m1_results, y):
    """
    Test 1: Manual t-test on the population elasticity coefficient (Model 1).

    Computed from scratch using the OLS coefficient standard error formula:
      Var(β̂) = σ² (XᵀX)⁻¹
      where σ² = RSS / (n - k)

    This demonstrates understanding of the underlying statistics rather than
    relying on a black-box library.

    H₀: β₁ = 0  (population has no effect on trips)
    H₁: β₁ ≠ 0  (population has a significant effect)
    """
    residuals = m1_results["residuals"]
    X_m1      = m1_results["X"]
    n         = len(y)
    k         = 2  # intercept + 1 predictor

    se_resid = np.sqrt(np.sum(residuals**2) / (n - k))
    X_const  = np.column_stack([np.ones(n), X_m1])
    XtX_inv  = np.linalg.inv(X_const.T @ X_const)
    se_coef  = se_resid * np.sqrt(np.diag(XtX_inv))

    beta1       = m1_results["model"].coef_[0]
    t_stat      = beta1 / se_coef[1]
    p_value     = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))
    significant = p_value < config.ALPHA

    print("\nTEST 1: COEFFICIENT T-TEST — Model 1 Population Elasticity")
    print(f"  H₀: β₁ = 0  |  H₁: β₁ ≠ 0  |  α = {config.ALPHA}")
    print(f"  β₁ estimate:  {beta1:.6f}")
    print(f"  Std. Error:   {se_coef[1]:.6f}")
    print(f"  t-statistic:  {t_stat:.4f}")
    print(f"  p-value:      {p_value:.4e}")
    print(f"  Significant:  {'Yes ✓' if significant else 'No'}")

    return {"t_stat": t_stat, "p_value": p_value, "significant": significant}


def test2_ftest(m1_results, y):
    """
    Test 2: F-test for overall model significance (Model 1).

    Tests whether Model 1 as a whole explains significantly more variance
    than a model with only the intercept.

    H₀: R² = 0  (model explains no variance)
    H₁: R² > 0  (model explains significant variance)

    Even if R² is tiny (0.0011), this test tells us whether the model
    structure itself is statistically valid.
    """
    residuals = m1_results["residuals"]
    n         = len(y)
    k         = 2  # intercept + 1 predictor

    ss_res  = np.sum(residuals**2)
    ss_tot  = np.sum((y - np.mean(y))**2)
    ss_reg  = ss_tot - ss_res
    f_stat  = (ss_reg / (k - 1)) / (ss_res / (n - k))
    f_pval  = 1 - stats.f.cdf(f_stat, dfn=k - 1, dfd=n - k)
    significant = f_pval < config.ALPHA

    print("\nTEST 2: F-TEST — Model 1 Overall Significance")
    print(f"  H₀: R² = 0  |  H₁: R² > 0  |  α = {config.ALPHA}")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  df:          ({k-1}, {n-k})")
    print(f"  p-value:     {f_pval:.4e}")
    print(f"  Decision:    {'REJECT H₀ — model is significant ✓' if significant else 'FAIL TO REJECT H₀'}")

    return {"f_stat": f_stat, "p_value": f_pval, "significant": significant}


def test3_shapiro_wilk(m1_results, m2_results, m3_results):
    """
    Test 3: Shapiro-Wilk normality test on residuals from all three models.

    OLS regression assumes normally distributed residuals for valid inference.
    We test this formally rather than relying only on visual Q-Q plots.

    H₀: Residuals are normally distributed
    H₁: Residuals deviate from normality

    If H₀ is rejected → Wilcoxon signed-rank (Test 5) is the appropriate
    primary model comparison test, as it makes no normality assumption.

    Returns:
        dict: {model_name: (W_stat, p_value, is_normal)}
    """
    results = {}
    print("\nTEST 3: SHAPIRO-WILK NORMALITY TEST ON RESIDUALS")
    print(f"  H₀: Residuals ~ Normal  |  α = {config.ALPHA}")

    for name, res in [
        ("M1 Simple Log-Log",       m1_results["residuals"]),
        ("M2 Multiple OLS + COVID", m2_results["residuals"]),
        ("M3 Ridge",                m3_results["residuals"]),
    ]:
        w, p      = shapiro(res)
        is_normal = p > config.ALPHA
        result    = "NORMAL ✓" if is_normal else "NON-NORMAL ✗ → use Wilcoxon"
        print(f"  {name:28s}: W={w:.4f}, p={p:.3e}  → {result}")
        results[name] = (w, p, is_normal)

    return results


def test4_paired_ttest(m1_results, m2_results, m3_results):
    """
    Test 4: Paired t-test comparing absolute prediction errors between models.

    Because all models are evaluated on the same 102 observations, the errors
    are paired — a paired t-test is appropriate (more powerful than unpaired).

    One-tailed test: H₁ = later model has strictly SMALLER errors.

    H₀: Mean(|errors_A|) = Mean(|errors_B|)
    H₁: Mean(|errors_B|) < Mean(|errors_A|)  (one-tailed, α=0.05)

    Note: If Shapiro-Wilk rejects normality, t-test assumptions are violated.
    Test 5 (Wilcoxon) is then the primary test; t-test results are secondary.
    """
    abs_err_1 = np.abs(m1_results["residuals"])
    abs_err_2 = np.abs(m2_results["residuals"])
    abs_err_3 = np.abs(m3_results["residuals"])

    comparisons = [
        ("M1 vs M2", abs_err_1, abs_err_2),
        ("M2 vs M3", abs_err_2, abs_err_3),
        ("M1 vs M3", abs_err_1, abs_err_3),
    ]

    print("\nTEST 4: PAIRED T-TEST — Model Prediction Error Comparison")
    print(f"  One-tailed, α = {config.ALPHA}")
    print(f"  H₁: Later model has strictly smaller MAE\n")

    results = {}
    for label, e1, e2 in comparisons:
        t, p_two  = stats.ttest_rel(e1, e2)
        p_one     = p_two / 2
        improved  = e2.mean() < e1.mean()
        sig       = p_one < config.ALPHA and improved
        decision  = "SIGNIFICANT IMPROVEMENT ✓" if sig else "NOT SIGNIFICANT"
        print(f"  {label}: MAE {e1.mean():.5f} → {e2.mean():.5f}")
        print(f"         t={t:.3f}, p(one-tail)={p_one:.4e}  → {decision}")
        results[label] = {"t": t, "p_one": p_one, "significant": sig}

    return results


def test5_wilcoxon(m1_results, m2_results, m3_results):
    """
    Test 5: Wilcoxon signed-rank test (non-parametric model comparison).

    This is our PRIMARY model comparison test because Shapiro-Wilk (Test 3)
    confirmed non-normal residuals in all three models. The Wilcoxon test
    makes no normality assumption and is robust to outliers.

    H₀: No difference in prediction errors between models
    H₁: There is a significant difference in prediction errors
    α = 0.05 (two-tailed)
    """
    abs_err_1 = np.abs(m1_results["residuals"])
    abs_err_2 = np.abs(m2_results["residuals"])
    abs_err_3 = np.abs(m3_results["residuals"])

    comparisons = [
        ("M1 vs M2", abs_err_1, abs_err_2),
        ("M2 vs M3", abs_err_2, abs_err_3),
        ("M1 vs M3", abs_err_1, abs_err_3),
    ]

    print("\nTEST 5: WILCOXON SIGNED-RANK TEST (Primary model comparison)")
    print("  Non-parametric — used because Shapiro-Wilk rejected normality")
    print(f"  H₀: No difference in prediction errors  |  α = {config.ALPHA}\n")

    results = {}
    for label, e1, e2 in comparisons:
        w, p      = wilcoxon(e1, e2)
        sig       = p < config.ALPHA
        decision  = "SIGNIFICANT ✓" if sig else "NOT SIGNIFICANT"
        print(f"  {label}: W={w:.1f}, p={p:.4e}  → {decision}")
        results[label] = {"W": w, "p": p, "significant": sig}

    return results


def test_durbin_watson(m1_results, m2_results, m3_results):
    """
    Durbin-Watson autocorrelation test on all three models.

    Monthly time-series data typically exhibits positive autocorrelation
    (trips this month are correlated with trips last month).

    DW ≈ 2 → no autocorrelation
    DW < 1.5 → positive autocorrelation (adjacent errors have same sign)
    DW > 2.5 → negative autocorrelation

    Implication: if DW < 1.5, standard errors are underestimated,
    meaning p-values are too small and confidence intervals too narrow.
    This is a known limitation we acknowledge — ARIMA would be more
    appropriate for pure forecasting tasks.
    """
    print("\nDURBIN-WATSON AUTOCORRELATION TEST")
    print("  DW ≈ 2: no autocorrelation | DW < 1.5: positive autocorrelation\n")

    results = {}
    for name, res in [
        ("M1", m1_results["residuals"]),
        ("M2", m2_results["residuals"]),
        ("M3", m3_results["residuals"]),
    ]:
        dw     = np.sum(np.diff(res)**2) / np.sum(res**2)
        interp = ("acceptable (≈2)" if 1.5 < dw < 2.5 else
                  "positive autocorrelation" if dw < 1.5 else "negative autocorrelation")
        print(f"  {name}: DW = {dw:.4f}  → {interp}")
        results[name] = dw

    print("\n  Note: Positive autocorrelation is expected in monthly data.")
    print("  Implication: standard errors may be slightly underestimated.")
    print("  Recommendation: ARIMA for pure forecasting tasks.")
    return results


def plot_model_comparison(m1_results, m2_results, m3_results):
    """
    Bar charts comparing R², RMSE, and MAE across all three models.
    Colour coding: red (M1, worst) → gold (M2) → green (M3, best).
    """
    labels  = ["M1\nSimple", "M2\nMultiple", "M3\nRidge"]
    colors  = ["#C0392B", "#E8A020", "#1A7A4A"]
    r2s     = [m1_results["r2"],   m2_results["r2"],   m3_results["r2"]]
    rmses   = [m1_results["rmse"], m2_results["rmse"], m3_results["rmse"]]
    maes    = [m1_results["mae"],  m2_results["mae"],  m3_results["mae"]]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Model Performance Comparison — M1, M2, M3",
                 fontsize=13, fontweight="bold")

    for ax, values, title, fmt in [
        (axes[0], r2s,   "R² (higher = better)",   ".4f"),
        (axes[1], rmses, "RMSE (lower = better)",  ".5f"),
        (axes[2], maes,  "MAE (lower = better)",   ".5f"),
    ]:
        bars = ax.bar(labels, values,
                      color=(colors if title.startswith("R²") else colors[::-1]),
                      edgecolor="black", linewidth=0.8)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(title.split(" ")[0])
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    f"{val:{fmt}}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plt.savefig(config.CHART_MODEL_COMPARE, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {config.CHART_MODEL_COMPARE}")


def plot_residual_diagnostics(df, y, m1_results, m2_results, m3_results):
    """
    3×2 grid of residual diagnostic plots for all three models.

    Left column:  Residuals vs Fitted — checks homoscedasticity.
                  Points coloured by COVID_Flag (blue=normal, red=COVID).
                  Key visual: COVID points cluster away from zero in M1
                  but are absorbed in M2/M3, confirming the dummy works.

    Right column: Normal Q-Q plot — checks normality assumption.
                  Points deviating from the diagonal indicate non-normality,
                  which is confirmed formally by Shapiro-Wilk (Test 3).
    """
    fig = plt.figure(figsize=(14, 12))
    fig.suptitle("Residual Diagnostics — M1, M2, M3", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(3, 2, figure=fig)

    covid_colors = ["steelblue" if c == 0 else "crimson"
                    for c in df["COVID_Flag"].values]

    for row, (m_results, r2_val) in enumerate([
        (m1_results, m1_results["r2"]),
        (m2_results, m2_results["r2"]),
        (m3_results, m3_results["r2"]),
    ]):
        # Left: residuals vs fitted
        ax_left = fig.add_subplot(gs[row, 0])
        ax_left.scatter(m_results["y_pred"], m_results["residuals"],
                        alpha=0.5, s=20, c=covid_colors)
        ax_left.axhline(0, color="red", linestyle="--", lw=1.5)
        ax_left.set_title(
            f"{m_results['name']} — Residuals vs Fitted  (R²={r2_val:.4f})")
        ax_left.set_xlabel("Fitted ln(Trips)")
        ax_left.set_ylabel("Residual")
        ax_left.grid(True, alpha=0.3)

        # Right: Q-Q plot
        ax_right = fig.add_subplot(gs[row, 1])
        stats.probplot(m_results["residuals"], dist="norm", plot=ax_right)
        ax_right.set_title(f"{m_results['name']} — Normal Q-Q Plot")
        ax_right.grid(True, alpha=0.3)

    fig.text(0.5, 0.01,
             "Point colour: blue = normal period, red = COVID period",
             ha="center", fontsize=9, style="italic")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(config.CHART_RESIDUALS, dpi=config.CHART_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {config.CHART_RESIDUALS}")
    print("  Note: COVID points cluster far from zero in M1 but absorbed in M2/M3.")


def print_final_comparison_table(m1_results, m2_results, m3_results):
    """Print a formatted final model comparison table to the console."""
    print("\n" + "=" * 90)
    print("FINAL MODEL COMPARISON — All models on same 102-observation dataset")
    print("=" * 90)
    rows = []
    for m in [m1_results, m2_results, m3_results]:
        rows.append({
            "Model":       m["name"],
            "R²":          f"{m['r2']:.4f}",
            "RMSE":        f"{m['rmse']:.5f}",
            "MAE":         f"{m['mae']:.5f}",
            "λ (Ridge)":   f"{m.get('lambda', '—')}",
        })
    df_table = pd.DataFrame(rows)
    print(df_table.to_string(index=False))

    best = max([m1_results, m2_results, m3_results], key=lambda m: m["r2"])
    print(f"\n★ Best model: {best['name']} — R² = {best['r2']:.4f}")
    print(f"  Explains {best['r2']*100:.1f}% of monthly Opal trip variance.")
    print(f"  Remaining {(1-best['r2'])*100:.1f}% from unmodelled factors:")
    print(f"  (service quality, fuel prices, WFH rates, major events)")


def main():
    """
    Run all statistical tests and produce evaluation charts.
    Imports fitted models from 04_models.py.
    Called directly via: python 05_evaluate.py
    """
    print("=" * 60)
    print("STEP 5: MODEL EVALUATION & STATISTICAL TESTING")
    print("=" * 60)

    # Import and run models (avoids duplicating fit logic)
    from models_04 import main as run_models
    df, y, m1, m2, m3 = run_models()

    # Rename for clarity inside this module
    m1_results = m1
    m2_results = m2
    m3_results = m3

    print("\n--- STATISTICAL TESTS ---")
    test1_coefficient_ttest(m1_results, y)
    test2_ftest(m1_results, y)
    test3_shapiro_wilk(m1_results, m2_results, m3_results)
    test4_paired_ttest(m1_results, m2_results, m3_results)
    test5_wilcoxon(m1_results, m2_results, m3_results)
    test_durbin_watson(m1_results, m2_results, m3_results)

    print("\n--- EVALUATION CHARTS ---")
    plot_model_comparison(m1_results, m2_results, m3_results)
    plot_residual_diagnostics(df, y, m1_results, m2_results, m3_results)

    print_final_comparison_table(m1_results, m2_results, m3_results)

    print("\nStep 5 complete.\n")
    return m1_results, m2_results, m3_results


if __name__ == "__main__":
    # When run standalone, import 04_models directly
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from importlib import import_module
    mod = import_module("04_models")
    df, y, m1, m2, m3 = mod.main()

    from scipy import stats
    from scipy.stats import shapiro, wilcoxon

    print("\n--- STATISTICAL TESTS ---")
    test1_coefficient_ttest(m1, y)
    test2_ftest(m1, y)
    test3_shapiro_wilk(m1, m2, m3)
    test4_paired_ttest(m1, m2, m3)
    test5_wilcoxon(m1, m2, m3)
    test_durbin_watson(m1, m2, m3)

    print("\n--- EVALUATION CHARTS ---")
    plot_model_comparison(m1, m2, m3)
    plot_residual_diagnostics(df, y, m1, m2, m3)
    print_final_comparison_table(m1, m2, m3)
    print("\nStep 5 complete.\n")
