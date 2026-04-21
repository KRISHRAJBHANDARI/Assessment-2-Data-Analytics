# NSW Opal Trip Demand Analysis

## PRT564 Data Analytics and Visualisation — Assessment 2

**Group Members:**

| Name              | Student ID | Contribution                                                                   |
| ----------------- | ---------- | ------------------------------------------------------------------------------ |
| Nasla Maharjan    | S398425    | ETL pipeline (`01_load_data.py`, `02_preprocess.py`)                       |
| Krish Rajbhandari | S395754    | Regression models & statistical testing (`04_models.py`, `05_evaluate.py`) |
| Suyog Kadariya    | S393829    | EDA visualisations & findings (`03_eda.py`, `06_findings.py`)              |

**Campus:** Sydney | **Unit:** PRT564 | **Year:** 2026

---

## Project Overview

This project analyses 9 years of NSW Opal tap-on data (2016–2025) combined with ABS population statistics to answer a core planning question for Transport for NSW:

> **Does population growth automatically drive public transport demand?**

**Key finding:** Population alone explains only 0.11% of monthly trip variance. Per-capita Opal trips are declining at −9.6 trips/1,000 residents/month even before COVID. Infrastructure investment — not population — is the primary driver of demand.

---

## Data Sources

| Source                                                                                                                                     | File               | Description                                                  |
| ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------ | ------------------------------------------------------------ |
| [data.gov.au](https://data.gov.au/data/dataset/nsw-2-opal-trips-all-modes)                                                                    | `opal_trips.csv` | Monthly Opal tap-on counts by mode and card type, 2016–2025 |
| [Australian Bureau of Statistics](https://www.abs.gov.au/statistics/people/population/national-state-and-territory-population/latest-release) | `3101051.xlsx`   | NSW population by age and sex (annual)                       |

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place data files

```
opal_transport_analysis/
└── data/
    ├── opal_trips.csv          ← download from data.gov.au
    └── 3101051.xlsx            ← download from ABS website
```

### 3. Run the full pipeline

```bash
python run_all.py
```

All outputs are saved to the `outputs/` folder automatically.

---

## Project Structure

```
opal_transport_analysis/
│
├── README.md               This file
├── requirements.txt        Python dependencies
├── config.py               All paths and constants
│
├── data/                   Input data files (not committed to GitHub)
│   ├── opal_trips.csv
│   └── 3101051.xlsx
│
├── outputs/                Generated charts and CSVs (auto-created)
│
├── 01_load_data.py         ETL: load, standardise, merge both data sources
├── 02_preprocess.py        Feature engineering, COVID flags, regression dataset
├── 03_eda.py               7 EDA visualisations
├── 04_models.py            3 regression models (OLS × 2, Ridge)
├── 05_evaluate.py          5 statistical tests + evaluation charts
├── 06_findings.py          Technical & non-technical findings output
└── run_all.py              Master script — runs everything in order
```

---

## Running Individual Steps

Each script can also be run independently:

```bash
python 01_load_data.py      # ETL only
python 02_preprocess.py     # Feature engineering only
python 03_eda.py            # EDA charts only (requires Step 1+2 outputs)
python 04_models.py         # Models only (requires Step 1+2 outputs)
python 05_evaluate.py       # Evaluation (requires Step 4 outputs)
python 06_findings.py       # Findings (requires all prior outputs)
```

---

## Models

| Model | Type                 | Features                                | R²    |
| ----- | -------------------- | --------------------------------------- | ------ |
| M1    | Simple Log-Log OLS   | ln(Population)                          | 0.0011 |
| M2    | Multiple OLS + COVID | ln(Pop) + Time + PopGrowth + COVID_Flag | 0.6266 |
| M3    | Ridge (L2, CV-tuned) | All M2 + Post_COVID_Recovery + Month    | 0.6491 |

**Best model:** M3 (Ridge) — R² = 0.649, λ = 0.005 selected by 5-fold cross-validation.

---

## Statistical Tests

| Test                          | Purpose                                | Result                               |
| ----------------------------- | -------------------------------------- | ------------------------------------ |
| T1 — Coefficient t-test (M1) | Is population elasticity ≠ 0?         | Significant (p < 0.05)               |
| T2 — F-test (M1)             | Does model structure explain variance? | Significant                          |
| T3 — Shapiro-Wilk            | Are residuals normally distributed?    | Non-normal → use Wilcoxon           |
| T4 — Paired t-test           | M1 vs M2 vs M3 accuracy comparison     | M1→M2 significant (p=2.3×10⁻¹¹) |
| T5 — Wilcoxon (primary)      | Non-parametric model comparison        | M1→M2: significant; M2→M3: p=0.12  |

---

## Key Findings

1. **Population explains almost nothing** — R² = 0.0011 for Model 1
2. **COVID caused a 47% demand collapse** — 53.8M → 28.5M monthly trips
3. **Recovery is incomplete** — 10.1M trips/month still missing vs pre-COVID
4. **Per-capita demand declining** — −9.6 trips/1,000 residents/month even pre-COVID
5. **Infrastructure drives demand** — Time_Index is the strongest predictor in M2/M3

---

## Outputs

| File                             | Description                                        |
| -------------------------------- | -------------------------------------------------- |
| `merged_opal_population.csv`   | Full merged dataset with all features              |
| `regression_full_dataset.csv`  | 102-observation regression dataset                 |
| `model_comparison_results.csv` | R², RMSE, MAE for all three models                |
| `correlation_matrix.png`       | Pearson correlations — confirms multicollinearity |
| `covid_timeseries.png`         | Annotated time-series with COVID structural break  |
| `per_capita_trips.png`         | Headline finding: declining per-capita demand      |
| `seasonal_heatmap.png`         | Month × mode demand heatmap                       |
| `mode_share.png`               | Mode share stacked area chart                      |
| `yoy_growth_comparison.png`    | Trip vs population YoY growth comparison           |
| `trips_by_mode.png`            | Absolute trip counts per mode over time            |
| `model_comparison.png`         | R², RMSE, MAE bar charts across models            |
| `residual_diagnostics_all.png` | 3×2 residual and Q-Q plots for all models         |

---

## References

- NSW Government (2026). *Transport for NSW*. https://www.transport.nsw.gov.au/about-us
- Australian Government (2025). *Opal Trip Counts Based on NRT – by month*. https://data.gov.au/data/dataset/nsw-2-opal-trips-all-modes
- Australian Bureau of Statistics (2026). *National, State and Territory Population*. https://www.abs.gov.au/statistics/people/population/national-state-and-territory-population/latest-release
- ScienceDirect (n.d.). *Poisson Regression*. https://www.sciencedirect.com/topics/psychology/poisson-regression
