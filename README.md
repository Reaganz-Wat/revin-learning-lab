# RevIN Learning Repository

A beginner-to-intermediate self-contained course on statistics fundamentals,
time series forecasting, and **Reversible Instance Normalization (RevIN)**.

---

## Project Goal

Walk from first principles (mean, variance, std) through the key ideas that
make RevIN necessary (distribution shift, non-stationarity) and into a
complete stock-price forecasting project that shows RevIN's advantage.

---

## Repository Structure

```
revin-learning/
│
├── data/
│   ├── fundamentals_data.csv        # 50-row dataset for stats basics
│   ├── stock_sample.csv             # 120-day synthetic OHLCV stock data
│   └── shifted_stock_sample.csv     # Same structure, different regime
│
├── notebooks/
│   ├── 01_mean_basics.ipynb
│   ├── 02_variance_basics.ipynb
│   ├── 03_standard_deviation.ipynb
│   ├── 04_normalization_basics.ipynb
│   ├── 05_zscore_and_scaling.ipynb
│   ├── 06_time_series_basics.ipynb
│   ├── 07_distribution_shift.ipynb
│   ├── 08_revin_from_scratch.ipynb
│   └── 09_revin_project_stock_forecasting.ipynb
│
├── models/
│   ├── baseline_model.pkl           # RandomForest without RevIN
│   └── revin_model.pkl              # RandomForest with RevIN
│
├── scripts/
│   ├── generate_data.py             # Re-generate all CSV files
│   ├── train_baseline.py            # Train and save baseline model
│   └── train_revin.py               # Train and save RevIN model
│
├── utils/
│   ├── __init__.py
│   ├── ml_utils.py                  # Stats, normalization, RevIN, metrics
│   └── plotting.py                  # All visualization helpers
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Learning Roadmap

| # | Notebook | Concepts |
|---|----------|----------|
| 01 | Mean Basics | Arithmetic mean, centering |
| 02 | Variance Basics | Spread, squared deviations, population vs sample |
| 03 | Standard Deviation | sqrt(variance), 68-95-99.7 rule |
| 04 | Normalization Basics | Why normalize, centering + scaling |
| 05 | Z-Score and Min-Max | Two normalization strategies, reversibility |
| 06 | Time Series Basics | Trend, seasonality, lag, sliding windows |
| 07 | Distribution Shift | Mean shift, variance shift, why it breaks models |
| 08 | RevIN from Scratch | Instance normalization, reversibility proof |
| 09 | Capstone Project | End-to-end forecasting: baseline vs RevIN |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate data (already done — run only to regenerate)

```bash
python scripts/generate_data.py
```

### 3. (Optional) Retrain models

```bash
python scripts/train_baseline.py
python scripts/train_revin.py
```

### 4. Open notebooks

```bash
cd notebooks
jupyter notebook
```

Work through them in order: `01 → 09`.

---

## Expected Learning Outcomes

After completing all nine notebooks you will be able to:

- Derive mean, variance, and standard deviation from scratch
- Explain why z-score and min-max normalization are reversible
- Identify trends, seasonality, and lag features in a time series
- Explain what distribution shift is and why it degrades model performance
- Implement RevIN from scratch in fewer than 20 lines of Python
- Build a complete supervised time series forecasting pipeline
- Quantitatively compare baseline and RevIN models under distribution shift

---

## Key Reference

> Kim, T., Kim, J., Tae, Y., Park, C., Choi, J. H., & Choo, J. (2022).
> **Reversible Instance Normalization for Accurate Time-Series Forecasting
> against Distribution Shift.**
> *International Conference on Learning Representations (ICLR 2022).*

---

## Notes

- All plotting logic lives in `utils/plotting.py` — notebooks stay clean.
- `utils/ml_utils.py` provides all math from scratch; NumPy is used for
  validation only.
- Data is fully synthetic — no external downloads required.
