# 🍽️ Zomato Data Analysis (Improved)

**Goal:** Turn simple EDA into *validated insights* using robust cleaning, hypothesis testing, and regression.

## Dataset
- Columns typically include: `online_order`, `book_table`, `rate`, `votes`, `approx_cost(for two people)`, `listed_in(type)`.
- Ratings parsed from strings like `4.2/5` → numeric and **winsorized** to reduce outlier impact.

## Approach
1. **Cleaning:** parse ratings, handle outliers (winsorization).
2. **EDA:** distributions, boxplots, correlations.
3. **Hypothesis test:** Welch’s t‑test for `online_order` vs rating.
4. **Effect size & 95% CI:** practical significance via **Cohen’s d** and CI.
5. **Regression (OLS):** control confounders: `book_table`, `listed_in(type)`, `votes`, `approx_cost(for two people)`.

## Results (from the notebook)
- Mean rating (online order **Yes**): **95**
- Mean rating (online order **No**): **95**
- Difference: **-** | t = **.** | p = **1**
- Cohen’s d: **-** | 95% CI: **boxplots, distributions**
- OLS coef for `online_order=Yes`: **.** (with other controls)

## How it’s done (step‑by‑step)
- Parse `rate` to numeric → winsorize 1–99%.
- Split groups by `online_order` → run **Welch’s t‑test**.
- Compute **Cohen’s d**, **95% CI**, and visualize with boxplots.
- Fit **OLS**: `rate_winz ~ C(online_order) + C(book_table) + C(listed_in(type)) + votes + approx_cost`.
- Interpret coefficients & p‑values for causal direction hints.

## Run
```bash
pip install -r requirements.txt
jupyter notebook Zomato_Analysis.ipynb
```
