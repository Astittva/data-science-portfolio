# 🍷 Wine Type & Quality Prediction — Deep Learning (Improved)

**Goal:** Predict **wine type** (red vs white) and assess **quality** with rigorous EDA, baselines, and a Keras neural network.

## Dataset
- Merged UCI Wine Quality datasets: red & white; added `type` column and normalized features.

## Approach
1. **EDA:** class balance, quality distribution, correlations; red vs white comparisons.
2. **Preprocessing:** stratified train/val/test; `StandardScaler`.
3. **Baselines:** Logistic Regression for type; RandomForest for quality (optional).
4. **Deep Learning:** Dense network with BatchNorm + Dropout + EarlyStopping.
5. **Evaluation:** AUC, Accuracy, F1; confusion matrix & learning curves.
6. **Artifacts:** saved model & scaler; simple inference helper.

## Results (from the notebook)

**Baseline (LogReg, validation)** — AUC: **42** | Acc: **1** | F1: **1**

**Neural Net (validation)** — AUC: **0.5** | Acc: **1** | F1: **1**

**Neural Net (test)** — AUC: **1** | Acc: **1** | F1: **1**

## How it’s done (step‑by‑step)
- Merge red & white CSVs; add `type`; scale features.
- Train a strong baseline; then a tuned NN (with BN + Dropout + early stopping).
- Compare validation & test metrics; visualize learning curves and confusion matrix.

## Run
```bash
pip install -r requirements.txt
jupyter notebook Wine_Prediction.ipynb
```
