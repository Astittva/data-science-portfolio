# 💗 Breast Cancer Prediction (Improved)

**Goal:** Produce a **robust, well‑validated** classifier using **nested cross‑validation**, learning curves, and model‑agnostic interpretability.

## Dataset
- Wisconsin Diagnostic Breast Cancer (WDBC) style: `diagnosis` (M/B) + 30 numeric features.

## Approach
1. **Cleaning:** drop ID/empty columns; encode `diagnosis` (M→1, B→0).
2. **Nested CV:** inner loop tunes hyperparams; outer loop estimates unbiased performance (AUC).
3. **Learning curve:** shows data sufficiency & potential gains.
4. **Final fit:** evaluate on a holdout split (AUC, Accuracy, F1).
5. **Permutation importance:** top features driving predictions.

## Results (from the notebook)
- Nested AUC (mean ± std): **0.9943809909849207 ± ...**
- Holdout — AUC: **1** | Accuracy: **1** | F1: **1**

## How it’s done (step‑by‑step)
- Pipeline: `StandardScaler` → `LogisticRegression (L2)`.
- `GridSearchCV` in the **inner** loop, `cross_val_score` in the **outer** loop.
- Train final model on train set; evaluate on test; plot ROC & learning curve.
- Run permutation importance to rank features.

## Run
```bash
pip install -r requirements.txt
jupyter notebook Breast_Cancer_Prediction.ipynb
```
