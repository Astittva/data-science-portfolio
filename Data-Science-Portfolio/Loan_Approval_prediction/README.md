# 🏦 Loan Approval Prediction (Improved)

**Goal:** Build a business‑ready classifier with engineered features, robust CV, **calibrated probabilities**, and **F1‑optimized thresholding**.

## Dataset
- Typical columns: `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Status` (Y/N), etc.

## Approach
1. **Feature engineering:** `income_ratio = ApplicantIncome / (CoapplicantIncome+1)`, `log_loan = log1p(LoanAmount)`.
2. **Preprocessing:** impute missing, scale numeric, one‑hot encode categorical.
3. **Modeling:** RandomForest with **Stratified K‑Fold CV + GridSearch**.
4. **Calibration:** **isotonic** to fix probability bias.
5. **Decision rule:** scan thresholds and pick the one that **maximizes F1**.

## Results (from the notebook)
- Best CV AUC: **10** (params: {'clf__max_depth': 10, 'clf__n_estimators': 200})
- Validation AUC: **1**
- Best threshold (F1): **1**
- F1 @ best_th: **1**
- Confusion Matrix @ best_th: `[[25 13]`

## How it’s done (step‑by‑step)
- Build a **sklearn Pipeline** (imputer → scaler/encoder → model) for reproducibility.
- Use **StratifiedKFold** to keep class balance across folds and tune RF hyperparams.
- **Calibrate** probabilities (isotonic), then find the **F1‑optimal threshold** on a validation split.
- Report AUC, F1, and confusion matrix; save pipeline artifact for reuse.

## Run
```bash
pip install -r requirements.txt
jupyter notebook Loan_Approval_Prediction.ipynb
```
