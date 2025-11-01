# 💼 Astittva Mandloi — Data Scientist  

**Email:** astittvamandloi@gmail.com  
**Role:** Data Scientist  

Welcome to my **Data Science Portfolio**, a collection of end-to-end notebook-based projects across **EDA**, **Machine Learning**, **Deep Learning**, and **Computer Vision**.  
Each project folder includes a clean `README.md`, the Jupyter notebook, and reproducible requirements.

---

## 📂 Projects  

| Project | Description | Tech Stack |
|:--|:--|:--|
| [🍽️ Zomato Data Analysis](./Zomato_Analysis) | Statistical EDA with validated insights — rating parsing, winsorization, Welch’s t-test, OLS with confounders. | Python, Pandas, Seaborn, SciPy, Statsmodels |
| [🏦 Loan Approval Prediction](./Loan_Approval_prediction) | Feature-engineered, calibrated classifier with F1-optimal threshold and confusion-matrix reporting. | Scikit-learn, Pipelines, Calibration |
| [💗 Breast Cancer Prediction](./Breast_Cancer_Prediction) | Robust evaluation via nested CV, learning curves, and permutation importance. | Scikit-learn, Logistic Regression |
| [🍷 Wine Type & Quality (Deep Learning)](./Prediction%20of%20Wine%20type%20using%20Deep%20Learning) | Deep learning model (Keras) for wine type and quality classification; baselines vs NN with learning curves. | TensorFlow/Keras, Scikit-learn |
| [✍️ OCR of Handwritten Digits (OpenCV)](./OCR%20of%20Handwritten%20digits%20OpenCV) | OpenCV segmentation → 28×28 normalization → CNN digit recognition with overlay visualizations. | OpenCV, TensorFlow/Keras |
| 📉 [Market Risk Simulation Dashboard](https://risk-dashboard.streamlit.app) | A Streamlit app demonstrating Monte Carlo and Parametric Value-at-Risk (VaR) analysis, backtesting, and portfolio optimization.



---

## 🚀 How to Run Locally  

```bash
# Clone your portfolio (replace with your actual repo URL)
git clone https://github.com/<your-username>/Data-Science-Portfolio.git
cd Data-Science-Portfolio

# Choose a project folder
cd "Zomato_Analysis"           # or any of the other folders

# Install dependencies and run the notebook
pip install -r requirements.txt
jupyter notebook

