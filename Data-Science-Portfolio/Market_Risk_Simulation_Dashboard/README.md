# 📉 Market Risk Simulation Dashboard — Monte Carlo & Parametric VaR

A fully interactive **Streamlit dashboard** for analyzing **portfolio market risk**, built with  
**Python · NumPy · Pandas · Plotly · SciPy · yFinance**.

This project demonstrates **real-world quantitative finance skills**, including:
- Value-at-Risk (VaR) estimation  
- Monte Carlo simulation (historical bootstrap)  
- Parametric (Normal) VaR  
- Expected Shortfall (CVaR)  
- Risk scaling  
- Backtesting  
- Portfolio optimization (Efficient Frontier)

---

## 🧠 Project Overview

The app lets users:
- Fetch historical market prices using Yahoo Finance  
- Build a multi-asset portfolio with custom weights  
- Simulate and visualize portfolio losses  
- Compute **Value-at-Risk (VaR)** and **Expected Shortfall (ES)** for different horizons  
- Evaluate **diversification effects** with correlation heatmaps  
- Backtest VaR to see how well the model fits actual data  
- Generate the **Efficient Frontier** to visualize optimal risk-return trade-offs

---

## ⚙️ Tech Stack

| Category | Tools |
|-----------|-------|
| Data | `yfinance`, `pandas`, `numpy` |
| Analytics | `scipy`, `statistics` |
| Visualization | `plotly`, `matplotlib` |
| Web App | `streamlit` |

---

## 🧩 Key Concepts

| Concept | Explanation |
|----------|--------------|
| **Log Returns** | Stable, additive measure of returns for time aggregation |
| **Monte Carlo (Bootstrap)** | Resampling past returns to simulate future loss distributions |
| **Parametric VaR** | Analytical VaR under a Normal distribution assumption |
| **Expected Shortfall (ES)** | Mean of losses worse than the VaR — measures tail risk |
| **√Time Rule** | Scales 1-day VaR to N-day assuming independence |
| **Backtesting** | Checking whether actual losses exceed predicted VaR |
| **Efficient Frontier** | Curve showing optimal portfolios at different risk levels |

---

## 🚀 How to Run Locally

### 1️⃣ Create a new environment
```bash
conda create -n riskdash python=3.11 -y
conda activate riskdash
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Streamlit app
```bash
streamlit run app.py
```

### 4️⃣ Open in browser
After running, open the provided **localhost URL** (e.g. `http://localhost:8501`) to access your dashboard.

---

## 🧮 Example Usage

1. Select tickers (e.g., AAPL, MSFT, TSLA)  
2. Choose start & end dates  
3. Adjust portfolio weights in the sliders  
4. Pick VaR type — Monte Carlo or Parametric  
5. Run analysis to view:
   - Portfolio risk metrics  
   - Correlation heatmap  
   - Historical VaR breaches  
   - Efficient Frontier chart  

---

## 📈 Outputs & Insights

| Section | Description |
|----------|--------------|
| **Data Summary** | Displays log-return statistics for each asset |
| **Correlation Heatmap** | Shows diversification potential |
| **VaR Distribution Chart** | Highlights potential losses at given confidence levels |
| **Backtesting Plot** | Shows where real returns breached VaR |
| **Efficient Frontier** | Visualizes optimal risk-return trade-offs |

---

## 🧾 Deployment (Streamlit Cloud)

1. Push your `app.py`, `requirements.txt`, and `README.md` to a **GitHub repo**  
   Example folder:
   ```
   Market_Risk_Simulation_Dashboard/
   ├── app.py
   ├── requirements.txt
   └── README.md
   ```

2. Go to [https://share.streamlit.io](https://data-science-portfoliomarket-risk-simula-2e5xgw.streamlit.app)

3. Click **New app → Paste GitHub repo link**

4. Fill out:
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** (your choice)

5. Click **Deploy**

---

## 🧑‍💻 Author

**Astitva Mandloi**  
*MSc Data Science and Analytics*  
📧 astittvamandloi@gmail.com  

> Designed as a demonstration of quantitative analysis and risk modelling skills for finance/data science roles.

---

## 📚 References

- Hull, J. *Risk Management and Financial Institutions*  
- Dowd, K. *Measuring Market Risk*  
- QuantStart: [https://www.quantstart.com](https://www.quantstart.com)  
- Streamlit Docs: [https://docs.streamlit.io](https://docs.streamlit.io)

---

### ⭐ Future Enhancements
- Add **GARCH volatility models**  
- Include **non-normal VaR (Cornish-Fisher expansion)**  
- Add **live data refresh**  
- Support **crypto assets**

---

### 🏁 Summary
This project is perfect for demonstrating:
✅ Quantitative modeling  
✅ Financial risk analysis  
✅ Streamlit app development  
✅ Interactive data visualization  
✅ Optimization and simulation in Python
