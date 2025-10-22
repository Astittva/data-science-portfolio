# 📄 app.py
# =============================================================================
# Market Risk Simulation Dashboard — Monte Carlo & Parametric VaR
# -----------------------------------------------------------------------------
# Clean and stable version with detailed explanations in comments.
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.stats import norm

# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Market Risk Simulation Dashboard", page_icon="📉", layout="wide")
st.title("📉 Market Risk Simulation Dashboard — Monte Carlo & Parametric VaR")
st.caption("Built with Python • Streamlit • NumPy • Pandas • Plotly • SciPy • yFinance")

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def fetch_prices(tickers, start, end):
    """Download adjusted close prices."""
    if not tickers:
        return pd.DataFrame()
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = data["Close"] if "Close" in data else data
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    return prices.dropna(how="all")

def to_log_returns(prices):
    """Compute daily log returns."""
    return np.log(prices / prices.shift(1)).dropna(how="any")

def normalize_weights(weights_dict):
    """Normalize weights to sum to 1."""
    total = sum(weights_dict.values())
    if total == 0:
        return {k: 1 / len(weights_dict) for k in weights_dict}
    return {k: v / total for k, v in weights_dict.items()}

def portfolio_returns(returns, weights):
    """Compute portfolio returns."""
    w = np.array([weights[t] for t in returns.columns])
    return returns.dot(w)

def historical_bootstrap(port_ret, n_sims):
    """Monte Carlo bootstrap sampling."""
    return np.random.choice(port_ret.values, size=n_sims, replace=True)

def var_es_from_samples(samples, alpha):
    """Compute VaR and ES from simulated samples."""
    q = np.quantile(samples, 1 - alpha)
    es = samples[samples <= q].mean() if np.any(samples <= q) else q
    return q, es

def parametric_one_day_var_es(mu, sigma, alpha):
    """Parametric VaR/ES (Normal distribution)."""
    z = norm.ppf(1 - alpha)
    var = mu + z * sigma
    es = mu - sigma * norm.pdf(z) / (1 - alpha)
    return var, es

def scale_to_n_days(one_day_val, n_days):
    """Scale 1-day VaR/ES to N-day via sqrt(time)."""
    return one_day_val * np.sqrt(n_days)

# -----------------------------------------------------------------------------
# Efficient Frontier (bonus section)
# -----------------------------------------------------------------------------
def optimize_for_return(target_mu, returns):
    n = returns.shape[1]
    mu_vec = returns.mean().values
    cov = returns.cov().values

    def obj(w): return w @ cov @ w
    cons = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: mu_vec @ w - target_mu},
    )
    bounds = [(0, 1)] * n
    w0 = np.ones(n) / n
    res = minimize(obj, w0, constraints=cons, bounds=bounds)
    return res

def efficient_frontier(returns, points=25):
    if returns.shape[1] < 2:
        return pd.DataFrame(columns=["mu", "sigma"])
    mu_vec = returns.mean().values
    min_mu, max_mu = mu_vec.min(), mu_vec.max()
    targets = np.linspace(min_mu, max_mu, points)
    out = []
    for t in targets:
        res = optimize_for_return(t, returns)
        if res.success:
            cov = returns.cov().values
            sigma_p = np.sqrt(res.x @ cov @ res.x)
            out.append([t, sigma_p])
    return pd.DataFrame(out, columns=["mu", "sigma"])

# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
st.sidebar.header("Portfolio Settings")

default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
tickers = st.sidebar.multiselect("Select tickers", default_tickers, default_tickers[:3])
start = st.sidebar.date_input("Start date", pd.to_datetime("2020-01-01"))
end = st.sidebar.date_input("End date", pd.to_datetime("today"))
n_days = st.sidebar.slider("VaR horizon (days)", 1, 30, 5)
num_sims = st.sidebar.slider("Monte Carlo samples", 1000, 50000, 10000, step=1000)
conf_label = st.sidebar.selectbox("Confidence level", ["95%", "97.5%", "99%"], index=0)
alpha = {"95%": 0.95, "97.5%": 0.975, "99%": 0.99}[conf_label]
method = st.sidebar.radio("VaR type", ["Monte Carlo (Bootstrap)", "Parametric (Normal)"])
run = st.sidebar.button("🚀 Run simulation")

# -----------------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------------
if run:
    with st.spinner("Fetching market data..."):
        prices = fetch_prices(tickers, start, end)

    if prices.empty:
        st.error("No data fetched. Try different tickers or dates.")
        st.stop()

    rets = to_log_returns(prices)

    st.subheader("1️⃣ Data Summary")
    st.write(f"Period: **{rets.index[0].date()} → {rets.index[-1].date()}**, {len(rets)} observations.")
    # ✅ Fixed line below — now transposes describe()
    st.dataframe(rets.describe().T[["mean", "std", "min", "max"]])

    # Custom weights
    st.subheader("2️⃣ Portfolio Weights")
    st.caption("Adjust weights; they’ll be normalized to sum to 1.")
    raw_weights = {}
    cols = st.columns(len(tickers)) if tickers else [st]
    for i, t in enumerate(tickers):
        raw_weights[t] = cols[i].number_input(f"{t}", value=1.0, min_value=0.0, step=0.1)
    weights = normalize_weights(raw_weights)
    st.write("Normalized Weights:", weights)

    # Portfolio returns
    port_ret = portfolio_returns(rets, weights)

    # Correlation heatmap
    st.subheader("3️⃣ Correlation Heatmap")
    fig_corr = px.imshow(rets.corr(), text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
    st.plotly_chart(fig_corr, use_container_width=True)

    # VaR computation
    st.subheader("4️⃣ VaR & ES Estimation")

    if method == "Monte Carlo (Bootstrap)":
        sims = historical_bootstrap(port_ret, num_sims)
        var_1d, es_1d = var_es_from_samples(sims, alpha)
        fig = px.histogram(pd.DataFrame({"Simulated Returns": sims}), x="Simulated Returns",
                           nbins=60, color_discrete_sequence=["#0072B2"])
        fig.add_vline(x=var_1d, line_dash="dash", annotation_text=f"VaR {conf_label}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        mu, sigma = port_ret.mean(), port_ret.std()
        var_1d, es_1d = parametric_one_day_var_es(mu, sigma, alpha)
        x = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 400)
        pdf = norm.pdf((x - mu) / sigma) / sigma
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=pdf, mode="lines", name="Normal PDF"))
        fig.add_vline(x=var_1d, line_dash="dash", annotation_text=f"VaR {conf_label}")
        fig.update_layout(title="Parametric Normal Distribution", yaxis_title="Density")
        st.plotly_chart(fig, use_container_width=True)

    # N-day scaling
    var_nd = scale_to_n_days(var_1d, n_days)
    es_nd = scale_to_n_days(es_1d, n_days)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"{conf_label} VaR (1-day)", f"{var_1d*100:.2f}%")
    c2.metric(f"{conf_label} VaR ({n_days}-day)", f"{var_nd*100:.2f}%")
    c3.metric(f"Expected Shortfall ({n_days}-day)", f"{es_nd*100:.2f}%")

    # Backtesting
    st.subheader("5️⃣ Backtesting — VaR Exceptions")
    mu_b, sigma_b = port_ret.mean(), port_ret.std()
    var_line = mu_b - norm.ppf(alpha) * sigma_b
    breaches = port_ret < var_line

    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=port_ret.index, y=port_ret.values*100, mode="lines", name="Daily Return (%)"))
    fig_bt.add_trace(go.Scatter(x=port_ret.index, y=np.full(len(port_ret), var_line*100),
                                mode="lines", name=f"{conf_label} VaR line", line=dict(color="red", dash="dash")))
    fig_bt.add_trace(go.Scatter(x=port_ret.index, y=np.where(breaches, port_ret.values*100, np.nan),
                                mode="markers", name="Breaches", marker=dict(color="red", size=6)))
    fig_bt.update_layout(title="Historical VaR Backtesting", yaxis_title="Return (%)")
    st.plotly_chart(fig_bt, use_container_width=True)

    st.write(f"Exceptions: **{int(breaches.sum())} days** (~{breaches.mean()*100:.2f}% "
             f"vs expected {(1-alpha)*100:.2f}%)")

    # Efficient Frontier
    st.subheader("6️⃣ Efficient Frontier (Mean-Variance)")
    ef = efficient_frontier(rets, points=25)
    if not ef.empty:
        ef_plot = ef.copy()
        ef_plot["mu_annual"] = ef_plot["mu"] * 252
        ef_plot["sigma_annual"] = ef_plot["sigma"] * np.sqrt(252)
        fig_ef = px.scatter(ef_plot, x="sigma_annual", y="mu_annual",
                            title="Efficient Frontier (Annualized)",
                            labels={"sigma_annual": "Volatility", "mu_annual": "Expected Return"})
        st.plotly_chart(fig_ef, use_container_width=True)
    else:
        st.info("Efficient frontier needs ≥ 2 tickers.")

    # Notes
    st.markdown("""
    ---
    ### 🧠 Concepts Recap
    - **Log Returns:** additive and stable for volatility modelling  
    - **Monte Carlo VaR:** samples from history (non-parametric)  
    - **Parametric VaR:** analytic under normality (fast but idealized)  
    - **√Time Rule:** scales risk to multiple days  
    - **Backtesting:** checks model calibration  
    - **Efficient Frontier:** shows optimal portfolios for given risk/return trade-off  
    ---
    """)
else:
    st.info("Use the sidebar to select parameters, then click **Run simulation**.")
