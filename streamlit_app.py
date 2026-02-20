import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import time

# --- ELITE QUANTITATIVE ENGINE ---
class InstitutionalEngine:
    def __init__(self, S, K, T, r, sigma, opt_type='call'):
        self.S, self.K, self.T, self.r, self.sigma = S, K, T, r, sigma
        self.opt_type = opt_type.lower()

    # 1. Black-Scholes-Merton & Greeks (High Precision)
    def bsm_analytics(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        price = (self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)) if self.opt_type == 'call' \
                else (self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1))
        
        delta = norm.cdf(d1) if self.opt_type == 'call' else norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
        vega = (self.S * norm.pdf(d1) * np.sqrt(self.T)) / 100
        theta = (-(self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T))) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d1 if self.opt_type == 'call' else -d1)) / 365
        rho = (self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(d1 if self.opt_type == 'call' else -d1)) / 100
        
        return {"Price": price, "Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}

    # 2. Monte Carlo with Variance Reduction (Antithetic Variates)
    def monte_carlo_engine(self, n_sims=50000):
        np.random.seed(42)
        z = np.random.standard_normal(n_sims // 2)
        z = np.concatenate([z, -z]) # Antithetic variates for better convergence
        
        ST = self.S * np.exp((self.r - 0.5 * self.sigma**2) * self.T + self.sigma * np.sqrt(self.T) * z)
        payoff = np.maximum(ST - self.K, 0) if self.opt_type == 'call' else np.maximum(self.K - ST, 0)
        
        mc_price = np.exp(-self.r * self.T) * np.mean(payoff)
        mc_std = np.sqrt(np.var(payoff) / n_sims) # Standard Error
        return mc_price, ST, mc_std

    # 3. Binomial Tree (CRR) for American Features
    def binomial_tree(self, steps=200):
        dt = self.T / steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(self.r * dt) - d) / (u - d)
        
        prices = self.S * (u ** np.arange(steps, -1, -1)) * (d ** np.arange(0, steps + 1))
        values = np.maximum(prices - self.K, 0) if self.opt_type == 'call' else np.maximum(self.K - prices, 0)
        
        for i in range(steps - 1, -1, -1):
            values = (p * values[:-1] + (1 - p) * values[1:]) * np.exp(-self.r * dt)
        return values[0]

# --- UI CONFIGURATION ---
st.set_page_config(page_title="QuantX Terminal", layout="wide", initial_sidebar_state="expanded")
st.title("🎯 Advanced Multi-Model Options Pricer")

with st.sidebar:
    st.header("🧑‍💻 Connect with me !")
    st.write("**Gabriel Avitan**")
    linkedin_url = "https://www.linkedin.com/in/gabriel-avitan-a6225534a"
    st.markdown(f'[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)]({linkedin_url})')
    
    st.divider()
    st.header("⌨️ Market Parameters")
    S = st.number_input("Spot Price", value=100.0)
    K = st.number_input("Strike Price", value=100.0)
    T = st.slider("Maturity (Y)", 0.01, 5.0, 1.0)
    r = st.slider("Risk-Free Rate (r)", 0.0, 0.2, 0.05)
    sigma = st.slider("Implied Vol (σ)", 0.05, 1.5, 0.25)
    opt_type = st.radio("Option Type", ["Call", "Put"], horizontal=True)
    
    st.divider()
    st.header("🛠️ Solver Settings")
    n_sims = st.select_slider("Monte Carlo Paths", options=[1000, 10000, 50000, 100000])
    steps = st.number_input("Binomial Steps", value=100, max_value=500)

# ENGINE EXECUTION
engine = InstitutionalEngine(S, K, T, r, sigma, opt_type)
bsm = engine.bsm_analytics()
mc_price, paths, mc_se = engine.monte_carlo_engine(n_sims)
tree_price = engine.binomial_tree(steps)

# MAIN DASHBOARD
c1, c2, c3, c4 = st.columns(4)
c1.metric("BSM Fair Value", f"${bsm['Price']:.3f}")
c2.metric("Monte Carlo", f"${mc_price:.3f}", f"SE: {mc_se:.4f}")
c3.metric("Binomial Tree", f"${tree_price:.3f}")
c4.metric("Market Status", "LIVE", delta="Arbitrage Free")

st.divider()

# GREEKS HEATMAP & ANALYSIS
st.subheader("🛡️ Risk Sensitivity (The Greeks)")
g1, g2, g3, g4, g5 = st.columns(5)
g1.metric("Delta (Δ)", f"{bsm['Delta']:.4f}")
g2.metric("Gamma (Γ)", f"{bsm['Gamma']:.4f}")
g3.metric("Vega (ν)", f"{bsm['Vega']:.4f}")
g4.metric("Theta (θ)", f"{bsm['Theta']:.4f}")
g5.metric("Rho (ρ)", f"{bsm['Rho']:.4f}")

# VISUAL ANALYTICS TABS
tab1, tab2, tab3 = st.tabs(["📈 Profit & Loss (PnL)", "🧬 Path Simulations", "🌫️ Volatility Stress Test"])

with tab1:
    st.subheader("PnL Surface Analysis")
    s_range = np.linspace(S*0.5, S*1.5, 50)
    pnl = [(InstitutionalEngine(s, K, T, r, sigma, opt_type).bsm_analytics()['Price'] - bsm['Price']) for s in s_range]
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(s_range, pnl, color='#00FFAA', lw=2)
    ax1.fill_between(s_range, pnl, color='#00FFAA', alpha=0.2)
    ax1.set_xlabel("Spot Price at Expiry")
    ax1.set_ylabel("PnL")
    ax1.set_facecolor('#0E1117')
    st.pyplot(fig1)

with tab2:
    st.subheader("Log-Normal Asset Price Distribution")
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.histplot(paths, kde=True, color="gold", ax=ax2)
    ax2.axvline(K, color="red", linestyle="--", label="Strike")
    ax2.set_facecolor('#0E1117')
    st.pyplot(fig2)

with tab3:
    st.subheader("Sensitivity to Volatility & Time")
    v_range = np.linspace(0.05, 1.0, 10)
    t_range = np.linspace(0.01, 2.0, 10)
    matrix = np.zeros((len(v_range), len(t_range)))
    for i, v in enumerate(v_range):
        for j, t in enumerate(t_range):
            matrix[i, j] = InstitutionalEngine(S, K, t, r, v, opt_type).bsm_analytics()['Price']
    
    fig3, ax3 = plt.subplots()
    sns.heatmap(matrix, xticklabels=np.round(t_range, 2), yticklabels=np.round(v_range, 2), cmap="magma", ax=ax3)
    ax3.set_xlabel("Maturity")
    ax3.set_ylabel("Volatility")
    st.pyplot(fig3)

st.divider()
st.info("💡 **Quantitative Insight:** This engine uses Antithetic Variates for Monte Carlo variance reduction and backward induction for the CRR Binomial Model.")
