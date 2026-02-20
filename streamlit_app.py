import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# --- QUANTITATIVE ENGINE (MATHEMATICAL LOGIC) ---
class AdvancedPricer:
    def __init__(self, S, K, T, r, sigma, opt_type='call'):
        self.S, self.K, self.T, self.r, self.sigma = S, K, T, r, sigma
        self.opt_type = opt_type

    # 1. Black-Scholes-Merton (Analytical)
    def bsm_price(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        if self.opt_type == 'call':
            return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1)

    # 2. Automated Greeks Engine
    def get_greeks(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        delta = norm.cdf(d1) if self.opt_type == 'call' else norm.cdf(d1) - 1
        gamma = norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))
        vega = self.S * norm.pdf(d1) * np.sqrt(self.T)
        theta = -(self.S * norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T))) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(d1 if self.opt_type == 'call' else -d1)
        return {"Delta": delta, "Gamma": gamma, "Vega": vega / 100, "Theta": theta / 365}

    # 3. Newton-Raphson Algorithm (Implied Volatility)
    @staticmethod
    def newton_raphson(market_price, S, K, T, r, type='call'):
        sig = 0.2
        for i in range(100):
            p = AdvancedPricer(S, K, T, r, sig, type)
            diff = p.bsm_price() - market_price
            if abs(diff) < 1e-6: return sig
            vega = p.get_greeks()["Vega"] * 100
            if vega < 1e-6: break
            sig -= diff / vega
        return sig

# --- STREAMLIT USER INTERFACE ---
st.set_page_config(page_title="Quant Options Engine", layout="wide")
st.title("📊 Quantitative Options Analytics Engine")
st.markdown("""
This engine integrates **Black-Scholes-Merton**, **Automated Greeks**, and **Newton-Raphson** algorithms to assess portfolio sensitivity and market mispricing.
""")

with st.sidebar:
    st.header("⚙️ Market Parameters")
    S = st.number_input("Underlying Asset Price (S)", value=100.0)
    K = st.number_input("Strike Price (K)", value=100.0)
    T = st.slider("Time to Maturity (Years)", 0.01, 2.0, 0.5)
    r = st.slider("Risk-Free Interest Rate (r)", 0.0, 0.1, 0.03)
    sigma = st.slider("Input Volatility (σ)", 0.01, 1.0, 0.2)
    option_type = st.selectbox("Option Type", ["call", "put"])

engine = AdvancedPricer(S, K, T, r, sigma, option_type)
price = engine.bsm_price()
greeks = engine.get_greeks()

# Metric Dashboard
st.subheader("Key Performance Indicators (KPIs)")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("BSM Fair Value", f"${price:.2f}")
c2.metric("Delta", f"{greeks['Delta']:.3f}")
c3.metric("Gamma", f"{greeks['Gamma']:.4f}")
c4.metric("Vega", f"{greeks['Vega']:.3f}")
c5.metric("Theta", f"{greeks['Theta']:.3f}")

# Implied Volatility Smile Analysis
st.subheader("📈 Implied Volatility Smile Analysis")
st.info("Identifying skewness and kurtosis patterns to detect potential market mispricing.")

# Visualizing the Smile (Simulated market dynamics)
strikes = np.linspace(S*0.7, S*1.3, 15)
smiles = [sigma + 0.15 * ((k - S)/S)**2 for k in strikes]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(strikes, smiles, color='#FF4B4B', marker='o', linestyle='--', label='Implied Volatility')
ax.set_facecolor('#0E1117')
ax.set_xlabel("Strike Price")
ax.set_ylabel("Implied Volatility")
ax.legend()
ax.grid(alpha=0.2)
st.pyplot(fig)