# Advanced Options Pricing Terminal

A multi-model derivatives valuation engine built in Python, integrating 
Black-Scholes-Merton, Cox-Ross-Rubinstein Binomial Tree, and Antithetic 
Monte Carlo simulations for cross-model validation and risk analysis.

---

## Motivation

During my internship at BGC Group (Euro Interest Rate Swaps desk, London, 
2025), I observed daily derivatives pricing workflows and the practical 
constraints of applying theoretical models in a real-time market environment. 
This project was built to consolidate that experience, creating a unified 
terminal that runs three pricing models simultaneously, extracts Implied 
Volatility from observed market prices, and stress-tests assumptions across 
the volatility surface.

---

## Problem Statement

Pricing equity derivatives accurately requires choosing between models with 
fundamentally different assumptions and computational trade-offs:
- BSM is fast but assumes constant volatility (misprices tails)
- Binomial Tree handles early exercise but is computationally expensive
- Monte Carlo is flexible but converges slowly without variance reduction

This terminal addresses all three simultaneously, enabling cross-model 
arbitrage detection and sensitivity analysis across the vol-time surface.

---

## Architecture & Implementation

Strictly modular object-oriented design (OOP). Three independent engines, 
one unified interface.

**1. Black-Scholes-Merton (Analytical)**
Closed-form pricing for European contracts. Greeks computed analytically 
(Delta, Gamma, Vega, Theta, Rho).

**2. Cox-Ross-Rubinstein Binomial Tree (N=200+ steps)**
Backward induction supporting both European and American options. 
Identifies optimal early-exercise boundaries for American puts/calls.

**3. Monte Carlo — Antithetic Variates (100,000+ paths)**
GBM-based path simulation with Antithetic Variates variance reduction, 
materially lowering standard error on the risk-neutral expectation. 
Confidence intervals reported on all outputs.

**Implied Volatility Solver**
Newton-Raphson iterative algorithm extracting IV from observed market 
prices by minimizing the cost function between model fair value and 
market quote. Convergence typically achieved in <10 iterations.

---

## Validation

Backtested against 25 SPY call and put options (strikes ranging from 
-15% to +15% moneyness, expiry March 2026), sourced from Yahoo Finance.
Market prices computed as bid-ask midpoint.

| Metric                         | Result        |
|--------------------------------|---------------|
| BSM vs market (mean abs. error)| 2.3%          |
| Monte Carlo std. error (N=100k)| 0.004         |
| IV solver convergence rate     | 97% of cases  |


---

## Risk & Stress-Testing

- **Greeks Engine:** First and second-order sensitivities (Delta, Gamma, 
  Vega, Theta, Rho), second-order Gamma critical for hedging nonlinear 
  exposure.
- **IV Smile Analysis:** IV plotted across strike range to identify skewness 
  and kurtosis patterns indicative of tail-risk mispricing.
- **Stress-Testing Heatmap:** 2D sensitivity visualization across the 
  vol-time surface via interactive Streamlit dashboard.

---

## Known Limitations

- BSM and CRR assume constant volatility : no vol surface / SABR modelling
- Monte Carlo runtime scales linearly with path count (not real-time optimized)
- No dividend adjustment currently implemented
- Next: Heston stochastic volatility model integration

---

## Stack

Python 3.11 | NumPy | SciPy | Pandas | Plotly | Streamlit
