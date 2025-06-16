# Contents of pages/Mean_Reverting.py
import streamlit as st
import numpy as np
from scipy.stats import norm

st.markdown("# Mean-Reverting Model 📙")
st.sidebar.markdown("### Mean-Reverting (OU)")

def mean_reverting_call_price(S, K, T, r, sigma, kappa, theta):
    f_t = theta + (S - theta) * np.exp(-kappa * T)
    sigma_eff = sigma * np.sqrt((1 - np.exp(-2 * kappa * T)) / (2 * kappa * T))
    d1 = (np.log(f_t / K) + 0.5 * sigma_eff**2 * T) / (sigma_eff * np.sqrt(T))
    d2 = d1 - sigma_eff * np.sqrt(T)
    return f_t * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

S = st.number_input("Spot Price (S)", value=100.0)
K = st.number_input("Strike Price (K)", value=100.0)
T = st.number_input("Maturity (T)", value=1.0)
r = st.number_input("Risk-Free Rate (r)", value=0.05)
sigma = st.number_input("Volatility (σ)", value=0.2)
kappa = st.number_input("Mean Reversion Speed (κ)", value=1.0)
theta = st.number_input("Long-term Mean (θ)", value=100.0)

if st.button("Calculate Price"):
    price = mean_reverting_call_price(S, K, T, r, sigma, kappa, theta)
    st.success(f"Mean-Reverting Approx. Call Price: {price:.4f}")
