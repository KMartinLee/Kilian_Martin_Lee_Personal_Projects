# Contents of pages/Mean_Reverting.py
import streamlit as st
import numpy as np
from scipy.stats import norm
#from utils.Black_76_utils import (
#    black76_delta,
#    black76_gamma,
#    black76_vega,
#    black76_theta,
#    black76_rho,
#    black76_Vanna,
#    black76_volga
#)'''

st.markdown("# Mean-Reverting Model ")
st.sidebar.markdown("### Mean-Reverting (Ornstein-Uhlenbeck)")
st.sidebar.markdown("<br><br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)

# --- LinkedIn Footer ---
st.sidebar.markdown("---")
st.sidebar.markdown("#### Kilian Martin Lee's LinkedIn")
st.sidebar.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/kilian-martin-lee-0093a6256/)")

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

#-------------------------------------------------------------------------------------------
def numerical_delta(price_func, S, *args, h=1e-4):
    """Numerical approximation of Delta"""
    return (price_func(S + h, *args) - price_func(S - h, *args)) / (2 * h)

def numerical_vega(price_func, S, sigma, *args, h=1e-4):
    """Numerical approximation of Vega"""
    args_with_sigma_up = (S, sigma + h) + args
    args_with_sigma_down = (S, sigma - h) + args
    return (price_func(*args_with_sigma_up) - price_func(*args_with_sigma_down)) / (2 * h)
#--------------------------------------------------------------------------------------------
def delta_mean_reverting(S, sigma, K, T, r, kappa, theta):
    return numerical_delta(mean_reverting_call_price, S, K, T, r, sigma, kappa, theta)

def vega_mean_reverting(S, sigma, K, T, r, kappa, theta):
    return numerical_vega(mean_reverting_call_price, S, sigma, K, T, r, kappa, theta)
#--------------------------------------------------------------------------------------------






if st.button("Calculate Price"):
    price = mean_reverting_call_price(S, K, T, r, sigma, kappa, theta)
    st.success(f"Mean-Reverting Approx. Call Price: {price:.4f}")
    
    st.markdown(f"**Delta (numerical):** {delta_mean_reverting(S, sigma, K, T, r, kappa, theta):.4f}")
    st.markdown(f"**Vega (numerical):** {vega_mean_reverting(S, sigma, K, T, r, kappa, theta):.4f}")

