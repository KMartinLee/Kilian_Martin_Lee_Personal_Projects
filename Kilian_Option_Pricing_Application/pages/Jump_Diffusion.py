# Contents of pages/Jump_Diffusion.py
import streamlit as st
import numpy as np
from scipy.stats import norm
import math
#from utils.Black_76_utils import (
#    black76_delta,
#    black76_gamma,
#    black76_vega,
#    black76_theta,
#    black76_rho,
#    black76_Vanna,
#    black76_volga
#)'''


st.markdown("# Jump Diffusion Model ")
st.sidebar.markdown("### Jump Diffusion")
st.sidebar.markdown("## Kilian Martin Lee's Linkedin")
st.sidebar.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/kilian-martin-lee-0093a6256/)")

def black76(f, k, t, r, sigma, option_type='call'):
    d1 = (np.log(f / k) + 0.5 * sigma**2 * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if option_type == 'call':
        return np.exp(-r * t) * (f * norm.cdf(d1) - k * norm.cdf(d2))
    else:
        return np.exp(-r * t) * (k * norm.cdf(-d2) - f * norm.cdf(-d1))

def jump_diffusion_price(S, K, T, r, sigma, lamb, mu_j, sigma_j, option_type='call', n_terms=40):
    price = 0
    for k in range(n_terms):
        sigma_k = np.sqrt(sigma**2 + (k * sigma_j**2) / T)
        r_k = r - lamb * (mu_j - 1) + (k * np.log(mu_j)) / T
        poisson_prob = np.exp(-lamb * T) * (lamb * T)**k / math.factorial(k)
        bs_price = black76(S, K, T, r_k, sigma_k, option_type)
        price += poisson_prob * bs_price
    return price

S = st.number_input("Spot Price (S)", value=100.0)
K = st.number_input("Strike Price (K)", value=100.0)
T = st.number_input("Time to Maturity (T)", value=1.0)
r = st.number_input("Risk-Free Rate (r)", value=0.05)
sigma = st.number_input("Volatility (σ)", value=0.2)
lamb = st.number_input("Jump Frequency (λ)", value=0.75)
mu_j = st.number_input("Jump Mean Size (μ_j)", value=1.0)
sigma_j = st.number_input("Jump Volatility (σ_j)", value=0.3)
opt_type = st.selectbox("Option Type", ["call", "put"])


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
def delta_jump(S, sigma, K, T, r, lamb, mu_j, sigma_j, option_type='call'):
    return numerical_delta(jump_diffusion_price, S, K, T, r, sigma, lamb, mu_j, sigma_j, option_type)

def vega_jump(S, sigma, K, T, r, lamb, mu_j, sigma_j, option_type='call'):
    return numerical_vega(jump_diffusion_price, S, sigma, K, T, r, lamb, mu_j, sigma_j, option_type)
#--------------------------------------------------------------------------------------------


if st.button("Calculate Price"):
    price = jump_diffusion_price(S, K, T, r, sigma, lamb, mu_j, sigma_j, opt_type)
    st.success(f"Jump Diffusion {opt_type} price: {price:.4f}")
    st.markdown(f"**Delta (numerical):** {delta_jump(S, sigma, K, T, r, lamb, mu_j, sigma_j, opt_type):.4f}")
    st.markdown(f"**Vega (numerical):** {vega_jump(S, sigma, K, T, r, lamb, mu_j, sigma_j, opt_type):.4f}")











