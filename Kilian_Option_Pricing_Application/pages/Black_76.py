import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib as plt
import matplotlib.pyplot as plt

from utils.Black_76_utils import (
    plot_price_vs_strike,
    plot_price_vs_time,
    plot_delta_vs_strike,
    render_latex_greek_explanations
)
from utils.Black_76_core import (
    black76_model,
    black76_delta,
)

#from utils.Black_76_utils import (
#    black76_delta,
#    black76_gamma,
#    black76_vega,
#    black76_theta,
#    black76_rho,
#    black76_Vanna,
#    black76_volga
#)'''


st.sidebar.markdown("### Black-76 Model")
st.sidebar.markdown("## Kilian Martin Lee's Linkedin")
st.sidebar.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/kilian-martin-lee-0093a6256/)")

"""
def black76_model(f, k, t, r, sigma, option = "call"):

    d1 = (np.log(f / k) + (1/2) * sigma**2 * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    if option == "call":
        return np.exp(-r * t) * (f * norm.cdf(d1) - k * norm.cdf(d2))
    else:
        return np.exp(-r * t) * (k * norm.cdf(-d2) - f * norm.cdf(-d1))

def black76_delta(f, k, t, r, sigma, option="call"):
    d1 = (np.log(f / k) + (1/2) * sigma**2 * t) / (sigma * np.sqrt(t))
    discount = np.exp(-r * t)
    if option == "call":
        return discount * norm.cdf(d1)
    else:
        return - discount * norm.cdf(- d1) 
    
def black76_gamma(f, k, t, r, sigma, option = "call"):
    d1 = (np.log(f / k) + (1/2) * sigma**2 * t) / (sigma * np.sqrt(t))
    return np.exp(-r*t) * (norm.pdf(d1)) / (f * sigma * np.sqrt(t))

def black76_vega(f, k, t, r, sigma, option = "call"):
    d1 = (np.log(f / k) + (1/2) * sigma**2 * t) / (sigma * np.sqrt(t))
    return f * np.exp(-r * t) * (norm.pdf(d1) * np.sqrt(t))

def black76_theta(f, k, t, r, sigma, option = "call"):
    d1 = (np.log(f / k) + (1/2) * sigma**2 * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if option == 'call':
        return (- (f * np.exp(-r*t) * norm.pdf(d1) * sigma) / 2* np.sqrt(t)) - r * k * np.exp(-r * t) * norm.cdf(d2) + r * f * np.exp(-r*t) * norm.cdf(d1)
    else:
        return (- (f * np.exp(-r * t) * norm.pdf(d1) * sigma) / 2* np.sqrt(t)) + r * k * np.exp(-r * t) * norm.cdf(-d2) - r * f * np.exp(-r*t) * norm.cdf(-d1)

def black76_rho(f, k, t, r, sigma, option = "call"):
    d1 = (np.log(f / k) + (1/2) * sigma**2 * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if option == 'call':
        return - t * np.exp(-r*t) * (f * norm.cdf(d1) - k * norm.cdf(d2))
    else: 
        return - t * np.exp(-r*t) * (k * norm.cdf(-d2) - f * norm.cdf(-d1))
    
def black76_Vanna(f, k, t, r, sigma, option = "call"):
    d1 = (np.log(f / k) + (1/2) * sigma**2 * t) / (sigma * np.sqrt(t))
    vega = black76_vega(f, k, t, r, sigma, option = "call")
    vanna = ( vega / f ) * (1 - (d1 / sigma * np.sqrt(t)))
    return vanna

def black76_volga(f, k, t, r, sigma, option = "call"):
    d1 = (np.log(f / k) + (1/2) * sigma**2 * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    vega = black76_vega(f, k, t, r, sigma, option = "call")
    return vega * (d1 * d2) / sigma

"""


st.markdown(r"""
            Black-76 Model
            """)

with st.expander("Black76 Formula"):
        st.markdown(r"""
            $\displaystyle Call = e^{-rT} \left[F N(d_1) - K N(d_2)\right]$ <br> <br>
            $\displaystyle Put = e^{-rT} \left[K N(-d_2) - F N(-d_1)\right]$ <br> <br>""", unsafe_allow_html=True)
        

        col1, col2 = st.columns([1, 5])  # Adjust ratio as needed

        with col1:
            st.latex(r"""
            \begin{array}{rl}
            \text{with} \quad d_1 &= \frac{\ln(F/K) + (\sigma^2 / 2) T}{\sigma \sqrt{T}} \\
                d_2 &= \frac{\ln(F/K) - (\sigma^2 / 2) T}{\sigma \sqrt{T}}
            \end{array} \\
                    
            
            \text{Assumptions:}\\
            \text{- No arbitrage and frictionless markets}\\
            \text{- Constant volatility and interest rate}\\
            \text{- European-style option on futures}\\
            """)



f = st.number_input("Futures Price (F)", min_value= 0.0, max_value = 10000.0, step = 10.0, value = 100.0)
k = st.number_input("Strike Price (K)", min_value = 0.0, max_value= 10000.0, step = 10.0, value = 100.0)
t = st.number_input("Time to Maturity (T) in years", min_value = 0.0, max_value = 100.0, step = 1.0, value = 1.0)
r = st.number_input("Risk free rate (r)", min_value = -0.05, max_value= 0.3, step = 0.01, value = 0.05)
sigma = st.number_input("Volatility", min_value=0.0, max_value=1.0, step=0.01, value = 0.2)
option = st.selectbox("Option Type", ["call", "put"])


if st.button("Calculate Price"):
    price = black76_model(f, k, t, r, sigma, option)
    st.success(f"Black 76 {option} option price: {price:.4f}")

    # Greeks & LaTeX
    render_latex_greek_explanations()
    
    # Plots
    plot_price_vs_strike(f, k, t, r, sigma, option)
    plot_price_vs_time(f, k, r, sigma, option)
    plot_delta_vs_strike(f, k, t, r, sigma, option)


