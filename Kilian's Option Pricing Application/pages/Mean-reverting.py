import streamlit as st
import numpy as np
from scipy.stats import norm


st.markdown("""
# Mean Reverting Process
""")
st.sidebar.markdown("### Mean-Reverting (OU)")

def mean_reverting_call_prices(S, K, T, r, sigma, kappa, theta):
    """
    Calculate the call option prices for a mean-reverting process.
    
    Parameters:
    S : float : Current stock price
    K : float : Strike price
    T : float : Time to maturity
    r : float : Risk-free interest rate"""

    f_t = 
