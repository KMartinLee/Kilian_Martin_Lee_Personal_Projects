import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm
from pages.Black_76 import black76_model, black76_delta  # import other greeks as needed

# --- Plot 1: Option Price vs Strike Price ---
def plot_price_vs_strike(f, k, t, r, sigma, option):
    K_values = np.linspace(60, 140, 100)
    prices = [black76_model(f, K_i, t, r, sigma, option) for K_i in K_values]

    plt.figure()
    plt.plot(K_values, prices, label=f'{option.title()} Price (f={f}, t={t}, r={r}, σ={sigma})')
    plt.axvline(k, color='gray', linestyle='--', label='Selected K')
    plt.xlabel("Strike Price")
    plt.ylabel("Option Price")
    plt.title("Option Price Sensitivity to Strike")
    plt.legend()
    st.pyplot(plt)


# --- Plot 2: Option Price vs Time ---
def plot_price_vs_time(f, k, r, sigma, option):
    T_values = np.linspace(0.01, 2, 100)
    prices = [black76_model(f, k, T_i, r, sigma, option) for T_i in T_values]

    plt.figure()
    plt.plot(T_values, prices)
    plt.xlabel("Time to Maturity (T)")
    plt.ylabel("Option Price")
    plt.title("Option Price Sensitivity to Time")
    plt.legend()
    st.pyplot(plt)


# --- Plot 3: Delta vs Strike Price ---
def plot_delta_vs_strike(f, k, t, r, sigma, option):
    K_values = np.linspace(60, 140, 100)
    deltas = [black76_delta(f, K_i, t, r, sigma, option) for K_i in K_values]

    plt.figure()
    plt.plot(K_values, deltas, label=f'{option.title()} Delta')
    plt.axvline(k, color='gray', linestyle='--', label='Selected K')
    plt.xlabel("Strike Price (K)")
    plt.ylabel("Delta")
    plt.title("Delta Sensitivity to Strike Price")
    plt.legend()
    st.pyplot(plt)


# --- Render LaTeX / Markdown Explanations ---
def render_latex_greek_explanations():
    st.markdown("### Greeks Summary")
    st.markdown("#### Delta")
    st.latex(r"""
    \Delta = \frac{\partial V}{\partial S}
    """)
    st.markdown("#### Gamma")
    st.latex(r"""
    \Gamma = \frac{\partial^2 V}{\partial S^2}
    """)
    # Add others as needed: Vega, Theta, Rho, Vanna, Volga
