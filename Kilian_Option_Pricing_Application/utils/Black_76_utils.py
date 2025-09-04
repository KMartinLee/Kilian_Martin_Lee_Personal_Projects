import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm
from utils.Black_76_core import (
    black76_model, 
    black76_delta, 
    black76_gamma,
    black76_vega,
    black76_theta,
    black76_rho,
    black76_Vanna,
    black76_volga   # import other greeks as needed
)

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


def render_latex_greek_explanations(f, k, t, r, sigma, option):
    st.markdown("### Greeks Summary")

    # --- Delta ---
    st.markdown("**Delta** – Sensitivity to underlying futures price")
    st.latex(r"""
    \Delta_{\text{call}} = e^{-rT} \cdot \Phi(d_1), \quad
    \Delta_{\text{put}} = -e^{-rT} \cdot \Phi(-d_1)
    """)
    st.markdown(f"**Delta:** `{black76_delta(f, k, t, r, sigma, option):.4f}`")

    # --- Gamma ---
    st.markdown("**Gamma** – Sensitivity of delta to underlying price")
    st.latex(r"""
    \Gamma = e^{-rT} \cdot \frac{\phi(d_1)}{F \sigma \sqrt{T}}
    """)
    st.markdown(f"**Gamma:** `{black76_gamma(f, k, t, r, sigma, option):.4f}`")

    # --- Vega ---
    st.markdown("**Vega** – Sensitivity to volatility")
    st.latex(r"""
    \mathcal{V} = F e^{-rT} \phi(d_1) \sqrt{T}
    """)
    st.markdown(f"**Vega:** `{black76_vega(f, k, t, r, sigma, option):.4f}`")

    # --- Theta ---
    st.markdown("**Theta** – Sensitivity to time decay")
    st.latex(r"""
    \theta_{\text{call}} = - \frac{F e^{-rT} \phi(d_1) \sigma}{2 \sqrt{T}} - r K e^{-rT} \Phi(d_2) + r F e^{-rT} \Phi(d_1)
    """)
    st.markdown(f"**Theta:** `{black76_theta(f, k, t, r, sigma, option):.4f}`")

    # --- Rho ---
    st.markdown("**Rho** – Sensitivity to interest rate")
    st.latex(r"""
    \rho_{\text{call}} = - T e^{-rT} \left[ F \Phi(d_1) - K \Phi(d_2) \right]
    """)
    st.markdown(f"**Rho:** `{black76_rho(f, k, t, r, sigma, option):.4f}`")

    # --- Vanna ---
    st.markdown("**Vanna** – Sensitivity of delta to volatility")
    st.latex(r"""
    \mathcal{V}_{\text{anna}} = \frac{\mathcal{V}}{F} \left[ 1 - \frac{d_1}{\sigma \sqrt{T}} \right]
    """)
    st.markdown(f"**Vanna:** `{black76_Vanna(f, k, t, r, sigma, option):.4f}`")

    # --- Volga ---
    st.markdown("**Volga** – Second-order sensitivity to volatility")
    st.latex(r"""
    \mathcal{V}_{\text{olga}} = \mathcal{V} \cdot \frac{d_1 d_2}{\sigma}
    """)
    st.markdown(f"**Volga:** `{black76_volga(f, k, t, r, sigma, option):.4f}`")
