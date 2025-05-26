import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib as plt
import matplotlib.pyplot as plt

st.markdown(r""""
            Black-76 Model
            
            """)
st.sidebar.markdown("### Black-76 Model")



def black76_model(f, k, t, r, sigma, option = "call"):

    d1 = (np.log(f / k) + (1/2) * sigma**2 * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    if option == "call":
        return np.exp(-r * t) * (f * norm.cdf(d1) - k * norm.cdf(d2))
    else:
        return np.exp(-r * t) * (k * norm.cdf(-d2) - f * norm.cdf(-d1))

def black76_delta(f, k, t, r, sigma, option="call"):
    d1 = (np.log(f / k) + (1/2) * sigma**2 * t) / (sigma * np.sqrt(t))
    if option == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1
    

f = st.number_input("Futures Price (F)", min_value= 0.0, max_value = 10000.0, step = 10.0, value = 100.0)
k = st.number_input("Strike Price (K)", min_value = 0.0, max_value= 10000.0, step = 10.0, value = 100.0)
t = st.number_input("Time to Maturity (T)", min_value = 0.0, max_value = 100.0, step = 1.0, value = 1.0)
r = st.number_input("Risk free rate (r)", min_value = -0.05, max_value= 0.3, step = 0.01, value = 0.05)
sigma = st.number_input("Volatility", min_value=0.0, max_value=1.0, step=0.01, value = 0.2)
option = st.selectbox("Option Type", ["call", "put"])

if st.button("Calculate Price"):
    price = black76_model(f, k, t, r, sigma, option)
    st.success(f"Black 76 {option} option price: {price:.4f}")


    st.markdown("""<br><br>""", unsafe_allow_html=True)
    st.markdown("""<br><br>""", unsafe_allow_html=True)


    #We are going to plot the relationship between price option and strike price
    st.subheader("Relationship between Option Price and Strike Price")
    K_values = np.linspace(60, 140, 100)

    prices_vs_K = [black76_model(f, K_i, t, r, sigma, option) for K_i in K_values]
    
    plt.plot( K_values, prices_vs_K, label=f'{option.title()} Price\n(f={f}, t={t}, r={r}, σ={sigma})')
    plt.axvline(k, color= 'gray', linestyle = '--', label ='Selected K')
    plt.ylabel("Option Price")
    plt.xlabel("Strike Price")
    plt.title("Option Price Sensitivity to Strike")
    plt.legend()
    st.pyplot(plt)


    st.markdown("""<br><br>""", unsafe_allow_html=True)
    st.markdown("""<br><br>""", unsafe_allow_html=True)

    #We are going to plot the relationship between price option and Time to maturity

    st.subheader("Option Price vs Time to Maturity")
    T_values = np.linspace(0.01, 2, 100)
    prices_vs_T = [black76_model(f, k, T_i, r, sigma, option) for T_i in T_values]
    plt.figure(figsize=(6, 4))
    plt.plot(T_values, prices_vs_T, label="Option Price")
    plt.xlabel("Time to Maturity (T)")
    plt.ylabel("Option Price")
    plt.title("Option Price Sensitivity to Time")
    plt.legend()
    st.pyplot(plt)


    st.markdown("""<br><br>""", unsafe_allow_html=True)
    st.markdown("""<br><br>""", unsafe_allow_html=True)


    st.subheader("Delta vs Strike Price")
    delta_values = [black76_delta(f, K_i, t, r, sigma, option ) for K_i in K_values]
    plt.figure(figsize=(6, 4))
    plt.plot(K_values, delta_values, label=f'{option.title()} Delta')
    plt.axvline(k, color='gray', linestyle='--', label='Selected K')
    plt.xlabel("Strike Price (K)")
    plt.ylabel("Delta")
    plt.title("Delta Sensitivity to Strike Price")
    plt.legend()
    st.pyplot(plt)
