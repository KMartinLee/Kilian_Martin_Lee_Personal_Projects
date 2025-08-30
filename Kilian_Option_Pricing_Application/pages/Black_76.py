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


'''
if st.button("Calculate Price"):
    price = black76_model(f, k, t, r, sigma, option)
    st.success(f"Black 76 {option} option price: {price:.4f}")

    
    #DELTA
    st.markdown("""
    **Delta of the option**  
    Rate of change in option price with respect to the underlying futures price (1st derivative).  
    Proxy for probability of the option expiring in the money.
    """)
    st.latex(r"""
    \Delta = \frac{\partial V}{\partial S}
    """)
    st.latex(r"""
    \Delta_{\text{call}} = e^{-rT} \cdot \Phi(d_1)
    """)
    st.latex(r"""
    \Delta_{\text{put}} = -e^{-rT} \cdot \Phi(-d_1)
    """)
    st.markdown(f"**Delta of this option:** {(black76_delta(f, k,t, r, sigma, option )):.4f}")

    
    #GAMMA
    st.markdown("""
    **Gamma of the option** <br>Rate of change in delta with respect to the underlying stock price (2nd derivative).""", unsafe_allow_html=True)
    st.latex(r"""
    \Gamma_{\text{call or put}} = \frac{\partial^2 V}{\partial S^2} 
    = \frac{\partial \Delta}{\partial S} 
    = e^{-rT} \cdot \frac{\phi(d_1)}{F \sigma \sqrt{T}}
    """)
    st.markdown(f"**Gamma of this option:** {black76_gamma(f, k, t, r, sigma, option):.4f}")


    #VEGA
    st.markdown("""
    **Vega of the option** <br>
    Rate of change in option price with respect to the volatility of underlying futures contract. <br>
    """,unsafe_allow_html=True)
    st.latex(r"""
    \mathcal{V} = \frac{\partial V}{\partial\sigma} = Fe^{-rT}\phi(d_1)\sqrt{T}
    """)
    st.markdown(f"""**Vega of this option is:** {black76_vega(f, k, t, r, sigma, option = "call")}
    """)

    #Theta
    st.markdown("""
    <strong>Theta of the option</strong> <br>
    Rate of change in option price with respect to time (i.e. time decay).
    """,unsafe_allow_html=True)
    st.latex(r"""
    \theta_{\text{call}} = \frac{\partial V}{\partial t} = - \frac{F e^{-rT} \phi(d_1) \sigma}{2 \sqrt{T}} - r K e^{-rT} \Phi(d_2) + r F e^{-rT} \Phi(d_1) \\[2ex]
    \theta_{\text{put}} = \frac{\partial V}{\partial t} = - \frac{F e^{-rT} \phi(d_1) \sigma}{2 \sqrt{T}} + r K e^{-rT} \Phi(-d_2) - r F e^{-rT} \Phi(-d_1)
    """)
    st.markdown(f""" **Theta of this option is:** {black76_theta(f, k, t, r, sigma, option = "call")}""")

    #Rho
    st.markdown("""
    <strong>Rho of the option</strong> <br>
    Rate of change in option price with respect to the risk-free rate.      
    """, unsafe_allow_html=True)
    st.latex(r"""
    \rho_{\text{call}} = \frac{\partial V}{\partial r} = - T e^{-rT} \left[ F \Phi(d_1) - K \Phi(d2) \right]  \\[2ex]
    \rho_{\text{put}}  = \frac{\partial V}{\partial r} = - T e^{-rT} \left[K \Phi(-d_2) - F \Phi(-d_1) \right]
    """)
    st.markdown(f"""**Rho of this option is:** {black76_rho(f, k, t, r, sigma, option = "call")}""")

    #Vanna
    st.markdown("""
    <strong>Vanna of the option</strong> <br>
    Sensitivity of delta with respect to change in volatility.
    """,unsafe_allow_html=True)
    st.latex(r"""
    \mathcal{V}_{\text{anna}}= \frac{\partial^2 V}{\partial S \, \partial \sigma} =\frac{\mathcal{V}}{F} \left[ 1 - \frac{d_1}{\sigma \sqrt{T}} \right]
    """)
    st.markdown(f"""
    **Vanna of the option is:** {black76_Vanna(f, k, t, r, sigma, option = "call")}
    """)

    #Volga
    st.markdown(f"""
    <strong>Volga of the option</strong> <br>
    2nd order sensitivity to volatility.
    """, unsafe_allow_html=True)
    st.latex(r"""
     \mathcal{V}_{\text{olga}}= \frac{\partial^2 V}{\partial \sigma ^2} = \mathcal{V} \cdot \frac{d_1 d_2}{\sigma}
    """)
    st.markdown(f"""
    **Volga of the option is:** {black76_volga(f, k, t, r, sigma, option = "call")}
    """)


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
'''
    
    





