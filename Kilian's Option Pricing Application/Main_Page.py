        # -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from io import StringIO
from PIL import Image

# --- Page Configuration ---
st.set_page_config(page_title="Commodity Option Pricing", layout="wide")
st.sidebar.markdown("## Main Page ")
st.title("Commodity Option Pricing Models")
st.markdown("""
            <br>
            Welcome to the commodity option pricing tool.

            You can use the side bar located on the left to choose the model adapted for your analysis. (More models are about to come soon)
            
            **Models:**
            - Black-76 for futures-based pricing
            - Mean-Reversion Models 
            - Merton-style Jump Diffusion
            
            Each pages allows you to input all the parameters to price the option. 
            """, unsafe_allow_html= True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
            <font size = "3"><ins><strong>1. Black-76 Model (for Futures-Based Pricing)</strong></ins> <br></font>
            The Black-76 model assumes that the underlying futures price follows a geometric Brownian motion (GBM) under the risk-neutral measure:
""", unsafe_allow_html=True)

#img_bs = Image.open('/Users/kilian_1/Pictures/Option Pricing App/bs_formula.png')

st.latex(r'\LARGE dF_t = \sigma F_t \, dW_t')

st.markdown(
        r"""
        $\sigma$: The constant volatility 
        $W_t$: standard Brownian motion under $\mathbb{Q}$ 
        <br><br>
        <strong>Assumptions for the Black76 Model</strong>
        - There exists a risk-free interest rate,
        - The underlying asset (e.g. an equity) follows a geometric Brownian motion,
        - The underlying asset does not pay a dividend,
        - It is possible to borrow and lend any amount of money (even fractional) at the risk-free rate,
        - It is possible to buy or sell any amount of the underlying (even fractional),
        - There are no arbitrage opportunities,
        - All the transactions above do not incur any fees or costs (i.e. frictionless market).
        """, unsafe_allow_html= True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown(r"""
            <ins><strong>2. Mean-Reversion Models </ins></strong> <br>
            Commodities in general and energy commodity prices in particular have been shown to have a mean reverting propreties, Gibson & Schwartz (1995), Brennan (1991), Cortazar & Schwartz (1994) and Schwatz (1997). Log Prices tend to revert to a fundamental value     
            $\mu$ over time, reflecting real-world dynamics like storage costs or equilibrium pricing in commodities. Commodities like Electricity, Nat Gas or Agricultural have a strong mean-reversion characteristics.
            """, unsafe_allow_html=True )

st.latex(r'\LARGE \frac{dS_t}{S_t} = \alpha(\mu - lnS_t)d_t + \sigma dW_t')
         
st.markdown(
    r"""
    $\alpha$: Mean reversion rate <br>
    $\mu$: long-term log-level to which $S_t$ reverts <br>
    $\sigma$: The constant volatility of chages in the spot price <br>
    $W_t$: standard Brownian motion <br>
    <br></br>
    In this model the difference between current level of log prices ($lnS_t$) and the long run log-price level ($\mu$) determines the direction of price change over the next period.
    - If current prices are lower than $\mu$, the expected change in price will be positive
    - If current prices are higer than $\mu$, the expected change in price will be negative
    <br><br>
""", unsafe_allow_html=True)
            

st.markdown(r"""
        <ins><strong>3. Merton Jump-Diffusion Model</ins></strong> <br>
        This model extends the Black-Scholes framework by introducing discrete jumps in addition to continuous diffusion:
""", unsafe_allow_html=True)

st.latex(r'\LARGE \frac{dS_t}{S_t} = \mu d_t + \sigma dW_t + kdq_t')

st.markdown(r"""
            $\mu$: Growth rate (drift). In the absence of any randomness or jumps, this term drives linear growth over time. <br>
            $\sigma$: The constant volatility of changes in the spot price <br>
            $dq_t$: Is a Poisson process that "jumps" from 0 to 1 with some small probability in a given time interval. <br>
            $k$: Proportional jump size i.e., it scales the jump impact on price. <br>
""",unsafe_allow_html=True)     
st.latex(r'prob(dq=1) = \phi d_t')
st.markdown(r"""$\phi$ is the jump intensity, representing the average number of jumps per unit time. 
            E.g., if $\phi = 2$ we expect 2 jumps per year (on average).""")


