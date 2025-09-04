import streamlit as st
import numpy as np
from scipy.stats import norm
import matplotlib as plt
import matplotlib.pyplot as plt


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