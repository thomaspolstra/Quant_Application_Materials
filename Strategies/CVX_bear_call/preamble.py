import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta, rho
import scipy.stats as stats
import pandas as pd
import yfinance as yf
import datetime as dt
from arch import arch_model
from statsmodels.tsa.stattools import adfuller
from scipy.stats import anderson, zscore, norm
from statsmodels.graphics.tsaplots import plot_acf
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from py_vollib.black_scholes.implied_volatility import implied_volatility as iv


def geo_paths(S, T, sigma, steps, N, r=0, q=0):
    """
    Parameters:
    S: Initial value of the asset/position
    T: Time in years
    r: Risk-free interest rate
    q: Continuous dividend yield
    sigma: Yearly volatility of the stock
    steps: Number of steps in a simulation
    N: Number of simulations
    strike_price: Strike price for options
    
    Output: Simulated geometric Brownian motion paths of assets/positions based on the inputs.
    """
    
    dt = T / steps
    dW = np.sqrt(dt) * np.random.normal(size=(steps, N))
    increments = (r - q - (sigma**2) / 2) * dt + sigma * dW
    log_returns = np.cumsum(increments, axis=0)
    ST = S * np.exp(log_returns)
    paths_with_strike = np.insert(ST, 0, S, axis=0)
    return paths_with_strike




def standard_error(payoffs, T, r=0):
    """
    Calculate the standard error of an array of payoffs.

    Parameters:
    payoffs (array-like): An array of payoffs or returns.
    T (float): Time in years.
    r (float, optional): Risk-free interest rate. Default is 0.

    Returns:
    float: The calculated standard error of the payoffs.

    Explanation:
    1. Calculate the mean payoff after discounting by the risk-free interest rate.
    2. Determine the number of payoffs (N) in the array.
    3. Calculate the sample standard deviation (sigma) of the payoffs.
    4. Compute the standard error (SE) as sigma divided by the square root of N.
    5. Return the calculated standard error.

    Note: This function assumes that the payoffs are independent and identically distributed.
    """

    # Calculate the mean payoff after discounting by the risk-free interest rate
    payoff = np.mean(payoffs) * np.exp(-r * T)

    # Determine the number of payoffs (N) in the array
    N = len(payoffs)

    # Calculate the sample standard deviation (sigma) of the payoffs
    sigma = np.sqrt(np.sum((payoffs - payoff)**2) / (N - 1))

    # Compute the standard error (SE) as sigma divided by the square root of N
    SE = sigma / np.sqrt(N)

    # Return the calculated standard error
    return SE

import numpy as np


def MC_delta(S, T, sigma, N, K, epsilon=1, r=0, q=0):
    """
    Estimate the deltas of a European call and put option using a central difference method based on Monte Carlo simulations.

    Parameters:
        S (float): Initial stock price.
        T (float): Time to maturity of the option.
        sigma (float): Volatility of the stock.
        N (int): Number of simulations.
        K (float): Strike price of the option.
        epsilon (float, optional): Small perturbation factor for delta estimation. Default is 1.
        r (float, optional): Risk-free interest rate. Default is 0.
        q (float, optional): Dividend yield. Default is 0.

    Returns:
        tuple: A tuple containing estimated deltas for the call and put options.
            delta_call (float): Estimated delta for the call option.
            delta_put (float): Estimated delta for the put option.
    """

    dW = np.sqrt(T) * np.random.normal(size=(1, N))
    increments = (r - q - (sigma**2) / 2) * T + sigma * dW
    log_returns = np.cumsum(increments, axis=0)
    ST = S * np.exp(log_returns)
    ST1 = (S - epsilon) * np.exp(log_returns)
    ST2 = (S + epsilon) * np.exp(log_returns)
    
    call_values = np.exp(-r*T)*np.maximum(ST[-1] - K, 0)
    call_values1 = np.exp(-r*T)*np.maximum(ST1[-1] - K, 0)
    call_values2 = np.exp(-r*T)*np.maximum(ST2[-1] - K, 0)
    
    put_values = np.exp(-r*T)*np.maximum(-ST[-1] + K, 0)
    put_values1 = np.exp(-r*T)*np.maximum(-ST1[-1] + K, 0)
    put_values2 = np.exp(-r*T)*np.maximum(-ST2[-1] + K, 0)
    
    delta_call =  np.mean([(call_values2 - call_values) / epsilon, (call_values - call_values1) / epsilon])
    delta_put =  np.mean([(put_values2 - put_values) / epsilon, (put_values - put_values1) / epsilon])
    
    all_call_deltas =  np.array([(call_values2 - call_values) / epsilon, (call_values - call_values1) / epsilon])
    all_put_deltas =  np.array([(put_values2 - put_values) / epsilon, (put_values - put_values1) / epsilon])
    
    
    

    # Calculate the sample standard deviation of deltas
    sigma_call = np.sqrt(np.sum((all_call_deltas - delta_call)**2) / (N - 1)) 
    sigma_put = np.sqrt(np.sum((all_put_deltas - delta_put)**2) / (N - 1)) 
    SE_call = sigma_call / np.sqrt(N)
    SE_put = sigma_put / np.sqrt(N)

    
    return {'delta_call':delta_call, 'delta_put': delta_put, 'all_call_deltas': all_call_deltas, 'SE_call': SE_call, 'SE_put': SE_put, 
            'all_put_deltas': all_put_deltas}


import numpy as np



def MC_accurate(S, T, sigma, steps, K, N, r=0, q=0, epsilon=1):
    """
    Estimate the prices of European call and put options using a more accurate Monte Carlo simulation method.
    
    This method involves simulating the stock price path and considering hedging strategies to calculate option prices.

    Parameters:
        S (float): Initial stock price.
        T (float): Time to maturity of the option.
        sigma (float): Volatility of the stock.
        steps (int): Number of time steps for the simulation.
        K (float): Strike price of the option.
        N (int): Number of simulations.
        r (float, optional): Risk-free interest rate. Default is 0.
        q (float, optional): Dividend yield. Default is 0.
        epsilon (float, optional): Small perturbation factor for delta estimation. Default is 1.

    Returns:
        dict: A dictionary containing estimated Monte Carlo prices and standard errors for the call and put options.
            MC_call (float): Estimated Monte Carlo price for the call option.
            MC_put (float): Estimated Monte Carlo price for the put option.
            SE_call (float): Standard error of the Monte Carlo estimate for the call option.
            SE_put (float): Standard error of the Monte Carlo estimate for the put option.
    """
    
    # Calculate time step and time to expiration for each step
    DT = T / steps
    TTE = [T - DT * i for i in range(0, steps + 1)]
    
    # Generate random increments and calculate log returns for stock price simulation
    dt = T / steps
    dW = np.sqrt(dt) * np.random.normal(size=(steps, N))
    increments = (r - q - (sigma**2) / 2) * dt + sigma * dW
    log_returns = np.cumsum(increments, axis=0)
    st = S * np.exp(log_returns)
    ST = np.insert(st, 0, S, axis=0)

    # Calculate average stock values and option values at each time step
    avg_stock_values = [np.exp(-r*(T-TTE[i]))*np.mean(ST[i]) for i in range(len(ST))]
    call_values = [np.exp(-r*(T-TTE[i]))*np.maximum(ST[i] - K, 0) for i in range(len(ST))]
    put_values = [np.exp(-r*(T-TTE[i]))*np.maximum(-ST[i] + K, 0) for i in range(len(ST))]

    # Calculate deltas using previously defined MC_delta function
    DELTAS = [MC_delta(avg_stock_values[i], TTE[i], sigma, 1000, K, epsilon, r) for i in range(len(ST))]
    call_deltas = [DELTAS[i]['delta_call'] for i in range(len(ST))]
    put_deltas = [DELTAS[i]['delta_put'] for i in range(len(ST))]

    # Calculate hedge values and new option values
    X_call = [-call_deltas[i] * ((ST[i + 1] - ST[i]) * np.exp(-r * (TTE[i + 1]))) for i in range(steps)]
    X_put = [-put_deltas[i] * ((ST[i + 1] - ST[i]) * np.exp(-r * (TTE[i + 1]))) for i in range(steps)]
    call_hedge_values = np.sum(X_call, axis=0)
    put_hedge_values = np.sum(X_put, axis=0)
    new_call_values = call_values[-1] + call_hedge_values
    new_put_values = put_values[-1] + put_hedge_values

    # Calculate Monte Carlo estimates and standard errors
    MC_call = np.mean(new_call_values)
    MC_put = np.mean(new_put_values)
    SE_call = standard_error(new_call_values, T)
    SE_put = standard_error(new_put_values, T)
    
    return {'MC_call': MC_call,
           'MC_put': MC_put,
           'SE_call': SE_call,
           'SE_put': SE_put}


def per_in_money_paths(returns):
    """
    Calculate the percentage of in-the-money paths based on the provided returns.

    :param returns: A list or array of returns or payoffs from financial transactions
    :return: Percentage of in-the-money paths
    
    This function calculates the percentage of in-the-money paths within a given set of returns or payoffs.
    It takes the provided returns, converts them to a NumPy array, and filters out the positive returns
    (i.e., in-the-money scenarios). The function then computes the ratio of in-the-money paths to the total
    number of paths, providing a measure of how often positive returns occur within the dataset.

    This percentage is useful for evaluating the likelihood of achieving profitable outcomes based on historical
    or simulated data. It is commonly used in options trading to assess the effectiveness of strategies and
    potential profitability.
    """
    M = len(returns)
    returns = np.array(returns)
    returns_positive = returns[returns > 0]
    return len(returns_positive) / M
