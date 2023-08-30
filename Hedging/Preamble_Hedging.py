import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks.analytical import delta, gamma, vega, theta, rho


def put_value(spot_price, strike_price):
    """
    Calculate the value of a put option.

    Parameters:
        spot_price (float): Current spot price of the underlying asset.
        strike_price (float): Strike price of the put option.

    Returns:
        float: Value of the put option.
    """
    a = strike_price - spot_price
    return np.max([a, 0])


def protective_put_value(asset_price, strike_price, put_premium, num_assets, num_puts):
    """
    Calculate the total value of a protective put position.

    Parameters:
        asset_price (float): Current asset price.
        strike_price (float): Strike price of the put option.
        put_premium (float): Premium paid for each put option.
        num_assets (int): Number of assets in the position.
        num_puts (int): Number of put options.

    Returns:
        float: Total value of the protective put position.
    """
    n = num_assets
    m = num_puts
    p = put_premium
    x = asset_price
    s = strike_price
    a = put_value(asset_price, strike_price)
    
    return n * x + m * a - m * p
    

def protective_put_profit_loss_graph(spot_prices, asset_price, strike_price, put_premium, num_assets, num_puts):
    """
    Generate a profit and loss graph for a protective put strategy.

    Parameters:
        spot_prices (array-like): Array of spot prices for graphing.
        asset_price (float): Current asset price.
        strike_price (float): Strike price of the put option.
        put_premium (float): Premium paid for each put option.
        num_assets (int): Number of assets in the position.
        num_puts (int): Number of put options.
    """
    n = num_assets
    m = num_puts
    p = put_premium
    x = asset_price
    s = strike_price
    
    investment = n * x + m * p
    put_values = np.array([put_value(x, asset_price) for x in spot_prices])
    position_value = num_assets * spot_prices + m * put_values - m * put_premium
    profit_loss = position_value - investment
    
    plt.figure(figsize=(8, 6))
    sns.set_style('dark')
    sns.set_palette('tab10')
    plt.plot(spot_prices, profit_loss, color='blue', lw=2, label='Profit and Loss')
    plt.axhline(0, color='black', ls='--', lw=2)
    plt.xlabel('Spot Price')
    plt.ylabel('Profit/Loss')
    plt.title('Protective Put Strategy: Profit and Loss Graph')
    plt.grid(True, color='black', linestyle='--', alpha=0.5)
    plt.legend()
    plt.show()
        

def geo_paths(S, T, sigma, steps, strike_price, N, r=0, q=0):
    """
    Parameters:
    S = Initial value of the asset/position
    T = Time in years
    r = Risk-free interest rate
    q = Continuous dividend yield
    sigma = Yearly volatility of the stock
    steps = Number of steps in a simulation
    N = Number of simulations
    strike_price = Strike price for options
    
    Output: Simulated geometric Brownian motion paths of assets/positions based on the inputs.
    """
    dt = T / steps
    ST = S * np.exp(np.cumsum(((r - q - sigma**2 / 2) * dt +
                               sigma * np.sqrt(dt) *
                               np.random.normal(size=(steps, N))), axis=0))
    paths_with_strike = np.insert(ST, 0, strike_price, axis=0)
    return paths_with_strike


# Calculate standard error of an array of payoffs
def standard_error(payoffs, T, r=0):
    payoff = np.mean(payoffs) * np.exp(-r * T)
    N = len(payoffs)
    sigma = np.sqrt(np.sum((payoffs - payoff)**2) / (N - 1))
    SE = sigma / np.sqrt(N)
    return SE


# Function to calculate delta
def calc_delta(flag, price, K, time, r, sigma, position='s'):
    """
    Calculate the delta of an option.
    
    Parameters:
    flag (str): Option type, 'c' for call and 'p' for put.
    price (float): Current price of the underlying stock.
    K (float): Strike price of the option.
    time (float): Time to maturity in years.
    r (float): Risk-free rate.
    sigma (float): Volatility of the underlying stock.
    position (str, optional): Position type, 's' for short and 'l' for long. Default is 's'.
    
    Returns:
    float: Delta value of the option.
    """
    if time == 0:
        return np.nan
    else:
        if position=='l':
            return int(delta(flag, price, K, time, r, sigma)*100)
        else:
            return -int(delta(flag, price, K, time, r, sigma)*100)
        
# Function to adjust position based on delta
def adjust(delta, total):
    """
    Adjust the position based on delta and total position.
    
    Parameters:
    delta (float): Delta value of the option.
    total (float): Total position quantity.
    
    Returns:
    str: Adjustment action, e.g., 'Buy 5', 'Sell 10', or 'None'.
    """
    if delta < 0:
        return 'Buy {0}'.format(abs(delta))
    elif delta > 0:
        return 'Sell {0}'.format(abs(delta))
    elif delta == 0:
        return 'None'
    else:
        if total < 0:
            return 'Sell {0}'.format(abs(total))
        elif total > 0:
            return 'Buy {0}'.format(abs(total))
        else:
            return np.nan

# Function to determine the total adjusted position
def totalAdj(counter,time):
    """
    Determine the adjusted total position type.
    
    Parameters:
    counter (float): Total position quantity.
    time (float): Time value.
    
    Returns:
    str: Adjusted total position description, e.g., 'Long 10', 'Short 5', or np.nan.
    """
    if time > 0:
        if counter < 0:
            return 'Long {0}'.format(abs(counter))
        elif counter > 0:
            return 'Short {0}'.format(abs(counter))
        else:
            return np.nan
    else:
        return np.nan
        
# Function to adjust cash based on delta and time
def cashAdj(delta, price, time, total):
    """
    Adjust cash based on delta, price, time, and total position.
    
    Parameters:
    delta (float): Delta value of the option.
    price (float): Current price of the underlying stock.
    time (float): Time value.
    total (float): Total position quantity.
    
    Returns:
    float: Adjusted cash amount.
    """
    if time > 0:
        return delta * price
    else:
        return -total * price
    
# Importing functions from py_vollib library
from py_vollib.black_scholes.implied_volatility import implied_volatility as iv
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks.analytical import vega, delta

# Function to calculate implied volatility
def implied_vol(S0, K, T, r, market_price, flag='c', tol=0.00001):
    """
    Calculate the implied volatility of a European option.
    
    Parameters:
    S0 (float): Stock price.
    K (float): Strike price.
    T (float): Time to maturity in years.
    r (float): Risk-free rate.
    market_price (float): Option price in the market.
    flag (str, optional): Option type, 'c' for call and 'p' for put. Default is 'c'.
    tol (float, optional): Tolerance for convergence. Default is 0.00001.
    
    Returns:
    float: Implied volatility value.
    """
    max_iter = 200 # max no. of iterations
    vol_old = 0.1 # initial guess 

    for k in range(max_iter):
        bs_price = bs(flag, S0, K, T, r, vol_old)
        Cprime = vega(flag, S0, K, T, r, vol_old)*100
        C = bs_price - market_price

        vol_new = vol_old - C/Cprime
        new_bs_price = bs(flag, S0, K, T, r, vol_new)
        if (abs(vol_old-vol_new) < tol or abs(new_bs_price-market_price) < tol):
            break

        vol_old = vol_new

    implied_vol = vol_new
    return implied_vol
