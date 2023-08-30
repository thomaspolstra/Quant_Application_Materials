import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from py_vollib.black_scholes.implied_volatility import implied_volatility as iv
from py_vollib.black_scholes import black_scholes as bs
from py_vollib.black_scholes.greeks.analytical import vega, delta, gamma
from yahooquery import Ticker
import yfinance as yf
import datetime as dt
from arch import arch_model
from itertools import combinations

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




def butterfly_strikes_calls(calls, puts, S0, T):

    calls['price'] = (calls['ask'] + calls['bid'])/2

    calls['iv'] = [iv(calls.iloc[i]['price'], S0, calls.iloc[i]['strike'], T, 
                                           0, 'c') for i in range(len(calls))]

    puts['price'] = (puts['ask'] + puts['bid'])/2

    puts['iv'] = [iv(puts.iloc[i]['price'], S0, puts.iloc[i]['strike'], T, 
                                           0, 'p') for i in range(len(puts))]

    strike_tuples_calls = list(combinations(calls.strike.values, 3))
    strike_tuples_puts = list(combinations(puts.strike.values, 3))

    reshaped_calls = calls.pivot_table(index='strike', columns='optionType', values='iv')
    reshaped_puts = puts.pivot_table(index='strike', columns='optionType', values='iv')


    butterfly_indicators_calls = {tup: reshaped_calls.loc[tup[0]][0] + reshaped_calls.loc[tup[2]][0] 
                             - 2*(reshaped_calls.loc[tup[1]][0]) for tup in strike_tuples_calls}

    butterfly_indicators_puts = {tup: reshaped_puts.loc[tup[0]][0] + reshaped_puts.loc[tup[2]][0] 
                             - 2*(reshaped_puts.loc[tup[1]][0]) for tup in strike_tuples_puts}


    long_max_calls = min([value for value in butterfly_indicators_calls.values()])
    short_max_calls = max([value for value in butterfly_indicators_calls.values()])

    long_max_puts = min([value for value in butterfly_indicators_puts.values()])
    short_max_puts = max([value for value in butterfly_indicators_puts.values()])

    short_butterfly_strikes_calls = {key: value for key, value in butterfly_indicators_calls.items() if value == long_max_calls}
    long_butterfly_strikes_calls = {key: value for key, value in butterfly_indicators_calls.items() if value == short_max_calls}

    short_butterfly_strikes_puts = {key: value for key, value in butterfly_indicators_puts.items() if value == long_max_puts}
    long_butterfly_strikes_puts = {key: value for key, value in butterfly_indicators_puts.items() if value == short_max_puts}
    
    return {'short_butterfly_call_strikes': short_butterfly_strikes_calls, 
            'long_butterfly_call_strikes': long_butterfly_strikes_calls,
           'short_butterfly_put_strikes': short_butterfly_strikes_puts,
           'long_butterfly_put_strikes': long_butterfly_strikes_puts}


def butterfly_strikes_calls(calls, S0, T,sigma):
    
    calls['price'] = (calls['ask'] + calls['bid'])/2

    calls['price_diff'] = (calls.price - calls.bs_price)


    strike_tuples_calls = list(combinations(calls.strike.values, 3))


    reshaped_calls = calls.pivot_table(index='strike', columns='optionType', values='price_diff')



    butterfly_indicators_calls = {tup: reshaped_calls.loc[tup[0]][0] + reshaped_calls.loc[tup[2]][0] 
                             - 2*(reshaped_calls.loc[tup[1]][0]) for tup in strike_tuples_calls}




    long_max_calls = max([value for value in butterfly_indicators_calls.values()])
    short_max_calls = min([value for value in butterfly_indicators_calls.values()])


    short_butterfly_strikes_calls = {key: value for key, value in butterfly_indicators_calls.items() if value == long_max_calls}
    long_butterfly_strikes_calls = {key: value for key, value in butterfly_indicators_calls.items() if value == short_max_calls}


    
    return {'short_butterfly_call_strikes': short_butterfly_strikes_calls, 
            'long_butterfly_call_strikes': long_butterfly_strikes_calls}

def butterfly_strikes_puts(calls, S0, T):

    calls['price'] = (calls['ask'] + calls['bid'])/2

    calls['iv'] = [iv(calls.iloc[i]['price'], S0, calls.iloc[i]['strike'], T, 
                                           0, 'p') for i in range(len(calls))]


    strike_tuples_calls = list(combinations(calls.strike.values, 3))


    reshaped_calls = calls.pivot_table(index='strike', columns='optionType', values='iv')



    butterfly_indicators_calls = {tup: reshaped_calls.loc[tup[0]][0] + reshaped_calls.loc[tup[2]][0] 
                             - 2*(reshaped_calls.loc[tup[1]][0]) for tup in strike_tuples_calls}




    long_max_calls = min([value for value in butterfly_indicators_calls.values()])
    short_max_calls = max([value for value in butterfly_indicators_calls.values()])


    short_butterfly_strikes_calls = {key: value for key, value in butterfly_indicators_calls.items() if value == long_max_calls}
    long_butterfly_strikes_calls = {key: value for key, value in butterfly_indicators_calls.items() if value == short_max_calls}


    
    return {'short_butterfly_put_strikes': short_butterfly_strikes_calls, 
            'long_butterfly_put_strikes': long_butterfly_strikes_calls}