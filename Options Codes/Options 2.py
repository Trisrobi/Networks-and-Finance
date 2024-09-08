# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 17:37:22 2024

@author: robin
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def black_scholes(S, K, T, r, sigma, option_type='call'):
        """
    Calculate the Black-Scholes price for a European option.
    
    Parameters:
    S : float : Current stock price
    K : float : Strike price
    T : float : Time to maturity (in years)
    r : float : Risk-free interest rate (annual)
    sigma : float : Volatility of the underlying asset (annual)
    option_type : str : 'call' or 'put' (default is 'call')
    
    Returns:
    float : Price of the option
    """

        d1=(np.log(S/K)+(r+0.05*sigma**2)*T)/(sigma*np.sqrt(T))
        d2=d1-(sigma*np.sqrt(T))

        if option_type=='call':
            price= S* norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type=='put':
            price= K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put' . ")
        return price

#now can put in parameters as needed

S = 100# Current stock price
K = 100  # Strike price
T = 2    # Time to maturity in years
r = 0.05 # Annual risk-free interest rate
sigma = 0.2  # Annual volatility of the underlying asset


underlying_prices=np.linspace(50,150,100)
# Calculate call and put option prices
call_prices = [black_scholes(S, K, T, r, sigma, 'call') for S in underlying_prices]
put_prices =[black_scholes(S, K, T, r, sigma, 'put') for S in underlying_prices]


# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(underlying_prices, call_prices, label='Call Option Price', color='blue')
plt.plot(underlying_prices, put_prices, label='Put Option Price', color='red')
plt.title('Option Prices vs. Underlying Asset Price')
plt.xlabel('Underlying Asset Price ($)')
plt.ylabel('Option Price ($)')
plt.legend()
plt.grid(True)
plt.show()

underlying_volatilities=np.linspace(0.1,0.5,100)
call_prices = [black_scholes(S, K, T, r, sigma, 'call') for sigma in underlying_volatilities]
put_prices =[black_scholes(S, K, T, r, sigma, 'put') for sigma in underlying_volatilities]

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(underlying_prices, call_prices, label='Call Option Price', color='blue')
plt.plot(underlying_prices, put_prices, label='Put Option Price', color='red')
plt.title('Option Prices vs. Underlying Asset volatility')
plt.xlabel('Underlying volatility')
plt.ylabel('Option Price ($)')
plt.legend()
plt.grid(True)
plt.show()