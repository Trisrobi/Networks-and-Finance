# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 17:37:22 2024

@author: robin
"""

import numpy as np
from scipy.stats import norm

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

# Calculate call and put option prices
call_price = black_scholes(S, K, T, r, sigma, 'call')
put_price = black_scholes(S, K, T, r, sigma, 'put')

print(f"Call Option Price: {call_price:.2f}")
print(f"Put Option Price: {put_price:.2f}")





def binomial_tree(S, K, T, r, sigma, n, option_type='call'):
    """
    Binomial Tree Model for pricing options.
    
    Parameters:
    S : float : Current stock price
    K : float : Strike price
    T : float : Time to maturity (in years)
    r : float : Risk-free interest rate (annual)
    sigma : float : Volatility of the underlying asset (annual)
    n : int : Number of time steps in the binomial tree
    option_type : str : 'call' or 'put' (default is 'call')
    
    Returns:
    float : Price of the option
    """

    dt= T / n
    u= np.exp(sigma*np.sqrt(dt))
    d= 1 / u
    p=(np.exp(r*dt)-d)/(u-d)

    asset_prices=np.zeros(n+1)
    option_values = np.zeros(n+1)

    for i in range(n+1):
            asset_prices[i] = S*(u**(n-1))*(d**i)
            if option_type=="call":
                    option_values[i]==max(0, asset_prices[i]-K)
            elif option_type=="put":
                    option_values[i]==max(0,K-asset_prices[i])
            for j in range(n-1, -1,-1):
                    for i in range(j+1):
                            asset_prices[i]=S*(u**(j-i))*(d**i)
                            option_values[i]=np.exp(-r*dt)*(p*option_values[i]+(1-p)*option_values[i+1])
                            if option_type=='call':
                                    option_values[i]=max(option_values[i],asset_prices[i]-K)
                            elif option_type=='put':
                                    option_values[i]=max(option_values[i],K-asset_prices[i])
            return option_values[0]
S = 100  # Current stock price
K = 100  # Strike price
T = 1    # Time to maturity (1 year)
r = 0.05 # Risk-free interest rate
sigma = 0.2  # Volatility
n = 100  # Number of time steps
call_price = binomial_tree(S, K, T, r, sigma, n, 'call')
put_price = binomial_tree(S, K, T, r, sigma, n, 'put')

print(f"American Call Option Price: {call_price:.2f}")
print(f"American Put Option Price: {put_price:.2f}")
