import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from arch import arch_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

stocks= ['TSLA', 'AAPL','MSFT','GOOGL','AMZN','NFLX']
data = yf.download(stocks, start='2020-01-01', end='2023-01-01')
print(data['Adj Close'].head())


'''GRAPH OF STOCK PRICES'''
plt.figure(figsize=(14, 8))
for stock in stocks:
    plt.plot(data['Adj Close'][stock], label=stock)
plt.title('Stock Adjusted Closing Prices')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price (USD)')
plt.legend()
plt.savefig('STOCKS Prices GRAPHS.png')
plt.close()

'''RETURNS SETUP'''
returns=data['Adj Close'].pct_change().dropna()
returns.index = pd.to_datetime(returns.index)
log_returns=np.log(data['Adj Close']/data['Adj Close'].shift(-1)).dropna()
''''SUMMARY STATS OF RETURNS'''
mean_returns=returns.mean()
variance_returns=returns.var()
skewness=returns.skew()
kurtosis=returns.kurtosis()
print("Mean Returns:\n", mean_returns)
print("\nVariance of Returns:\n", variance_returns)
print("\nSkewness of Returns:\n", skewness)
print("\nKurtosis of Returns:\n", kurtosis)

''''RETURNS GRAPH'''
plt.figure(figsize=(14, 8))
for i, ticker in enumerate(stocks, 1):
    plt.subplot(3, 3, i)
    plt.hist(returns[stock], bins=50, alpha=0.75)
    plt.title(f'{stock} Returns Histogram')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('GRAPH of Returns.png')
plt.close()

'''ADF TESTS'''
adf_results={}
for stock in stocks:
    result=adfuller(returns[stock])
    adf_results[stock] = {'ADF Statistic': result[0], 'p-value': result[1]}
print("\nADF Test Results:")
for stock, res in adf_results.items():
    print(f"{stock}: ADF Statistic = {res['ADF Statistic']}, p-value = {res['p-value']}")

'''AUTOCORRELATION AND PACF GRAPH'''
plt.figure(figsize=(14, 12))
for i, stock in enumerate(stocks,1):
    plt.subplot(6,3,3*i-1)
    plot_acf(returns[stock],ax=plt.gca(),lags=20)
    plt.title(f"{stock} Autocorrelation")
    plt.subplot(6,3,3*i-1)
    plot_pacf(returns[stock],ax=plt.gca(),lags=20)
    plt.title(f"{stock} Partial Autocorrelation")
    
plt.tight_layout()
plt.savefig('ACFPACF GRAPHS.png')
plt.close()

'''
# Differencing the data
diff_returns = returns.diff().dropna()


adf_results2={}
for stock in stocks:
    result=adfuller(diff_returns[stock])
    adf_results2[stock] = {'ADF Statistic': result[0], 'p-value': result[1]}
    
# Display ADF test results
print("\nADF Test Results:")
for stock, res in adf_results2.items():
    print(f"{stock}: ADF Statistic = {res['ADF Statistic']}, p-value = {res['p-value']}")
'''

rescaled_returns = returns * 100  # Rescale by multiplying by 100
'''FIT GARCH MODEL and display summary'''
garch_models = {}
for stock in stocks:
    model = arch_model(rescaled_returns[stock], vol='Garch', p=1, q=1)
    garch_fit = model.fit(disp='off')
    garch_models[stock] = garch_fit
print("\nGARCH Model Summary for AAPL:")
print(garch_models['AAPL'].summary())

'''PLOTTING GARCH '''
conditional_volatility = garch_models['AAPL'].conditional_volatility
plt.figure(figsize=(10, 6))
plt.plot(conditional_volatility)
plt.title('Conditional Volatility of GARCH Model for AAPL')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.savefig('conditional_volatility_garch_aapl.png')
plt.close()


'''FORECASTING GRAPH'''
forecast_steps = 30 
garch_forecast = garch_models['AAPL'].forecast(horizon=forecast_steps)
forecast_volatility = np.sqrt(garch_forecast.variance.values[-1, :])

plt.figure(figsize=(10, 6))
plt.plot(forecast_volatility, label='Forecasted Volatility', color='red')
plt.title('GARCH Forecast for AAPL Volatility')
plt.xlabel('Steps Ahead')
plt.ylabel('Volatility')
plt.legend()
plt.savefig('AAPL Garch Forecast.png')
plt.close()

'''
# Manually set the index for prediction results
forecast_steps = 10  # Number of steps to forecast
for stock in stocks:
    forecast = arima_models[stock].get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=returns.index[-1], periods=forecast_steps+1, closed='right')
    forecast.predicted_mean.index = forecast_index
    print(f"\nForecast for {stock}:")
    print(forecast.predicted_mean)
'''
