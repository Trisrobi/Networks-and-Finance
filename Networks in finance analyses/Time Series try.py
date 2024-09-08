import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from arch import arch_model

# Set the matplotlib backend to avoid issues
import matplotlib
matplotlib.use('Agg')  # Change the backend
import matplotlib.pyplot as plt


# Define the list of stock tickers
stocks= ['TSLA', 'AAPL','MSFT','GOOGL','AMZN','NFLX']

# Fetch historical stock data for each ticker
data = yf.download(stocks, start='2020-01-01', end='2023-01-01')

# Display the first few rows of the adjusted close prices
print(data['Adj Close'].head())

# Plot the adjusted close price over time for each ticker
plt.figure(figsize=(14, 8))
for stock in stocks:
    plt.plot(data['Adj Close'][stock], label=stock)

plt.title('Stock Adjusted Closing Prices')
plt.xlabel('Date')
plt.ylabel('Adjusted Close Price (USD)')
plt.legend()
plt.show()

returns=data['Adj Close'].pct_change().dropna()

returns.index = pd.to_datetime(returns.index)
log_returns=np.log(data['Adj Close']/data['Adj Close'].shift(-1)).dropna()

mean_returns=returns.mean()
variance_returns=returns.var()

skewness=returns.skew()
kurtosis=returns.kurtosis()

print("Mean Returns:\n", mean_returns)
print("\nVariance of Returns:\n", variance_returns)
print("\nSkewness of Returns:\n", skewness)
print("\nKurtosis of Returns:\n", kurtosis)


'''
plt.figure(figsize=(14, 8))
for i, ticker in enumerate(stocks, 1):
    plt.subplot(2, 2, i)
    plt.hist(returns[stock], bins=50, alpha=0.75)
    plt.title(f'{stock} Returns Histogram')
    plt.xlabel('Return')
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
'''

adf_results={}
for stock in stocks:
    result=adfuller(returns[stock])
    adf_results[stock] = {'ADF Statistic': result[0], 'p-value': result[1]}

# Display ADF test results
print("\nADF Test Results:")
for stock, res in adf_results.items():
    print(f"{stock}: ADF Statistic = {res['ADF Statistic']}, p-value = {res['p-value']}")


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

plt.figure(figsize=(14, 12))

for i, stock in enumerate(stocks,1):
    plt.subplot(6,3,3*i-1)
    plot_acf(returns[stock],ax=plt.gca(),lags=20)
    plt.title(f"{stock} Autocorrelation")

    plt.subplot(6,3,3*i-1)
    plot_pacf(returns[stock],ax=plt.gca(),lags=20)
    plt.title(f"{stock} Partial Autocorrelation")

plt.tight_layout()
plt.show()


'''ARIMA MODELLING, 2-0-2 model seems best fit from acf and pacf graphs'''

arima_models = {}
for stock in stocks:
    model = ARIMA(returns[stock], order=(2, 0, 2))  # ARIMA(p, d, q) with d=0 for stationary series
    arima_fit = model.fit()
    arima_models[stock] = arima_fit

print("\nARIMA Model Summary for AAPL:")
print(arima_models['AAPL'].summary())


residuals = arima_models['AAPL'].resid
plt.figure(figsize=(10, 6))
plt.plot(residuals)
plt.title('Residuals of ARIMA Model for AAPL')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.savefig('residuals_arima_aapl.png')  # Save the figure
plt.close()

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plot_acf(residuals, ax=plt.gca(), lags=20)
plt.title('ACF of Residuals')
plt.subplot(1, 2, 2)
plot_pacf(residuals, ax=plt.gca(), lags=20)
plt.title('PACF of Residuals')
plt.tight_layout()
plt.savefig('acf_pacf_residuals_arima_aapl.png')  # Save the figure
plt.close()



rescaled_returns = returns * 100  # Rescale by multiplying by 100
# Fit GARCH(1,1) model for each ticker with rescaled returns
garch_models = {}
for stock in stocks:
    model = arch_model(rescaled_returns[stock], vol='Garch', p=1, q=1)
    garch_fit = model.fit(disp='off')
    garch_models[stock] = garch_fit

# Display summary of GARCH model for a specific ticker (e.g., AAPL)
print("\nGARCH Model Summary for AAPL:")
print(garch_models['AAPL'].summary())

# Plot the conditional volatility of the GARCH model for AAPL
conditional_volatility = garch_models['AAPL'].conditional_volatility
plt.figure(figsize=(10, 6))
plt.plot(conditional_volatility)
plt.title('Conditional Volatility of GARCH Model for AAPL')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.savefig('conditional_volatility_garch_aapl.png')  # Save the figure
plt.close()



# Forecast future values using ARIMA model
forecast_steps = 30  # Number of steps to forecast
forecast = arima_models['AAPL'].get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# Plot the forecasted values
plt.figure(figsize=(10, 6))
plt.plot(returns['AAPL'], label='Historical Returns')
plt.plot(forecast_mean, label='Forecasted Returns', color='red')
plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='red', alpha=0.3)
plt.title('ARIMA Forecast for AAPL')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
plt.savefig('AAPL Forecast.png')
plt.close()



# Forecast future volatility using GARCH model
forecast_steps = 30  # Number of steps to forecast
garch_forecast = garch_models['AAPL'].forecast(horizon=forecast_steps)
forecast_volatility = np.sqrt(garch_forecast.variance.values[-1, :])
# Plot the forecasted volatility
plt.figure(figsize=(10, 6))
plt.plot(forecast_volatility, label='Forecasted Volatility', color='red')
plt.title('GARCH Forecast for AAPL Volatility')
plt.xlabel('Steps Ahead')
plt.ylabel('Volatility')
plt.legend()
plt.savefig('AAPL Garch Forecast.png')
plt.close()

# Manually set the index for prediction results
forecast_steps = 10  # Number of steps to forecast
for stock in stocks:
    forecast = arima_models[stock].get_forecast(steps=forecast_steps)
    forecast_index = pd.date_range(start=returns.index[-1], periods=forecast_steps+1, closed='right')
    forecast.predicted_mean.index = forecast_index
    print(f"\nForecast for {stock}:")
    print(forecast.predicted_mean)
