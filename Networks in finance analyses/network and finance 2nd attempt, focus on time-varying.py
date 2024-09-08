import yfinance as yf
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community
from scipy.optimize import minimize
import scipy.stats as stats
plt.switch_backend('Agg')

def sharpe_ratio(returns, rf=0.01):
    mean_return=np.mean(returns)
    returns_std=np.std(returns)
    return (mean_return-rf)/returns_std

def calculate_entropy(values):
    values=np.array(values)
    probabilities= values/np.sum(values)
    entropy_value = -np.sum(probabilities* np.log(probabilities + 1e-10))
    return entropy_value


def gini_coefficient(values):
    values=np.sort(values)
    n=len(values)
    cumulative_values=np.cumsum(values)
    relative_mean_diff=(2* np.sum(cumulative_values)- values[-1])/(n*np.sum(values))
    return 1-(relative_mean_diff/(n-1))

def portfolio_variance(weights, cov_matrix):
    return weights.T @ cov_matrix @ weights

def calculate_min_var_portfolio(cov_matrix, allow_short_selling):
    #calculate variance
    n = len(cov_matrix)
    init_guess = np.ones(n) / n
    bounds=[(-1, 1) if allow_short_selling else(0,1) for _ in range(n)]
    constraints=({'type': 'eq', 'fun': lambda weights: np.sum(weights)-1})
    #use minimizer of scikit-learn to find the portfolio which minimizes the variance
    result=minimize(portfolio_variance, init_guess, args=(cov_matrix,), method='SLSQP', bounds=bounds, constraints=constraints)
    #how to implement the network in this#
    return result.x





stocks= ['TSLA', 'AAPL','MSFT','GOOGL','AMZN','NFLX']
data_frames={}


summary_stats={}
for stock in stocks:
    df=yf.download(stock, start='2010-01-01', end='2023-01-01')
    df.reset_index(inplace=True)
    df['Log_Returns']=np.log(df['Close']/df['Close'].shift(1))
    df.dropna(inplace=True)
    data_frames[stock]= df[['Date', 'Log_Returns']]


# Merge data on 'Date'
merged_data= data_frames[stocks[0]].rename(columns={'Log_Returns':stocks[0]})
for stock in stocks[1:]:
    merged_data = pd.merge(merged_data, data_frames[stock].rename(columns={'Log_Returns':stock}), on='Date')

log_returns_df=merged_data.drop(columns=['Date'])
expected_returns=log_returns_df.mean()*252
cov_matrix= log_returns_df.cov()*252


time_windows =pd.date_range(start='2010-01-01', end='2023-01-01', freq='YE')

threshold= 0.3
average_correlations = []
network_densities = []
num_edges_list = []
gini_coefficients = []
entropies=[]
modularities=[]
standard_portfolio_weights=[]
network_portfolio_weights=[]
standard_variances = []
network_variances = []
standard_sharpe_ratios= []
network_sharpe_ratios=[]


for i in range(len(time_windows)-1):
    window_data=merged_data[(merged_data['Date']>= time_windows[i])& (merged_data['Date']<time_windows[i+1])]
    if not window_data.empty:
        correlation_matrix =window_data[stocks].corr()

        average_corr= correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape),k=1).astype(bool)).mean().mean()
        average_correlations.append(average_corr)

        G=nx.Graph()
        for col in correlation_matrix.columns:
            G.add_node(col)
        for j in range(len(correlation_matrix.columns)):
            for k in range(j+1, len(correlation_matrix.columns)):
                weight= correlation_matrix.iloc[j,k]
                if abs(weight)>=threshold:
                    G.add_edge(correlation_matrix.columns[j], correlation_matrix.columns[k], weight=weight)
        communities=list(nx.community.girvan_newman(G))
        modularity = nx.community.modularity(G, communities[0])
        modularities.append(modularity)

        network_densities.append(nx.density(G))
        num_edges_list.append(len(G.edges()))

        degrees=np.array([degree for node, degree in G.degree()])
        gini_coefficients.append(gini_coefficient(degrees))
        entropies.append(calculate_entropy(degrees))

        window_log_returns_df = window_data.drop(columns=['Date'])
        window_cov_matrix = window_log_returns_df.cov() * 252  # Annualized covariance matrix


        centrality=nx.degree_centrality(G)
        centrality_series=pd.Series(centrality).reindex(correlation_matrix.index)
        adjusted_cov_matrix= cov_matrix*centrality_series.values[:, None]*centrality_series.values[None,:]

        weights_standard=calculate_min_var_portfolio(window_cov_matrix, allow_short_selling=True)
        weights_network=calculate_min_var_portfolio(adjusted_cov_matrix, allow_short_selling=True)
        
        log_returns_window = window_data.drop(columns=['Date'])
        portfolio_returns_standard = log_returns_window.dot(weights_standard)
        portfolio_returns_network = log_returns_window.dot(weights_network)
        
        standard_sharpe_ratios.append(sharpe_ratio(portfolio_returns_standard))
        network_sharpe_ratios.append(sharpe_ratio(portfolio_returns_network))

        standard_variances.append(portfolio_variance(weights_standard, window_cov_matrix))
        network_variances.append(portfolio_variance(weights_network, adjusted_cov_matrix))
        
        standard_portfolio_weights.append(weights_standard)
        network_portfolio_weights.append(weights_network)

standard_weights_df = pd.DataFrame(standard_portfolio_weights, index=time_windows[:-1], columns=stocks)
network_weights_df = pd.DataFrame(network_portfolio_weights, index=time_windows[:-1], columns=stocks)



'''
        modularity_df=pd.DataFrame([
            [k+1, nx.community.modularity(G, communities[k])]
            for k in range(len(communities))
            ],
            columns=["k","Modularity"],
                                   )
        print(modularity_df)
    '''
# Plotting the average correlation and network density over time
plt.figure(figsize=(12, 8))
plt.plot(time_windows[:-1], average_correlations, label='Average Correlation')
plt.plot(time_windows[:-1], network_densities, label='Network Density')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()
plt.title('Average Correlation and Network Density Over Time')
plt.savefig('average_correlation_network_density.png')

plt.figure(figsize=(12, 8))
plt.plot(time_windows[:-1], num_edges_list, label='Number of Significant Correlations (Edges)')
plt.xlabel('Year')
plt.ylabel('Number of Edges')
plt.legend()
plt.title('Number of Significant Correlations (Edges) Over Time')
plt.savefig('num_significant_correlations.png')

plt.figure(figsize=(12, 8))
plt.plot(time_windows[:-1], gini_coefficients, label='Gini Coefficient')
plt.xlabel('Year')
plt.ylabel('Gini Coefficient')
plt.legend()
plt.title('Gini Coefficient of Degree Distribution Over Time')
plt.savefig('gini_coefficient.png')

plt.figure(figsize=(12, 8))
plt.plot(time_windows[:-1], entropies, label='Entropy')
plt.xlabel('Year')
plt.ylabel('Entropy')
plt.legend()
plt.title('Entropy of Degree Distribution Over Time')
plt.savefig('entropy_degrees.png')

plt.figure(figsize=(12,8))
plt.plot(time_windows[:-1], modularities, label='Modularity')
plt.xlabel('Year')
plt.ylabel('Community Modularity')
plt.legend()
plt.title('Modularity Over Time')
plt.savefig('Modularity.png')

for stock in stocks:
    plt.figure(figsize=(12, 8))
    plt.plot(time_windows[:-1], standard_weights_df[stock], label=f'Standard {stock}', color='blue')
    plt.plot(time_windows[:-1], network_weights_df[stock], linestyle='--', label=f'Network {stock}', color='red')
    plt.xlabel('Year')
    plt.ylabel('Weight')
    plt.legend()
    plt.title(f'Portfolio Weights Over Time: {stock}')
    plt.savefig(f'portfolio_weights_{stock}.png')

# Plotting portfolio variances over time
plt.figure(figsize=(12, 8))
plt.plot(time_windows[:-1], standard_variances, label='Standard Portfolio Variance')
plt.plot(time_windows[:-1], network_variances, label='Network-Adjusted Portfolio Variance')
plt.xlabel('Year')
plt.ylabel('Portfolio Variance')
plt.legend()
plt.title('Portfolio Variance Over Time')
plt.savefig('portfolio_variance_comparison.png')
print(network_variances)

# Plotting Sharpe Ratios over time
plt.figure(figsize=(12, 8))
plt.plot(time_windows[:-1], standard_sharpe_ratios, label='Standard Portfolio Sharpe Ratio')
plt.plot(time_windows[:-1], network_sharpe_ratios, label='Network-Adjusted Portfolio Sharpe Ratio')
plt.xlabel('Year')
plt.ylabel('Sharpe Ratio')
plt.legend()
plt.title('Sharpe Ratio Over Time')
plt.savefig('sharpe_ratio_comparison.png')

