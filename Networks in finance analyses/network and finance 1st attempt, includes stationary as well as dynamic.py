import yfinance as yf
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community

'''
def calculate_entropy(values):
    avg_prob=
    np.sum()
'''

def gini_coefficient(values):
    values=np.sort(values)
    n=len(values)
    cumulative_values=np.cumsum(values)
    relative_mean_diff=(2* np.sum(cumulative_values)- values[-1])/(n*np.sum(values))
    return 1-relative_mean_diff

stocks= ['TSLA', 'AAPL','MSFT','GOOGL','AMZN','NFLX']

data_frames={}
for stock in stocks:
    df=yf.download(stock, start='2010-01-01', end='2023-01-01')
    df.reset_index(inplace=True)
    df['Log_Returns']=np.log(df['Close']/df['Close'].shift(1))
    df.dropna(inplace=True)
    data_frames[stock]= df[['Date', 'Log_Returns']]


# Merge data on 'Date'
merged_data= data_frames[stocks[0]].rename(columns={'Log_Returns':stocks[0]})
for stock in stocks[1:]:
    merged_data = pd.merge(merged_data, data_frames[stock].rename(columns={'Log_Returns':stock}),
                           on='Date')



# Calculate correlation matrix
correlation_matrix = merged_data[stocks].corr()
print(correlation_matrix)

# Creating a network graph from the correlation matrix
G = nx.Graph()

# Adding nodes
for stock in correlation_matrix.columns:
    G.add_node(stock)

# Adding edges with weights (correlation values)
threshold=0.5
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        weight=correlation_matrix.iloc[i,j]
        if abs(weight)>=threshold:
            G.add_edge(correlation_matrix.columns[i], correlation_matrix.columns[j], weight=weight)


degree_centrality=nx.degree_centrality(G)
betweenness_centrality=nx.betweenness_centrality(G)
closeness_centrality=nx.closeness_centrality(G)

print("Degree Centrality:", degree_centrality)
print("Betweenness Centrality:", betweenness_centrality)
print("Closeness Centrality:", closeness_centrality)

node_size =[v*1000 for v in degree_centrality.values()]

pos = nx.spring_layout(G)
edges = G.edges(data=True)
weights = [edge[2]['weight'] for edge in edges]

min_weight = min(weights)
max_weight = max(weights)
norm_weights = [(weight - min_weight) / (max_weight - min_weight) for weight in weights]


communities = community.girvan_newman(G)
top_level_communities= next(communities)
second_level_communities= next(communities)

top_level_communities = list(sorted(c) for c in top_level_communities)
second_level_communities= list(sorted(c) for c in second_level_communities)

print("Top Level Communities:", top_level_communities)
print("Second Level Communities:", second_level_communities)

time_windows =pd.date_range(start='2010-01-01', end='2023-01-01', freq='YE')

average_correlations = []
network_densities = []
num_edges_list = []
gini_entropies = []

for i in range(len(time_windows)-1):
    window_data=merged_data[(merged_data['Date']>= time_windows[i])& (merged_data['Date']<time_windows[i+1])]
    if not window_data.empty:
        correlation_matrix =window_data[stocks].corr()

        average_corr= correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape),k=1).astype(bool)).mean().mean()
        average_correlations.append(average_corr)

        G=nx.Graph()
        for col in correlation_matrix.columns:
            G.add_node(col)
        for i in range(len(correlation_matrix.columns)):
            weight= correlation_matrix.iloc[i,j]
            if abs(weight)>=threshold:
                G.add_edge(correlation_matrix.columns[i], correlation_matrix.columns[j], weight=weight)

        network_densities.append(nx.density(G))
        num_edges_list.append(len(G.edges()))

        gini_coefficient=gini_coefficient(G.degree)

# Plotting the average correlation and network density over time
plt.figure(figsize=(12, 8))
plt.plot(time_windows[:-1], average_correlations, label='Average Correlation')
plt.plot(time_windows[:-1], network_densities, label='Network Density')
plt.xlabel('Year')
plt.ylabel('Value')
plt.legend()
plt.title('Average Correlation and Network Density Over Time')
plt.show()

plt.figure(figsize=(12, 8))
plt.plot(time_windows[:-1], num_edges_list, label='Number of Significant Correlations (Edges)')
plt.xlabel('Year')
plt.ylabel('Number of Edges')
plt.legend()
plt.title('Number of Significant Correlations (Edges) Over Time')
plt.show()
