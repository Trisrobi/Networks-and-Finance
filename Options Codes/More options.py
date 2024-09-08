# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 17:46:52 2024

@author: robin
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import numpy as np
from scipy.stats import norm

app = dash.Dash(__name__)


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
    
    
app.layout = html.Div([
    html.H1("Option Pricing Dashboard"),
    
    dcc.Slider(
        id='S-slider',
        min=50,
        max=150,
        step=1,
        value=100,
        marks={i: f"${i}" for i in range(50, 151, 10)},
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    html.Label("Underlying Asset Price (S)"),
    
    dcc.Slider(
        id='sigma-slider',
        min=0.1,
        max=0.5,
        step=0.01,
        value=0.2,
        marks={i/100: f"{i}%" for i in range(10, 51, 5)},
        tooltip={"placement": "bottom", "always_visible": True}
    ),
    html.Label("Volatility (σ)"),
    
    dcc.Graph(id='heatmap')
])

@app.callback(
    Output('heatmap', 'figure'),
    [Input('S-slider', 'value'),
     Input('sigma-slider', 'value')]
)
def update_heatmap(S, sigma):
    K_values = np.linspace(50, 150, 50)
    T_values = np.linspace(0.01, 2, 50)
    
    call_prices = np.array([[black_scholes(S, K, T, 0.05, sigma, 'call') for K in K_values] for T in T_values])

    fig = go.Figure(data=go.Heatmap(
        z=call_prices,
        x=K_values,
        y=T_values,
        colorscale='Reds'
    ))

    fig.update_layout(
        title='Call Option Prices',
        xaxis_title='Strike Price (K)',
        yaxis_title='Time to Maturity (T)',
        xaxis_nticks=10,
        yaxis_nticks=10
    )
    
    return fig

if __name__ == '__main__':
    import webbrowser
    app.run_server(debug=True)
    print("Dash is running on http://127.0.0.1:8050/")
    webbrowser.open('http://127.0.0.1:8050/')