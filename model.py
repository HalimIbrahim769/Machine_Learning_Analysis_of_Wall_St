from torch import nn
import torch
import matplotlib.pyplot as plt
import yfinance as yf

#Manual Seed to Assist with Replication
torch.manual_seed(2008)

# Creation of a Very Simple Module
class Stock_module(nn.Module):
    def __init__(self):
        super().__init__()

        #Addition of Linear Layers to produce graphical data due to the mathematical state of the stock market
        self.layer_1 = nn.Linear(in_features=1, out_features=16)
        self.layer_2 = nn.Linear(in_features=16, out_features=16)
        self.layer_3 = nn.Linear(in_features=16, out_features=1)

        #Due to the stock market being stochastic the non-linear layer allows for more flexibility with out module
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

#Visualization of our module
stock_analysis = Stock_module()
print(stock_analysis.state_dict())

# Reading Ticker file
X, y = [], []

with open(file='Tickers.txt',mode='r') as f:
    tickers = f.readlines()
    for ticker in tickers:
        
        #getting data from yahoofinance api
        ticker_data = yf.Ticker(ticker=ticker)
        y.append(float(ticker_data.history('max')['Close'].to_list()))