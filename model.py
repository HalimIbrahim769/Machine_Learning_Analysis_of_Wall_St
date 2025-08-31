from torch import nn
import torch
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.model_selection import train_test_split

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
#print(stock_analysis.state_dict())

# Prepare Data

#Reading Ticker File to Get Ticker
i = 0
# with open(file='Tickers.txt',mode='r') as f:
#     tickers = f.readlines()
#     for ticker in tickers:
#         try:
#             #getting data from yahoofinance api
#             ticker_data = yf.Ticker(ticker=ticker).history(period='max').reset_index()
#             dates = ticker_data['Date'].dt.date.apply(lambda x: x.toordinal())
#             price = ticker_data['Close']

#             X = torch.tensor(dates.values, dtype=torch.float32).unsqueeze(1)
#             y = torch.tensor(price.values, dtype=torch.float32).unsqueeze(1)

#             #split the data into training and testing data
#             X.numpy()
#             y.numpy()
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, train_size=.8, random_state=42)

#             combined_train_tensor = [i, X_train, y_train]
#             combined_test_tensor = [i, X_test, y_test]
#             i += 1
#             torch.save(combined_train_tensor, 'train_tensors.pt')
#             torch.save(combined_test_tensor, 'test_tensors.pt')
#         except:
#             print(f'{ticker} is not available')

#I will use specific tickers because it is more efficient and faster however the code above does allow for multiple different
#Tickers they just are not all available and that takes time and computing power
Ticker = str(input('What Stock would you like to analyze (Ticker): '))
ticker_data = yf.Ticker(ticker=Ticker).history(period='max').reset_index()
dates = ticker_data['Date'].dt.date.apply(lambda x: x.toordinal())
price = ticker_data['Close']

X = torch.tensor(dates.values, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(price.values, dtype=torch.float32).unsqueeze(1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, train_size=.8, random_state=42)

#Training and Testing
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(stock_analysis.parameters(), lr = .00001)

epochs = 1000
for epoch in range(epochs):
    stock_analysis.train()
    y_logits = stock_analysis(X_train)

    loss = loss_fn(y_logits, y_train)

    optimizer.zero_grad()

    loss.backward()

    stock_analysis.eval()
    with torch.inference_mode():
        test_logits = stock_analysis(X_test)

        test_loss = loss_fn(test_logits, y_test)
    
    if epoch % 10 == 0:
        print(f'Epoch: {epoch}/Loss: {loss}/Test loss: {test_loss}')