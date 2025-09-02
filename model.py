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
torch.manual_seed(2008)
stock_analysis = Stock_module()
#print(stock_analysis.state_dict())

#Reading Ticker File to Get Ticker
#i = 0
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

#**I will use specific tickers because it is more efficient and faster however the code above does allow for multiple different
#Tickers they just are not all available and that takes time and computing power

#Ticker Choice by user
torch.manual_seed(2008)
Ticker = str(input('What Stock would you like to analyze? (Ticker): '))
#Ticker = "AAPL"

#Gathering Ticker Data
ticker_data = yf.Ticker(ticker=Ticker).history(period='max').reset_index()
dates = ticker_data['Date'].dt.date.apply(lambda x: x.toordinal())
price = ticker_data['Close']
print(ticker_data['Date'][0])
#Organizing Data
X = torch.tensor(dates.values, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(price.values, dtype=torch.float32).unsqueeze(1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, train_size=.8, random_state=2008)

#Reducing data so the model can process the loss

X_mean, X_std = X_train.mean(), X_train.std()
y_mean, y_std = y_train.mean(), y_train.std()

X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

y_train = (y_train - y_mean) / y_std
y_test = (y_test - y_mean) / y_std


#Training and Testing
torch.manual_seed(2008)
loss_fn = nn.MSELoss()

torch.manual_seed(2008)
optimizer = torch.optim.SGD(stock_analysis.parameters(), lr = .1)

# Learning Rate Scheduler
torch.manual_seed(2008)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1000, min_lr=1e-5)
 

y_logits = stock_analysis(X_test)
epochs = 10000
torch.manual_seed(2008)
compiled_model = torch.compile(stock_analysis) # helps run the model faster 

import torch
from sklearn.metrics import r2_score

torch.manual_seed(2008)

for epoch in range(epochs):
    # Training mode
    stock_analysis.train()
    
    # Forward pass
    y_logits = stock_analysis(X_train)
    
    # Compute training loss
    loss = loss_fn(y_logits, y_train)
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Evaluation mode
    stock_analysis.eval()
    with torch.inference_mode():
        test_logits = stock_analysis(X_test)
        test_loss = loss_fn(test_logits, y_test)
    
    # Optional: Learning rate scheduler
    scheduler.step(test_loss)
    
    # Every 1000 epochs, print R² score and losses
    if epoch % 1000 == 0:
        r2 = r2_score(y_true=y_test.cpu().numpy(), y_pred=test_logits.cpu().numpy())
        print(f'Epoch: {epoch} | R2: {r2:.2f} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}')
print("\n")

#Testing the model on real time data
# NY GMT is -4
from datetime import datetime
import pytz as time
import time as tt


def real_time_price(stock_code):
    i = 0
    # Get the current time in NY
    utc_now = datetime.now(time.utc)
    gmt_minus_4_timezone = time.timezone("Etc/GMT+4")
    gmt_minus_4_time = utc_now.astimezone(gmt_minus_4_timezone)
    time_string = gmt_minus_4_time.strftime("%H:%M")
    while time_string != '9:30':
        #if the stock market closes
        print('It is not time yet...')
        if time_string == '4:30':
            print('Stock Market is Closed Today')
            break

        i += 1
        tt.sleep(60)
        if i % 10 == 0:
            print('The stock market has not opened yet...')
        utc_now = datetime.now(time.utc)
        gmt_minus_4_timezone = time.timezone("Etc/GMT+4")
        gmt_minus_4_time = utc_now.astimezone(gmt_minus_4_timezone)
        time_string = gmt_minus_4_time.strftime("%H:%M")

    #IMPORT LIBRARIES
    from bs4 import BeautifulSoup
    import requests
    #REQUEST WEBPAGE AND STORE IT AS A VARIABLE
    page_to_scrape = requests.get(f"https://www.google.com/finance/quote/{stock_code}:NASDAQ?hl=en")
    #USE BEAUTIFULSOUP TO PARSE THE HTML AND STORE IT AS A VARIABLE
    soup = BeautifulSoup(page_to_scrape.text, 'html.parser')
    #FIND ALL THE ITEMS IN THE PAGE WITH A CLASS ATTRIBUTE OF 'TEXT'
    #AND STORE THE LIST AS A VARIABLE
    quotes = soup.findAll('div', attrs={'class':'YMlKec fxKbKc'})
    for quote in quotes:
        import re
        trim = re.compile(r'[^\d.,]+')
        price = trim.sub('', quote.text)
        return price
    if quotes == []: #if wrong stock exchange (NYSE)
        page_to_scrape = requests.get(f"https://www.google.com/finance/quote/{stock_code}:NYSE?hl=en")
        #NYSE
        #USE BEAUTIFULSOUP TO PARSE THE HTML AND STORE IT AS A VARIABLE
        soup = BeautifulSoup(page_to_scrape.text, 'html.parser')
        #FIND ALL THE ITEMS IN THE PAGE WITH A CLASS ATTRIBUTE OF 'TEXT'
        #AND STORE THE LIST AS A VARIABLE
        num = soup.findAll('div', attrs={'class':'YMlKec fxKbKc'})
        for nums in num:
            import re
            trim = re.compile(r'[^\d.,]+')
            price = trim.sub('', nums.text)
            return float(price)

#get the current date since we analyze by the day not the hour/min
from datetime import datetime
from zoneinfo import ZoneInfo
time_now = datetime.now(ZoneInfo("America/New_York"))
now_ny = time_now.toordinal()
now_ny = (now_ny - X_mean) / X_std

torch.manual_seed(2008)
stock_analysis.eval()
with torch.inference_mode():
    test_logits = stock_analysis(torch.tensor([[float(now_ny)]]))
    test_pred = (test_logits * y_std) + y_mean
    print(f"The Model predicts that the price of {Ticker} is {test_pred} at {time_now}")

    price = torch.tensor([[real_time_price(Ticker)]])
    print(f"The actual price of {Ticker} today is {price}")