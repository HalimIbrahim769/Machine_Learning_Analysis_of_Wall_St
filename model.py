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
Ticker = str(input('What Stock would you like to analyze (Ticker): '))

#Gathering Ticker Data
ticker_data = yf.Ticker(ticker=Ticker).history(period='max').reset_index()
dates = ticker_data['Date'].dt.date.apply(lambda x: x.toordinal())
price = ticker_data['Close']

#Organizing Data
X = torch.tensor(dates.values, dtype=torch.float32).unsqueeze(1)
y = torch.tensor(price.values, dtype=torch.float32).unsqueeze(1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, train_size=.8, random_state=42)

#Reducing data so the model can process the loss
X_mean, X_std = X_train.mean(), X_train.std()
y_mean, y_std = y_train.mean(), y_train.std()

X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

y_train = (y_train - y_mean) / y_std
y_test = (y_test - y_mean) / y_std


#Training and Testing
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(stock_analysis.parameters(), lr = .1)

#Allows for the learning rate to change so that the model performs better
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

y_logits = stock_analysis(X_test)
epochs = 10000

compiled_model = torch.compile(stock_analysis) # helps run the model faster 
from sklearn.metrics import r2_score

for epoch in range(epochs):
    stock_analysis.train()
    y_logits = stock_analysis(X_train)
    loss = loss_fn(y_logits, y_train)


    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    scheduler.step()

    stock_analysis.eval()
    with torch.inference_mode():
        test_logits = stock_analysis(X_test)

        test_pred = test_logits.detach().numpy()
        y_true = y_test.numpy()

        test_loss = loss_fn(test_logits, y_test)

    #Checking accuracy
    if epoch % 1000 == 0:
        #the closer the R2 is to 1 the better the model is
        print(f'Epoch: {epoch+1000} R2: {r2_score(y_true, test_pred):.2f} Loss: {loss:.2f} Test loss: {test_loss:.2f}')
print("\n")

#Testing the model on real time data
# NY GMT is -4
from datetime import datetime
import pytz as time
import time as tt


# Get the current time in NY
utc_now = datetime.now(time.utc)
gmt_minus_4_timezone = time.timezone("Etc/GMT+4")
gmt_minus_4_time = utc_now.astimezone(gmt_minus_4_timezone)
time_string = gmt_minus_4_time.strftime("%H:%M")


    
def real_time_price(stock_code):
    i = 0
    while time_string != '9:30':
        #if the stock market closes
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
now_ny = datetime.now(ZoneInfo("America/New_York")).toordinal()

stock_analysis.eval()
with torch.inference_mode():
    test_logits = stock_analysis(torch.tensor(now_ny))
    test_pred = test_logits * y_std + y_mean
    price = torch.tensor(real_time_price(Ticker))

    r2 = r2_score(price, test_pred)
    loss = loss_fn(test_logits, price)
    print(f'R2: {r2:.2f}, Loss: {loss:.2f}')