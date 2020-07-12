#Importing Libraries  
import numpy as np
import pandas as pd
import talib

#Setting the random seed to a fixed number
import random
random.seed(42)

#Importing the dataset
dataset = pd.read_csv('RELIANCE.NS.csv')
print()
dataset = dataset.dropna()
dataset = dataset[['Open', 'High', 'Low', 'Close']]

#Preparing the dataset
dataset['H-L'] = dataset['High'] - dataset['Low']
dataset['C-O'] = dataset['Close'] - dataset['Open']
dataset['3 day Mean Avg'] = dataset['Close'].shift(1).rolling(window = 3).mean()
dataset['10 day Mean Avg'] = dataset['Close'].shift(1).rolling(window = 10).mean()
dataset['30 day Mean Avg'] = dataset['Close'].shift(1).rolling(window = 30).mean()
dataset['Std_deviance']= dataset['Close'].rolling(5).std()
dataset['RSI'] = talib.RSI(dataset['Close'].values, timeperiod = 9)
dataset['Williams %R'] = talib.WILLR(dataset['High'].values, dataset['Low'].values, dataset['Close'].values, 7)

dataset['Price_Rise'] = np.where(dataset['Close'].shift(-1) > dataset['Close'], 1, 0)

dataset = dataset.dropna()

X = dataset.iloc[:, 4:-1]
y = dataset.iloc[:, -1]

#Splitting the dataset
split = int(len(dataset)*0.8)
X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Building the Artificial Neural Network
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential()

classifier.add(Dense(
        units = 128, 
        kernel_initializer = 'uniform', 
        activation = 'relu', 
        input_dim = X.shape[1]
        ))
classifier.add(Dense(
        units = 128, 
        kernel_initializer = 'uniform', 
        activation = 'relu'
        ))
classifier.add(Dense(
        units = 1, 
        kernel_initializer = 'uniform', 
        activation = 'sigmoid'
        ))
classifier.compile(
                   optimizer = 'sgd', 
                   loss = 'mean_squared_logarithmic_error', 
                   metrics = ['accuracy']
                   )

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


#Predicting the movement of the stock
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

dataset['y_pred'] = np.NaN
dataset.iloc[(len(dataset) - len(y_pred)):,-1:] = y_pred
trd_dataset = dataset.dropna()

#Computing Strategy Returns
trd_dataset['Tomorrows Returns'] = 0.
trd_dataset['Tomorrows Returns'] = np.log(trd_dataset['Close']/trd_dataset['Close'].shift(1))
trd_dataset['Tomorrows Returns'] = trd_dataset['Tomorrows Returns'].shift(-1)

trd_dataset['Strategy Returns'] = 0.
trd_dataset['Strategy Returns'] = np.where(
                    trd_dataset['y_pred'] == True, 
                    trd_dataset['Tomorrows Returns'], 
                    - trd_dataset['Tomorrows Returns']
                    )

trd_dataset['Total Market Returns'] = np.cumsum(trd_dataset['Tomorrows Returns'])
trd_dataset['Total Strategy Returns'] = np.cumsum(trd_dataset['Strategy Returns'])

#Plotting the graph of returns
import matplotlib.pyplot as srikplt
srikplt.figure(figsize=(10,5))
srikplt.plot(trd_dataset['Total Market Returns'], color='r', label='Market Returns')
srikplt.plot(trd_dataset['Total Strategy Returns'], color='g', label='Expected Strategy Returns')
srikplt.legend()
srikplt.show()