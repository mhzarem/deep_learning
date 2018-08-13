import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

training_set = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = training_set.iloc[:, 1:2].values
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)
X_train = training_set[0:1257]
Y_train = training_set[1:1258]
X_train = np.reshape(X_train, (1257, 1, 1))
# making RNN
RNN = Sequential()
RNN.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1)))
RNN.add(Dense(units=1))
RNN.compile(optimizer='adam', loss='mean_squared_error')
RNN.fit(X_train, Y_train, batch_size=32, epochs=200)
