import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show
data_set = pd.read_csv('Credit_Card_Applications.csv')
X = data_set.iloc[:, :-1].values
Y = data_set.iloc[:, -1].values

# Feature Scaling
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

# Training the SOM
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(X, num_iteration=100)

# Visualizing the results
bone()
pcolor(som.distance_map().T)
colorbar()
marker = ['o', 's']
colors = ['r', 'g']
for i, item in enumerate(X):
    w = som.winner(item)
    plot(w[0]+0.5, w[1]+0.5, marker[Y[i]], markeredgecolor=colors[Y[i]], markerfacecolor='None', markersize=10, markeredgewidth=2)
plt.show()

# Finding the frauds
mapping = som.win_map(X)
frauds = np.concatenate((mapping[(8, 1)],  mapping[(6, 8)]), axis=0)
frauds = sc.inverse_transform(frauds)


