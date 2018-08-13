# importing libs
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
# importing data_set
data_set = pd.read_csv('Churn_Modelling.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# print(data_set.memory_usage())
# print(data_set.describe())

# splitting data to input and output
X = data_set.iloc[:, 3:13].values
Y = data_set.iloc[:, 13].values
# print(X)
# print(Y)

# converting categorical data to numeric data
label_encoder_x2 = LabelEncoder()
X[:, 1] = label_encoder_x2.fit_transform(X[:, 1])
# print(X[:, 1])

label_encoder_x2 = LabelEncoder()
X[:, 2] = label_encoder_x2.fit_transform(X[:, 2])
# print(X[:, 2])
# print(X)
X = pd.DataFrame(X)
# print(X.dtypes)
# dummy variable correction
one_hot_encoder = OneHotEncoder(categorical_features=[1])
X = one_hot_encoder.fit_transform(X).toarray()
# print(X)
X = X[:, 1:]
# print(X)

X = pd.DataFrame(X)
# print(X)

# splitting to train and test part
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
# print(X_train)
# creating ANN
ANN = Sequential()
# input layer
ANN.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
ANN.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
ANN.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
ANN.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
ANN.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ANN.fit(X_train, Y_train, batch_size=10, epochs=10)
y_predict = ANN.predict(X_test)
print(y_predict)
y_predict = (y_predict > 0.45)
print(y_predict)

cm = confusion_matrix(Y_test, y_predict)
print(cm)
print(ANN.summary())
