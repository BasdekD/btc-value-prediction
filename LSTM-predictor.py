# Library Imports
import json
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
from keras.backend import flatten
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

import conf

plt.style.use("ggplot")
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation, RepeatVector, TimeDistributed
import utilities

# endpoint = 'https://min-api.cryptocompare.com/data/histoday'      # TODO: UNCOMMENT FOR LIVE DATA
# res = requests.get(endpoint + '?fsym=BTC&tsym=EUR&limit=500')
# df = pd.DataFrame(json.loads(res.content)['Data'])

df = pd.read_csv(conf.CURR_DIR + '\\BTC-EUR.csv')

# Data Preprocessing        # TODO: UNCOMMENT FOR LIVE DATA PROCESSING
# Setting the datetime index as the date, only selecting the 'Close' column, then only the last 500 closing prices.
# df = df.set_index("time")
# df = df.set_index(pd.to_datetime(df.index, unit='s'))

df = df.set_index("Date")       # TODO: CHANGE DATE TO "time" FOR LIVE DATA
df = df.set_index(pd.to_datetime(df.index))

# Normalizing/Scaling the Data
# scaler = StandardScaler()
# scaler = RobustScaler()
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# Plotting the Closing Prices
df['Close'].plot(figsize=(14, 8))       # TODO: TOGGLE {Close, close}
plt.title("BTC Closing Prices")
plt.ylabel("Price (Normalized)")
plt.show()

# How many periods looking back to learn
n_per_in = 30

# How many periods to predict
n_per_out = 10

# Features (in this case it's 1 because there is only one feature: price)
n_features = 6

# Splitting the data into appropriate sequences
X, y = utilities.split_sequence(df, n_per_in, n_per_out, n_features)
print(X)
print(y)

X = np.array(X)
# Reshaping the X variable from 2D to 3D
X = X.reshape((X.shape[0], n_per_in, n_features))
y = np.array(y)
y = y.reshape((y.shape[0], n_per_out, n_features))
model = None

if not conf.TRAINED_MODEL.exists():

    # Instatiating the model
    model = Sequential()

    # Activation
    activ = "softsign"

    # Input layer
    model.add(LSTM(30, activation=activ, return_sequences=False, input_shape=(n_per_in, n_features)))

    model.add(RepeatVector(n_per_out))
    # Hidden layers
    utilities.layer_maker(model, n_layers=6, n_nodes=12, activation=activ)

    # Final Hidden layer
    model.add(LSTM(10, activation=activ, return_sequences=True))

    # Output layer
    # model.add(Dense(n_per_out))
    model.add(TimeDistributed(Dense(n_features)))

    # Model summary
    model.summary()

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    res = model.fit(X, y, batch_size=32, epochs=800, validation_split=0.1)

    utilities.visualize_training_results(res)

    # save the model to disk
    pickle.dump(model, open(conf.TRAINED_MODEL, 'wb'))

elif conf.TRAINED_MODEL.exists():
    model = pickle.load(open(conf.TRAINED_MODEL, 'rb'))

plt.figure(figsize=(12, 5))

# Getting predictions by predicting from the last available X variable
yhat = model.predict(X[-1].reshape(1, n_per_in, n_features)).tolist()[0]

# Transforming values back to their normal prices
yhat = scaler.inverse_transform(np.array(yhat).reshape(-1, n_features)).tolist()
yhat = pd.DataFrame(yhat)


# Getting the actual values from the last available y variable which correspond to its respective X variable
actual = scaler.inverse_transform(y[-1].reshape(-1, n_features))
actual = pd.DataFrame(actual)
print(mean_absolute_error(yhat, actual))

# Printing and plotting those predictions
print("Predicted Prices:\n", yhat.loc[::, 3])
plt.plot(yhat.loc[::, 3], label='Predicted')    # TODO: Change from 3 to 0

# Printing and plotting the actual values
print("\nActual Prices:\n", actual.loc[::, 3])
plt.plot(actual.loc[::, 3], label='Actual')     # TODO: Change from 3 to 0

plt.title(f"Predicted vs Actual Closing Prices")
plt.ylabel("Price")
plt.legend()
plt.show()

# Predicting off of y because it contains the most recent dates
yhat = model.predict(np.array(df.tail(n_per_in)).reshape(1, n_per_in, n_features)).tolist()[0]

# Transforming the predicted values back to their original prices
yhat = scaler.inverse_transform(np.array(yhat).reshape(-1, n_features)).tolist()

# Creating a DF of the predicted prices
preds = pd.DataFrame(yhat, index=pd.date_range(start=df.index[-1], periods=len(yhat), freq="D"), columns=df.columns)


# Printing the predicted prices
print(preds)

# Number of periods back to visualize the actual values
pers = 10

# Transforming the actual values to their original price
actual = pd.DataFrame(scaler.inverse_transform(df.tail(pers)), index=df.tail(pers).index, columns=df.columns).append(preds.head(1))

# Plotting
plt.figure(figsize=(20, 10))

plt.plot(actual['Close'], label="Actual Prices")        # TODO: TOGGLE {Close, close}
plt.plot(preds['Close'], label="Predicted Prices")
plt.ylabel("Price")
plt.xlabel("Date")
plt.title(f"Forecasting the next {len(yhat)} days")
plt.legend()
plt.show()


