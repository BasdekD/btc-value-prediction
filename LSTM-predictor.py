# Library Imports
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
plt.style.use("ggplot")
from keras.models import Sequential
from keras.layers import LSTM, Dense
import utilities

# Loading/Reading in the Data
# df = pd.read_csv("BTC-USD.csv")

endpoint = 'https://min-api.cryptocompare.com/data/histoday'
res = requests.get(endpoint + '?fsym=BTC&tsym=EUR&limit=500')
df = pd.DataFrame(json.loads(res.content)['Data'])

# Data Preprocessing
# Setting the datetime index as the date, only selecting the 'Close' column, then only the last 500 closing prices.
df = df.set_index("time").tail(500)
df = df.set_index(pd.to_datetime(df.index, unit='s'))

# Normalizing/Scaling the Data
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# Plotting the Closing Prices
df['close'].plot(figsize=(14, 8))
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
X = X.reshape((X.shape[0], 30, n_features))
y = np.array(y)

# Instatiating the model
model = Sequential()

# Activation
activ = "softsign"

# Input layer
model.add(LSTM(30, activation=activ, return_sequences=True, input_shape=(n_per_in, n_features)))

# Hidden layers
utilities.layer_maker(model, n_layers=6, n_nodes=12, activation=activ)

# Final Hidden layer
model.add(LSTM(60, activation=activ))

# Output layer
model.add(Dense(60))

# Model summary
model.summary()

# Compiling the data with selected specifications
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

res = model.fit(X, y, epochs=130, batch_size=32, validation_split=0.1)

utilities.visualize_training_results(res)

plt.figure(figsize=(12, 5))

# Getting predictions by predicting from the last available X variable
yhat = model.predict(X[-1].reshape(1, n_per_in, n_features)).tolist()[0]

# Transforming values back to their normal prices
yhat = scaler.inverse_transform(np.array(yhat).reshape(-1, 6)).tolist()
yhat = pd.DataFrame(yhat)


# Getting the actual values from the last available y variable which correspond to its respective X variable
actual = scaler.inverse_transform(y[-1].reshape(-1, 6))
actual = pd.DataFrame(actual)

# Printing and plotting those predictions
print("Predicted Prices:\n", yhat.loc[::, 0])
plt.plot(yhat.loc[::, 0], label='Predicted')

# Printing and plotting the actual values
print("\nActual Prices:\n", actual.loc[::, 0])
plt.plot(actual.loc[::, 0], label='Actual')

plt.title(f"Predicted vs Actual Closing Prices")
plt.ylabel("Price")
plt.legend()
plt.show()

# Predicting off of y because it contains the most recent dates
yhat = model.predict(np.array(df.tail(n_per_in)).reshape(1, n_per_in, n_features)).tolist()[0]

# Transforming the predicted values back to their original prices
yhat = scaler.inverse_transform(np.array(yhat).reshape(-1, 6)).tolist()

# Creating a DF of the predicted prices
preds = pd.DataFrame(yhat, index=pd.date_range(start=df.index[-1], periods=len(yhat), freq="D"), columns=df.columns)

# Printing the predicted prices
print(preds)

# Number of periods back to visualize the actual values
pers = 10

# Transforming the actual values to their original price
actual = pd.DataFrame(scaler.inverse_transform(df.tail(pers)), index=df.tail(pers).index, columns=df.columns).append(preds.head(1))

# Plotting
plt.figure(figsize=(16, 6))
plt.plot(actual['close'], label="Actual Prices")
plt.plot(preds['close'], label="Predicted Prices")
plt.ylabel("Price")
plt.xlabel("Dates")
plt.title(f"Forecasting the next {len(yhat)} days")
plt.legend()
# plt.savefig("BTC_predictions.png")
plt.show()


