import numpy as np
import pandas as pd
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import requests
from datetime import datetime, timedelta

class DebugCallback(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        print(f"Batch {batch} finished, LSTM state reset")

def get_binance_data(symbol="BTCUSDT", interval="1h", limit=87600):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url).json()
    df = pd.DataFrame(response, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                         'close_time', 'quote_asset_volume', 'trades', 'taker_buy_base',
                                         'taker_buy_quote', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  
    df['close'] = df['close'].astype(float)  
    return df[['timestamp', 'close']]


def prepare_data(data, time_steps=300):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data[['close']])

    X, y, dates = [], [], []
    for i in range(len(data_scaled) - time_steps):
        X.append(data_scaled[i:i+time_steps])
        y.append(data_scaled[i+time_steps])
        dates.append(data['timestamp'].iloc[i+time_steps])  

    return np.array(X), np.array(y), scaler, dates

def create_lstm_model(input_shape):
    model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(100, return_sequences=True),
    LSTM(50, return_sequences=False),
    Dense(50, activation="relu"),
    Dense(25, activation="relu"),
    Dense(1)
])
    model.compile(optimizer='adam', loss='mse')
    return model


df = get_binance_data()
X, y, scaler, dates = prepare_data(df)


split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
dates_train, dates_test = dates[:split], dates[split:]


model = create_lstm_model((X.shape[1], 1))
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_test, y_test), callbacks=[DebugCallback()])

checkpoint_dir = "saved_models"
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "custom_predict_model.h5")
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy')

model.save(checkpoint_path)
print(f"Модель сохранена по пути: {checkpoint_path}")


y_pred = model.predict(X_test)
y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))


future_steps = 3  
future_predictions = []
last_sequence = X[-1] 

future_dates = [dates[-1] + timedelta(hours=i+1) for i in range(future_steps)]  

for _ in range(future_steps):
    next_pred = model.predict(last_sequence.reshape(1, X.shape[1], 1))  
    future_predictions.append(next_pred[0, 0]) 


    last_sequence = np.roll(last_sequence, -1)
    last_sequence[-1] = next_pred

future_predictions_actual = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

plt.figure(figsize=(50, 6))
plt.plot(dates_test, y_test_actual, label="Реальная цена", color="blue")
plt.plot(dates_test, y_pred_actual, label="Предсказанная цена", color="red")
plt.plot(future_dates, future_predictions_actual, label="Прогноз (48ч вперед)", color="green", linestyle="dashed")

plt.xlabel("Дата")
plt.ylabel("Цена BTC (USDT)")
plt.legend()
plt.title("Прогноз цены биткойна на основе LSTM")
plt.xticks(rotation=45)
plt.grid()
plt.show()
