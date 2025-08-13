import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import joblib

# Load time-series data
df_ts = pd.read_csv("data/processed/daily_demand_data.csv")
df_ts['date'] = pd.to_datetime(df_ts['date'])
df_ts = df_ts.sort_values('date').reset_index(drop=True)

# Use 'delay_rate' as target for forecasting
data = df_ts['delay_rate'].values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences
def create_sequences(data, seq_length=30):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 30
X, y = create_sequences(data_scaled, seq_length)

# Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build LSTM
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train
model.fit(X_train, y_train, batch_size=32, epochs=20, validation_data=(X_test, y_test))

# Save model and scaler
model.save("models/lstm_model.h5")
joblib.dump(scaler, "models/lstm_scaler.pkl")
print("✅ LSTM model and scaler saved!")