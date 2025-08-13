# src/fix_lstm.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
import os

print("🔧 Loading and preparing data...")

# Load data
df = pd.read_csv("data/processed/daily_demand_data.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# Feature engineering
df['delay_7d_lag'] = df['delay_rate'].shift(7)
df['delay_7d_ma'] = df['delay_rate'].rolling(7).mean()
df.dropna(inplace=True)

# Prepare data
scaler = MinMaxScaler()
data = df[['delay_rate', 'delay_7d_lag', 'delay_7d_ma']].values
data_scaled = scaler.fit_transform(data)

def create_sequences(data, seq_length=7):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])  # Predict delay_rate
    return np.array(X), np.array(y)

seq_length = 7
X, y = create_sequences(data_scaled, seq_length)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Build model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 3)),
    Dropout(0.3),
    LSTM(50),
    Dropout(0.3),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')

# Train
print("🚀 Training LSTM model...")
history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                    validation_data=(X_test, y_test), verbose=1)

# ✅ Save: Architecture (JSON) + Weights (.weights.h5) + Scaler
os.makedirs("models", exist_ok=True)

# Save architecture
with open("models/lstm_model_fixed.json", "w") as f:
    f.write(model.to_json())
print("✅ Model architecture saved as JSON")

# Save weights (correct extension)
model.save_weights("models/lstm_model_fixed.weights.h5")
print("✅ Model weights saved as .weights.h5")

# Save scaler
joblib.dump(scaler, "models/lstm_scaler_fixed.pkl")
print("✅ Scaler saved")

print("🎉 Fixed LSTM model training and saving completed!")