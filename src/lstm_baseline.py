# src/lstm_baseline.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load data
df = pd.read_csv("data/processed/daily_demand_data.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# 2. Use only delay_rate
data = df['delay_rate'].values.reshape(-1, 1)

# 3. Scale
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 4. Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 14
X, y = create_sequences(data_scaled, seq_length)

# 5. Train-test split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 6. Build model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer=Adam(learning_rate=0.001), loss='mae')

# 7. Train
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# 8. Predict
y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled)
y_true = scaler.inverse_transform(y_test.reshape(-1, 1))

# 9. Metrics
mae = np.mean(np.abs(y_true - y_pred))
rmse = np.sqrt(np.mean((y_true - y_pred)**2))
r2 = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - y_true.mean())**2))

print("📊 Baseline LSTM Metrics:")
print(f"   MAE:  {mae:.4f}")
print(f"   RMSE: {rmse:.4f}")
print(f"   R²:   {r2:.4f}")

# 10. Save
os.makedirs("models", exist_ok=True)
model.save("models/lstm_baseline.h5")
joblib.dump(scaler, "models/lstm_baseline_scaler.pkl")