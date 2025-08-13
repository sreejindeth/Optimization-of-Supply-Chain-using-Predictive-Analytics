# src/lstm_viz.py
"""
Visualize LSTM Model Performance
- Loads trained LSTM model
- Loads scalers and STL components
- Reconstructs full delay rate from trend + seasonal + predicted residual
- Plots Actual vs Predicted
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
from tensorflow.keras.models import model_from_json

# --- 1. Load Model ---
def load_lstm_model():
    print("🧠 Loading LSTM model architecture and weights...")
    with open("models/lstm_improved.json", "r") as f:
        model = model_from_json(f.read())
    model.load_weights("models/lstm_improved.weights.h5")
    model.compile(optimizer="adam", loss="mae")
    return model

# --- 2. Load Data and Components ---
def load_data_and_components():
    print("📊 Loading data and STL components...")

    # Load raw time-series data
    df = pd.read_csv("data/processed/daily_demand_data.csv")
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # Load STL components
    components = joblib.load("models/stl_components.pkl")  # From improved_lstm.py
    components = components.reset_index()  # Make 'date' a column
    components['date'] = pd.to_datetime(components['date'])

    # Merge components with original data
    df = df.merge(components[['date', 'trend', 'seasonal', 'resid']], on='date', how='left')
    df = df.dropna()

    return df

# --- 3. Create Sequences ---
def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
    return np.array(X)

# --- 4. Main Visualization ---
def main():
    # Load model
    model = load_lstm_model()

    # Load data
    df = load_data_and_components()

    # Load scalers
    scalers = joblib.load("models/lstm_scalers_improved.pkl")

    # Parameters
    seq_length = 7

    # Scale the 'resid' for prediction
    resid_scaled = scalers['resid'].transform(df[['resid']])
    trend = df['trend'].values
    seasonal = df['seasonal'].values

    # Create sequences for prediction
    X_scaled = create_sequences(resid_scaled, seq_length)

    # Predict residuals
    print("🔮 Generating predictions...")
    y_pred_resid_scaled = model.predict(X_scaled, verbose=0)
    y_pred_resid = scalers['resid'].inverse_transform(y_pred_resid_scaled)

    # Reconstruct full prediction
    y_true_full = trend[seq_length:] + seasonal[seq_length:] + df['resid'].values[seq_length:]
    y_pred_full = trend[seq_length:] + seasonal[seq_length:] + y_pred_resid.flatten()

    # Ground truth for actual delay_rate
    y_true_actual = df['delay_rate'].values[seq_length:]

    # Plot
    print("📈 Plotting results...")
    plt.figure(figsize=(14, 6))
    plt.plot(y_true_actual, label='Actual Delay Rate', alpha=0.8)
    plt.plot(y_pred_full, label='Predicted Delay Rate', alpha=0.8)
    plt.title('LSTM Model Performance: Actual vs Predicted Delay Rate (Improved)')
    plt.xlabel('Time (Days)')
    plt.ylabel('Delay Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    os.makedirs("results/performance", exist_ok=True)
    plt.savefig("results/performance/lstm_improved_performance.png")
    plt.show()

    # ✅ Add this: Print Final Metrics
    print("📊 Calculating final metrics...")

    mae = np.mean(np.abs(y_true_actual - y_pred_full))
    rmse = np.sqrt(np.mean((y_true_actual - y_pred_full)**2))
    r2 = 1 - (np.sum((y_true_actual - y_pred_full)**2) / np.sum((y_true_actual - y_true_actual.mean())**2))

    print("📊 Final Metrics:")
    print(f"   MAE:  {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²:   {r2:.4f}")

if __name__ == "__main__":
    main()