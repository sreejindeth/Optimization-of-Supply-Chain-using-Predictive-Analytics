# src/tune_lstm.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import optuna
import joblib
import os

# --- 1. Load Processed Time-Series Data ---
def load_data():
    ts_file = "data/processed/daily_demand_data.csv"
    df = pd.read_csv(ts_file)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df

# --- 2. Prepare Sequences ---
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

# --- 3. Objective Function for Optuna ---
def objective(trial):
    # Load and prepare data
    df = load_data()
    target = 'delay_rate'  # Forecast system-wide delay rate
    raw_data = df[target].values.reshape(-1, 1)

    # Normalize
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(raw_data)

    # Hyperparameters to tune
    seq_length = trial.suggest_int('seq_length', 15, 30)
    n_lstm1 = trial.suggest_int('n_lstm1', 32, 128)
    n_lstm2 = trial.suggest_int('n_lstm2', 32, 128)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])

    # Create sequences
    X, y = create_sequences(data_scaled, seq_length)
    
    # Train-test split (80-20)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build model
    model = Sequential([
        LSTM(n_lstm1, return_sequences=True, input_shape=(seq_length, 1)),
        Dropout(dropout),
        LSTM(n_lstm2, return_sequences=False),
        Dropout(dropout),
        Dense(25, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=0
    )

    # Return validation loss
    return history.history['val_loss'][-1]

# --- 4. Run Tuning ---
if __name__ == "__main__":
    print("🔍 Starting LSTM Hyperparameter Tuning with Optuna...")

    # Create study
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)  # Increase trials for better results

    print("\n✅ Best Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"Best Validation Loss: {study.best_value:.6f}")

    # Save study results
    os.makedirs("models", exist_ok=True)
    joblib.dump(study, "models/lstm_optuna_study.pkl")
    print("💾 Optuna study saved to models/lstm_optuna_study.pkl")

    # Optional: Retrain best model (you can move this to lstm_model.py)
    print("\n🔁 Retraining best model...")
    best_params = study.best_params

    # Re-run data prep with best seq_length
    df = load_data()
    raw_data = df['delay_rate'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(raw_data)

    X, y = create_sequences(data_scaled, best_params['seq_length'])
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build best model
    best_model = Sequential([
        LSTM(best_params['n_lstm1'], return_sequences=True, input_shape=(best_params['seq_length'], 1)),
        Dropout(best_params['dropout']),
        LSTM(best_params['n_lstm2']),
        Dropout(best_params['dropout']),
        Dense(25, activation='relu'),
        Dense(1)
    ])

    best_model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mse')
    best_model.fit(X_train, y_train, 
                   batch_size=best_params['batch_size'], 
                   epochs=50, 
                   validation_data=(X_test, y_test),
                   callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
                   verbose=1)

    # Save best model and scaler
    # Save model architecture as JSON
    os.makedirs("models", exist_ok=True)
    with open("models/lstm_model_tuned.json", "w") as f:
          f.write(best_model.to_json())

    # ✅ Save weights with correct extension
    best_model.save_weights("models/lstm_model_tuned.weights.h5")

    # ✅ Save scaler
    joblib.dump(scaler, "models/lstm_scaler_tuned.pkl")

    print("✅ Best LSTM model saved as JSON + weights (.weights.h5)")
    print("✅ Scaler saved to models/lstm_scaler_tuned.pkl")