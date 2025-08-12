# src/visualize_lstm_performance.py
"""
Visualization of LSTM Model Performance for Demand Forecasting.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import load_model
import joblib
from src.model_utils import load_lstm_model_safe # Use robust loader

def create_output_dirs():
    """Create necessary directories for visualization outputs."""
    dirs = ['results/visualization']
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
        print(f"Ensured directory exists: {dir}")

def load_daily_demand_data():
    """Load daily demand data."""
    try:
        df_demand_file = 'data/processed/daily_demand_data.csv'
        if not os.path.exists(df_demand_file):
            df_demand_file = 'data/processed/daily_demand_data_corrected.csv'
            if not os.path.exists(df_demand_file):
                 raise FileNotFoundError(f"Daily demand data not found at {df_demand_file} or fallback.")
        
        df_demand = pd.read_csv(df_demand_file)
        df_demand['order_date'] = pd.to_datetime(df_demand['order_date'])
        df_demand.set_index('order_date', inplace=True)
        df_demand.sort_index(inplace=True)
        return df_demand
    except Exception as e:
        print(f"Error loading daily demand data: {e}")
        raise

def prepare_lstm_sequences(data, n_steps_in=14, n_steps_out=7, target_col='Sales'):
    """Prepare sequences for LSTM."""
    X, y = [], []
    for i in range(len(data) - n_steps_in - n_steps_out + 1):
        X.append(data[i:(i + n_steps_in)])
        y.append(data[(i + n_steps_in):(i + n_steps_in + n_steps_out), 0])
    return np.array(X), np.array(y)

def plot_actual_vs_predicted(y_true, y_pred, model_name='LSTM'):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title(f'{model_name} Actual vs Predicted Sales')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = f'results/visualization/{model_name.lower()}_actual_vs_predicted.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Actual vs Predicted plot saved to {plot_path}")

def plot_residuals_over_time(dates, residuals, model_name='LSTM'):
    """Plot residuals over time."""
    plt.figure(figsize=(12, 6))
    plt.plot(dates, residuals, marker='o', linestyle='-', alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title(f'{model_name} Residuals Over Time')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = f'results/visualization/{model_name.lower()}_residuals_over_time.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Residuals over time plot saved to {plot_path}")

def plot_time_series_forecast(actual_dates, actual_values, forecast_dates, forecast_values, model_name='LSTM'):
    """Plot time series forecast."""
    plt.figure(figsize=(15, 8))
    plt.plot(actual_dates, actual_values, label='Actual Sales', color='blue', linewidth=1)
    plt.plot(forecast_dates, forecast_values, label='Forecasted Sales', color='red', linestyle='--', linewidth=1)
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title(f'{model_name} Time Series Forecast')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = f'results/visualization/{model_name.lower()}_time_series_forecast.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Time series forecast plot saved to {plot_path}")

def main():
    """Main function to generate LSTM performance visualizations."""
    print("="*60)
    print("LSTM PERFORMANCE VISUALIZATION")
    print("="*60)
    
    create_output_dirs()
    
    try:
        # Load daily demand data
        print("Loading daily demand data...")
        df_demand = load_daily_demand_data()
        
        # Load LSTM model and scaler
        print("Loading LSTM model and scaler...")
        model_path = 'models/lstm_tuning_trial_9310.h5' # Use your best tuned model
        scaler_path = 'models/scaler_for_lstm_demand_forecast_sales.pkl'
        lstm_model = load_lstm_model_safe(model_path)
        lstm_scaler = joblib.load(scaler_path)
        
        # Prepare data
        print("Preparing data for LSTM evaluation...")
        target_col = 'Sales'
        feature_cols = ['Sales', 'Order Item Quantity', 'num_orders']
        
        if not all(col in df_demand.columns for col in feature_cols):
            raise ValueError(f"Required feature columns {feature_cols} not found in data.")
        
        data = df_demand[feature_cols].values
        scaled_data = lstm_scaler.transform(data)
        
        n_steps_in = 21 # Match your tuned model
        n_steps_out = 7
        X_seq, y_seq = prepare_lstm_sequences(scaled_data, n_steps_in, n_steps_out, target_col)
        
        # Time-based split
        split_index = int(len(X_seq) * 0.8)
        X_test_seq = X_seq[split_index:]
        y_test_seq = y_seq[split_index:]
        
        # Make predictions
        print("Making LSTM predictions...")
        y_pred_seq = lstm_model.predict(X_test_seq, verbose=0)
        
        # Inverse transform
        print("Inverting predictions...")
        num_predictions = y_pred_seq.shape[0] * y_pred_seq.shape[1]
        y_pred_flat = y_pred_seq.flatten()
        y_test_flat = y_test_seq.flatten()
        
        # Create dummy arrays for inverse transform
        dummy_data_for_inverse_pred = np.zeros((num_predictions, lstm_scaler.n_features_in_))
        dummy_data_for_inverse_test = np.zeros((num_predictions, lstm_scaler.n_features_in_))
        dummy_data_for_inverse_pred[:, 0] = y_pred_flat
        dummy_data_for_inverse_test[:, 0] = y_test_flat
        
        y_pred_actual = lstm_scaler.inverse_transform(dummy_data_for_inverse_pred)[:, 0]
        y_test_actual = lstm_scaler.inverse_transform(dummy_data_for_inverse_test)[:, 0]
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
        mae = mean_absolute_error(y_test_actual, y_pred_actual)
        print(f"LSTM Evaluation Metrics:")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        
        # Generate visualizations
        print("Generating visualizations...")
        
        # Actual vs Predicted
        plot_actual_vs_predicted(y_test_actual, y_pred_actual, 'LSTM')
        
        # Residuals over time
        # Get test dates
        test_dates = df_demand.index[split_index + n_steps_in:]
        residuals = y_test_actual - y_pred_actual
        plot_residuals_over_time(test_dates[:len(residuals)], residuals, 'LSTM')
        
        # Time series forecast (simplified: show last N actual vs forecasted)
        n_show = min(30, len(y_test_actual))
        last_actual_dates = df_demand.index[-n_show:]
        last_actual_values = df_demand[target_col].tail(n_show).values
        last_forecast_dates = df_demand.index[-n_show:-n_show+n_steps_out] if len(df_demand.index) > n_show else df_demand.index[:n_steps_out]
        last_forecast_values = y_pred_actual[-n_show:n_show+n_steps_out] if len(y_pred_actual) > n_show else y_pred_actual
        
        plot_time_series_forecast(
            df_demand.index[-n_show:], 
            df_demand[target_col].tail(n_show).values,
            df_demand.index[-n_show:-n_show+n_steps_out] if len(df_demand.index) > n_show else df_demand.index[:n_steps_out],
            y_pred_actual[-n_show:n_show+n_steps_out] if len(y_pred_actual) > n_show else y_pred_actual,
            'LSTM'
        )
        
        print("\nLSTM performance visualizations completed successfully!")
        print("Plots saved to results/visualization/")
        
    except Exception as e:
        print(f"Error during LSTM visualization: {e}")
        raise

if __name__ == "__main__":
    main()