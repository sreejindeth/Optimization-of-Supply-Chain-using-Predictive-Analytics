# src/tune.py
import os
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid # For generating hyperparameter combinations
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
# Ensure TensorFlow logs are not too verbose
tf.get_logger().setLevel('ERROR')

# Inside src/tune.py

# --- Configuration ---
DAILY_DEMAND_DATA_PATH = 'data/processed/daily_demand_data.csv'
DAILY_DEMAND_DATA_PATH_FALLBACK = 'data/processed/daily_demand_data_corrected.csv'
RESULTS_DIR = 'results/model'
MODELS_DIR = 'models'

def load_daily_demand_data():
    """Load the daily demand data needed for LSTM forecasting."""
    try:
        df_demand_file = DAILY_DEMAND_DATA_PATH
        if not os.path.exists(df_demand_file):
            df_demand_file = DAILY_DEMAND_DATA_PATH_FALLBACK
            if not os.path.exists(df_demand_file):
                 raise FileNotFoundError(f"Daily demand data not found at {DAILY_DEMAND_DATA_PATH} or fallback {DAILY_DEMAND_DATA_PATH_FALLBACK}.")

        df_demand = pd.read_csv(df_demand_file)
        df_demand['order_date'] = pd.to_datetime(df_demand['order_date'])
        df_demand.set_index('order_date', inplace=True)
        df_demand.sort_index(inplace=True) # Ensure order
        print(f"Daily demand data loaded successfully. Shape: {df_demand.shape}")
        return df_demand
    except Exception as e:
        print(f"Error loading daily demand  {e}")
        raise

def prepare_lstm_data_for_tuning(df_demand, n_steps_in=14, n_steps_out=7, target_cols=['Sales']):
    """Prepare sequences for LSTM demand forecasting (tuning version)."""
    print(f"--- Preparing Data for LSTM Tuning (n_steps_in={n_steps_in}, n_steps_out={n_steps_out}) ---")
    
    if not all(col in df_demand.columns for col in target_cols):
        raise ValueError(f"Target columns {target_cols} not found in df_demand. Available: {list(df_demand.columns)}")
    data = df_demand[target_cols].values

    # Standardize the data for LSTM
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    X_seq, y_seq = [], []
    for i in range(len(scaled_data) - n_steps_in - n_steps_out + 1):
        X_seq.append(scaled_data[i:(i + n_steps_in)])
        y_seq.append(scaled_data[(i + n_steps_in):(i + n_steps_in + n_steps_out), 0]) # Predict first target (e.g., Sales)

    X_seq, y_seq = np.array(X_seq), np.array(y_seq)
    print(f"LSTM Sequences created. X shape: {X_seq.shape}, y shape: {y_seq.shape}")

    # Time-based split (e.g., last 20% for testing)
    split_index = int(len(X_seq) * 0.8)
    X_train_seq, X_test_seq = X_seq[:split_index], X_seq[split_index:]
    y_train_seq, y_test_seq = y_seq[:split_index], y_seq[split_index:]

    print(f"LSTM Train set: X {X_train_seq.shape}, y {y_train_seq.shape}")
    print(f"LSTM Test set: X {X_test_seq.shape}, y {y_test_seq.shape}")

    return X_train_seq, X_test_seq, y_train_seq, y_test_seq, scaler

# Inside src/tune.py

def build_lstm_model_for_tuning(hp_config, n_steps_out=7):
    """
    Build an LSTM model based on hyperparameter configuration.
    
    Args:
        hp_config (dict): Dictionary containing hyperparameters.
                         Example: {'lstm_units': 50, 'dropout_rate': 0.2, 'n_layers': 1}
        n_steps_out (int): Number of future steps to predict.
        
    Returns:
        model: Compiled Keras LSTM model.
    """
    print(f"Building LSTM model with config: {hp_config}")
    
    model = Sequential()
    
    # First LSTM layer
    model.add(LSTM(
        units=hp_config['lstm_units'],
        activation='relu',
        input_shape=(hp_config['n_steps_in'], hp_config['n_features']), # n_features=3 for ['Sales', 'Order Item Quantity', 'num_orders']
        return_sequences=True if hp_config['n_layers'] > 1 else False # Return sequences if more layers follow
    ))
    model.add(Dropout(rate=hp_config['dropout_rate']))
    
    # Additional LSTM layers (if specified)
    for i in range(1, hp_config['n_layers']):
        model.add(LSTM(
            units=hp_config['lstm_units'],
            activation='relu',
            return_sequences=True if i < hp_config['n_layers'] - 1 else False # Return sequences if not last layer
        ))
        model.add(Dropout(rate=hp_config['dropout_rate']))
    
    # Output layer
    model.add(Dense(n_steps_out)) # Predict n_steps_out values
    
    # Compile model
    model.compile(optimizer='adam', loss='mse') # Use 'mae' if preferred
    
    print("LSTM model built and compiled.")
    return model

# Inside src/tune.py

def train_and_evaluate_lstm_model(hp_config, X_train, y_train, X_test, y_test, scaler, n_steps_out=7, model_name_base='lstm_tuning_trial'):
    """
    Train and evaluate an LSTM model with given hyperparameters.
    
    Args:
        hp_config (dict): Hyperparameter configuration.
        X_train, y_train, X_test, y_test: Training and testing data.
        scaler: Fitted StandardScaler.
        n_steps_out (int): Number of steps out.
        model_name_base (str): Base name for saving model/results.
        
    Returns:
        dict: Dictionary containing results (config, metrics, paths).
    """
    print(f"--- Training and Evaluating LSTM Model ---")
    
    try:
        # --- Build Model ---
        model = build_lstm_model_for_tuning(hp_config, n_steps_out)
        
        # --- Train Model ---
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        print("Starting LSTM training...")
        history = model.fit(
            X_train, y_train,
            epochs=100, # Use early stopping
            batch_size=hp_config['batch_size'],
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=0 # Suppress training output
        )
        print("LSTM training finished.")
        
        # --- Evaluate Model ---
        print("Evaluating LSTM model...")
        y_pred_seq = model.predict(X_test, verbose=0)
        
        # Inverse transform for meaningful metrics
        try:
            # Flatten predictions and actuals for inverse transform
            y_test_flat = y_test.reshape(-1, 1)
            y_pred_flat = y_pred_seq.reshape(-1, 1)
            
            # Create dummy arrays for inverse transform (assume scaler was fit on 3 features)
            dummy_data_for_inverse_test = np.zeros((y_test_flat.shape[0], scaler.n_features_in_))
            dummy_data_for_inverse_pred = np.zeros((y_pred_flat.shape[0], scaler.n_features_in_))
            dummy_data_for_inverse_test[:, 0] = y_test_flat.flatten()
            dummy_data_for_inverse_pred[:, 0] = y_pred_flat.flatten()
            
            y_test_actual = scaler.inverse_transform(dummy_data_for_inverse_test)[:, 0]
            y_pred_actual = scaler.inverse_transform(dummy_data_for_inverse_pred)[:, 0]
            
            # Calculate metrics on actual scale
            rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
            mae = mean_absolute_error(y_test_actual, y_pred_actual)
            
            print(f"LSTM Evaluation (on actual scale for primary target):")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAE: {mae:.2f}")
            
        except Exception as e:
            print(f"Warning: Could not calculate metrics on inverse-scaled data: {e}")
            # Fallback metrics on scaled data
            mse_scaled = mean_squared_error(y_test, y_pred_seq)
            mae_scaled = mean_absolute_error(y_test, y_pred_seq)
            print(f"LSTM Evaluation (on scaled data):")
            print(f"MSE (scaled): {mse_scaled:.4f}")
            print(f"MAE (scaled): {mae_scaled:.4f}")
            rmse = np.nan
            mae = np.nan # Indicate error in actual scale metrics
            
        # --- Save Model and Metrics ---
        config_name = f"{model_name_base}_{hash(str(hp_config)) % 10000}" # Simple hash for unique name
        model_path = f'{MODELS_DIR}/{config_name}.h5'
        metrics_path = f'{RESULTS_DIR}/{config_name}_metrics.csv'
        
        model.save(model_path)
        metrics_df = pd.DataFrame({'Metric': ['RMSE', 'MAE'], 'Value': [rmse, mae]})
        metrics_df.to_csv(metrics_path, index=False)
        print(f"LSTM model saved to {model_path}")
        print(f"LSTM metrics saved to {metrics_path}")
        
        return {
            'config': hp_config,
            'rmse': rmse,
            'mae': mae,
            'model_path': model_path,
            'metrics_path': metrics_path,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"Error training/evaluating LSTM model with config {hp_config}: {e}")
        return {
            'config': hp_config,
            'rmse': np.nan,
            'mae': np.nan,
            'model_path': None,
            'metrics_path': None,
            'status': 'failed',
            'error': str(e)
        }
    
# Inside src/tune.py

def main():
    """Main function to run the LSTM tuning pipeline."""
    print("="*60)
    print("LSTM HYPERPARAMETER TUNING PIPELINE")
    print("="*60)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Started at: {timestamp}")
    
    # Ensure directories exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # --- 1. Load Data ---
    df_demand = load_daily_demand_data()
    
    # --- 2. Define Hyperparameter Grid ---
    # These are the values to test for each hyperparameter
    param_grid = {
        'lstm_units': [50, 100],      # Test 50 and 100 units
        'dropout_rate': [0.2, 0.3],   # Test 20% and 30% dropout
        'n_layers': [1, 2],           # Test 1 and 2 LSTM layers
        'batch_size': [32, 64],       # Test batch sizes 32 and 64
        'n_steps_in': [14, 21],       # Test looking back 14 or 21 days
        # n_steps_out is fixed for this tuning run
        # learning_rate is not directly tunable here, but you could add it
    }
    n_steps_out_fixed = 7 # Fixed forecast horizon
    
    # --- 3. Generate All Combinations ---
    param_combinations = list(ParameterGrid(param_grid))
    print(f"Total hyperparameter combinations to test: {len(param_combinations)}")
    
    # --- 4. Prepare Data (using default n_steps_in for now, will adjust per config) ---
    # We'll prepare data inside the loop for each config to handle varying n_steps_in
    # But let's prepare a base dataset with a common n_steps_in
    base_n_steps_in = 14
    feature_cols = ['Sales', 'Order Item Quantity', 'num_orders']
    X_train_base, X_test_base, y_train_base, y_test_base, scaler_base = prepare_lstm_data_for_tuning(
        df_demand, n_steps_in=base_n_steps_in, n_steps_out=n_steps_out_fixed, target_cols=feature_cols
    )
    
    # --- 5. Run Tuning Loop ---
    results = []
    for i, params in enumerate(param_combinations):
        print(f"\n--- Testing Configuration {i+1}/{len(param_combinations)} ---")
        print(f"Parameters: {params}")
        
        # --- Adjust data preparation based on n_steps_in in params ---
        current_n_steps_in = params['n_steps_in']
        if current_n_steps_in != base_n_steps_in:
            print(f"Re-preparing data with n_steps_in={current_n_steps_in}...")
            X_train, X_test, y_train, y_test, scaler = prepare_lstm_data_for_tuning(
                df_demand, n_steps_in=current_n_steps_in, n_steps_out=n_steps_out_fixed, target_cols=feature_cols
            )
        else:
            # Use base prepared data
            X_train, X_test, y_train, y_test, scaler = X_train_base, X_test_base, y_train_base, y_test_base, scaler_base
            
        # Add derived parameters needed by the model builder
        params['n_features'] = len(feature_cols) # Add number of features
        
        # --- Train and Evaluate ---
        result = train_and_evaluate_lstm_model(
            params, X_train, y_train, X_test, y_test, scaler,
            n_steps_out=n_steps_out_fixed, model_name_base='lstm_tuning_trial'
        )
        results.append(result)
        
        # Print interim result
        if result['status'] == 'success':
            print(f"  -> Result: RMSE={result['rmse']:.2f}, MAE={result['mae']:.2f}")
        else:
            print(f"  -> Result: FAILED - {result['error']}")
    
    # --- 6. Summarize Results ---
    print("\n" + "="*60)
    print("LSTM TUNING RESULTS SUMMARY")
    print("="*60)
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        # Filter successful runs
        successful_results = results_df[results_df['status'] == 'success'].copy()
        if not successful_results.empty:
            # Sort by RMSE (ascending) to find best configuration
            successful_results_sorted = successful_results.sort_values('rmse', ascending=True)
            print(successful_results_sorted[['config', 'rmse', 'mae']].to_string(index=False))
            
            # Save full results
            results_summary_path = f'{RESULTS_DIR}/lstm_tuning_results_summary.csv'
            results_df.to_csv(results_summary_path, index=False)
            print(f"\nFull tuning results saved to {results_summary_path}")
            
            # Identify best configuration
            best_result = successful_results_sorted.iloc[0]
            print(f"\nBest Configuration Found:")
            print(f"  RMSE: {best_result['rmse']:.2f}")
            print(f"  MAE: {best_result['mae']:.2f}")
            print(f"  Config: {best_result['config']}")
            print(f"  Model saved to: {best_result['model_path']}")
        else:
            print("No successful tuning runs completed.")
    else:
        print("No tuning results to summarize.")
        
    end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nFinished at: {end_timestamp}")
    print("="*60)
    print("LSTM TUNING PIPELINE COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()