# src/model_utils.py
"""
Utility functions for loading and using the trained LSTM and XGBoost models.
Handles model loading complexities and provides easy-to-use prediction functions.
"""
import os
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
# --- For robust LSTM loading (copy the fixed load_lstm_model_safe function from visualize_lstm_predictions.py) ---
# You might need to copy the load_lstm_model_safe function here or import it.
# For simplicity, let's assume the standard load works or you use the fixed version.
# If you still have issues, copy the load_lstm_model_safe function here.

# --- 1. Load XGBoost Model ---
def load_xgboost_model(model_path='models/xgboost_late_delivery_risk.json',
                       imputer_path='models/imputer_for_xgboost_late_delivery_risk.pkl'):
    """
    Load the trained XGBoost model and its associated imputer.
    """
    try:
        import xgboost as xgb
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"XGBoost model file not found at {model_path}")
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        print(f"XGBoost model loaded from {model_path}")

        imputer = None
        if os.path.exists(imputer_path):
            imputer = joblib.load(imputer_path)
            print(f"XGBoost imputer loaded from {imputer_path}")
        else:
            print(f"Warning: XGBoost imputer not found at {imputer_path}. Ensure no imputation is needed or handle separately.")

        return model, imputer
    except Exception as e:
        print(f"Error loading XGBoost model or imputer: {e}")
        raise

# --- 2. Load LSTM Model (Use the robust version) ---
# Define or copy the load_lstm_model_safe function here.
# Placeholder for now, assuming it's defined or imported.
# You should copy the working `load_lstm_model_safe` function from your visualize script.
# Add this import at the top of src/model_utils.py if not already there

def load_lstm_model_safe(model_path):
    """
    Attempt to load the LSTM model, handling potential deserialization issues.
    Tries standard load first, then falls back to reconstructing the model
    and loading weights only.
    """
    print(f"Attempting to load model from {model_path}...")
    try:
        # Try 1: Standard load (most common and preferred)
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        print("Model loaded successfully using standard load_model.")
        return model
    except ValueError as e:
        error_message = str(e)
        if "Could not deserialize" in error_message and ("mse" in error_message or "mae" in error_message):
            print(f"Deserialization error encountered: {e}")
            print("Attempting alternative loading method (reconstructing model and loading weights)...")

            # --- Fallback: Reconstruct model architecture and load weights ---
            try:
                # --- Attempt 1: Load config and weights separately (if possible) ---
                # This sometimes works if the main issue is in the compiled metadata
                import h5py
                try:
                    with h5py.File(model_path, 'r') as f:
                        # Check if model config is readable
                        if 'model_config' in f.attrs:
                            print("Model config found in file attributes.")
                        # The weights are in the 'model_weights' group
                        # We still need to reconstruct the architecture manually.
                except Exception as h5e:
                    print(f"Could not inspect model file structure with h5py: {h5e}")

                # --- Attempt 2: Manual Reconstruction based on common patterns ---
                # Based on the error message `Weight expects shape (3, 200). Received saved weight with shape (1, 200)`,
                # it's highly likely the model was trained with input_shape=(timesteps, 1 feature).
                # Let's try reconstructing a common architecture with this assumption.
                # You might need to adjust these parameters if you know the exact training setup.

                # Common parameters used in modeling.py (adjust if needed based on memory)
                # These are GUESSES based on common usage. If they don't work, you'll need to find the correct ones.
                n_steps_in_guess = 14  # Common: Look back 14 days
                n_features_guess = 1   # Based on the (1, 200) weight shape error
                n_steps_out_guess = 7  # Common: Predict next 7 days
                lstm_units_guess = 50  # Common: 50 units in LSTM layers
                # Assume the architecture from modeling.py
                print(f"Trying to reconstruct model with guessed parameters: "
                      f"input_shape=({n_steps_in_guess}, {n_features_guess}), "
                      f"n_steps_out={n_steps_out_guess}, lstm_units={lstm_units_guess}")

                # Reconstruct the model architecture
                model = Sequential()
                # Input shape must match the weights
                model.add(LSTM(lstm_units_guess, activation='relu',
                               input_shape=(n_steps_in_guess, n_features_guess),
                               return_sequences=True))
                model.add(Dropout(0.2))
                model.add(LSTM(lstm_units_guess, activation='relu')) # return_sequences=False
                model.add(Dropout(0.2))
                model.add(Dense(n_steps_out_guess)) # Output layer for forecasting

                # Crucially, compile the model *before* loading weights
                # Use standard loss/metrics to avoid the deserialization issue
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                print("Model architecture reconstructed and compiled.")

                # Now, load only the weights (this bypasses the problematic config/metrics)
                model.load_weights(model_path)
                print("Model weights loaded successfully into reconstructed model.")
                return model

            except Exception as e2:
                print(f"Failed to reconstruct model and load weights: {e2}")
                # If manual reconstruction fails, there's not much else we can do
                # automatically without knowing the exact training architecture.

        print(f"Standard loading failed: {e}")
        raise e # Re-raise the original error if all fallbacks fail
    except Exception as e:
        print(f"Unexpected error loading model: {e}")
        raise e


# --- Add this function to src/model_utils.py if it's missing ---
def load_lstm_components(model_path='models/lstm_tuning_trial_9310.h5',
                         scaler_path='models/scaler_for_lstm_demand_forecast_sales.pkl'):
    """
    Load the trained LSTM model and its associated scaler.
    """
    try:
        # --- Use the robust loader we defined ---
        model = load_lstm_model_safe(model_path) # Use the robust loader
        print(f"LSTM model loaded from {model_path}")

        scaler = None
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print(f"LSTM scaler loaded from {scaler_path}")
        else:
            print(f"Warning: LSTM scaler not found at {scaler_path}. Cannot inverse transform predictions.")

        return model, scaler
    except Exception as e:
        print(f"Error loading LSTM model or scaler: {e}")
        raise


# Inside src/model_utils.py, in the predict_risk_xgboost function
# Find the section where imputation is applied and replace it with this:

def predict_risk_xgboost(model, imputer, order_features_df, feature_names_used_by_model):
    """
    Predict late delivery risk for a DataFrame of new orders.

    Args:
        model: Loaded XGBoost model.
        imputer: Fitted SimpleImputer (if needed).
        order_features_df (pd.DataFrame): DataFrame containing features for new orders.
                                    Must contain the columns specified in feature_names_used_by_model.
        feature_names_used_by_model (list): List of feature names the model was trained on.

    Returns:
        np.array: Array of risk probabilities (0 to 1) for each order.
    """
    try:
        # --- Ensure correct features are present ---
        missing_features = set(feature_names_used_by_model) - set(order_features_df.columns)
        if missing_features:
            raise ValueError(f"Missing features in input data for XGBoost: {missing_features}")

        # Select and order features to match the model's expectation
        X_new = order_features_df[feature_names_used_by_model]

        # --- Apply imputation if necessary ---
        if imputer is not None:
            print("Applying imputation to new order data for XGBoost...")

            # --- Crucial Fix for Feature Name Mismatch ---
            # The imputer was fitted on a DataFrame with specific column names.
            # It expects the DataFrame passed to transform to have the exact same column names.
            # The error message told us 'Product Description' was missing.
            # We need to ensure the DataFrame passed to the imputer has all the original column names,
            # including any that were empty and dropped during fitting.

            # Create a DataFrame for imputation with the correct columns
            # Start with the selected features
            X_new_df_for_imputation = X_new.copy()

            # Add any missing columns that the imputer expects (like 'Product Description')
            # with NaN values. The imputer should handle these correctly.
            # We can inspect the imputer to see what feature names it expects.
            # The imputer has an attribute that stores the feature names it was fit on.
            if hasattr(imputer, 'feature_names_in_'):
                expected_feature_names = imputer.feature_names_in_
                print(f"XGBoost Imputer was fitted on features: {list(expected_feature_names)}")

                # Add missing columns with NaN
                for expected_col in expected_feature_names:
                    if expected_col not in X_new_df_for_imputation.columns:
                        print(f"  -> Adding missing column '{expected_col}' with NaN values for imputation.")
                        X_new_df_for_imputation[expected_col] = np.nan # Fill with NaN

            # Ensure columns are in the order the imputer saw during fit (if it matters)
            # This might not be strictly necessary, but it's good practice.
            # The key is having the right columns with the right names.
            if hasattr(imputer, 'feature_names_in_'):
                 X_new_df_for_imputation = X_new_df_for_imputation[imputer.feature_names_in_]

            # Now pass this DataFrame to the imputer
            # This should resolve the "Feature names should match" error.
            X_new_imputed = imputer.transform(X_new_df_for_imputation)
            print(f"XGBoost input data shape after imputation: {X_new_imputed.shape}")

            # Use the imputed data (numpy array) for prediction
            X_new = X_new_imputed

        # --- Make prediction ---
        print("Predicting late delivery risk using XGBoost...")
        risk_probabilities = model.predict_proba(X_new)[:, 1] # Probability of class 1 (Late)
        print("XGBoost risk prediction completed.")
        return risk_probabilities

    except Exception as e:
        print(f"Error during XGBoost prediction: {e}")
        raise


# --- LSTM Prediction ---
# Inside src/model_utils.py
def prepare_lstm_sequences_for_forecast(scaler, latest_data_df, n_steps_in=14, feature_cols=['Sales']):
    """
    Prepare the latest data sequence for LSTM forecasting.
    This function takes the most recent 'n_steps_in' days of data and prepares it for prediction.

    Args:
        scaler: The fitted StandardScaler (determines expected number of features).
        latest_data_df (pd.DataFrame): DataFrame containing recent daily data.
        n_steps_in (int): Number of past days the model uses as input.
        feature_cols (list): List of feature column names intended for use.
                             Note: The scaler dictates the actual number used.
    """
    try:
        # --- Crucial: Align features with scaler's expectation ---
        # The scaler was fit on a specific number of features (likely 1, e.g., just 'Sales').
        # We must pass data with that many features to the scaler's transform method.
        expected_n_features = scaler.n_features_in_
        print(f"LSTM Scaler expects {expected_n_features} features.")

        # Assume the first 'expected_n_features' from feature_cols are the ones the scaler knows.
        # This is a simplification. Ideally, we'd know the exact column names the scaler was fit on.
        # For now, let's assume it's just 'Sales' if expected_n_features is 1.
        if expected_n_features == 1:
            scaler_feature_cols = ['Sales'] # Hardcoded assumption based on error
        else:
            # If it's 3, use the original list or a subset
            scaler_feature_cols = feature_cols[:expected_n_features]

        print(f"Using features for LSTM scaler: {scaler_feature_cols}")

        if not all(col in latest_data_df.columns for col in scaler_feature_cols):
             raise ValueError(f"Required scaler feature columns {scaler_feature_cols} not found in data. Available: {list(latest_data_df.columns)}")

        # Get the last n_steps_in rows for the features the scaler expects
        data_for_sequence = latest_data_df[scaler_feature_cols].tail(n_steps_in)

        if len(data_for_sequence) < n_steps_in:
            raise ValueError(f"Not enough data points to create a sequence of length {n_steps_in}. Available: {len(data_for_sequence)}")

        # Scale the data using the scaler (which expects data with its fitted feature count)
        scaled_data = scaler.transform(data_for_sequence)

        # Reshape for LSTM input (1 sample, n_steps_in, n_features_for_scaler)
        # The LSTM model itself might expect a different number of input features.
        # The reconstructed model expects (n_steps_in, 1) based on previous loading.
        # The X_seq shape should be (1, n_steps_in, expected_n_features)
        X_seq = scaled_data.reshape(1, n_steps_in, expected_n_features)

        print(f"LSTM input sequence prepared. Shape: {X_seq.shape}")
        return X_seq

    except Exception as e:
        print(f"Error preparing LSTM sequence: {e}")
        raise

def predict_demand_lstm(model, scaler, latest_data_df, n_steps_out=7, n_steps_in=14, feature_cols=['Sales']):
    """
    Predict future demand using the LSTM model.

    Args:
        model: Loaded LSTM model.
        scaler: Fitted StandardScaler.
        latest_data_df (pd.DataFrame): DataFrame with recent daily data (must include feature_cols and be indexed by date).
        n_steps_out (int): Number of future days to predict.
        n_steps_in (int): Number of past days the model uses as input.
        feature_cols (list): List of feature column names used by the model.

    Returns:
        dict: Dictionary with forecasted values for the primary target (e.g., 'Sales').
              Keys could be 'predicted_sales', 'predicted_dates', etc.
    """
    try:
        # --- Prepare input sequence ---
        X_seq = prepare_lstm_sequences_for_forecast(scaler, latest_data_df, n_steps_in, feature_cols)

        # --- Make prediction ---
        print("Predicting future demand using LSTM...")
        y_pred_scaled = model.predict(X_seq, verbose=0)
        print("LSTM demand prediction completed.")

        # --- Inverse Transform Predictions ---
        print("Inverting LSTM predictions...")
        # The model predicts n_steps_out values for the first feature (e.g., Sales).
        # y_pred_scaled shape: (1, n_steps_out)
        num_predictions = y_pred_scaled.shape[1] # n_steps_out
        predicted_first_feature_flat = y_pred_scaled.flatten() # Shape (n_steps_out,)

        # The scaler was fit on data with scaler.n_features_in_ features (e.g., 1 or 3).
        # To inverse transform, we need an array of shape (num_predictions, n_features_original).
        # We only have predictions for one feature (the first one the model was trained to predict).
        # We need to reconstruct an array compatible with the scaler.
        n_features_original = scaler.n_features_in_ # Get the number of features the scaler expects
        print(f"LSTM scaler expects {n_features_original} features for inverse transform.")

        # Create a dummy array for inverse transform.
        # Fill the first column with our predictions, rest can be zeros or NaNs.
        # The scaler should only use the first column if that's what it was fit on,
        # or it should handle the full array if it was fit on multiple features.
        dummy_data_for_inverse = np.zeros((num_predictions, n_features_original))
        dummy_data_for_inverse[:, 0] = predicted_first_feature_flat # Put predictions in the first column

        # Inverse transform to get predictions back to original scale
        y_pred_original_scale_full = scaler.inverse_transform(dummy_data_for_inverse)
        # Take the first column (corresponding to 'Sales')
        y_pred_original_scale = y_pred_original_scale_full[:, 0]

        print("LSTM predictions inverse transformed.")

        # --- Generate future dates (conceptual) ---
        # This requires knowing the last date in latest_data_df
        last_date = latest_data_df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_steps_out, freq='D')

        forecast_result = {
            'predicted_sales': y_pred_original_scale,
            'predicted_dates': future_dates,
            # Add other predicted features if your model predicts them
            'predicted_quantity': None,
            'predicted_orders': None
        }
        print("LSTM predictions inverse transformed and dates generated.")
        return forecast_result

    except Exception as e:
        print(f"Error during LSTM prediction: {e}")
        raise
