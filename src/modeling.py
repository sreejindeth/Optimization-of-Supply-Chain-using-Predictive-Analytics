# src/modeling.py
"""
Hybrid LSTM-XGBoost Model for E-Commerce Supply Chain Optimization
Modeling Script
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import warnings
warnings.filterwarnings('ignore')

# --- Scikit-learn imports ---
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer # As per scikit-learn docs for univariate imputation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_squared_error, mean_absolute_error

# --- XGBoost import ---
import xgboost as xgb

# --- TensorFlow/Keras imports for LSTM ---
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
# Ensure TensorFlow logs are not too verbose
tf.get_logger().setLevel('ERROR')

# --- Ensure output directories exist ---
def create_output_dirs():
    """Create necessary directories for models and results."""
    dirs = [
        'models',
        'results/model',
        'results/predictions'
    ]
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
        print(f"Ensured directory exists: {dir}")

# --- 1. LOAD DATA ---
def load_data():
    """
    Load the cleaned supply chain data and daily demand data.
    """
    print("--- Loading Data ---")
    try:
        # Load main cleaned dataset (for XGBoost and potential features for LSTM)
        df_main = pd.read_csv('data/processed/cleaned_supply_chain_data.csv')
        print(f"Loaded main data: {df_main.shape}")

        # Load daily demand data (for LSTM)
        # IMPORTANT: Ensure this file is correctly formatted with 'order_date', 'Sales', etc.
        df_demand_file_path = 'data/processed/daily_demand_data.csv'
        if not os.path.exists(df_demand_file_path):
             # Fallback to the corrected file if the standard one is missing/corrupt
             df_demand_file_path = 'data/processed/daily_demand_data_corrected.csv'
             if not os.path.exists(df_demand_file_path):
                 raise FileNotFoundError(f"Neither 'daily_demand_data.csv' nor 'daily_demand_data_corrected.csv' found in 'data/processed/'. Please ensure the aggregation step produced a valid file.")

        df_demand = pd.read_csv(df_demand_file_path)
        # Ensure 'order_date' is datetime and set as index for time series
        df_demand['order_date'] = pd.to_datetime(df_demand['order_date'])
        df_demand.sort_values('order_date', inplace=True) # Ensure order
        df_demand.set_index('order_date', inplace=True)
        print(f"Loaded daily demand data: {df_demand.shape}")
        print(f"Demand data columns: {list(df_demand.columns)}")
        print(f"Demand data date range: {df_demand.index.min()} to {df_demand.index.max()}")

        return df_main, df_demand

    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error loading data: {e}")
        raise

# --- 2. PREPARE DATA FOR LSTM (Demand Forecasting) ---
def prepare_lstm_data(df_demand, n_steps_in=14, n_steps_out=7, target_cols=['Sales']):
    """
    Prepare sequences for LSTM demand forecasting.
    Assumes df_demand is already sorted by date and has 'order_date' as index.
    """
    print("--- Preparing Data for LSTM ---")
    print(f"Using {n_steps_in} days to predict next {n_steps_out} days for targets: {target_cols}")

    # Select target variables
    if not all(col in df_demand.columns for col in target_cols):
        raise ValueError(f"Target columns {target_cols} not found in df_demand. Available: {list(df_demand.columns)}")
    data = df_demand[target_cols].values

    # Standardize the data for LSTM
    scaler_lstm = StandardScaler()
    scaled_data = scaler_lstm.fit_transform(data)

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

    return X_train_seq, X_test_seq, y_train_seq, y_test_seq, scaler_lstm


# --- 3. BUILD AND TRAIN LSTM MODEL ---
def build_train_lstm(X_train, y_train, X_test, y_test, scaler, n_steps_out, model_name='lstm_demand_forecast'):
    """
    Build, train, and save the LSTM model.
    Also includes basic evaluation.
    """
    print("--- Building and Training LSTM Model ---")
    model = Sequential()
    # Example architecture - you can tune this
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(n_steps_out)) # Output layer for predicting n_steps_out values
    model.compile(optimizer='adam', loss='mse') # Use 'mae' if preferred
    model.summary()

    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    print("Starting LSTM training...")
    history = model.fit(
        X_train, y_train,
        epochs=100, # Adjust based on validation loss and early stopping
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    print("LSTM training finished.")

    # --- Evaluate LSTM ---
    print("--- Evaluating LSTM Model ---")
    y_pred_seq = model.predict(X_test)

    # Inverse transform predictions and actuals for meaningful metrics
    # This is a simplification. For multi-step, multi-feature forecasting,
    # correctly inverting the scaling for the specific target can be complex.
    # Here, we assume we are predicting the first target column (e.g., Sales) scaled data.
    # We need to create arrays compatible with the scaler for inverse transform.
    # A robust way is to inverse transform the entire sequence if all features were predicted,
    # or reconstruct the necessary shape if only one is predicted.
    # For simplicity, let's assume scaler was fit on the target column data only if it's univariate,
    # or we focus on the first feature if multivariate.
    # Let's assume scaler was fit on the target_cols data (e.g., just 'Sales' if that's all we predict)
    # OR, if scaler was fit on multiple cols, we need to be careful.
    # Let's assume scaler_lstm was fit on df_demand[target_cols].values

    # --- Simple Inverse Transform (Assumes scaler was fit on target data shape) ---
    # If target_cols = ['Sales'], scaler is fitted on (N, 1) data.
    # y_test and y_pred_seq are (N, n_steps_out). We need to inverse transform each step.
    # This is a common simplification. For production, more care is needed.
    try:
        # Reshape for inverse transform if needed (depends on scaler fit)
        # If scaler was fit on (N, 1) data (e.g., just 'Sales'):
        y_test_flat = y_test.reshape(-1, 1) # Shape (N*n_steps_out, 1)
        y_pred_flat = y_pred_seq.reshape(-1, 1) # Shape (N*n_steps_out, 1)

        y_test_actual = scaler.inverse_transform(np.concatenate([y_test_flat, np.zeros((y_test_flat.shape[0], scaler.n_features_in_ - 1))], axis=1))[:, 0]
        y_pred_actual = scaler.inverse_transform(np.concatenate([y_pred_flat, np.zeros((y_pred_flat.shape[0], scaler.n_features_in_ - 1))], axis=1))[:, 0]

        # Calculate metrics on the actual scale
        rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
        mae = mean_absolute_error(y_test_actual, y_pred_actual)

        print(f"LSTM Evaluation (on actual scale for primary target):")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")

        # Save metrics
        metrics_df = pd.DataFrame({'Metric': ['RMSE', 'MAE'], 'Value': [rmse, mae]})
        metrics_df.to_csv(f'results/model/{model_name}_metrics.csv', index=False)
        print(f"LSTM metrics saved to results/model/{model_name}_metrics.csv")

    except Exception as e:
        print(f"Warning: Could not calculate metrics on inverse-scaled data due to shape/fit mismatch: {e}")
        # Fallback metrics on scaled data
        mse_scaled = mean_squared_error(y_test, y_pred_seq)
        mae_scaled = mean_absolute_error(y_test, y_pred_seq)
        print(f"LSTM Evaluation (on scaled data):")
        print(f"MSE (scaled): {mse_scaled:.4f}")
        print(f"MAE (scaled): {mae_scaled:.4f}")


    # Save the model
    model_path = f'models/{model_name}.h5'
    model.save(model_path)
    joblib.dump(scaler, f'models/scaler_for_{model_name}.pkl') # Save scaler too
    print(f"LSTM model saved to {model_path}")
    print(f"LSTM scaler saved to models/scaler_for_{model_name}.pkl")

    return model, history


# --- 4. PREPARE DATA FOR XGBOOST (Late Delivery Risk) ---
def prepare_xgboost_data(df_main, test_size=0.2, random_state=42):
    """
    Prepare data for XGBoost late delivery risk prediction.
    """
    print("--- Preparing Data for XGBoost ---")
    # Define features and target
    target_col = 'Late_delivery_risk'
    if target_col not in df_main.columns:
        raise ValueError(f"Target column '{target_col}' not found in df_main.")

    # --- Feature Selection ---
    # Start with numerical features
    feature_cols = df_main.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude target and potentially leaky features
    exclude_features = [target_col, 'Delivery Status', 'Days for shipping (real)'] # Avoid data leakage
    feature_cols = [col for col in feature_cols if col not in exclude_features]
    print(f"Selected {len(feature_cols)} features for XGBoost: {feature_cols}")

    X = df_main[feature_cols]
    y = df_main[target_col]

        # --- Handle Missing Values (using scikit-learn SimpleImputer as before) ---
    # Even though cleaned, good practice to ensure no new NaNs for XGBoost
    # Use the same strategy as preprocessing
    imputer_xgb = SimpleImputer(strategy='median') # For numerical features
    # Note: If you had categorical features encoded for XGBoost, you'd need to handle them too.
    # For now, assuming numerical features are sufficient or already encoded.
    X_imputed = imputer_xgb.fit_transform(X)
    print(f"XGBoost data imputed. Shape: {X_imputed.shape}. Missing values: {np.isnan(X_imputed).sum()}")

    # --- Crucial: Align feature names with imputed data ---
    # SimpleImputer might drop columns that are entirely NaN.
    # The imputed data shape tells us how many features were actually kept.
    num_features_after_imputation = X_imputed.shape[1]
    if num_features_after_imputation != len(feature_cols):
        print(f"Warning: Feature count changed after imputation: {len(feature_cols)} -> {num_features_after_imputation}")
        print(f"Original features: {feature_cols}")
        # Assume SimpleImputer dropped features from the end or based on internal logic.
        # A robust way is complex. A simple heuristic (often correct if empty cols are at the end):
        # However, we don't know which specific columns were dropped.
        # The safest way for feature importance is to get names that correspond to the columns *used*.
        # Let's pass the number of features used, and the caller can handle names if needed.
        # For now, let's try to find the correct names.
        # Check if any feature in feature_cols is entirely NaN in the original X
        # This replicates the imputer's logic slightly to get names.
        # A simpler, often sufficient way: assume the first N features are the ones kept.
        # This works if the dropped column(s) were at the end or if only one was dropped predictably.
        # Given the list, 'Product Description' is a likely candidate for being dropped if it was all NaN numerically.
        # Let's check and adjust feature_cols intelligently.

        # Check for columns in original X that might have been dropped
        # (This requires keeping track of X.columns if it were a DataFrame, but X is numpy here)
        # X was df_main[feature_cols]. Let's check df_main[feature_cols] for all NaN columns
        # before imputation.
        # This is getting complex. Let's use a simpler approach based on the output shape.
        # We will pass the number of features, and the build function will slice feature_names.

        # For immediate fix, pass the correct number of feature names based on output shape.
        # We need to determine which original features correspond to the columns in X_imputed.
        # This is tricky without inverse mapping from imputer.
        # Heuristic: Assume columns are kept in order, and only empty ones are dropped.
        # Find columns in original df_main[feature_cols] that are NOT all NaN.
        # We need to re-calculate this logic slightly.
        # Let's re-calculate the feature_cols based on non-all-NaN in the original data subset.

        # Recalculate feature_cols based on non-empty columns in the original subset X (before imputation numpy array)
        # Actually, X = df_main[feature_cols].values. So we need to check df_main[feature_cols] for all NaN columns.
        # Let's do it correctly here:
        original_feature_data = df_main[feature_cols]
        # Check for columns that are NOT all NaN (these are the ones imputer would keep)
        non_empty_feature_mask = ~original_feature_data.isnull().all(axis=0)
        final_feature_cols = [feature_cols[i] for i in range(len(feature_cols)) if non_empty_feature_mask.iloc[i]]
        print(f"Features kept after imputation check: {final_feature_cols}")
        if len(final_feature_cols) != num_features_after_imputation:
             # This should ideally not happen, but a final safeguard
             print(f"Error: Mismatch between calculated kept features ({len(final_feature_cols)}) and imputed data shape ({num_features_after_imputation}). Using first {num_features_after_imputation} original names.")
             final_feature_cols = feature_cols[:num_features_after_imputation]

    else:
        final_feature_cols = feature_cols

    # --- Train/Test Split ---
    # Option 1: Simple random split (often okay for risk prediction)
    # Option 2: Time-based split (if order date is relevant, e.g., predict risk on newer orders)
    # Let's use time-based split if 'order_date' is available and sorted
    if 'order_date' in df_main.columns:
        df_main_sorted = df_main.sort_values('order_date')
        split_index = int(len(df_main_sorted) * (1 - test_size))
        train_indices = df_main_sorted.index[:split_index]
        test_indices = df_main_sorted.index[split_index:]

        X_train = X_imputed[train_indices]
        X_test = X_imputed[test_indices]
        y_train = y.loc[train_indices]
        y_test = y.loc[test_indices]
        print("Used time-based split for XGBoost data.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print("Used random split for XGBoost data.")

    print(f"XGBoost Train set: X {X_train.shape}, y {y_train.shape}")
    print(f"XGBoost Test set: X {X_test.shape}, y {y_test.shape}")
    print(f"XGBoost Test set class distribution:\n{pd.Series(y_test).value_counts()}")

    # Return the final_feature_cols which align with X_train/X_test shape
    return X_train, X_test, y_train, y_test, imputer_xgb, final_feature_cols

# --- 5. BUILD AND TRAIN XGBOOST MODEL ---
# (Inside your src/modeling.py file)
def build_train_xgboost(X_train, y_train, X_test, y_test, imputer, feature_names, model_name='xgboost_late_delivery_risk'):
    """
    Build, train, evaluate, and save the XGBoost model.
    Simplified for compatibility with various XGBoost versions.
    Updated to handle potential feature name mismatches.
    """
    print("--- Building and Training XGBoost Model ---")

    # --- Crucial Check: Align feature_names with X_train shape ---
    # The model.feature_importances_ will have length equal to X_train.shape[1]
    # feature_names must match this length.
    num_features_used = X_train.shape[1]
    if len(feature_names) != num_features_used:
        print(f"Warning: Mismatch in feature names length ({len(feature_names)}) and features used ({num_features_used}).")
        print("Attempting to align feature names...")
        if len(feature_names) > num_features_used:
            # Likely features were dropped, slice the names
            aligned_feature_names = feature_names[:num_features_used]
            print(f"Using first {num_features_used} feature names.")
        else:
            # This case is less likely but handle it
            print(f"Error: Not enough feature names provided ({len(feature_names)}) for features used ({num_features_used}). Padding with generic names.")
            aligned_feature_names = feature_names + [f"Unknown_Feature_{i}" for i in range(len(feature_names), num_features_used)]
    else:
        aligned_feature_names = feature_names

    # Define model (rest of the model definition remains the same)
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100, # Example, tune this
        max_depth=6,      # Example, tune this
        learning_rate=0.1, # Example, tune this
        random_state=42
    )

    # --- Prepare parameters for .fit() (rest remains the same) ---
    fit_params = {
        'X': X_train,
        'y': y_train,
        'verbose': False # Removed eval_set due to previous compatibility issues
    }

    # Train the model (rest remains the same, without early stopping for now)
    print("Starting XGBoost training...")
    model.fit(**fit_params) # Simplified fit call
    print("XGBoost training finished.")

    # --- Evaluate XGBoost (rest remains the same) ---
    print("--- Evaluating XGBoost Model ---")
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of class 1 (Late)
    y_pred_class = model.predict(X_test) # Predicted class

    # Calculate metrics (rest remains the same)
    accuracy = accuracy_score(y_test, y_pred_class)
    precision = precision_score(y_test, y_pred_class, zero_division=0)
    recall = recall_score(y_test, y_pred_class, zero_division=0)
    f1 = f1_score(y_test, y_pred_class, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        auc = np.nan # In case of only one class in y_test
        print("Warning: Could not calculate AUC (only one class in test set?)")

    print("XGBoost Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}") # Important for late delivery prediction
    print(f"Recall: {recall:.4f}")     # Important for late delivery prediction
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")

    # Save metrics (rest remains the same)
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
        'Value': [accuracy, precision, recall, f1, auc]
    })
    metrics_df.to_csv(f'results/model/{model_name}_metrics.csv', index=False)
    print(f"XGBoost metrics saved to results/model/{model_name}_metrics.csv")

    # --- Feature Importance (Updated to use aligned_feature_names) ---
    importances = model.feature_importances_
    # Create a DataFrame for importance using the aligned names
    # Ensure lengths match (they should now)
    if len(aligned_feature_names) != len(importances):
         print(f"Error: Length mismatch after alignment. Names: {len(aligned_feature_names)}, Importances: {len(importances)}. Using generic names.")
         feature_importance_names = [f"Feature_{i}" for i in range(len(importances))]
    else:
         feature_importance_names = aligned_feature_names

    importance_df = pd.DataFrame({'feature': feature_importance_names, 'importance': importances})
    importance_df = importance_df.sort_values('importance', ascending=False)
    importance_df.to_csv(f'results/model/{model_name}_feature_importance.csv', index=False)
    print(f"XGBoost feature importances saved to results/model/{model_name}_feature_importance.csv")

    # Save the model and imputer (rest remains the same)
    model.save_model(f'models/{model_name}.json') # XGBoost native format
    joblib.dump(imputer, f'models/imputer_for_{model_name}.pkl')
    print(f"XGBoost model saved to models/{model_name}.json")
    print(f"XGBoost imputer saved to models/imputer_for_{model_name}.pkl")

    return model


# --- MAIN EXECUTION ---
def main():
    """Main function to run the modeling pipeline."""
    print("="*50)
    print("HYBRID LSTM-XGBOOST MODELING PIPELINE")
    print("="*50)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Started at: {timestamp}")

    create_output_dirs()

    # --- Load Data ---
    df_main, df_demand = load_data()

    # --- MODEL 1: LSTM for Demand Forecasting ---
    print("\n" + "="*30)
    print("MODEL 1: LSTM - Demand Forecasting")
    print("="*30)
    # Prepare data for LSTM
    # You can experiment with different target columns and sequence lengths
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, scaler_lstm = prepare_lstm_data(
        df_demand, n_steps_in=14, n_steps_out=7, target_cols=['Sales'] # Start with 'Sales' forecasting
    )
    # Build and train LSTM
    lstm_model, lstm_history = build_train_lstm(
        X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, scaler_lstm, n_steps_out=7, model_name='lstm_demand_forecast_sales'
    )

    # --- MODEL 2: XGBoost for Late Delivery Risk ---
    print("\n" + "="*30)
    print("MODEL 2: XGBoost - Late Delivery Risk Prediction")
    print("="*30)
    # Prepare data for XGBoost
    X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb, imputer_xgb, feature_cols_xgb = prepare_xgboost_data(df_main)
    # Build and train XGBoost
    xgb_model = build_train_xgboost(
        X_train_xgb, y_train_xgb, X_test_xgb, y_test_xgb, imputer_xgb, feature_cols_xgb, model_name='xgboost_late_delivery_risk'
    )

    # --- Finish ---
    end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nFinished at: {end_timestamp}")
    print("="*50)
    print("MODELING PIPELINE COMPLETED")
    print("="*50)
    print("\nNext steps:")
    print("- Analyze model metrics in 'results/model/'")
    print("- Review XGBoost feature importances")
    print("- (Optional) Perform hyperparameter tuning")
    print("- (Optional) Integrate model predictions for supply chain actions")


if __name__ == "__main__":
    main()