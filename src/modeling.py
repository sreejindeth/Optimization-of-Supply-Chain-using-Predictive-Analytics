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
    # Input shape: (n_timesteps, n_features). Assuming X_train is (samples, timesteps, features)
    model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu')) # return_sequences=False by default for the last LSTM layer
    model.add(Dropout(0.2))
    model.add(Dense(n_steps_out)) # Output layer for predicting n_steps_out values for the FIRST target feature
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
    y_pred_seq = model.predict(X_test, verbose=0) # Shape: (num_test_samples, n_steps_out)

    # --- Inverse Transform Predictions ---
    print("Inverting LSTM predictions...")
    # --- Corrected Logic for Inverse Transform ---
    # 1. Get the number of features the scaler was fitted on.
    #    Based on the error message, it's 1 (likely just 'Sales').
    n_features_original = scaler.n_features_in_
    print(f"LSTM scaler expects {n_features_original} features for inverse transform.")

    # 2. Flatten the predictions. y_pred_seq is (N, n_steps_out).
    #    We want to inverse transform each predicted value.
    num_predictions = y_pred_seq.shape[0] * y_pred_seq.shape[1] # Total number of predicted values
    predicted_first_feature_flat = y_pred_seq.flatten() # Shape (num_predictions,)
    print(f"Flattened LSTM predictions. Shape: {predicted_first_feature_flat.shape}")

    # 3. Create a dummy array for the scaler.
    #    The scaler needs an array of shape (num_predictions, n_features_original).
    #    We only have predictions for the FIRST feature (e.g., Sales).
    #    We need to create an array compatible with the scaler.
    #    Fill the first column with our predictions, rest can be zeros or NaNs.
    #    The scaler should only use the first column if that's what it was fit on,
    #    or it should handle the full array if it was fit on multiple features.
    dummy_data_for_inverse = np.zeros((num_predictions, n_features_original))
    dummy_data_for_inverse[:, 0] = predicted_first_feature_flat # Put predictions in the first column
    print(f"Dummy data array for inverse transform created. Shape: {dummy_data_for_inverse.shape}")

    # 4. Inverse transform to get predictions back to original scale
    try:
        y_pred_original_scale_full = scaler.inverse_transform(dummy_data_for_inverse)
        # 5. Take the first column (corresponding to 'Sales')
        y_pred_original_scale = y_pred_original_scale_full[:, 0]
        print("LSTM predictions inverse transformed successfully.")

        # --- Simple Evaluation on Flattened Arrays (for demonstration) ---
        # Note: This is a simplification for multi-step forecasting.
        # A more robust evaluation would compare corresponding forecast horizons.
        # Flatten y_test to match the flattened predictions for basic metrics.
        y_test_flat = y_test.flatten()
        if len(y_test_flat) == len(y_pred_original_scale):
            rmse = np.sqrt(mean_squared_error(y_test_flat, y_pred_original_scale))
            mae = mean_absolute_error(y_test_flat, y_pred_original_scale)
            print(f"LSTM Evaluation (on actual scale for primary target, flattened):")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAE: {mae:.2f}")

            # Save metrics
            metrics_df = pd.DataFrame({'Metric': ['RMSE', 'MAE'], 'Value': [rmse, mae]})
            metrics_df.to_csv(f'results/model/{model_name}_metrics.csv', index=False)
            print(f"LSTM metrics saved to results/model/{model_name}_metrics.csv")
        else:
             print(f"Warning: Shape mismatch for evaluation. y_test_flat: {y_test_flat.shape}, y_pred_original_scale: {y_pred_original_scale.shape}")

    except Exception as e:
        print(f"Warning: Could not calculate metrics on inverse-scaled data due to shape/fit mismatch: {e}")
        # Fallback metrics on scaled data
        mse_scaled = mean_squared_error(y_test, y_pred_seq)
        mae_scaled = mean_absolute_error(y_test, y_pred_seq)
        print(f"LSTM Evaluation (on scaled data):")
        print(f"MSE (scaled): {mse_scaled:.4f}")
        print(f"MAE (scaled): {mae_scaled:.4f}")

        # Save fallback metrics
        metrics_df = pd.DataFrame({'Metric': ['MSE_Scaled', 'MAE_Scaled'], 'Value': [mse_scaled, mae_scaled]})
        metrics_df.to_csv(f'results/model/{model_name}_metrics.csv', index=False)
        print(f"LSTM fallback metrics saved to results/model/{model_name}_metrics.csv")


    # Save the model
    model_path = f'models/{model_name}.h5'
    model.save(model_path)
    joblib.dump(scaler, f'models/scaler_for_{model_name}.pkl') # Save scaler too
    print(f"LSTM model saved to {model_path}")
    print(f"LSTM scaler saved to models/scaler_for_{model_name}.pkl")

    return model, history



# Inside src/modeling.py, replace the prepare_xgboost_data_fixed function

# --- 4. PREPARE DATA FOR XGBOOST (Late Delivery Risk) WITH FIXES ---
def prepare_xgboost_data_fixed(df_main, test_size=0.2, random_state=42):
    """
    Prepare data for XGBoost late delivery risk prediction with correct categorical handling.
    """
    print("--- Preparing Data for XGBoost (FIXED) ---")
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
    print(f"Selected {len(feature_cols)} initial numerical features for XGBoost: {feature_cols[:10]}...") # Show first 10

    X = df_main[feature_cols]
    y = df_main[target_col]

    # --- Identify Categorical Features based on the provided CSV ---
    # Load the feature importance CSV to get all feature names expected by the model
    importance_file_path = 'results/model/xgboost_late_delivery_risk_feature_importance.csv'
    if not os.path.exists(importance_file_path):
        raise FileNotFoundError(f"Feature importance file not found at {importance_file_path}. Cannot determine model features.")

    importance_df = pd.read_csv(importance_file_path)
    all_model_features = importance_df['feature'].tolist()
    print(f"All features expected by the model: {len(all_model_features)}")

    # Identify categorical features that should have been in the original data
    # Based on the CSV content and dataset description, these are likely categorical:
    categorical_feature_candidates = [
        'Type', 'Delivery Status', 'Category Name', 'Customer Segment',
        'Department Name', 'Market', 'Order City', 'Order Country',
        'Order Region', 'Order State', 'Product Description', 'Product Name',
        'Product Status', 'Shipping Mode'
    ]
    # Filter to only those that are actually in the model features and the main dataframe
    categorical_features_for_xgb_model = [col for col in categorical_feature_candidates if col in all_model_features and col in df_main.columns]
    print(f"Identified categorical features for XGBoost model: {categorical_features_for_xgb_model}")

    # Add categorical features to X if they exist in df_main and are expected by the model
    for cat_col in categorical_features_for_xgb_model:
        if cat_col in df_main.columns:
            X[cat_col] = df_main[cat_col]
            if cat_col in feature_cols:
                feature_cols.remove(cat_col) # Remove from numerical list if it was there
            feature_cols.append(cat_col) # Add to the end of feature_cols
    print(f"Final feature list for XGBoost: {len(feature_cols)} features")

    # Re-select X with the updated feature list to ensure correct order and inclusion
    X = df_main[feature_cols]

    # --- Handle Categorical Features (Label Encoding BEFORE Imputation) ---
    label_encoders = {}
    X_encoded = X.copy() # Work on a copy to avoid modifying original data

    for col in categorical_features_for_xgb_model:
        if col in X_encoded.columns:
            print(f"  -> Processing categorical feature: '{col}'")
            # Handle potential NaNs in categorical columns
            # Fill with a placeholder to represent missing categories
            X_encoded[col].fillna('Missing_Category_XGB', inplace=True)
            le = LabelEncoder()
            # Fit and transform, storing the encoder for later use in prediction
            try:
                X_encoded[col] = le.fit_transform(X_encoded[col])
                label_encoders[col] = le # Save the encoder
                print(f"    -> Applied Label Encoding to '{col}'. Classes: {le.classes_}")
            except ValueError as e:
                print(f"    -> Error encoding '{col}': {e}. Dropping column.")
                X_encoded.drop(columns=[col], inplace=True)
                if col in feature_cols:
                    feature_cols.remove(col)

    # Update feature_cols after potential column drops
    final_feature_cols_before_imputation = X_encoded.columns.tolist()
    print(f"Features after categorical encoding: {len(final_feature_cols_before_imputation)}")

    # --- Handle Missing Values (using scikit-learn SimpleImputer as before) ---
    # Even though cleaned, good practice to ensure no new NaNs for XGBoost
    # Use the same strategy as preprocessing
    imputer_xgb = SimpleImputer(strategy='median') # For numerical features (and now encoded categoricals)
    X_imputed = imputer_xgb.fit_transform(X_encoded)
    print(f"XGBoost data imputed. Shape: {X_imputed.shape}. Missing values: {np.isnan(X_imputed).sum()}")

    # --- Crucial Fix: Align final features with imputed data shape ---
    # The imputer might have dropped columns that were entirely empty (all NaN) or constant.
    # The shape of X_imputed tells us how many features were actually kept.
    num_features_after_imputation = X_imputed.shape[1]

    # The imputer has an attribute that tells us the feature names it was fit on (if available in newer versions)
    # and more importantly, how it handled dropping columns.
    # However, a robust way is to check which columns from the original X_encoded led to the final shape.
    # SimpleImputer usually keeps columns unless they are entirely NaN. Let's assume it dropped one or more.
    # We need to determine the correct final_feature_cols that correspond to the columns in X_imputed.

    # Check if the imputer provides information about dropped features (available in newer sklearn versions)
    # This is the most direct way if the imputer supports it.
    if hasattr(imputer_xgb, 'n_features_in_') and hasattr(imputer_xgb, 'feature_names_in_'):
        # If the imputer was fit on a DataFrame, it might remember input features.
        # However, SimpleImputer usually doesn't drop columns just based on being constant.
        # Let's check if the number of input features matches the output shape.
        # n_features_in_ is the number of features the imputer was fitted on.
        if imputer_xgb.n_features_in_ == num_features_after_imputation:
            # No features were dropped, use the original feature list (filtered)
            final_feature_cols = final_feature_cols_before_imputation[:num_features_after_imputation]
        else:
            # Features were likely dropped. This is trickier without explicit mapping.
            # A common scenario is dropping the last column(s) if they were entirely empty.
            # A safer, more general approach:
            # Assume the first N=num_features_after_imputation columns from the original list are the ones kept.
            # This is often correct if SimpleImputer drops columns from the end due to being all NaN.
            final_feature_cols = final_feature_cols_before_imputation[:num_features_after_imputation]
            print(f"Warning: Imputer dropped features. Assuming first {num_features_after_imputation} features were kept.")
    else:
        # Older sklearn version or imputer doesn't provide direct feature info easily.
        # Use the robust heuristic: assume imputer kept the first N features from the input DataFrame.
        final_feature_cols = final_feature_cols_before_imputation[:num_features_after_imputation]
        print(f"Heuristic: Assuming first {num_features_after_imputation} features were kept by imputer.")

    print(f"Final feature list after imputation check: {len(final_feature_cols)}")

    # Convert back to DataFrame for clarity (optional, but good practice)
    # Crucially, use the CORRECT final_feature_cols that match X_imputed.shape[1]
    try:
        X_imputed_df = pd.DataFrame(X_imputed, columns=final_feature_cols, index=X_encoded.index)
        print("Successfully created DataFrame for imputed data.")
    except ValueError as e:
        # This catch should ideally not trigger if final_feature_cols length matches X_imputed.shape[1]
        # But if it does, it's a critical shape mismatch.
        print(f"Critical Error creating DataFrame: {e}")
        print(f"  X_imputed shape: {X_imputed.shape}")
        print(f"  Length of final_feature_cols: {len(final_feature_cols)}")
        print(f"  final_feature_cols: {final_feature_cols}")
        raise # Re-raise to stop the process as this indicates a fundamental problem


    # --- Train/Test Split ---
    # Option 1: Simple random split (often okay for risk prediction)
    # Option 2: Time-based split (if order date is relevant, e.g., predict risk on newer orders)
    # Let's use time-based split if 'order_date' is available and sorted
    if 'order_date' in df_main.columns:
        df_main_sorted = df_main.sort_values('order_date')
        split_index = int(len(df_main_sorted) * (1 - test_size))
        train_indices = df_main_sorted.index[:split_index]
        test_indices = df_main_sorted.index[split_index:]

        X_train = X_imputed_df.loc[train_indices]
        X_test = X_imputed_df.loc[test_indices]
        y_train = y.loc[train_indices]
        y_test = y.loc[test_indices]
        print("Used time-based split for XGBoost data.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed_df, y, test_size=test_size, random_state=random_state, stratify=y
        )
        print("Used random split for XGBoost data.")

    print(f"XGBoost Train set: X {X_train.shape}, y {y_train.shape}")
    print(f"XGBoost Test set: X {X_test.shape}, y {y_test.shape}")
    print(f"XGBoost Test set class distribution:\n{pd.Series(y_test).value_counts()}")

    # Return the final_feature_cols which align with X_train/X_test shape
    # Also return the imputer and label_encoders for saving
    return X_train, X_test, y_train, y_test, imputer_xgb, label_encoders, final_feature_cols # Return encoders and final feature list

# --- 5. BUILD AND TRAIN XGBOOST MODEL (FIXED) ---
def build_train_xgboost_fixed(X_train, y_train, X_test, y_test, imputer, label_encoders, feature_names, model_name='xgboost_late_delivery_risk_fixed'):
    """
    Build, train, evaluate, and save the FIXED XGBoost model.
    """
    print("--- Building and Training FIXED XGBoost Model ---")

    # --- Align feature names with X_train shape (sanity check) ---
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

    # --- Define and Train Model ---
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100, # Example, tune this
        max_depth=6,      # Example, tune this
        learning_rate=0.1, # Example, tune this
        random_state=42
    )

    # --- Prepare parameters for .fit() ---
    fit_params = {
        'X': X_train,
        'y': y_train,
        'verbose': False # Removed eval_set due to previous compatibility issues
    }

    # Train the model
    print("Starting FIXED XGBoost training...")
    model.fit(**fit_params) # Simplified fit call
    print("FIXED XGBoost training finished.")

    # --- Evaluate XGBoost ---
    print("--- Evaluating FIXED XGBoost Model ---")
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of class 1 (Late)
    y_pred_class = model.predict(X_test) # Predicted class

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_class)
    precision = precision_score(y_test, y_pred_class, zero_division=0)
    recall = recall_score(y_test, y_pred_class, zero_division=0)
    f1 = f1_score(y_test, y_pred_class, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except ValueError:
        auc = np.nan # In case of only one class in y_test
        print("Warning: Could not calculate AUC (only one class in test set?)")

    print("FIXED XGBoost Evaluation Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}") # Important for late delivery prediction
    print(f"Recall: {recall:.4f}")     # Important for late delivery prediction
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")


    # Save metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
        'Value': [accuracy, precision, recall, f1, auc]
    })
    metrics_df.to_csv(f'results/model/{model_name}_metrics.csv', index=False)
    print(f"FIXED XGBoost metrics saved to results/model/{model_name}_metrics.csv")

    # --- Feature Importance ---
    importances = model.feature_importances_
    # Create a DataFrame for importance
    # Ensure lengths match (they should now)
    if len(aligned_feature_names) != len(importances):
         print(f"Error: Length mismatch after alignment. Names: {len(aligned_feature_names)}, Importances: {len(importances)}. Using generic names.")
         feature_importance_names = [f"Feature_{i}" for i in range(len(importances))]
    else:
         feature_importance_names = aligned_feature_names

    importance_df = pd.DataFrame({'feature': feature_importance_names, 'importance': importances})
    importance_df = importance_df.sort_values('importance', ascending=False)
    importance_df.to_csv(f'results/model/{model_name}_feature_importance.csv', index=False)
    print(f"FIXED XGBoost feature importances saved to results/model/{model_name}_feature_importance.csv")

    # --- Save Model Components ---
    model.save_model(f'models/{model_name}.json') # XGBoost native format
    joblib.dump(imputer, f'models/imputer_for_{model_name}.pkl')
    # --- Crucially, save the fitted LabelEncoders ---
    joblib.dump(label_encoders, f'models/label_encoders_for_{model_name}.pkl') # Save encoders!
    print(f"FIXED XGBoost model saved to models/{model_name}.json")
    print(f"FIXED XGBoost imputer saved to models/imputer_for_{model_name}.pkl")
    print(f"FIXED LabelEncoders saved to models/label_encoders_for_{model_name}.pkl") # Confirm save

    return model


# --- MAIN EXECUTION ---
def main():
    """Main function to run the modeling pipeline."""
    print("="*50)
    print("HYBRID LSTM-XGBoost MODELING PIPELINE")
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

    # --- MODEL 2: XGBoost for Late Delivery Risk (FIXED) ---
    print("\n" + "="*30)
    print("MODEL 2: XGBoost - Late Delivery Risk Prediction (FIXED)")
    print("="*30)
    # Prepare data for FIXED XGBoost
    X_train_xgb_fixed, X_test_xgb_fixed, y_train_xgb_fixed, y_test_xgb_fixed, imputer_xgb_fixed, label_encoders_xgb_fixed, feature_cols_xgb_fixed = prepare_xgboost_data_fixed(df_main)
    # Build and train FIXED XGBoost
    xgb_model_fixed = build_train_xgboost_fixed(
        X_train_xgb_fixed, y_train_xgb_fixed, X_test_xgb_fixed, y_test_xgb_fixed, 
        imputer_xgb_fixed, label_encoders_xgb_fixed, feature_cols_xgb_fixed, 
        model_name='xgboost_late_delivery_risk_fixed'
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
    print("- Update src/model_utils.py and src/app.py to use the new FIXED model components")
    print("- Test the /predict/risk/fixed endpoint in your API")


if __name__ == "__main__":
    main()
