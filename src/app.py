# src/app.py
"""
Flask API for Hybrid LSTM-XGBoost Supply Chain Optimization Model.
Exposes endpoints for demand forecasting and late delivery risk prediction.
"""
import os
import logging
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

# Import your model utility functions
# Make sure the path is correct relative to where this script runs
# Assuming this script is run from the project root (LSTM/)
from src.model_utils import (
    load_xgboost_model,
    load_lstm_components, # This should use the robust loader internally
    predict_risk_xgboost,
    predict_demand_lstm
)

# Import the supply chain optimizer
# Make sure this import is correct based on your file structure
# If supply_chain_optimizer.py is in src/, this should work
try:
    from src.supply_chain_optimizer import OrderPriorityEngine
    OPTIMIZER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import OrderPriorityEngine: {e}. /recommend/action endpoint will be disabled.")
    OPTIMIZER_AVAILABLE = False
    OrderPriorityEngine = None # Define a placeholder

# --- Configuration ---
# Define paths to your saved models and preprocessors relative to the project root
XGBOOST_MODEL_PATH = 'models/xgboost_late_delivery_risk.json'
XGBOOST_IMPUTER_PATH = 'models/imputer_for_xgboost_late_delivery_risk.pkl'
LSTM_MODEL_PATH = 'models/lstm_tuning_trial_9310.h5'
LSTM_SCALER_PATH = 'models/scaler_for_lstm_demand_forecast_sales.pkl'
DAILY_DEMAND_DATA_PATH = 'data/processed/daily_demand_data.csv' # Or corrected version
# Fallback path if main file is corrupt/bad
DAILY_DEMAND_DATA_PATH_FALLBACK = 'data/processed/daily_demand_data_corrected.csv'

# --- Initialize Flask App ---
app = Flask(__name__)
# Configure logging
app.logger.setLevel(logging.INFO)

# --- Global variables to hold loaded models and optimizer ---
# Loading models can take time, so we do it once when the app starts
xgb_model = None
xgb_imputer = None
lstm_model = None
lstm_scaler = None
feature_names_for_xgb = None # Will be loaded dynamically
order_priority_engine = None # Global instance of the optimizer

# --- Helper Functions ---
def get_xgboost_feature_names():
    """
    Retrieve the feature names used by the XGBoost model in the correct order.
    This is crucial for aligning new data with the model's expectations.
    """
    try:
        importance_file = 'results/model/xgboost_late_delivery_risk_feature_importance.csv'
        if os.path.exists(importance_file):
            importance_df = pd.read_csv(importance_file)
            # The 'feature' column contains the names IN THE ORDER THE MODEL USES THEM
            feature_names = importance_df['feature'].tolist()
            app.logger.info(f"Retrieved XGBoost feature names (in model order). Count: {len(feature_names)}")
            return feature_names
        else:
            app.logger.error(f"XGBoost feature importance file not found at {importance_file}.")
            raise FileNotFoundError(f"XGBoost feature importance file not found at {importance_file}. "
                                    f"Cannot determine feature names and order required by the model and its imputer.")
    except Exception as e:
        app.logger.error(f"Error retrieving XGBoost feature names: {e}")
        raise

def load_daily_demand_data():
    """
    Load the daily demand data needed for LSTM forecasting.
    """
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
        app.logger.info("Daily demand data loaded successfully.")
        return df_demand
    except Exception as e:
        app.logger.error(f"Error loading daily demand  {e}")
        raise

# --- Load Models on App Startup ---
def load_models():
    """
    Load all necessary models, preprocessors, and the optimizer when the application starts.
    """
    global xgb_model, xgb_imputer, lstm_model, lstm_scaler, feature_names_for_xgb, order_priority_engine
    try:
        app.logger.info("Starting to load models...")

        # --- Load XGBoost Model and Imputer ---
        app.logger.info("Loading XGBoost model and imputer...")
        xgb_model, xgb_imputer = load_xgboost_model(XGBOOST_MODEL_PATH, XGBOOST_IMPUTER_PATH)
        app.logger.info("XGBoost model and imputer loaded successfully.")

        # --- Load LSTM Model and Scaler (using robust loader) ---
        app.logger.info("Loading LSTM model and scaler...")
        # Use the robust loader from model_utils
        lstm_model, lstm_scaler = load_lstm_components(LSTM_MODEL_PATH, LSTM_SCALER_PATH)
        app.logger.info("LSTM model and scaler loaded successfully.")

        # --- Load XGBoost Feature Names ---
        app.logger.info("Loading XGBoost feature names...")
        feature_names_for_xgb = get_xgboost_feature_names()
        app.logger.info("XGBoost feature names loaded successfully.")

        # --- Load the Order Priority Optimizer (if available) ---
        if OPTIMIZER_AVAILABLE:
            app.logger.info("Loading Order Priority Engine...")
            # Determine the correct path for daily demand data for the optimizer's baseline
            baseline_data_path = DAILY_DEMAND_DATA_PATH
            if not os.path.exists(baseline_data_path):
                baseline_data_path = DAILY_DEMAND_DATA_PATH_FALLBACK
                if not os.path.exists(baseline_data_path):
                     app.logger.warning(f"Optimizer baseline data not found at {DAILY_DEMAND_DATA_PATH} or fallback {DAILY_DEMAND_DATA_PATH_FALLBACK}. Using default baseline.")
                     baseline_data_path = None # Optimizer will use default

            order_priority_engine = OrderPriorityEngine(baseline_volume_data_path=baseline_data_path)
            app.logger.info("Order Priority Engine loaded successfully.")
        else:
            app.logger.info("Order Priority Engine not available/disabled.")
            order_priority_engine = None

        app.logger.info("All models and dependencies loaded successfully!")
    except Exception as e:
        app.logger.error(f"Failed to load models/optimizer during startup: {e}")
        # Depending on your deployment strategy, you might want to exit here
        # or handle the error differently (e.g., return 503 Service Unavailable)
        # For now, we'll let the app start but routes will fail if models aren't loaded
        # raise # Re-raise if you want the app to crash on startup failure

# Load models when the script is executed
# This ensures they are loaded once when the WSGI server starts the app
# (e.g., when using `flask run` or `gunicorn`)
load_models()

# --- API Routes ---

@app.route('/')
def home():
    """Simple home route to check if the API is running."""
    optimizer_status = "Available" if OPTIMIZER_AVAILABLE and order_priority_engine is not None else "Not Available"
    return jsonify({"message": "Hybrid LSTM-XGBoost Supply Chain Optimization API is running!",
                    "optimizer_status": optimizer_status,
                    "endpoints": {
                        "/predict/risk": "POST - Predict late delivery risk for an order",
                        "/forecast/demand": "GET/POST - Get demand forecast for the next N days",
                        "/recommend/action": "POST - Get integrated action recommendations based on risk, context, and inventory" if OPTIMIZER_AVAILABLE else "DISABLED - Optimizer not loaded"
                    }})

@app.route('/health')
def health_check():
    """Health check endpoint to verify models are loaded."""
    models_loaded = (xgb_model is not None and xgb_imputer is not None and
                     lstm_model is not None and lstm_scaler is not None and
                     feature_names_for_xgb is not None)
    
    optimizer_loaded = OPTIMIZER_AVAILABLE and order_priority_engine is not None
    
    if models_loaded and (not OPTIMIZER_AVAILABLE or optimizer_loaded):
        return jsonify({"status": "healthy", "models_loaded": True, "optimizer_loaded": optimizer_loaded}), 200
    else:
        return jsonify({"status": "unhealthy", "models_loaded": False, "optimizer_loaded": optimizer_loaded,
                        "message": "One or more models/optimizer failed to load. Check server logs."}), 503

@app.route('/predict/risk', methods=['POST'])
def predict_late_delivery_risk():
    """
    Predict the late delivery risk for a new order.
    Expects JSON input with order features.
    Returns JSON with risk score and recommended actions.
    """
    # Use global variables for models and features
    global xgb_model, xgb_imputer, feature_names_for_xgb

    if xgb_model is None or xgb_imputer is None or feature_names_for_xgb is None:
        return jsonify({"error": "Models are not loaded. Health check failed."}), 503

    try:
        # --- 1. Get JSON data from request ---
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided in request body."}), 400

        app.logger.info(f"Received data for risk prediction: {list(data.keys())}")

        # --- 2. Convert JSON to DataFrame ---
        # The input should be a single order's features.
        # Flask's request.get_json() for a single object returns a dict.
        # We wrap it in a list to create a single-row DataFrame.
        order_df = pd.DataFrame([data])
        app.logger.info(f"Order data DataFrame shape: {order_df.shape}")

        # --- 3. Feature Alignment ---
        # Ensure the DataFrame has the features the model expects
        # Select and reorder the features in the sample data to match the model's expectation
        # Handle missing columns gracefully
        available_features_in_request = set(order_df.columns)
        required_features_for_xgb = set(feature_names_for_xgb)

        missing_features_for_xgb = required_features_for_xgb - available_features_in_request
        extra_features_in_request = available_features_in_request - required_features_for_xgb

        if extra_features_in_request:
            app.logger.info(f"Warning: Request data contains extra features not used by XGBoost: {extra_features_in_request}. These will be ignored.")
            # Optionally, filter the DataFrame to only known features
            # order_df = order_df[feature_names_for_xgb] # This would drop extras, but let's be lenient

        if missing_features_for_xgb:
            app.logger.warning(f"Request data is missing features required by XGBoost: {missing_features_for_xgb}")
            # Depending on your model's robustness, you might want to reject the request
            # or impute missing numerical features with 0 or NaN.
            # For now, we'll proceed and let the imputer handle it if possible.
            # Filter feature_names_for_xgb to only those present
            feature_names_filtered = [f for f in feature_names_for_xgb if f in available_features_in_request]
            app.logger.info(f"Proceeding with {len(feature_names_filtered)}/{len(feature_names_for_xgb)} available features.")
            if not feature_names_filtered:
                return jsonify({"error": "No required XGBoost features found in request data."}), 400
        else:
            feature_names_filtered = feature_names_for_xgb

        # Select the relevant columns from the sample data, in the order the model expects
        df_order_for_prediction = order_df[feature_names_filtered].copy()
        app.logger.info(f"Order data prepared for XGBoost prediction. Shape: {df_order_for_prediction.shape}")

        # --- 4. Make Prediction ---
        # Use the predict_risk_xgboost function from model_utils
        # It handles the imputation internally based on the robust logic we developed.
        # This function should now correctly handle the feature name alignment for the imputer,
        # as discussed previously, by ensuring the DataFrame passed to imputer.transform
        # has the correct column names that match what the imputer saw during fit.
        risk_scores = predict_risk_xgboost(xgb_model, xgb_imputer, df_order_for_prediction, feature_names_filtered)

        # Assuming single prediction
        risk_score = float(risk_scores[0]) if len(risk_scores) > 0 else None

        # --- 5. Determine Actions ---
        # Define risk thresholds (these could be configurable or loaded from a config file)
        HIGH_RISK_THRESHOLD = 0.7
        MEDIUM_RISK_THRESHOLD = 0.4

        if risk_score is not None:
            if risk_score > HIGH_RISK_THRESHOLD:
                classification = "HIGH RISK"
                actions = ["Prioritize order in packing queue", "Consider expedited shipping option", "Assign to experienced staff"]
            elif risk_score > MEDIUM_RISK_THRESHOLD:
                classification = "MEDIUM RISK"
                actions = ["Standard processing with monitoring"]
            else:
                classification = "LOW RISK"
                actions = ["Standard processing"]
        else:
            classification = "UNKNOWN"
            actions = ["Unable to determine risk - check input data"]
            risk_score = -1 # Indicate error

        # --- 6. Return JSON Response ---
        response = {
            "risk_score": risk_score,
            "classification": classification,
            "recommended_actions": actions,
            "message": "Prediction successful" if risk_score != -1 else "Prediction failed"
        }
        app.logger.info(f"Risk prediction completed. Score: {risk_score}, Classification: {classification}")
        return jsonify(response), 200

    except Exception as e:
        app.logger.error(f"Error during XGBoost risk prediction: {e}")
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500


@app.route('/forecast/demand', methods=['GET', 'POST'])
def forecast_demand():
    """
    Forecast demand for the next N days using the LSTM model.
    Can accept parameters via GET query strings or POST JSON.
    Returns JSON with forecasted dates and sales.
    """
    # Use global variables for models and scaler
    global lstm_model, lstm_scaler

    if lstm_model is None or lstm_scaler is None:
        return jsonify({"error": "LSTM models are not loaded. Health check failed."}), 503

    try:
        # --- 1. Get parameters (n_days, etc.) ---
        n_steps_out = 7 # Default forecast horizon
        n_steps_in = 14 # Default lookback window (should match training)

        if request.method == 'POST':
            data = request.get_json()
            if data:
                n_steps_out = data.get('n_days', n_steps_out)
                n_steps_in = data.get('lookback_days', n_steps_in)
        elif request.method == 'GET':
            # Get parameters from query string
            n_steps_out = request.args.get('n_days', default=n_steps_out, type=int)
            n_steps_in = request.args.get('lookback_days', default=n_steps_in, type=int)

        # Validate parameters
        if not (1 <= n_steps_out <= 30):
             return jsonify({"error": "n_days must be between 1 and 30."}), 400
        # Note: n_steps_in should ideally match the training value. We can enforce or warn.
        # For simplicity, we'll use the default 14 which matches common training setup.

        app.logger.info(f"Received demand forecast request. n_days: {n_steps_out}, lookback_days: {n_steps_in}")

        # --- 2. Load Daily Demand Data ---
        df_daily_demand = load_daily_demand_data()

        # --- 3. Make Forecast ---
        # Define parameters (should ideally match training, but we use defaults)
        feature_cols = ['Sales', 'Order Item Quantity', 'num_orders'] # Match training

        forecast = predict_demand_lstm(
            lstm_model, lstm_scaler, df_daily_demand,
            n_steps_out=n_steps_out, n_steps_in=n_steps_in, feature_cols=feature_cols
        )

        # --- 4. Format Response ---
        if forecast and 'predicted_sales' in forecast and 'predicted_dates' in forecast:
            forecast_list = []
            for date, sales in zip(forecast['predicted_dates'], forecast['predicted_sales']):
                forecast_list.append({
                    "date": date.strftime('%Y-%m-%d'),
                    "predicted_sales": round(float(sales), 2) # Round for cleaner JSON
                })

            response = {
                "forecast": forecast_list,
                "parameters": {
                    "n_days": n_steps_out,
                    "lookback_days": n_steps_in
                },
                "message": "Forecast generated successfully"
            }
            app.logger.info(f"LSTM demand forecast completed for {n_steps_out} days.")
            return jsonify(response), 200
        else:
            app.logger.error("LSTM forecast returned unexpected format or None.")
            return jsonify({"error": "Failed to generate forecast. Internal model error."}), 500

    except FileNotFoundError as e:
        app.logger.error(f"Data file not found for LSTM forecast: {e}")
        return jsonify({"error": "Required data file for forecasting not found."}), 500
    except Exception as e:
        app.logger.error(f"Error during LSTM demand forecast: {e}")
        return jsonify({"error": f"An error occurred during forecasting: {str(e)}"}), 500


# --- New Integrated Endpoint for Action Recommendations ---
@app.route('/recommend/action', methods=['POST'])
def recommend_action():
    """
    Integrated endpoint for recommending actions based on risk, demand context, and inventory.
    Combines XGBoost risk prediction with business rules.
    Expects JSON with 'order_data', and optionally 'inventory_data', 'context'.
    Returns JSON with risk score, classification, and recommended actions.
    """
    # Use global variables for models, features, and optimizer
    global xgb_model, xgb_imputer, feature_names_for_xgb, lstm_model, lstm_scaler, order_priority_engine

    # --- Health Checks ---
    if xgb_model is None or xgb_imputer is None or feature_names_for_xgb is None:
        return jsonify({"error": "XGBoost models are not loaded. Health check failed."}), 503
    if lstm_model is None or lstm_scaler is None:
        return jsonify({"error": "LSTM models are not loaded. Health check failed."}), 503
    if not OPTIMIZER_AVAILABLE or order_priority_engine is None:
        return jsonify({"error": "Order Priority Engine is not loaded. Health check failed."}), 503

    try:
        # --- 1. Get JSON data from request ---
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided in request body."}), 400

        app.logger.info("Received data for action recommendation.")

        order_data = data.get('order_data')
        inventory_data = data.get('inventory_data', {}) # Optional
        context = data.get('context', {}) # Optional, e.g., {'high_demand_period': True}

        if not order_data:
            return jsonify({"error": "Missing 'order_data' in request body."}), 400

        # --- 2. Prepare Order Data for XGBoost Prediction ---
        order_df = pd.DataFrame([order_data])

        # --- 3. Feature Alignment (reuse logic from /predict/risk) ---
        available_features_in_request = set(order_df.columns)
        required_features_for_xgb = set(feature_names_for_xgb)
        missing_features_for_xgb = required_features_for_xgb - available_features_in_request

        if missing_features_for_xgb:
            app.logger.warning(f"Request data is missing features required by XGBoost: {missing_features_for_xgb}")
            feature_names_filtered = [f for f in feature_names_for_xgb if f in available_features_in_request]
            if not feature_names_filtered:
                return jsonify({"error": "No required XGBoost features found in request data."}), 400
        else:
            feature_names_filtered = feature_names_for_xgb

        df_order_for_prediction = order_df[feature_names_filtered].copy()

        # --- 4. Make XGBoost Prediction (Internal Call) ---
        try:
            risk_scores = predict_risk_xgboost(xgb_model, xgb_imputer, df_order_for_prediction, feature_names_filtered)
            xgboost_risk_score = float(risk_scores[0]) if len(risk_scores) > 0 else None
            if xgboost_risk_score is None:
                return jsonify({"error": "Failed to generate XGBoost risk prediction internally."}), 500
        except Exception as pred_error:
            app.logger.error(f"Error during internal XGBoost risk prediction: {pred_error}")
            return jsonify({"error": f"An error occurred during internal XGBoost prediction: {str(pred_error)}"}), 500

        # --- 5. Get LSTM Demand Forecast for Context (Internal Call) ---
        try:
            # Load daily demand data for context
            df_daily_demand = load_daily_demand_data()

            # Estimate shipment date context (simplified)
            # Assume 'Days for shipment (scheduled)' is in order_data to estimate shipment date
            days_for_shipment_scheduled = order_data.get('Days for shipment (scheduled)', 3) # Default estimate

            # Get forecast for next 7 days to estimate volume for the shipment date
            n_steps_out_context = 7
            n_steps_in_context = 14
            feature_cols_context = ['Sales', 'Order Item Quantity', 'num_orders'] # Match training

            forecast_context = predict_demand_lstm(
                lstm_model, lstm_scaler, df_daily_demand,
                n_steps_out=n_steps_out_context, n_steps_in=n_steps_in_context, feature_cols=feature_cols_context
            )

            if forecast_context and 'predicted_sales' in forecast_context:
                # Get the predicted volume for the estimated shipment day
                # If scheduled days is 3, take the 3rd day of the forecast (index 2)
                forecast_horizon_index = min(days_for_shipment_scheduled - 1, len(forecast_context['predicted_sales']) - 1)
                lstm_predicted_volume_for_shipment_date = float(forecast_context['predicted_sales'][forecast_horizon_index])
                app.logger.info(f"LSTM context volume retrieved. Estimated volume for shipment day: {lstm_predicted_volume_for_shipment_date}")
            else:
                 app.logger.error("Failed to get LSTM forecast for context (internal).")
                 lstm_predicted_volume_for_shipment_date = 500.0 # Default fallback volume

        except Exception as forecast_error:
            app.logger.error(f"Error during internal LSTM demand forecast call: {forecast_error}")
            # Fallback to a default or average volume if forecast fails internally
            lstm_predicted_volume_for_shipment_date = 500.0 # Or load a default average

        # --- 6. Calculate Unified Priority using Optimizer ---
        app.logger.info("Calculating unified priority score using OrderPriorityEngine...")
        # Pass the XGBoost score and LSTM volume to the optimizer
        # Assume shipment date window is 1 day for simplicity in this context
        priority_result = order_priority_engine.calculate_priority_score(
            xgboost_risk_score,
            lstm_predicted_volume_for_shipment_date,
            shipment_date_window_days=1
        )

        # --- 7. Get Recommended Actions from Optimizer ---
        recommended_actions = order_priority_engine.get_recommended_actions(priority_result['final_tier'])

        # --- 8. Return JSON Response (WITH FIX FOR SERIALIZATION) ---
        response = {
            "xgboost_risk_score": float(xgboost_risk_score) if xgboost_risk_score is not None else None,
            "lstm_contextual_volume_forecast": float(lstm_predicted_volume_for_shipment_date) if lstm_predicted_volume_for_shipment_date is not None else None,
            "priority_score": float(priority_result.get('priority_score', -1)) if priority_result.get('priority_score') is not None else -1,
            "base_classification": str(priority_result.get('base_classification', 'UNKNOWN')),
            "contextual_classification": str(priority_result.get('contextual_classification', 'UNKNOWN')),
            "final_tier": str(priority_result.get('final_tier', 'UNKNOWN')),
            # --- FIX: Ensure boolean is Python-native for JSON serialization ---
            "is_high_volume_context": bool(priority_result.get('is_high_volume_context', False)),
            "recommended_actions": list(recommended_actions) if recommended_actions else [],
            "message": "Unified order priority and actions calculated successfully"
        }
        app.logger.info(f"Unified order priority calculation completed. Tier: {priority_result.get('final_tier', 'UNKNOWN')}")
        return jsonify(response), 200

    except Exception as e:
        app.logger.error(f"Error during unified order priority calculation: {e}")
        return jsonify({"error": f"An error occurred during priority calculation: {str(e)}"}), 500


# --- Run the App (for development only) ---
# This block is only executed if the script is run directly (not imported)
if __name__ == '__main__':
    # In production, you would typically use a WSGI server like Gunicorn
    # and not run the app directly with app.run()
    app.run(debug=True, host='127.0.0.1', port=5000) # Default Flask port
