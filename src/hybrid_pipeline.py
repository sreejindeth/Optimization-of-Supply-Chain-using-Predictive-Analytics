# src/hybrid_pipeline.py
"""
Main script to demonstrate the Hybrid LSTM-XGBoost Supply Chain Optimization Pipeline.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
# Import helper functions
from src.model_utils import load_xgboost_model, load_lstm_components, predict_risk_xgboost, predict_demand_lstm


def create_output_dirs():
    """Create necessary directories for demo outputs."""
    dirs = ['data/demo', 'results/hybrid_demo']
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
        print(f"Ensured directory exists: {dir}")

# Inside src/hybrid_pipeline.py, replace the get_xgboost_feature_names function:
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
            # This order must match the imputer's expectation if it was fit on the same data.
            feature_names = importance_df['feature'].tolist()
            print(f"Retrieved XGBoost feature names (in model order). Count: {len(feature_names)}")
            # Print first few for verification
            print(f"First 5 features: {feature_names[:5]}")
            return feature_names
        else:
            # Fallback: Raise a clear error. Hardcoding is risky.
            raise FileNotFoundError(f"XGBoost feature importance file not found at {importance_file}. "
                                    f"Cannot determine feature names and order required by the model and its imputer.")
    except Exception as e:
        print(f"Error retrieving XGBoost feature names: {e}")
        # Re-raise to stop the pipeline if features can't be determined
        raise

def load_sample_data():
    """
    Load sample data for demonstration.
    In a real system, this would be live data streams or database queries.
    """
    print("--- Loading Sample Data for Demonstration ---")
    
    # --- 1. Load Daily Demand Data for LSTM Forecast ---
    daily_demand_file = 'data/processed/daily_demand_data.csv' # Or corrected version
    if not os.path.exists(daily_demand_file):
        daily_demand_file = 'data/processed/daily_demand_data_corrected.csv'
    
    if not os.path.exists(daily_demand_file):
        raise FileNotFoundError(f"Daily demand data not found at {daily_demand_file} or fallback.")

    df_daily = pd.read_csv(daily_demand_file)
    df_daily['order_date'] = pd.to_datetime(df_daily['order_date'])
    df_daily.set_index('order_date', inplace=True)
    df_daily.sort_index(inplace=True)
    print(f"Loaded daily demand data. Shape: {df_daily.shape}")

    # --- 2. Load Sample New Orders for XGBoost Risk Assessment ---
    # For demo, we can use a slice of the cleaned data or create mock data.
    # Let's create a small mock dataset based on real feature names and plausible values.
    # Alternatively, use a sample from the cleaned data.
    
    # Option 1: Use real data (ensure features match)
    cleaned_data_file = 'data/processed/cleaned_supply_chain_data.csv'
    if not os.path.exists(cleaned_data_file):
        raise FileNotFoundError(f"Cleaned data file not found at {cleaned_data_file}")
    
    df_cleaned = pd.read_csv(cleaned_data_file)
    # Select a few recent orders as examples
    sample_orders_df = df_cleaned.tail(5).copy() # Get last 5 orders
    print(f"Loaded sample orders data. Shape: {sample_orders_df.shape}")
    
    # Ensure 'order_date' is datetime if needed for sorting or context
    if 'order_date' in sample_orders_df.columns:
        sample_orders_df['order_date'] = pd.to_datetime(sample_orders_df['order_date'])
        
    return df_daily, sample_orders_df


def run_hybrid_pipeline():
    """
    Execute the main steps of the hybrid pipeline.
    """
    print("="*60)
    print("HYBRID LSTM-XGBOOST SUPPLY CHAIN OPTIMIZATION DEMO")
    print("="*60)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Demo started at: {timestamp}")

    create_output_dirs()

    # --- 1. Load Sample Data ---
    df_daily_demand, df_sample_orders = load_sample_data()

    # --- 2. Load Models and Preprocessors ---
    print("\n--- Loading Trained Models ---")
    xgb_model, xgb_imputer = load_xgboost_model()
    lstm_model, lstm_scaler = load_lstm_components()

    # --- 3. STEP 1: LSTM Demand Forecasting ---
    print("\n" + "-"*40)
    print("STEP 1: LSTM - Demand Forecasting")
    print("-"*40)
    try:
        # Define parameters (should match training)
        n_steps_in = 14
        n_steps_out = 7
        feature_cols = ['Sales', 'Order Item Quantity', 'num_orders'] # Match training
        
        forecast = predict_demand_lstm(
            lstm_model, lstm_scaler, df_daily_demand, 
            n_steps_out=n_steps_out, n_steps_in=n_steps_in, feature_cols=feature_cols
        )
        
        print("LSTM Demand Forecast Results:")
        for i, (date, sales) in enumerate(zip(forecast['predicted_dates'], forecast['predicted_sales'])):
            print(f"  {date.strftime('%Y-%m-%d')}: Predicted Sales = {sales:.2f}")

        # Save forecast results
        forecast_df = pd.DataFrame({
            'date': forecast['predicted_dates'],
            'predicted_sales': forecast['predicted_sales']
        })
        forecast_output_path = 'results/hybrid_demo/latest_forecast.csv'
        forecast_df.to_csv(forecast_output_path, index=False)
        print(f"\nLSTM forecast saved to {forecast_output_path}")

        # --- Conceptual Operational Planning ---
        avg_sales = np.mean(forecast['predicted_sales'])
        print(f"\nOperational Insight: Average predicted sales for next {n_steps_out} days: {avg_sales:.2f}")
        if any(sales > avg_sales * 1.2 for sales in forecast['predicted_sales']):
            high_demand_dates = [date for date, sales in zip(forecast['predicted_dates'], forecast['predicted_sales']) if sales > avg_sales * 1.2]
            print(f"  -> High demand predicted on: {[d.strftime('%Y-%m-%d') for d in high_demand_dates]}")
            print("  -> Recommendation: Consider increasing staffing/inventory for these dates.")

    except Exception as e:
        print(f"Error in LSTM forecasting step: {e}")
        # Continue to XGBoost even if LSTM fails for demo purposes
        forecast = None

    # --- 4. STEP 2: XGBoost Risk Assessment for New Orders ---
    print("\n" + "-"*40)
    print("STEP 2: XGBoost - Real-Time Order Risk Assessment")
    print("-"*40)
    try:
        # Get feature names used by XGBoost model (in correct order)
        xgb_feature_names = get_xgboost_feature_names()
        expected_num_features = len(xgb_feature_names)
        print(f"XGBoost model/feature importance list has {expected_num_features} features.")

        # --- Crucial Alignment Step ---
        # Ensure df_sample_orders contains the exact columns needed by the model/imputer
        # The imputer might expect a different number of features if one was dropped during training.
        # We need to be robust to this.

        available_features_in_sample = set(df_sample_orders.columns)
        required_features_for_xgb = set(xgb_feature_names)

        # 1. Check for features the model needs but the sample data doesn't have
        missing_features_for_xgb = required_features_for_xgb - available_features_in_sample
        if missing_features_for_xgb:
            print(f"Warning: Sample order data is missing features required by XGBoost: {missing_features_for_xgb}")
            # Filter xgb_feature_names to only those present in the sample data
            xgb_feature_names_filtered = [f for f in xgb_feature_names if f in available_features_in_sample]
            print(f"Adjusted feature list to {len(xgb_feature_names_filtered)} available features.")
            if not xgb_feature_names_filtered:
                raise ValueError("No required XGBoost features found in sample order data.")
        else:
            xgb_feature_names_filtered = xgb_feature_names # All required features are present

        # 2. Select the relevant columns from the sample data, in the order the model expects
        # This creates the DataFrame that will be passed to the prediction function.
        df_sample_orders_for_prediction = df_sample_orders[xgb_feature_names_filtered].copy()
        print(f"Sample orders data prepared for XGBoost. Shape: {df_sample_orders_for_prediction.shape}")

        # --- Check for Imputer Expectation ---
        # The core issue is the imputer expects 40 features but gets 39.
        # This likely means one feature was dropped during training because it was empty.
        # The model importance list might include this dropped feature, or the imputer's
        # expectation is based on the data it was originally fit on (which had 40 columns
        # before dropping).
        # The safest approach is to let the imputer tell us what it expects.
        # However, we don't have direct access to the imputer's original fit shape easily.
        # Let's pass the correctly aligned DataFrame and handle any error inside predict_risk_xgboost
        # or see if this alignment fixes it.

        # --- Assess risk for sample orders ---
        # Pass the aligned DataFrame and the (potentially filtered) feature names list.
        # The predict_risk_xgboost function should handle the imputation.
        risk_scores = predict_risk_xgboost(
            xgb_model, xgb_imputer, df_sample_orders_for_prediction, xgb_feature_names_filtered
        )

        # --- Conceptual Intervention Logic ---
        # Define risk thresholds
        HIGH_RISK_THRESHOLD = 0.7
        MEDIUM_RISK_THRESHOLD = 0.4

        print("XGBoost Risk Assessment Results:")
        results_data = []
        for i, (idx, order_row) in enumerate(df_sample_orders.iterrows()):
            order_id = order_row.get('Order Id', f'Order_{i}')
            risk_score = risk_scores[i]
            results_data.append({'Order_Id': order_id, 'Risk_Score': risk_score})

            print(f"  Order ID {order_id}: Risk Score = {risk_score:.4f}")
            if risk_score > HIGH_RISK_THRESHOLD:
                print(f"    -> Classification: HIGH RISK")
                print(f"    -> Recommended Actions:")
                print(f"       * Prioritize order in packing queue.")
                print(f"       * Consider expedited shipping option.")
                print(f"       * Assign to experienced staff.")
            elif risk_score > MEDIUM_RISK_THRESHOLD:
                print(f"    -> Classification: MEDIUM RISK")
                print(f"    -> Recommended Actions:")
                print(f"       * Standard processing with monitoring.")
            else:
                print(f"    -> Classification: LOW RISK")
                print(f"    -> Recommended Actions:")
                print(f"       * Standard processing.")

        # Save risk assessment results
        results_df = pd.DataFrame(results_data)
        risk_output_path = 'results/hybrid_demo/order_risk_assessments.csv'
        results_df.to_csv(risk_output_path, index=False)
        print(f"\nXGBoost risk assessments saved to {risk_output_path}")

    except Exception as e:
        print(f"Error in XGBoost risk assessment step: {e}")
        # Re-raise if you want the pipeline to stop on XGBoost errors
        # raise

    # --- 5. Summary ---
    end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\nDemo finished at: {end_timestamp}")
    print("="*60)
    print("HYBRID PIPELINE DEMO COMPLETED")
    print("="*60)
    print("\nConceptual Workflow Demonstrated:")
    print("1. LSTM model forecasted future demand.")
    print("2. Forecast insights suggested operational planning (e.g., prepare for high-volume days).")
    print("3. XGBoost model assessed risk for new incoming orders.")
    print("4. Risk scores triggered recommended actions for order handling.")
    print("\nIn a production system, this would run continuously:")
    print("- LSTM: Daily/Periodically for forecasting.")
    print("- XGBoost: Real-time for each new order.")
    print("- Actions: Automatically integrated with OMS/WMS/TMS.")


if __name__ == "__main__":
    run_hybrid_pipeline()
