# src/api_client_demo.py
"""
Simple client script to demonstrate interaction with the
Hybrid LSTM-XGBoost Supply Chain Optimization API.
Modified to use real data from the cleaned dataset for risk prediction.
"""
import requests
import json
import pandas as pd
import random
import os

# --- Configuration ---
# The base URL of your Flask API
API_BASE_URL = "http://127.0.0.1:5000"

# Path to the cleaned supply chain data
CLEANED_DATA_PATH = 'data/processed/cleaned_supply_chain_data.csv'

# --- Helper Functions ---
def make_request(method, endpoint, data=None, params=None):
    """
    Makes an HTTP request to the API and handles the response.
    """
    url = f"{API_BASE_URL}{endpoint}"
    headers = {'Content-Type': 'application/json'} if data else {}

    try:
        if method == 'GET':
            response = requests.get(url, params=params)
        elif method == 'POST':
            response = requests.post(url, headers=headers, json=data)
        else:
            print(f"Unsupported HTTP method: {method}")
            return None

        print(f"\n--- Request: {method} {url} ---")
        if params:
            print(f"Params: {params}")
        if data:
            # Truncate data print for readability
            data_str = json.dumps(data, indent=2)
            if len(data_str) > 500:
                data_str = data_str[:500] + "... (truncated)"
            print(f"Data: {data_str}")

        print(f"Status Code: {response.status_code}")
        
        # Try to parse JSON response
        try:
            response_data = response.json()
            print("Response:")
            # Truncate response print for readability if it's the forecast
            if endpoint == '/forecast/demand' and 'forecast' in response_data:
                # Print summary
                print(f"  Message: {response_data.get('message', 'N/A')}")
                print(f"  Parameters: {response_data.get('parameters', 'N/A')}")
                forecast_list = response_data.get('forecast', [])
                print(f"  Forecast Points: {len(forecast_list)} (showing first 3)")
                for point in forecast_list[:3]:
                    print(f"    {point}")
                if len(forecast_list) > 3:
                    print("    ...")
            else:
                print(json.dumps(response_data, indent=2))
            return response_data
        except requests.exceptions.JSONDecodeError:
            print("Response (Non-JSON):")
            print(response.text[:500] + ("..." if len(response.text) > 500 else "")) # Truncate long text
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error making request to {url}: {e}")
        return None

# --- API Interaction Functions ---
def check_health():
    """Check the health of the API."""
    return make_request('GET', '/health')

def get_demand_forecast(n_days=None, lookback_days=None):
    """
    Get demand forecast from the API.
    Can use GET with query parameters or POST with JSON data.
    """
    params = {}
    if n_days is not None:
        params['n_days'] = n_days
    if lookback_days is not None:
        params['lookback_days'] = lookback_days

    # Example using GET with parameters
    if params:
        return make_request('GET', '/forecast/demand', params=params)
    else:
        return make_request('GET', '/forecast/demand')

def predict_late_delivery_risk(order_features):
    """
    Send order features to the API to predict late delivery risk.
    """
    return make_request('POST', '/predict/risk', data=order_features)

def load_real_orders(num_orders=3):
    """
    Load real orders from the cleaned supply chain data.
    Selects a few recent orders and prepares them for API consumption.
    """
    try:
        if not os.path.exists(CLEANED_DATA_PATH):
            raise FileNotFoundError(f"Cleaned data file not found at {CLEANED_DATA_PATH}")

        print(f"\nLoading real orders from {CLEANED_DATA_PATH}...")
        # Load the data
        df = pd.read_csv(CLEANED_DATA_PATH)
        print(f"Loaded {len(df)} records from cleaned data.")

        # Select a few recent orders (tail) as examples
        # In a real application, you'd get the *new* order data as it arrives.
        sample_df = df.tail(num_orders).copy()
        print(f"Selected {len(sample_df)} recent orders for demonstration.")

        # Load the list of features the XGBoost model expects
        importance_file = 'results/model/xgboost_late_delivery_risk_feature_importance.csv'
        if not os.path.exists(importance_file):
            raise FileNotFoundError(f"Feature importance file not found at {importance_file}")
            
        importance_df = pd.read_csv(importance_file)
        expected_features = importance_df['feature'].tolist()
        print(f"API expects {len(expected_features)} features for XGBoost model.")

        real_orders_list = []
        for index, row in sample_df.iterrows():
            order_dict = {}
            order_id = row.get('Order Id', f'Order_{index}')
            
            # For each expected feature, try to get its value from the row
            missing_features = []
            for feature in expected_features:
                if feature in row:
                    order_dict[feature] = row[feature]
                else:
                    # Handle potential mismatches in column names (e.g., spaces, case)
                    # This is a simple heuristic, might need refinement based on your data
                    # Check for feature name in row's columns (case-insensitive)
                    found = False
                    for col in df.columns:
                        if col.lower().replace(" ", "_") == feature.lower().replace(" ", "_"):
                            order_dict[feature] = row[col]
                            found = True
                            break
                    if not found:
                        # If feature is not found, we have a problem.
                        # The API requires all features. We could impute or raise an error.
                        # For demo, let's use a default value (0.0) and warn.
                        # A production system would need a more robust solution.
                        missing_features.append(feature)
                        order_dict[feature] = 0.0 # Default value
            
            if missing_features:
                print(f"Warning: Order {order_id} is missing data for features: {missing_features}. Using default value 0.0.")
            
            real_orders_list.append({'order_id': order_id, 'features': order_dict})
            
        print(f"Prepared {len(real_orders_list)} orders with required features.")
        return real_orders_list

    except FileNotFoundError as e:
        print(f"Error loading real orders: {e}")
        return None
    except Exception as e:
        print(f"Error processing real orders: {e}")
        return None

# --- Main Demo Function ---
def main():
    """
    Main function to run the client demo using real data.
    """
    print("=" * 60)
    print("DEMO: Interacting with Hybrid LSTM-XGBoost API (Using Real Data)")
    print("=" * 60)
    print(f"Target API URL: {API_BASE_URL}")

    # 1. Check API Health
    print("\n1. Checking API Health...")
    health_data = check_health()
    if not health_data or health_data.get('status') != 'healthy':
        print("API is not healthy. Exiting demo.")
        return

    # 2. Get Demand Forecast (Default)
    print("\n2. Getting Default Demand Forecast...")
    forecast_data_default = get_demand_forecast()

    # 3. Get Demand Forecast (Custom Parameters)
    print("\n3. Getting Custom Demand Forecast (3 days)...")
    forecast_data_custom = get_demand_forecast(n_days=3)

    # 4. Predict Late Delivery Risk using Real Data
    print("\n4. Predicting Late Delivery Risk for Real Orders from Dataset...")
    # Load real orders
    real_orders = load_real_orders(num_orders=3)
    if real_orders:
        for i, order_info in enumerate(real_orders):
            order_id = order_info['order_id']
            order_features = order_info['features']
            print(f"\n--- Predicting Risk for Real Order {i+1} (ID: {order_id}) ---")
            # Predict risk
            risk_prediction = predict_late_delivery_risk(order_features)
            # Note: We don't break on error to try predicting for other orders
    else:
        print("Could not load real orders. Skipping risk prediction.")

    print("\n" + "=" * 60)
    print("DEMO COMPLETED")
    print("=" * 60)

if __name__ == "__main__":
    main()