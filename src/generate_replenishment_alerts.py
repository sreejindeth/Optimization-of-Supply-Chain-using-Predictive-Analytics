"""
Script to generate predictive replenishment alerts using the LSTM forecast from the API.
Compares forecasted demand against current (simulated) inventory levels.
"""
import requests
import json
import pandas as pd
import os
from datetime import datetime, timedelta

# --- Configuration ---
API_BASE_URL = "http://127.0.0.1:5000"
# Simulated inventory data (in a real system, this would come from an ERP database)
# Key: Product Identifier (e.g., Product Card Id), Value: Current Stock Level
# This is a placeholder. You would need to map your forecast items to these keys.
SIMULATED_INVENTORY = {
    "PRODUCT_A": 150, # Example product
    "PRODUCT_B": 20,  # Example product with potentially low stock
    "12345": 75,      # Example using a numeric ID as string
    # Add more products as needed for testing
}
DEFAULT_SAFETY_STOCK = 50
DEFAULT_LEAD_TIME_DAYS = 5 # How many days it takes to receive new stock
FORECAST_HORIZON_DAYS = 7 # Should match or be less than your LSTM forecast

# Path to save the alerts report
ALERTS_REPORT_DIR = 'results/reports'
os.makedirs(ALERTS_REPORT_DIR, exist_ok=True)

def get_demand_forecast(n_days=FORECAST_HORIZON_DAYS):
    """
    Fetches the demand forecast from the Flask API.
    """
    try:
        params = {'n_days': n_days}
        response = requests.get(f"{API_BASE_URL}/forecast/demand", params=params)
        response.raise_for_status()
        forecast_data = response.json()
        print(f"Successfully fetched {n_days}-day demand forecast from API.")
        return forecast_data.get('forecast', [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching demand forecast from API: {e}")
        return None

def calculate_forecasted_demand_during_lead_time(forecast_list, lead_time_days):
    """
    Calculates the total forecasted demand over the lead time period.
    Note: This is simplified. A real system would map forecasts to specific products/SKUs.
    For this demo, we assume the forecast is for a generic "demand unit" or a key product.
    """
    if not forecast_list:
        return 0
    
    # Sum the predicted sales (or quantity) for the first 'lead_time_days' of the forecast
    # This assumes the forecast list is ordered chronologically.
    demand_during_lead_time = sum(
        item['predicted_sales'] for item in forecast_list[:min(lead_time_days, len(forecast_list))]
    )
    # If forecasting quantity, you'd sum 'predicted_quantity' instead.
    return demand_during_lead_time

def check_replenishment_needed(product_id, current_stock, forecasted_demand, safety_stock, lead_time_days):
    """
    Determines if a replenishment alert is needed for a product.
    This is the core logic for the alert.
    """
    # Calculate expected stock at the end of the lead time period
    # Assuming consumption happens during the lead time
    expected_stock_at_lead_time_end = current_stock - forecasted_demand

    alert_needed = expected_stock_at_lead_time_end < safety_stock
    recommended_po_quantity = 0
    if alert_needed:
        # Simple PO calculation: cover lead time demand plus replenish safety stock
        # PO_Qty = Forecasted_Demand + Safety_Stock - Current_Stock
        # Ensure PO quantity is not negative
        recommended_po_quantity = max(0, forecasted_demand + safety_stock - current_stock)
        
    return alert_needed, expected_stock_at_lead_time_end, recommended_po_quantity

def generate_alerts():
    """
    Main function to generate replenishment alerts.
    """
    print("=" * 60)
    print("GENERATING PREDICTIVE REPLENISHMENT ALERTS")
    print("=" * 60)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # 1. Fetch demand forecast
    forecast_list = get_demand_forecast(n_days=FORECAST_HORIZON_DAYS)
    if not forecast_list:
        print("Failed to get demand forecast. Exiting.")
        return

    # 2. Load current inventory (simulated)
    print(f"\nUsing simulated inventory data: {SIMULATED_INVENTORY}")
    print(f"Using default lead time: {DEFAULT_LEAD_TIME_DAYS} days")
    print(f"Using default safety stock: {DEFAULT_SAFETY_STOCK} units")

    # 3. Process each product in inventory
    alerts = []
    print("\n--- Checking Inventory for Replenishment Needs ---")
    for product_id, current_stock in SIMULATED_INVENTORY.items():
        print(f"\nAnalyzing Product ID: {product_id}")
        print(f"  Current Stock: {current_stock}")

        # 4. Calculate forecasted demand during lead time (simplified)
        # NOTE: This is a key simplification. In reality, you'd map the API's
        # forecast (likely total sales) to forecasted demand for *this specific product*.
        # For demo, we'll use the total forecasted sales as a proxy.
        # A more accurate system would have the LSTM forecast demand for specific products/SKUs.
        forecasted_demand = calculate_forecasted_demand_during_lead_time(forecast_list, DEFAULT_LEAD_TIME_DAYS)
        print(f"  Forecasted Demand (Total Proxy) over {DEFAULT_LEAD_TIME_DAYS} days: {forecasted_demand:.2f}")

        # 5. Check if replenishment is needed
        alert_needed, expected_stock, po_qty = check_replenishment_needed(
            product_id, current_stock, forecasted_demand, DEFAULT_SAFETY_STOCK, DEFAULT_LEAD_TIME_DAYS
        )

        # 6. Record Alert
        if alert_needed:
            alert_message = (
                f"ALERT: Product {product_id} needs replenishment.\n"
                f"  Reason: Expected stock ({expected_stock:.2f}) < Safety Stock ({DEFAULT_SAFETY_STOCK}).\n"
                f"  Forecasted demand during {DEFAULT_LEAD_TIME_DAYS}-day lead time: {forecasted_demand:.2f}\n"
                f"  Current stock: {current_stock}\n"
                f"  Recommended PO quantity: {po_qty:.2f}"
            )
            print(f"  -> {alert_message}")
            alerts.append({
                "timestamp": timestamp,
                "product_id": product_id,
                "current_stock": current_stock,
                "forecasted_demand": round(forecasted_demand, 2),
                "lead_time_days": DEFAULT_LEAD_TIME_DAYS,
                "safety_stock": DEFAULT_SAFETY_STOCK,
                "expected_stock_at_lead_time_end": round(expected_stock, 2),
                "alert_needed": True,
                "recommended_po_quantity": round(po_qty, 2),
                "message": alert_message
            })
        else:
            print(f"  -> Status: OK. No replenishment needed.")
            alerts.append({
                "timestamp": timestamp,
                "product_id": product_id,
                "current_stock": current_stock,
                "forecasted_demand": round(forecasted_demand, 2),
                "lead_time_days": DEFAULT_LEAD_TIME_DAYS,
                "safety_stock": DEFAULT_SAFETY_STOCK,
                "expected_stock_at_lead_time_end": round(expected_stock, 2),
                "alert_needed": False,
                "recommended_po_quantity": 0,
                "message": "Stock level is sufficient."
            })

    # 7. Save Alerts to Report
    if alerts:
        alerts_df = pd.DataFrame(alerts)
        report_filename = f"replenishment_alerts_{timestamp}.csv"
        report_filepath = os.path.join(ALERTS_REPORT_DIR, report_filename)
        alerts_df.to_csv(report_filepath, index=False)
        print(f"\n--- Alerts Generated ---")
        print(f"Total alerts checked: {len(alerts)}")
        num_alerts_triggered = len([a for a in alerts if a['alert_needed']])
        print(f"Alerts triggered: {num_alerts_triggered}")
        print(f"Full report saved to: {report_filepath}")
        
        # Also print triggered alerts to console
        print("\n--- Triggered Alerts ---")
        for alert in alerts:
            if alert['alert_needed']:
                print(alert['message'])
                print("-" * 20)
    else:
        print("\nNo inventory data found or no alerts generated.")

    print("\n" + "=" * 60)
    print("PREDICTIVE REPLENISHMENT ALERT GENERATION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    # Ensure the Flask API is running before executing this script
    generate_alerts()