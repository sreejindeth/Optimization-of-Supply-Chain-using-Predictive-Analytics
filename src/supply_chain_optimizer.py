# src/supply_chain_optimizer.py
"""
Module containing optimization logic for the supply chain.
1. Dynamic Safety Stock Calculation
2. Intelligent Order Prioritization & Resource Allocation Engine
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
# Import your model utilities if needed for internal calls
# from src.model_utils import load_lstm_components, load_xgboost_model, predict_demand_lstm, predict_risk_xgboost

# --- Configuration (Could be moved to a config file) ---
HIGH_RISK_THRESHOLD = 0.7
MEDIUM_RISK_THRESHOLD = 0.4
HIGH_VOLUME_THRESHOLD_FACTOR = 1.2 # 20% above average predicted volume is "High Volume"
PRIORITY_BOOST_WEIGHT = 0.2 # Weight for workload context boost
BASELINE_VOLUME_WINDOW_DAYS = 30 # Window to calculate average baseline volume

# --- 1. Intelligent Order Prioritization & Resource Allocation Engine ---

class OrderPriorityEngine:
    """
    Calculates a unified priority score for orders based on risk and predicted workload.
    """

    def __init__(self, baseline_volume_data_path=None):
        """
        Initializes the engine.
        Args:
            baseline_volume_data_path (str, optional): Path to historical volume data
                (e.g., daily_demand_data.csv) to calculate baseline average volume.
                If None, a simple average from recent predictions might be used.
        """
        self.baseline_volume_data_path = baseline_volume_data_path
        self.baseline_average_daily_orders = self._calculate_baseline_volume()

    def _calculate_baseline_volume(self):
        """
        Calculates the baseline average daily order volume.
        This could be a simple average or a moving average.
        """
        baseline_avg = 500 # Default placeholder
        if self.baseline_volume_data_path and os.path.exists(self.baseline_volume_data_path):
            try:
                # Load historical daily demand data
                df_baseline = pd.read_csv(self.baseline_volume_data_path)
                df_baseline['order_date'] = pd.to_datetime(df_baseline['order_date'])
                df_baseline.set_index('order_date', inplace=True)
                df_baseline.sort_index(inplace=True)

                # Calculate average 'num_orders' over the defined window
                # Or a simple overall average if the window is too long for the data
                if 'num_orders' in df_baseline.columns:
                    baseline_avg = df_baseline['num_orders'].mean() # Simple average
                    # baseline_avg = df_baseline['num_orders'].rolling(window=BASELINE_VOLUME_WINDOW_DAYS).mean().iloc[-1] # Recent MA (needs enough data)
                    print(f"Calculated baseline average daily orders: {baseline_avg:.2f}")
                else:
                    print(f"Warning: 'num_orders' column not found in {self.baseline_volume_data_path}. Using default baseline.")
            except Exception as e:
                print(f"Warning: Could not calculate baseline volume from {self.baseline_volume_data_path}: {e}. Using default.")
        else:
            print(f"Warning: Baseline volume data path not provided or not found. Using default baseline of {baseline_avg}.")
        return baseline_avg

    def get_base_classification(self, risk_score):
        """
        Classifies order risk based on XGBoost prediction.
        """
        if risk_score > HIGH_RISK_THRESHOLD:
            return "HIGH RISK"
        elif risk_score > MEDIUM_RISK_THRESHOLD:
            return "MEDIUM RISK"
        else:
            return "LOW RISK"

    def is_high_volume_period(self, predicted_volume_for_period, period_days=1):
        """
        Determines if a predicted period is high volume based on baseline.
        """
        # Average daily volume for the period
        avg_daily_volume = predicted_volume_for_period / period_days
        is_high_vol = avg_daily_volume > (self.baseline_average_daily_orders * HIGH_VOLUME_THRESHOLD_FACTOR)
        if is_high_vol:
            print(f"Detected High Volume Period: Avg Daily Vol ({avg_daily_volume:.2f}) > Baseline ({self.baseline_average_daily_orders * HIGH_VOLUME_THRESHOLD_FACTOR:.2f})")
        return is_high_vol

    def calculate_priority_score(self, xgboost_risk_score, lstm_predicted_volume_for_shipment_date, shipment_date_window_days=1):
        """
        Calculates the unified priority score for an order.

        Args:
            xgboost_risk_score (float): Risk score from XGBoost model (0 to 1).
            lstm_predicted_volume_for_shipment_date (float): Predicted total orders for the shipment date/day.
            shipment_date_window_days (int): Number of days the volume prediction covers (default 1 for daily).

        Returns:
            dict: Contains 'priority_score', 'base_classification', 'contextual_classification', 'final_tier'.
        """
        base_classification = self.get_base_classification(xgboost_risk_score)
        print(f"Base Risk Classification: {base_classification} (Score: {xgboost_risk_score:.4f})")

        is_high_vol_context = self.is_high_volume_period(lstm_predicted_volume_for_shipment_date, shipment_date_window_days)
        
        # --- Calculate Unified Priority Score ---
        contextual_boost = PRIORITY_BOOST_WEIGHT if is_high_vol_context and base_classification != "LOW RISK" else 0.0
        # For MEDIUM RISK in high volume, maybe a smaller boost
        if is_high_vol_context and base_classification == "MEDIUM RISK":
             contextual_boost = PRIORITY_BOOST_WEIGHT / 2.0 # Half boost for medium risk

        priority_score = min(1.0, xgboost_risk_score + contextual_boost) # Cap at 1.0
        print(f"Priority Score Calculation: Base ({xgboost_risk_score:.4f}) + Boost ({contextual_boost:.4f}) = {priority_score:.4f}")

        # --- Determine Final Tier ---
        if priority_score > HIGH_RISK_THRESHOLD:
            final_tier = "CRITICAL" # Highest priority
            contextual_classification = f"{base_classification} (Elevated due to High Volume Context)"
        elif priority_score > MEDIUM_RISK_THRESHOLD:
            final_tier = "HIGH"
            if is_high_vol_context:
                contextual_classification = f"{base_classification} (Elevated due to High Volume Context)"
            else:
                contextual_classification = base_classification
        else:
            final_tier = "STANDARD"
            contextual_classification = base_classification

        return {
            "priority_score": round(priority_score, 4),
            "base_classification": base_classification,
            "contextual_classification": contextual_classification,
            "final_tier": final_tier,
            "is_high_volume_context": is_high_vol_context,
            "contextual_boost_applied": contextual_boost > 0
        }

    def get_recommended_actions(self, final_tier):
        """
        Maps the final priority tier to recommended actions.
        """
        actions_map = {
            "CRITICAL": [
                "PRIORITIZE_ORDER_IN_PACKING_QUEUE",
                "CONSIDER_EXPEDITED_SHIPPING_OPTION",
                "ASSIGN_TO_EXPERIENCED_STAFF",
                "FLAG_FOR_PROACTIVE_CUSTOMER_COMMUNICATION"
            ],
            "HIGH": [
                "PRIORITY_HANDLING_IN_WAREHOUSE",
                "MONITOR_PROGRESS_CLOSLEY",
                "ALLOCATE_DEDICATED_PACKING_SLOT_IF_POSSIBLE"
            ],
            "STANDARD": [
                "STANDARD_PROCESSING"
            ]
        }
        return actions_map.get(final_tier, ["STANDARD_PROCESSING"]) # Default action


# --- Placeholder for Dynamic Safety Stock Calculation Logic ---
# We will implement this after we have a better understanding of how to structure
# the data flow for getting LSTM forecasts and calculating variances.
# This might involve more complex data aggregation and historical analysis.

def calculate_dynamic_safety_stock(avg_demand, demand_std, avg_lead_time, lead_time_std, service_level=0.95):
    """
    Placeholder function for Dynamic Safety Stock calculation.
    This would be fleshed out in Phase 2.
    Formula: SS = Z * sqrt((L * σ_D²) + (D² * σ_L²))
    """
    from scipy import stats
    z_score = stats.norm.ppf(service_level)
    ss = z_score * np.sqrt((avg_lead_time * demand_std**2) + (avg_demand**2 * lead_time_std**2))
    return ss

# --- Example/Demo Function ---
def demo_priority_engine():
    """
    Demonstrates how the OrderPriorityEngine could be used.
    This would typically be called by the API or another process.
    """
    print("\n--- Demo: Intelligent Order Prioritization Engine ---")
    
    # Simulate loading the engine with baseline data
    engine = OrderPriorityEngine(baseline_volume_data_path='data/processed/daily_demand_data.csv')
    
    # --- Simulate Inputs (from API predictions) ---
    print("\n--- Simulating API Prediction Inputs ---")
    simulated_xgboost_risk_score = 0.75 # High risk order
    simulated_lstm_predicted_orders_for_shipment_day = 650.0 # High predicted volume for that day
    simulated_shipment_date = datetime.today().date() + timedelta(days=2) # Shipment in 2 days
    
    print(f"Simulated Order XGBoost Risk Score: {simulated_xgboost_risk_score}")
    print(f"Simulated LSTM Predicted Orders for {simulated_shipment_date}: {simulated_lstm_predicted_orders_for_shipment_day}")
    
    # --- Calculate Priority ---
    print("\n--- Calculating Priority ---")
    priority_result = engine.calculate_priority_score(
        simulated_xgboost_risk_score,
        simulated_lstm_predicted_orders_for_shipment_day,
        shipment_date_window_days=1 # Assuming daily prediction
    )
    
    print("\n--- Priority Calculation Result ---")
    for key, value in priority_result.items():
        print(f"  {key}: {value}")
    
    # --- Get Actions ---
    print("\n--- Determining Recommended Actions ---")
    actions = engine.get_recommended_actions(priority_result['final_tier'])
    print(f"Recommended Actions for Tier '{priority_result['final_tier']}':")
    for action in actions:
        print(f"  - {action}")

if __name__ == '__main__':
    demo_priority_engine()
    print("\n--- End of Demo ---")
