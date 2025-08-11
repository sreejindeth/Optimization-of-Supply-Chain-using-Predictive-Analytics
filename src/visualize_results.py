# src/visualize_results.py
"""
Visualization of Modeling Results for Hybrid LSTM-XGBoost Model
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Ensure output directory exists ---
def create_output_dirs():
    """Create necessary directories for visualization outputs."""
    dirs = ['results/visualization']
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
        print(f"Ensured directory exists: {dir}")

# --- 1. Visualize XGBoost Feature Importances ---
def plot_xgboost_feature_importance(top_n=20):
    """
    Plot the top N feature importances from the XGBoost model.
    """
    print("--- Plotting XGBoost Feature Importances ---")
    try:
        # Load feature importances
        importance_file = 'results/model/xgboost_late_delivery_risk_feature_importance.csv'
        if not os.path.exists(importance_file):
            print(f"Warning: Feature importance file not found at {importance_file}")
            return

        importance_df = pd.read_csv(importance_file)
        
        if importance_df.empty:
            print("Warning: Feature importance data is empty.")
            return

        # Sort and get top N
        importance_df = importance_df.sort_values(by='importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance_df, y='feature', x='importance', palette='viridis')
        plt.title(f'Top {top_n} XGBoost Feature Importances for Late Delivery Risk')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plot_path = 'results/visualization/xgboost_feature_importance.png'
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"XGBoost feature importance plot saved to {plot_path}")

    except Exception as e:
        print(f"Error plotting XGBoost feature importances: {e}")

# --- 2. Visualize LSTM Demand Forecast (if data is available) ---
def plot_lstm_forecast():
    """
    Plot actual vs predicted demand from LSTM model (conceptual).
    This requires loading the model and making predictions on test data,
    or having saved the predictions. For now, we'll just check for saved metrics.
    """
    print("--- Checking LSTM Results ---")
    try:
        # Load LSTM metrics
        metrics_file = 'results/model/lstm_demand_forecast_sales_metrics.csv'
        if not os.path.exists(metrics_file):
            print(f"Warning: LSTM metrics file not found at {metrics_file}")
            return

        metrics_df = pd.read_csv(metrics_file)
        print("LSTM Metrics:")
        print(metrics_df.to_string(index=False))
        # Note: Plotting actual vs predicted time series would require
        # loading the model, scaler, test data, and making predictions.
        # This is more involved. For now, we report the metrics.

    except Exception as e:
        print(f"Error checking LSTM results: {e}")

# --- 3. Visualize Model Performance Metrics ---
def plot_model_performance_comparison():
    """
    Compare key performance metrics of LSTM and XGBoost.
    """
    print("--- Plotting Model Performance Comparison ---")
    try:
        # Load metrics
        xgb_metrics_file = 'results/model/xgboost_late_delivery_risk_metrics.csv'
        lstm_metrics_file = 'results/model/lstm_demand_forecast_sales_metrics.csv'
        
        metrics_data = {}
        
        if os.path.exists(xgb_metrics_file):
            xgb_metrics = pd.read_csv(xgb_metrics_file)
            # Pivot for easier plotting (Metric, Value)
            xgb_metrics_pivot = xgb_metrics.set_index('Metric')['Value']
            metrics_data['XGBoost (Late Delivery Risk)'] = xgb_metrics_pivot
        else:
            print(f"Warning: XGBoost metrics file not found at {xgb_metrics_file}")

        # Note: LSTM metrics are error metrics (RMSE, MAE) which are on a different scale
        # than classification metrics (Accuracy, Precision, etc.). Comparing them directly
        # on a bar chart isn't very meaningful. We can list them or plot them separately.
        # For classification-like summary, we focus on XGBoost for now.
        # If you want to visualize LSTM error metrics, that's a separate plot.
        
        if metrics_data:
            # Combine metrics into a single DataFrame for plotting
            combined_metrics_df = pd.DataFrame(metrics_data)
            # Only include metrics that make sense to compare (if any)
            # For now, just XGBoost metrics are plotted.
            if 'XGBoost (Late Delivery Risk)' in combined_metrics_df.columns:
                xgb_plot_data = combined_metrics_df['XGBoost (Late Delivery Risk)'].dropna()
                if not xgb_plot_data.empty:
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x=xgb_plot_data.index, y=xgb_plot_data.values, palette='Set2')
                    plt.title('XGBoost Model Performance Metrics (Late Delivery Risk Prediction)')
                    plt.ylabel('Score')
                    plt.xlabel('Metric')
                    plt.ylim(0, 1.1) # Assuming metrics are between 0 and 1
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plot_path = 'results/visualization/xgboost_performance_metrics.png'
                    plt.savefig(plot_path, dpi=300)
                    plt.close()
                    print(f"XGBoost performance metrics plot saved to {plot_path}")
                else:
                    print("Warning: No valid XGBoost metrics to plot.")

    except Exception as e:
        print(f"Error plotting model performance comparison: {e}")

# --- 4. (Optional/Advanced) Load and Visualize Actual vs Predicted for LSTM ---
# This is more complex and requires reloading the model, scaler, and test data.
# It's a good next step after basic visualization.

# --- Main Execution ---
def main():
    """Main function to run the visualization pipeline."""
    print("="*50)
    print("VISUALIZATION OF MODELING RESULTS")
    print("="*50)

    create_output_dirs()

    # --- Visualize XGBoost Feature Importances ---
    plot_xgboost_feature_importance(top_n=20)

    # --- Check/Report LSTM Results ---
    plot_lstm_forecast()

    # --- Visualize Model Performance ---
    plot_model_performance_comparison()

    print("\n" + "="*50)
    print("VISUALIZATION PIPELINE COMPLETED")
    print("="*50)
    print("\nNext steps:")
    print("- Review plots in 'results/visualization/'")
    print("- Analyze the key features driving late delivery risk predictions")
    print("- (Optional) Implement advanced visualizations (LSTM actual vs predicted)")

if __name__ == "__main__":
    main()