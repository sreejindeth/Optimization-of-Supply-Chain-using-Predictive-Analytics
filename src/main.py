# src/main.py
import os
import pandas as pd
import sys

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import load_data, clean_data, aggregate_data, save_processed_data

def create_directories():
    """
    Create necessary directories for the project
    """
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'results/eda',
        'results/predictions',
        'results/performance'
    ]
    
    for directory in directories:
        full_path = os.path.join(os.path.dirname(__file__), '..', directory)
        os.makedirs(full_path, exist_ok=True)
        print(f"Ensured directory exists: {full_path}")

def main():
    print("Starting supply chain data preprocessing pipeline...")
    
    # Create directories if they don't exist
    create_directories()
    
    # Define file paths
    raw_data_path = os.path.join('data', 'raw', 'DataCoSupplyChainDataset.csv')
    processed_data_path = os.path.join('data', 'processed', 'cleaned_supply_chain_data.csv')
    aggregated_data_path = os.path.join('data', 'processed', 'daily_demand_data.csv')
    
    # Check if raw data file exists
    if not os.path.exists(raw_data_path):
        print(f"Error: Raw data file not found at {raw_data_path}")
        print("Please make sure your data files are in the correct location.")
        print("Current working directory:", os.getcwd())
        return
    
    # Load raw data
    print("Loading raw data...")
    df = load_data(raw_data_path)
    if df is None:
        return
    
    # Preprocess data
    print("Cleaning data...")
    cleaned_df = clean_data(df)
    
    # Save processed data
    print("Saving processed data...")
    save_processed_data(cleaned_df, processed_data_path)
    
    # Aggregate data for time series analysis
    print("Aggregating data for time series analysis...")
    aggregated_df = aggregate_data(cleaned_df)
    
    if aggregated_df is not None:
        # Save aggregated data
        save_processed_data(aggregated_df, aggregated_data_path)
        print(f"Aggregated data saved to {aggregated_data_path}")
    
    print("Preprocessing pipeline completed successfully!")

if __name__ == "__main__":
    main()