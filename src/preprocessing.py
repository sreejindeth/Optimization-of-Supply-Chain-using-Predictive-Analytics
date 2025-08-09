import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os # Import os for directory creation

warnings.filterwarnings('ignore')

def load_data(file_path):
    """
    Load the supply chain dataset with proper encoding handling
    """
    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16']
    
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully loaded data with encoding: {encoding}")
            print(f"Dataset shape: {df.shape}")
            return df
        except UnicodeDecodeError:
            print(f"Failed to load with encoding: {encoding}")
            continue
        except Exception as e:
            print(f"Error loading data with {encoding}: {e}")
            continue
    
    print("Error: Could not load data with any of the attempted encodings")
    return None

def clean_data(df):
    """
    Clean and preprocess the supply chain data
    """
    print("Starting data cleaning process...")
    
    # Make a copy to avoid modifying original data
    df_clean = df.copy()
    
    # Handle missing values using scikit-learn's SimpleImputer best practices
    # (As discussed, median for numerical, mode for categorical)
    # Note: The logic here is similar to SimpleImputer's approach for univariate imputation.
    print("Handling missing values...")
    
    # For numerical columns, fill with median
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    for col in numerical_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)
    
    # For categorical columns, fill with mode
    categorical_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            # Mode can return multiple values, take the first one
            mode_value = df_clean[col].mode()
            if not mode_value.empty:
                df_clean[col].fillna(mode_value[0], inplace=True)
            else:
                # If mode is empty (e.g., all values are NaN), fill with a placeholder
                # This aligns with handling empty features conceptually
                print(f"Warning: All values missing in categorical column '{col}', filling with 'Unknown'.")
                df_clean[col].fillna('Unknown', inplace=True) # Or drop if appropriate

    
    # Convert date columns to datetime
    date_columns = ['order date (DateOrders)', 'Shipping date (DateOrders)']
    for col in date_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    # Create time-based features
    if 'order date (DateOrders)' in df_clean.columns:
        df_clean['order_date'] = df_clean['order date (DateOrders)']
        df_clean['order_year'] = df_clean['order_date'].dt.year
        df_clean['order_month'] = df_clean['order_date'].dt.month
        df_clean['order_day'] = df_clean['order_date'].dt.day
        # Encode weekday as number for easier use in models
        df_clean['order_weekday'] = df_clean['order_date'].dt.weekday # 0=Monday, 6=Sunday
        df_clean['order_quarter'] = df_clean['order_date'].dt.quarter
        df_clean['order_week'] = df_clean['order_date'].dt.isocalendar().week
    
    # Calculate delivery performance metrics
    if 'Days for shipping (real)' in df_clean.columns and 'Days for shipment (scheduled)' in df_clean.columns:
        df_clean['delivery_delay'] = df_clean['Days for shipping (real)'] - df_clean['Days for shipment (scheduled)']
        df_clean['delivery_delay_category'] = pd.cut(df_clean['delivery_delay'], 
                                                   bins=[-np.inf, -1, 1, np.inf], 
                                                   labels=['Early', 'On Time', 'Late'])
    
    # Calculate order value metrics
    if 'Sales' in df_clean.columns and 'Order Item Quantity' in df_clean.columns:
        # Add small value to avoid division by zero
        df_clean['avg_item_price'] = df_clean['Sales'] / (df_clean['Order Item Quantity'] + 1e-8)
        df_clean['sales_per_item'] = df_clean['Sales'] / (df_clean['Order Item Quantity'] + 1e-8)
    
    # Create customer-level features
    if 'Customer Id' in df_clean.columns:
        try:
            customer_stats = df_clean.groupby('Customer Id').agg({
                'Sales': ['count', 'sum', 'mean'],
                'Order Item Quantity': 'sum'
            }).reset_index()
            
            customer_stats.columns = ['Customer Id', 'total_orders', 'total_sales', 'avg_order_value', 'total_items']
            df_clean = df_clean.merge(customer_stats, on='Customer Id', how='left')
        except Exception as e:
            print(f"Warning: Could not create customer features: {e}")
    
    print(f"Data cleaning completed. Final shape: {df_clean.shape}")
    return df_clean

def aggregate_data(df):
    """
    Aggregate data for time series analysis.
    Ensures the date column is explicitly included in the output DataFrame.
    """
    print("Starting data aggregation...")
    
    if 'order_date' not in df.columns:
        print("Error: 'order_date' column not found in dataframe")
        return None

    # Daily aggregation for demand forecasting
    daily_demand = df.groupby(df['order_date'].dt.date).agg({
        'Sales': 'sum',
        'Order Item Quantity': 'sum',
        'Order Id': 'nunique' # Count unique orders per day
    }).rename(columns={'Order Id': 'num_orders'})

    # Crucial Step: Ensure the date is a column, not just the index
    # This makes saving/loading straightforward and prevents the corruption issue.
    daily_demand_with_date = daily_demand.reset_index()
    # Rename the date column appropriately
    daily_demand_with_date.rename(columns={'index': 'order_date'}, inplace=True)
    # If the groupby date column has a specific name, use it. Let's check:
    # The groupby key is df['order_date'].dt.date, the resulting index name might be None or 'date'.
    # reset_index() should create a column named 'date' or similar. Let's be explicit.
    # The most reliable way is to ensure the column name after reset_index.
    if daily_demand_with_date.index.name is None and 'index' in daily_demand_with_date.columns:
         # If reset_index created a column named 'index'
        daily_demand_with_date.rename(columns={'index': 'order_date'}, inplace=True)
    elif daily_demand_with_date.index.name is not None:
        # If the index had a name (e.g., 'order_date' or 'date')
        # reset_index() would turn it into a column with that name
        # We just need to make sure it's named 'order_date' consistently
        # Let's assume reset_index() worked, and the column might be named 'date' or similar
        # Check the columns and rename if necessary
        pass # We'll handle naming after reset

    # More robust way: Get the column name created by groupby and reset_index
    # Grouping by df['order_date'].dt.date should ideally result in a column named something like 'date'
    # after reset_index(). Let's make sure it's named 'order_date'.
    daily_demand_with_date = daily_demand.reset_index() # This creates a column from the index
    # Find the column that was the index (likely named something like 'date' or 'level_0' if unnamed)
    # The first column is usually the one created from the groupby key
    potential_date_col = daily_demand_with_date.columns[0]
    # Rename it to 'order_date' for consistency
    daily_demand_with_date.rename(columns={potential_date_col: 'order_date'}, inplace=True)

    # Ensure 'order_date' column is of datetime type
    daily_demand_with_date['order_date'] = pd.to_datetime(daily_demand_with_date['order_date'])

    # Sort by date to ensure chronological order
    daily_demand_with_date = daily_demand_with_date.sort_values('order_date').reset_index(drop=True)

    print(f"Data aggregation completed. Final shape: {daily_demand_with_date.shape}")
    # Print column names to verify
    print(f"Aggregated data columns: {list(daily_demand_with_date.columns)}")
    return daily_demand_with_date


def save_processed_data(df, file_path):
    """
    Save processed data to CSV, ensuring directories exist.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save DataFrame. If it has 'order_date' as a column, it will be saved.
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"Data saved successfully to {file_path}")
        print(f"Saved data shape: {df.shape}")
        if not df.empty:
             print(f"First few rows columns: {list(df.head().columns)}")
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")

# Example usage function
def process_supply_chain_data(input_file, output_file_main, output_file_ts):
    """
    Complete pipeline for processing supply chain data.
    Now saves both the main cleaned data and the aggregated time series data.
    """
    # Load data
    df = load_data(input_file)
    if df is None:
        print("Failed to load data. Exiting pipeline.")
        return None, None
    
    # Clean data
    print("\n--- Cleaning Data ---")
    df_clean = clean_data(df)
    if df_clean is None:
        print("Data cleaning failed. Exiting pipeline.")
        return None, None
    
    # Save cleaned data
    print("\n--- Saving Cleaned Data ---")
    save_processed_data(df_clean, output_file_main)
    
    # Aggregate data for time series
    print("\n--- Aggregating Data for Time Series ---")
    df_ts = aggregate_data(df_clean)
    if df_ts is None:
        print("Data aggregation failed.")
        return df_clean, None
    
    # Save aggregated time series data
    print("\n--- Saving Aggregated Time Series Data ---")
    save_processed_data(df_ts, output_file_ts)
    
    print("\n--- Preprocessing Pipeline Completed ---")
    return df_clean, df_ts

if __name__ == "__main__":
    # This can be used for testing the full pipeline
    print("Running preprocessing pipeline test...")
    main_data_file = 'data/raw/DataCoSupplyChainDataset.csv'
    cleaned_data_file = 'data/processed/cleaned_supply_chain_data.csv'
    ts_data_file = 'data/processed/daily_demand_data.csv' 

    if os.path.exists(main_data_file):
        df_cleaned, df_time_series = process_supply_chain_data(main_data_file, cleaned_data_file, ts_data_file)
        if df_cleaned is not None:
            print(f"\nMain data processing successful. Shape: {df_cleaned.shape}")
        if df_time_series is not None:
            print(f"Time series data processing successful. Shape: {df_time_series.shape}")
            print("Sample of time series data:")
            print(df_time_series.head())
    else:
        print(f"Input data file not found at {main_data_file}. Please check the path.")
    print("Preprocessing module test run completed.")