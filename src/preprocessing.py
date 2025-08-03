# src/preprocessing.py
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
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
    
    # Handle missing values
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
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
    
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
        df_clean['order_weekday'] = df_clean['order_date'].dt.day_name()
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
        df_clean['avg_item_price'] = df_clean['Sales'] / (df_clean['Order Item Quantity'] + 1e-8)  # Add small value to avoid division by zero
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
    Aggregate data for time series analysis
    """
    print("Starting data aggregation...")
    
    if 'order_date' not in df.columns:
        print("Error: 'order_date' column not found in dataframe")
        return None
    
    # Daily aggregation for demand forecasting
    daily_demand = df.groupby(df['order_date'].dt.date).agg({
        'Sales': 'sum',
        'Order Item Quantity': 'sum',
        'Order Id': 'nunique'
    }).rename(columns={'Order Id': 'num_orders'})
    
    daily_demand.index = pd.to_datetime(daily_demand.index)
    daily_demand = daily_demand.sort_index()
    
    print(f"Data aggregation completed. Final shape: {daily_demand.shape}")
    return daily_demand

def save_processed_data(df, file_path):
    """
    Save processed data to CSV
    """
    try:
        # Ensure the directory exists
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        df.to_csv(file_path, index=False, encoding='utf-8')
        print(f"Data saved successfully to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")

# Example usage function
def process_supply_chain_data(input_file, output_file):
    """
    Complete pipeline for processing supply chain data
    """
    # Load data
    df = load_data(input_file)
    if df is None:
        return None
    
    # Clean data
    df_clean = clean_data(df)
    
    # Save cleaned data
    save_processed_data(df_clean, output_file)
    
    return df_clean

if __name__ == "__main__":
    # This can be used for testing
    print("Preprocessing module loaded successfully!")