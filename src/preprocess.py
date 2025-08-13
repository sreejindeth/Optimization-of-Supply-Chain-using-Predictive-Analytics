# src/preprocess.py
"""
Supply Chain Data Preprocessing Pipeline
- Cleans raw data
- Creates features for XGBoost and LSTM
- Outputs:
  1. cleaned_supply_chain_data.csv → for XGBoost
  2. daily_demand_data.csv → for LSTM (clean delay_rate)
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def load_data(file_path, encodings=['latin-1', 'utf-8']):
    """Load data with fallback encodings"""
    print("🚀 Starting preprocessing pipeline...")
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            print(f"✅ Successfully loaded data with encoding: {enc}")
            return df
        except Exception as e:
            print(f"❌ Failed with encoding: {enc}")
    raise ValueError("Could not load CSV with any encoding")

def clean_data(df):
    """Clean and enrich the dataset"""
    print("🧹 Starting data cleaning and enrichment...")
    
    # Strip column names
    df.columns = df.columns.str.strip()
    
    # --- 1. Handle Missing Values ---
    print("🔧 Handling missing values...")
    df = df.fillna({
        'Customer Fname': 'Unknown',
        'Customer Lname': 'Unknown',
        'Customer Segment': 'Consumer',
        'Product Status': 0,
        'Order Region': 'Unknown',
        'Market': 'Unknown'
    })
    
    # --- 2. Convert Date Columns ---
    print("📅 Converting date columns...")
    date_cols = ['order date (DateOrders)', 'shipping date (DateOrders)']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Create order_date and shipping_date
    df['order_date'] = df['order date (DateOrders)'].copy()
    df['shipping_date'] = df['shipping date (DateOrders)'].copy()
    
    # Drop original date columns
    df.drop(columns=date_cols, errors='ignore', inplace=True)
    
    # --- 3. Calculate Delivery Metrics ---
    print("🚚 Calculating delivery metrics...")
    # Create is_delayed from Delivery Status
    df['is_delayed'] = (df['Delivery Status'] == 'Late delivery').astype(int)
    
    # Calculate delivery delay in days
    df['delivery_delay'] = (df['shipping_date'] - df['order_date']).dt.days
    df['delivery_delay'] = df['delivery_delay'].fillna(0).astype(int)
    
    # --- 4. Create Customer and Product Features ---
    print("📊 Creating customer and product features...")
    
    # Customer-level stats
    if 'Customer Id' in df.columns:
        cust_stats = df.groupby('Customer Id').agg({
            'Sales': ['count', 'sum', 'mean'],
            'Order Item Quantity': 'sum'
        }).reset_index()
        cust_stats.columns = ['Customer Id', 'total_orders', 'total_sales', 'avg_order_value', 'total_items']
        df = df.merge(cust_stats, on='Customer Id', how='left')
    
    # Product-level stats
    if 'Product Name' in df.columns:
        prod_stats = df.groupby('Product Name').agg({
            'Sales': 'sum',
            'Order Item Quantity': 'sum'
        }).reset_index()
        prod_stats.rename(columns={'Sales': 'total_sales_product'}, inplace=True)
        df = df.merge(prod_stats, on='Product Name', how='left')
    
    # --- 5. Add Geospatial Features ---
    print("🌍 Adding geospatial features...")
    # Mark international orders
    domestic_markets = ['USCA', 'LATAM', 'Pacific Asia']  # Adjust based on your data
    df['is_international'] = (~df['Market'].isin(['USCA'])).astype(int)
    
    # --- 6. Encode Categorical Variables ---
    print("🏷️ Encoding categorical variables...")
    cat_cols = [
        'Shipping Mode', 'Market', 'Order Region', 'Category Name',
        'Customer Segment', 'Product Status', 'Type'
    ]
    
    for col in cat_cols:
        if col in df.columns:
            df[f"{col}_enc"] = df[col].astype('category').cat.codes
    
    # --- 7. Prepare for Cold-Start Scenarios ---
    print("🧊 Preparing for cold-start scenarios...")
    # Fill any remaining NaN in encoded columns
    enc_cols = [c for c in df.columns if c.endswith('_enc')]
    df[enc_cols] = df[enc_cols].fillna(-1)
    
    # --- 8. Create Time Features ---
    df['order_year'] = df['order_date'].dt.year
    df['order_month'] = df['order_date'].dt.month
    df['order_day'] = df['order_date'].dt.day
    df['order_weekday'] = df['order_date'].dt.dayofweek
    df['order_week'] = df['order_date'].dt.isocalendar().week.astype(int)
    
    # Average item price
    df['avg_item_price'] = df['Sales'] / df['Order Item Quantity']
    df['avg_item_price'].fillna(0, inplace=True)
    
    print(f"✅ Cleaning completed. Final shape: {df.shape}")
    return df

def aggregate_data(df):
    """Aggregate data for LSTM: daily delay rate"""
    print("📈 Aggregating data for LSTM...")
    
    if 'order_date' not in df.columns:
        raise ValueError("❌ 'order_date' column not found")
    
    # Ensure order_date is datetime
    df['order_date'] = pd.to_datetime(df['order_date'])
    
    # Group by day
    daily_agg = df.groupby(df['order_date'].dt.date).agg({
        'is_delayed': 'mean',  # % of orders delayed
        'Sales': 'sum',
        'Order Item Quantity': 'sum',
        'Order Id': 'nunique'
    }).reset_index()
    
    # Rename
    daily_agg.columns = ['date', 'delay_rate', 'total_sales', 'total_quantity', 'num_orders']
    
    # Clip delay_rate to [0, 1] (should already be)
    daily_agg['delay_rate'] = daily_agg['delay_rate'].clip(0, 1)
    
    # Ensure date is datetime
    daily_agg['date'] = pd.to_datetime(daily_agg['date'])
    daily_agg = daily_agg.sort_values('date').reset_index(drop=True)
    
    print(f"✅ Aggregation completed. Shape: {daily_agg.shape}")
    return daily_agg

def process_supply_chain_data(input_file, output_main, output_ts):
    """Main pipeline"""
    # Load
    df = load_data(input_file)
    
    # Clean
    df_clean = clean_data(df)
    
    # Save cleaned data
    os.makedirs(os.path.dirname(output_main), exist_ok=True)
    df_clean.to_csv(output_main, index=False)
    print(f"💾 Saved to {output_main}")
    
    # Aggregate for LSTM
    df_ts = aggregate_data(df_clean)
    
    # Save time-series data
    os.makedirs(os.path.dirname(output_ts), exist_ok=True)
    df_ts.to_csv(output_ts, index=False)
    print(f"💾 Saved to {output_ts}")
    
    print("🎉 Preprocessing pipeline completed!")
    return df_clean, df_ts

if __name__ == "__main__":
    input_file = "data/raw/supply_chain_data.csv"
    output_main = "data/processed/cleaned_supply_chain_data.csv"
    output_ts = "data/processed/daily_demand_data.csv"
    
    try:
        df_cleaned, df_time_series = process_supply_chain_data(input_file, output_main, output_ts)
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")