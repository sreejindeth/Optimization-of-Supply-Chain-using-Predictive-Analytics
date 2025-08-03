# src/utils.py
import pandas as pd
import numpy as np
import os

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
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def check_data_quality(df):
    """
    Check data quality and print summary statistics
    """
    print("="*50)
    print("DATA QUALITY REPORT")
    print("="*50)
    print(f"Dataset shape: {df.shape}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    
    if 'order_date' in df.columns:
        print(f"Date range: {df['order_date'].min()} to {df['order_date'].max()}")
    
    if 'Customer Id' in df.columns:
        print(f"Unique customers: {df['Customer Id'].nunique()}")
    
    if 'Order Id' in df.columns:
        print(f"Total orders: {df['Order Id'].nunique()}")

if __name__ == "__main__":
    print("Utils module loaded successfully!")