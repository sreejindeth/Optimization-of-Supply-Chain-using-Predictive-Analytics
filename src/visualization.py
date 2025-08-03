# src/visualization.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

def load_processed_data():
    """Load the processed data for visualization"""
    try:
        df = pd.read_csv('data/processed/cleaned_supply_chain_data.csv')
        df['order_date'] = pd.to_datetime(df['order_date'])
        return df
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return None

def load_daily_demand_data():
    """Load the aggregated daily demand data"""
    try:
        df = pd.read_csv('data/processed/daily_demand_data.csv')
        df['order_date'] = pd.to_datetime(df['order_date'])
        df.set_index('order_date', inplace=True)
        return df
    except Exception as e:
        print(f"Error loading daily demand data: {e}")
        return None

def plot_sales_distribution(df):
    """Plot sales distribution"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['Sales'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Distribution of Sales')
    plt.xlabel('Sales Amount')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(np.log1p(df['Sales']), bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Distribution of Log(Sales + 1)')
    plt.xlabel('Log(Sales + 1)')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('results/eda/sales_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_time_series_analysis():
    """Plot time series analysis of daily demand"""
    df = load_daily_demand_data()
    if df is None:
        return
    
    plt.figure(figsize=(15, 12))
    
    # Daily Sales
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['Sales'], linewidth=1, color='blue')
    plt.title('Daily Sales Over Time')
    plt.ylabel('Sales')
    plt.grid(True, alpha=0.3)
    
    # Daily Order Quantity
    plt.subplot(3, 1, 2)
    plt.plot(df.index, df['Order Item Quantity'], linewidth=1, color='green')
    plt.title('Daily Order Quantity Over Time')
    plt.ylabel('Quantity')
    plt.grid(True, alpha=0.3)
    
    # Daily Number of Orders
    plt.subplot(3, 1, 3)
    plt.plot(df.index, df['num_orders'], linewidth=1, color='red')
    plt.title('Daily Number of Orders Over Time')
    plt.ylabel('Number of Orders')
    plt.xlabel('Date')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/eda/time_series_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_category_analysis(df):
    """Plot category-wise analysis"""
    if 'Category Name' not in df.columns:
        print("Category Name column not found")
        return
    
    plt.figure(figsize=(15, 10))
    
    # Top categories by sales
    plt.subplot(2, 2, 1)
    category_sales = df.groupby('Category Name')['Sales'].sum().sort_values(ascending=False).head(10)
    sns.barplot(x=category_sales.values, y=category_sales.index, palette='viridis')
    plt.title('Top 10 Categories by Total Sales')
    plt.xlabel('Total Sales')
    
    # Top categories by order count
    plt.subplot(2, 2, 2)
    category_orders = df.groupby('Category Name')['Order Id'].nunique().sort_values(ascending=False).head(10)
    sns.barplot(x=category_orders.values, y=category_orders.index, palette='plasma')
    plt.title('Top 10 Categories by Number of Orders')
    plt.xlabel('Number of Orders')
    
    # Delivery status distribution
    plt.subplot(2, 2, 3)
    if 'Delivery Status' in df.columns:
        delivery_status = df['Delivery Status'].value_counts()
        plt.pie(delivery_status.values, labels=delivery_status.index, autopct='%1.1f%%', startangle=90)
        plt.title('Delivery Status Distribution')
    
    # Late delivery risk
    plt.subplot(2, 2, 4)
    if 'Late_delivery_risk' in df.columns:
        late_risk = df['Late_delivery_risk'].value_counts()
        bars = plt.bar(['On Time', 'Late'], [late_risk.get(0, 0), late_risk.get(1, 0)], 
                      color=['green', 'red'], alpha=0.7)
        plt.title('Late Delivery Risk Distribution')
        plt.ylabel('Number of Orders')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('results/eda/category_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def plot_seasonal_analysis(df):
    """Plot seasonal and temporal analysis"""
    if 'order_month' not in df.columns:
        print("Time-based columns not found")
        return
    
    plt.figure(figsize=(15, 10))
    
    # Monthly sales trend
    plt.subplot(2, 2, 1)
    monthly_sales = df.groupby('order_month')['Sales'].sum()
    plt.plot(monthly_sales.index, monthly_sales.values, marker='o', linewidth=2, markersize=8)
    plt.title('Monthly Sales Trend')
    plt.xlabel('Month')
    plt.ylabel('Total Sales')
    plt.grid(True, alpha=0.3)
    
    # Weekly pattern
    plt.subplot(2, 2, 2)
    if 'order_weekday' in df.columns:
        weekday_sales = df.groupby('order_weekday')['Sales'].sum()
        # Reorder weekdays
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_sales = weekday_sales.reindex(weekdays)
        plt.bar(range(len(weekday_sales)), weekday_sales.values, color='orange', alpha=0.7)
        plt.title('Weekly Sales Pattern')
        plt.xlabel('Day of Week')
        plt.ylabel('Total Sales')
        plt.xticks(range(len(weekdays)), [day[:3] for day in weekdays])
    
    # Shipping mode analysis
    plt.subplot(2, 2, 3)
    if 'Shipping Mode' in df.columns:
        shipping_sales = df.groupby('Shipping Mode')['Sales'].sum().sort_values(ascending=False)
        sns.barplot(x=shipping_sales.values, y=shipping_sales.index, palette='coolwarm')
        plt.title('Sales by Shipping Mode')
        plt.xlabel('Total Sales')
    
    # Customer segment analysis
    plt.subplot(2, 2, 4)
    if 'Customer Segment' in df.columns:
        segment_sales = df.groupby('Customer Segment')['Sales'].sum().sort_values(ascending=False)
        plt.pie(segment_sales.values, labels=segment_sales.index, autopct='%1.1f%%', startangle=90)
        plt.title('Sales by Customer Segment')
    
    plt.tight_layout()
    plt.savefig('results/eda/seasonal_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def generate_summary_report(df):
    """Generate a summary report of the data"""
    print("="*60)
    print("SUPPLY CHAIN DATA SUMMARY REPORT")
    print("="*60)
    
    print(f"Dataset Period: {df['order_date'].min()} to {df['order_date'].max()}")
    print(f"Total Days: {(df['order_date'].max() - df['order_date'].min()).days}")
    print(f"Total Orders: {df['Order Id'].nunique():,}")
    print(f"Total Customers: {df['Customer Id'].nunique():,}")
    print(f"Total Sales: ${df['Sales'].sum():,.2f}")
    print(f"Average Order Value: ${df['Sales'].mean():.2f}")
    print(f"Total Products Sold: {df['Order Item Quantity'].sum():,}")
    
    if 'Category Name' in df.columns:
        print(f"Product Categories: {df['Category Name'].nunique()}")
    
    if 'Order Region' in df.columns:
        print(f"Regions Covered: {df['Order Region'].nunique()}")
    
    if 'Late_delivery_risk' in df.columns:
        late_percentage = (df['Late_delivery_risk'].sum() / len(df)) * 100
        print(f"Late Delivery Rate: {late_percentage:.2f}%")
    
    print("="*60)

def main():
    """Main function to run all visualizations"""
    print("Starting EDA visualization pipeline...")
    
    # Load data
    df = load_processed_data()
    if df is None:
        return
    
    # Generate summary report
    generate_summary_report(df)
    
    # Create visualizations
    print("Creating sales distribution plot...")
    plot_sales_distribution(df)
    
    print("Creating time series analysis...")
    plot_time_series_analysis()
    
    print("Creating category analysis...")
    plot_category_analysis(df)
    
    print("Creating seasonal analysis...")
    plot_seasonal_analysis(df)
    
    print("EDA pipeline completed successfully!")
    print("Visualizations saved to results/eda/ directory")

if __name__ == "__main__":
    main()