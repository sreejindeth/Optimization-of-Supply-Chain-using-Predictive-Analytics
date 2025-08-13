# src/diagnose_lstm.py
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load daily data
df = pd.read_csv("data/processed/daily_demand_data.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

y_true = df['delay_rate'].values

# 1. Baseline: Predict Yesterday's Value (Naive Forecast)
y_pred_naive = np.roll(y_true, 1)  # t-1 → t
y_pred_naive[0] = y_true[0]  # Fix first value

# 2. Baseline: Predict Rolling Mean (7-day)
y_pred_rm = df['delay_rate'].rolling(7, center=False).mean().fillna(method='bfill').values

# 3. Baseline: Predict Global Mean
y_pred_mean = np.full_like(y_true, y_true.mean())

# Metrics
def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    r2 = r2_score(y_true, y_pred)
    print(f"{name}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}\n")

evaluate(y_true, y_pred_naive, "📉 Naive Forecast (y_t = y_t-1)")
evaluate(y_true, y_pred_rm,    "📈 Rolling Mean (7-day)")
evaluate(y_true, y_pred_mean,  "🎯 Global Mean")