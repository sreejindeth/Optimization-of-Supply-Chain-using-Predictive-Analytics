import pandas as pd
# Verify cleaned_supply_chain_data.csv
df_clean = pd.read_csv("data/processed/cleaned_supply_chain_data.csv")
print(df_clean.head())
print(df_clean.describe())

# Verify daily_demand_data.csv
df_daily = pd.read_csv("data/processed/daily_demand_data.csv")
print(df_daily.head())
print(df_daily.describe())
