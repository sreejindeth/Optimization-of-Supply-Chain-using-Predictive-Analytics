import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import joblib

# Load cleaned data
df = pd.read_csv("data/processed/cleaned_supply_chain_data.csv")

# Define features and target
target = 'Late_delivery_risk'
# Or use 'is_delayed' if 'Late_delivery_risk' doesn't exist
if 'Late_delivery_risk' not in df.columns:
    target = 'is_delayed'

features = [
    'Sales', 'Order Item Quantity_x', 'Product Price',
    'Shipping Mode_enc', 'Market_enc', 'Order Region_enc',
    'Category Name_enc', 'Customer Segment_enc', 'Product Status_enc',
    'delivery_delay', 'avg_item_price', 'total_orders', 'total_sales',
    'is_international', 'order_month', 'order_weekday'
]

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("XGBoost Model Performance")
print(classification_report(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_pred_proba))

# Save model
joblib.dump(model, "models/xgboost_model.pkl")
print("✅ XGBoost model saved to models/xgboost_model.pkl")