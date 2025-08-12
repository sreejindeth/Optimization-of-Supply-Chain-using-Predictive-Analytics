# src/visualize_xgboost_performance.py
"""
Visualization of XGBoost Model Performance for Late Delivery Risk Prediction.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import joblib
from src.model_utils import load_xgboost_model # Assuming this function exists and works

def create_output_dirs():
    """Create necessary directories for visualization outputs."""
    dirs = ['results/visualization']
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)
        print(f"Ensured directory exists: {dir}")

def plot_confusion_matrix(y_true, y_pred, model_name='XGBoost'):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['On Time', 'Late'], 
                yticklabels=['On Time', 'Late'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plot_path = f'results/visualization/{model_name.lower()}_confusion_matrix.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Confusion matrix saved to {plot_path}")

def plot_roc_curve(y_true, y_pred_proba, model_name='XGBoost'):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = f'results/visualization/{model_name.lower()}_roc_curve.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"ROC curve saved to {plot_path}")

def plot_feature_importance(importance_df, model_name='XGBoost', top_n=20):
    """Plot feature importance."""
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
    plt.title(f'Top {top_n} {model_name} Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plot_path = f'results/visualization/{model_name.lower()}_feature_importance.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Feature importance plot saved to {plot_path}")

def load_test_data():
    """Load test data for evaluation."""
    # This assumes you have saved your test data during modeling
    # You might need to adjust this path based on how you saved it
    test_data_path = 'data/processed/cleaned_supply_chain_data.csv' # Or a specific test split file
    if os.path.exists(test_data_path):
        df_test = pd.read_csv(test_data_path)
        # Assuming you have a way to identify test set (e.g., time-based split)
        # For demo, let's take a sample
        df_test_sample = df_test.tail(10000) # Adjust sample size as needed
        return df_test_sample
    else:
        raise FileNotFoundError(f"Test data not found at {test_data_path}")

def get_xgboost_feature_names():
    """Get feature names used by XGBoost model."""
    try:
        importance_file = 'results/model/xgboost_late_delivery_risk_feature_importance.csv'
        if os.path.exists(importance_file):
            importance_df = pd.read_csv(importance_file)
            feature_names = importance_df['feature'].tolist()
            return feature_names, importance_df
        else:
            raise FileNotFoundError(f"Feature importance file not found at {importance_file}")
    except Exception as e:
        print(f"Error loading XGBoost feature names: {e}")
        raise

def main():
    """Main function to generate XGBoost performance visualizations."""
    print("="*60)
    print("XGBOOST PERFORMANCE VISUALIZATION")
    print("="*60)
    
    create_output_dirs()
    
    try:
        # Load test data
        print("Loading test data...")
        df_test = load_test_data()
        
        # Load XGBoost model and imputer
        print("Loading XGBoost model and imputer...")
        model_path = 'models/xgboost_late_delivery_risk.json'
        imputer_path = 'models/imputer_for_xgboost_late_delivery_risk.pkl'
        xgb_model, xgb_imputer = load_xgboost_model(model_path, imputer_path)
        
        # Get feature names
        print("Loading XGBoost feature names...")
        feature_names, importance_df = get_xgboost_feature_names()
        
        # Prepare test data
        print("Preparing test data for prediction...")
        target_col = 'Late_delivery_risk'
        if target_col not in df_test.columns:
            raise ValueError(f"Target column '{target_col}' not found in test data.")
        
        y_test = df_test[target_col]
        
        # Select features
        available_features = set(df_test.columns)
        required_features = set(feature_names)
        missing_features = required_features - available_features
        
        if missing_features:
            print(f"Warning: Missing features in test data: {missing_features}")
            feature_names_filtered = [f for f in feature_names if f in available_features]
        else:
            feature_names_filtered = feature_names
            
        X_test = df_test[feature_names_filtered]
        
        # Apply imputation
        if xgb_imputer is not None:
            print("Applying imputation to test data...")
            # Handle feature name alignment for imputer (robust approach)
            if hasattr(xgb_imputer, 'feature_names_in_'):
                expected_features = xgb_imputer.feature_names_in_
                for expected_col in expected_features:
                    if expected_col not in X_test.columns:
                        print(f"  -> Adding missing column '{expected_col}' with NaN for imputation.")
                        X_test[expected_col] = np.nan
                X_test = X_test[expected_features]
            
            X_test_imputed = xgb_imputer.transform(X_test)
            X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns, index=X_test.index)
        
        # Make predictions
        print("Making predictions...")
        y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
        y_pred = xgb_model.predict(X_test)
        
        # Generate visualizations
        print("Generating visualizations...")
        
        # Confusion Matrix
        plot_confusion_matrix(y_test, y_pred, 'XGBoost')
        
        # ROC Curve
        plot_roc_curve(y_test, y_pred_proba, 'XGBoost')
        
        # Feature Importance
        plot_feature_importance(importance_df, 'XGBoost', top_n=20)
        
        print("\nXGBoost performance visualizations completed successfully!")
        print("Plots saved to results/visualization/")
        
    except Exception as e:
        print(f"Error during XGBoost visualization: {e}")
        raise

if __name__ == "__main__":
    main()