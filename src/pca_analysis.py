# src/pca_analysis.py
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

def generate_scree_plot(df, save_plot=True):
    """
    Generate scree plot for PCA analysis on supply chain dataset with proper NaN handling
    """
    print("Generating scree plot with proper NaN handling...")
    
    # Select numerical features for PCA
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove date-related columns and other non-informative columns
    exclude_cols = ['order_date', 'order_year', 'order_month', 'order_day', 
                   'order_quarter', 'order_week', 'delivery_delay', 'Unnamed']
    numerical_features = [col for col in numerical_features 
                         if col not in exclude_cols and 'Unnamed' not in col and col in df.columns]
    
    print(f"Analyzing {len(numerical_features)} numerical features")
    print("Features for PCA:", numerical_features[:10], "..." if len(numerical_features) > 10 else "")
    
    # Extract numerical data
    numerical_data = df[numerical_features]
    
    # Check for completely empty columns
    empty_columns = numerical_data.columns[numerical_data.isnull().all()].tolist()
    if empty_columns:
        print(f"Warning: Completely empty columns found: {empty_columns}")
        # Remove empty columns
        numerical_features = [col for col in numerical_features if col not in empty_columns]
        numerical_data = df[numerical_features]
    
    print(f"After removing empty columns: {len(numerical_features)} features")
    
    # Handle missing values using scikit-learn's SimpleImputer
    print("Handling missing values...")
    print(f"Missing values before imputation: {numerical_data.isnull().sum().sum()}")
    
    # Apply imputation based on scikit-learn best practices
    imputer = SimpleImputer(strategy='median', keep_empty_features=False)  # This will drop completely empty features
    imputed_data = imputer.fit_transform(numerical_data)
    
    # Get the actual feature names after imputation (some might have been dropped)
    # The imputer drops columns that are completely NaN
    non_empty_features = [col for col in numerical_features 
                         if not numerical_data[col].isnull().all()]
    
    print(f"Features after imputation: {len(non_empty_features)}")
    print(f"Missing values after imputation: {pd.DataFrame(imputed_data).isnull().sum().sum()}")
    
    # Convert back to DataFrame with correct column names
    imputed_df = pd.DataFrame(imputed_data, columns=non_empty_features, index=df.index)
    
    # Standardize the data (important for PCA)
    print("Standardizing data...")
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_df)
    
    # Apply PCA
    print("Applying PCA...")
    pca = PCA()
    pca.fit(scaled_data)
    
    # Calculate explained variance ratios
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # Create scree plot
    plt.figure(figsize=(15, 10))
    
    # Scree plot - Individual explained variance
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Scree Plot - Individual Component Variance')
    plt.grid(True, alpha=0.3)
    
    # Scree plot - Zoomed view (first 20 components)
    plt.subplot(2, 2, 2)
    n_components_show = min(20, len(explained_variance_ratio))
    plt.plot(range(1, n_components_show + 1), explained_variance_ratio[:n_components_show], 
             'ro-', linewidth=2, markersize=8)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f'Scree Plot - First {n_components_show} Components')
    plt.grid(True, alpha=0.3)
    
    # Cumulative explained variance
    plt.subplot(2, 2, 3)
    plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 
             'go-', linewidth=2, markersize=8)
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Variance')
    plt.axhline(y=0.9, color='b', linestyle='--', label='90% Variance')
    plt.axhline(y=0.95, color='g', linestyle='--', label='95% Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Cumulative Explained Variance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Cumulative variance - Zoomed view
    plt.subplot(2, 2, 4)
    n_cumulative_show = min(50, len(cumulative_variance_ratio))
    plt.plot(range(1, n_cumulative_show + 1), cumulative_variance_ratio[:n_cumulative_show], 
             'mo-', linewidth=2, markersize=8)
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Variance')
    plt.axhline(y=0.9, color='b', linestyle='--', label='90% Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title(f'Cumulative Variance (First {n_cumulative_show} Components)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('results/eda/scree_plot.png', dpi=300, bbox_inches='tight')
        print("Scree plot saved to results/eda/scree_plot.png")
    
    plt.show()
    
    # Print summary statistics
    print("\nPCA Analysis Summary:")
    print(f"Total number of components: {len(explained_variance_ratio)}")
    print(f"Original features: {len(non_empty_features)}")
    
    # Handle case where cumulative variance might not reach certain thresholds
    try:
        components_80 = np.argmax(cumulative_variance_ratio >= 0.8) + 1 if any(cumulative_variance_ratio >= 0.8) else len(cumulative_variance_ratio)
        print(f"Components for 80% variance: {components_80}")
    except:
        print("Could not determine components for 80% variance")
    
    try:
        components_90 = np.argmax(cumulative_variance_ratio >= 0.9) + 1 if any(cumulative_variance_ratio >= 0.9) else len(cumulative_variance_ratio)
        print(f"Components for 90% variance: {components_90}")
    except:
        print("Could not determine components for 90% variance")
    
    try:
        components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1 if any(cumulative_variance_ratio >= 0.95) else len(cumulative_variance_ratio)
        print(f"Components for 95% variance: {components_95}")
    except:
        print("Could not determine components for 95% variance")
    
    # Show top components
    print("\nTop 10 components explained variance:")
    for i in range(min(10, len(explained_variance_ratio))):
        print(f"PC{i+1}: {explained_variance_ratio[i]:.4f} ({cumulative_variance_ratio[i]:.4f} cumulative)")
    
    return pca, scaler, imputer, explained_variance_ratio, cumulative_variance_ratio, non_empty_features

def elbow_method_analysis(df):
    """
    Apply elbow method to determine optimal number of components
    """
    # Select numerical features
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['order_date', 'order_year', 'order_month', 'order_day', 
                   'order_quarter', 'order_week', 'delivery_delay', 'Unnamed']
    numerical_features = [col for col in numerical_features 
                         if col not in exclude_cols and 'Unnamed' not in col and col in df.columns]
    
    # Remove completely empty columns
    numerical_data = df[numerical_features]
    empty_columns = numerical_data.columns[numerical_data.isnull().all()].tolist()
    if empty_columns:
        numerical_features = [col for col in numerical_features if col not in empty_columns]
    
    # Extract and impute data
    numerical_data = df[numerical_features]
    imputer = SimpleImputer(strategy='median')
    imputed_data = imputer.fit_transform(numerical_data)
    
    # Standardize and apply PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_data)
    
    pca = PCA()
    pca.fit(scaled_data)
    
    # Calculate explained variance ratios
    explained_variance = pca.explained_variance_
    
    # Plot elbow method
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, 'bo-')
    plt.xlabel('Principal Component')
    plt.ylabel('Eigenvalues')
    plt.title('Elbow Method - Eigenvalues')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # Second derivative to find elbow point (handle potential issues)
    if len(explained_variance) > 2:
        second_diff = np.diff(explained_variance, 2)
        if len(second_diff) > 0:
            plt.plot(range(3, len(second_diff) + 3), second_diff, 'ro-')
            plt.xlabel('Principal Component')
            plt.ylabel('Second Difference of Eigenvalues')
            plt.title('Elbow Point Detection')
            plt.grid(True, alpha=0.3)
        else:
            plt.text(0.5, 0.5, 'Insufficient data for second difference', 
                    ha='center', va='center', transform=plt.gca().transAxes)
            plt.title('Elbow Point Detection - Insufficient Data')
    else:
        plt.text(0.5, 0.5, 'Insufficient components for analysis', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Elbow Point Detection - Insufficient Data')
    
    plt.tight_layout()
    plt.savefig('results/eda/elbow_method.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pca, scaler, imputer

def feature_importance_analysis(df, n_components=10):
    """
    Analyze which original features contribute most to the first few principal components
    """
    # Select numerical features
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['order_date', 'order_year', 'order_month', 'order_day', 
                   'order_quarter', 'order_week', 'delivery_delay', 'Unnamed']
    numerical_features = [col for col in numerical_features 
                         if col not in exclude_cols and 'Unnamed' not in col and col in df.columns]
    
    # Remove completely empty columns
    numerical_data = df[numerical_features]
    empty_columns = numerical_data.columns[numerical_data.isnull().all()].tolist()
    if empty_columns:
        numerical_features = [col for col in numerical_features if col not in empty_columns]
    
    # Extract and impute data
    numerical_data = df[numerical_features]
    imputer = SimpleImputer(strategy='median')
    imputed_data = imputer.fit_transform(numerical_data)
    
    # Standardize and apply PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_data)
    
    pca = PCA(n_components=n_components)
    pca.fit(scaled_data)
    
    # Create feature importance dataframe
    components_df = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=numerical_features
    )
    
    # Plot top contributing features for first few components
    plt.figure(figsize=(15, 12))
    n_plot_components = min(6, n_components)
    
    for i in range(n_plot_components):
        plt.subplot(2, 3, i+1)
        # Get absolute loadings for this component
        feature_importance = components_df[f'PC{i+1}'].abs().sort_values(ascending=False).head(10)
        sns.barplot(x=feature_importance.values, y=feature_importance.index)
        plt.title(f'Top 10 Features in PC{i+1}\n(Explained Variance: {pca.explained_variance_ratio_[i]:.3f})')
        plt.xlabel('Absolute Component Loading')
    
    plt.tight_layout()
    plt.savefig('results/eda/pca_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return components_df, pca, scaler, imputer

# Main execution
if __name__ == "__main__":
    # Load your processed data
    try:
        df = pd.read_csv('data/processed/cleaned_supply_chain_data.csv')
        print("Dataset loaded successfully!")
        print(f"Dataset shape: {df.shape}")
        
        # Generate scree plot with proper NaN handling
        pca_model, scaler, imputer, exp_var, cum_var, features_used = generate_scree_plot(df)
        
        # Apply elbow method analysis
        elbow_pca, elbow_scaler, elbow_imputer = elbow_method_analysis(df)
        
        # Feature importance analysis
        components_df, feature_pca, feature_scaler, feature_imputer = feature_importance_analysis(df)
        components_df.to_csv('results/eda/pca_components_analysis.csv')
        
        print(f"\nFinal analysis used {len(features_used)} features:")
        print(features_used)
        print("\nPCA analysis completed successfully!")
        print("Results saved to results/eda/ directory")
        
    except FileNotFoundError:
        print("Error: Could not find processed data file. Please run preprocessing first.")
    except Exception as e:
        print(f"Error during PCA analysis: {e}")
        import traceback
        traceback.print_exc()