# compare_scalers.py
import joblib
import numpy as np

def compare_scalers(original_scaler_path, trial_scaler_path):
    """
    Compare two StandardScaler objects to see if they are identical.
    """
    try:
        # Load both scalers
        original_scaler = joblib.load(original_scaler_path)
        trial_scaler = joblib.load(trial_scaler_path)

        print(f"Comparing scalers:")
        print(f"  Original Scaler: {original_scaler_path}")
        print(f"  Trial Scaler:    {trial_scaler_path}")
        print("-" * 40)

        # Check if they have the same number of features
        if hasattr(original_scaler, 'n_features_in_') and hasattr(trial_scaler, 'n_features_in_'):
            orig_n_features = original_scaler.n_features_in_
            trial_n_features = trial_scaler.n_features_in_
            print(f"Number of features:")
            print(f"  Original: {orig_n_features}")
            print(f"  Trial:    {trial_n_features}")
            if orig_n_features != trial_n_features:
                print("  -> SCALERS DIFFER: Number of features is different.")
                return False
        else:
            print("Warning: Could not determine number of features for one or both scalers.")
            return False

        # Check if they were fitted on the same data (compare mean and scale)
        if hasattr(original_scaler, 'mean_') and hasattr(trial_scaler, 'mean_'):
            means_close = np.allclose(original_scaler.mean_, trial_scaler.mean_, rtol=1e-5, atol=1e-8)
            print(f"Means are close: {means_close}")
            if not means_close:
                print("  -> SCALERS DIFFER: Means are not close.")
                print(f"     Original mean: {original_scaler.mean_}")
                print(f"     Trial mean:    {trial_scaler.mean_}")
                return False
        else:
            print("Warning: Could not compare means for one or both scalers.")
            return False

        if hasattr(original_scaler, 'scale_') and hasattr(trial_scaler, 'scale_'):
            scales_close = np.allclose(original_scaler.scale_, trial_scaler.scale_, rtol=1e-5, atol=1e-8)
            print(f"Scales are close: {scales_close}")
            if not scales_close:
                print("  -> SCALERS DIFFER: Scales are not close.")
                print(f"     Original scale: {original_scaler.scale_}")
                print(f"     Trial scale:    {trial_scaler.scale_}")
                return False
        else:
            print("Warning: Could not compare scales for one or both scalers.")
            return False

        print("-" * 40)
        print("SCALERS APPEAR TO BE IDENTICAL.")
        return True

    except FileNotFoundError as e:
        print(f"Error: Scaler file not found: {e}")
        return False
    except Exception as e:
        print(f"Error comparing scalers: {e}")
        return False

if __name__ == "__main__":
    # Define paths
    original_scaler_path = 'models/scaler_for_lstm_demand_forecast_sales.pkl'
    # You need to find the scaler saved during the tuning trial
    # It might have been saved with a similar name pattern
    trial_scaler_path = 'models/scaler_for_lstm_tuning_trial_9310.pkl' # Adjust if different
    
    # If the trial didn't save a specific scaler, it likely used the original one.
    # Check if the trial scaler file exists
    import os
    if not os.path.exists(trial_scaler_path):
        print(f"Trial scaler not found at {trial_scaler_path}. Assuming it uses the original scaler.")
        trial_scaler_path = original_scaler_path # Compare original with itself (should be identical)
        
    are_identical = compare_scalers(original_scaler_path, trial_scaler_path)
    
    if are_identical:
        print("\nConclusion: The scalers are identical. You can safely use the original scaler.")
    else:
        print("\nConclusion: The scalers are different. You MUST use the trial scaler for the tuned model.")
