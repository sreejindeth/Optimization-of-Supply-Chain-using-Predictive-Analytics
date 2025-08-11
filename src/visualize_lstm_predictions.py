from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense

def load_lstm_model_safe(model_path):
    """
    Attempt to load the LSTM model, handling potential deserialization issues.
    """
    print(f"Attempting to load model from {model_path}...")
    try:
        # Try 1: Standard load (most common)
        model = load_model(model_path)
        print("Model loaded successfully using standard load_model.")
        return model
    except ValueError as e:
        if "Could not deserialize" in str(e) and ("mse" in str(e) or "mae" in str(e)):
            print(f"Deserialization error encountered: {e}")
            print("Attempting alternative loading method (reconstructing model and loading weights)...")
            
            try:
                # We're skipping the unused with-statement here since it's not required.
                # --- Try to reconstruct the model architecture ---
                print("Reconstructing a common LSTM architecture...")

                # Infer shapes based on error messages or assumptions
                n_steps_in_guess = 14  # Common value
                n_features_guess = 1   # Based on weight shapes
                n_steps_out = 7        # Common value from modeling.py

                print(f"Trying reconstruction with input_shape=({n_steps_in_guess}, {n_features_guess})")
                
                model = Sequential()
                model.add(LSTM(50, activation='relu', input_shape=(n_steps_in_guess, n_features_guess), return_sequences=True))
                model.add(Dropout(0.2))
                model.add(LSTM(50, activation='relu'))  # return_sequences=False by default
                model.add(Dropout(0.2))
                model.add(Dense(n_steps_out))  # Predict n_steps_out values for the first target feature

                model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                model.load_weights(model_path)
                print("Model reconstructed with guessed architecture and weights loaded successfully.")
                return model

            except Exception as e2:
                print(f"Failed to reconstruct model with guessed architecture: {e2}")
        
        print(f"Standard loading failed: {e}")
        raise e  # Re-raise the original error if alternatives fail
    except Exception as e:
        print(f"Unexpected error loading model: {e}")
        raise e
