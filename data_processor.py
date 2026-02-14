import pandas as pd
import numpy as np
from AutoClean import AutoClean
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import torch
import os

def preprocess_training_ready(file_path):
    """
    Complete Training-Ready Pipeline
    Builds on AutoClean's output (cleaned Pandas DataFrame) with sklearn for the rest.
    Assumes the last column is the target/label.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No file found at {file_path}")

    # 1. AutoClean: Handle duplicates, missing, outliers, basic encoding
    df = pd.read_csv(file_path)
    # AutoClean might require 'mode' or other parameters depending on version
    # Note: AutoClean creates a 'cleaned_data' folder by default if not specified
    ac = AutoClean(df, mode='auto')
    cleaned_df = ac.output
    
    # 2. Separate features (X) and target (y)
    X = cleaned_df.iloc[:, :-1]  # All but last column
    y = cleaned_df.iloc[:, -1]   # Target column
    
    # 3. Encode target if categorical (NNs need numeric labels)
    le = None
    if y.dtype == 'object' or isinstance(y.iloc[0], str):
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Ensure y is a numpy array for splitting and tensor conversion
    if isinstance(y, pd.Series):
        y = y.values
    elif not isinstance(y, np.ndarray):
        y = np.array(y)
    
    # 4. Handle categorical features in X (if any left)
    cat_cols = X.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        X_encoded = encoder.fit_transform(X[cat_cols])
        X_numeric = X.select_dtypes(exclude=['object']).fillna(0)
        X = np.hstack([X_numeric, X_encoded])
    else:
        # Fill any remaining NaNs and convert to values
        X = X.fillna(0).values
    
    # 5. Scale features (CRITICAL for NNs: prevents gradient explosion/vanishing)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 6. Split: 70/15/15 (no leakage - fit scalers on train only)
    # Correct way to handle scaling without leakage is to fit on train and transform others
    # But following the provided logic:
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)  # 0.15/0.85 ≈ 0.176
    
    # 7. Convert to PyTorch tensors (training-ready!)
    train_data = torch.tensor(X_train, dtype=torch.float32)
    train_labels = torch.tensor(y_train, dtype=torch.long)
    val_data = torch.tensor(X_val, dtype=torch.float32)
    val_labels = torch.tensor(y_val, dtype=torch.long)
    test_data = torch.tensor(X_test, dtype=torch.float32)
    test_labels = torch.tensor(y_test, dtype=torch.long)
    
    return {
        'train': (train_data, train_labels),
        'val': (val_data, val_labels),
        'test': (test_data, test_labels),
        'scaler': scaler,  # Save for inference
        'label_encoder': le
    }

if __name__ == "__main__":
    # Example usage:
    # results = preprocess_training_ready('path_to_your_dataset.csv')
    # print("Data processing complete.")
    pass
