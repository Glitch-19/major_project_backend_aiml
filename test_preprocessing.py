import pandas as pd
import numpy as np
import os

def create_sample_dataset(file_path='sample_data.csv'):
    """Creates a sample dataset with missing values, duplicates, and categorical data."""
    data = {
        'Feature1': [1.0, 2.0, np.nan, 4.0, 5.0, 1.0, 7.0, 8.0, 9.0, 10.0],
        'Feature2': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'B'],
        'Feature3': [0.5, 0.2, 0.1, 0.8, 0.9, 0.5, 0.3, 0.4, 0.7, 0.6],
        'Target': ['Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes']
    }
    df = pd.DataFrame(data)
    # Add a duplicate row
    df.to_csv(file_path, index=False)
    print(f"Sample dataset created at {file_path}")
    return file_path

if __name__ == "__main__":
    from data_processor import preprocess_training_ready
    
    # 1. Create a "dirty" dataset
    csv_path = create_sample_dataset()
    
    try:
        # 2. Run the automatic preprocessing
        print("Starting automatic preprocessing...")
        output = preprocess_training_ready(csv_path)
        
        # 3. Verify the output
        print("\n--- Preprocessing Results ---")
        for set_name, (data, labels) in [('Train', output['train']), ('Val', output['val']), ('Test', output['test'])]:
            print(f"{set_name} data shape: {data.shape}, labels shape: {labels.shape}")
            print(f"{set_name} data type: {data.dtype}, labels type: {labels.dtype}")
        
        print("\nScaler check:", output['scaler'])
        print("Label encoder check:", output['label_encoder'])
        print("\nSUCCESS: Dataset processed automatically and converted to Tensors.")
        
    except Exception as e:
        print(f"\nERROR during preprocessing: {e}")
    finally:
        # Cleanup
        if os.path.exists(csv_path):
            os.remove(csv_path)
            print(f"\nCleaned up {csv_path}")
