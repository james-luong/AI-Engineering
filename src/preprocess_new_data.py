import os
import sys
import json
import numpy as np
import pandas as pd

print("="*70)
print("PREPROCESSING NEW DATA (Standalone)")
print("="*70)

def preprocess_new_data():
    """Preprocess new data without joblib"""
    
    # Check if new data exists
    new_data_path = 'data/new_data.csv'
    if not os.path.exists(new_data_path):
        print(" No new data found at data/new_data.csv")
        return False
    
    # Load new data
    print(f"\n Loading new data from {new_data_path}...")
    df = pd.read_csv(new_data_path)
    print(f" Loaded {len(df)} rows with {df.shape[1]} columns")
    
    # Load feature columns (JSON, not pickle)
    feature_columns_path = 'artifacts/preprocessing/feature_columns.json'
    if not os.path.exists(feature_columns_path):
        # Try old location
        feature_columns_path = 'artifacts/feature_columns.json'
        if not os.path.exists(feature_columns_path):
            print(f" Feature columns not found at {feature_columns_path}")
            return False
    
    with open(feature_columns_path, 'r') as f:
        feature_columns = json.load(f)
    print(f" Loaded {len(feature_columns)} feature columns")
    
    # Extract target column
    print("\n🎯 Processing data...")
    if 'target' in df.columns:
        y_new = df['target'].values
        X_df = df.drop('target', axis=1)
    elif 'y' in df.columns:
        y_new = df['y'].values
        X_df = df.drop('y', axis=1)
    else:
        # Assume last column is target
        y_new = df.iloc[:, -1].values
        X_df = df.iloc[:, :-1]
    
    print(f" Target shape: {y_new.shape}")
    
    # Match features
    print("\n🔍 Matching features...")
    available_features = [col for col in feature_columns if col in X_df.columns]
    missing_features = [col for col in feature_columns if col not in X_df.columns]
    
    print(f" Available: {len(available_features)}/{len(feature_columns)}")
    if missing_features:
        print(f" Missing {len(missing_features)} features, filling with 0")
        for col in missing_features:
            X_df[col] = 0
    
    # Ensure correct column order
    X_new = X_df[feature_columns].values
    
    # Handle NaN values
    if np.isnan(X_new).any():
        print(f" Found NaN in features, filling with 0")
        X_new = np.nan_to_num(X_new, nan=0.0)
    
    # Try to load scaler, but skip if corrupted
    print("\n Applying scaling...")
    scaler_path = 'artifacts/preprocessing/scaler.pkl'
    if os.path.exists(scaler_path):
        try:
            import joblib
            scaler = joblib.load(scaler_path)
            X_new_scaled = scaler.transform(X_new)
            print(" Scaler applied")
        except Exception as e:
            print(f" Could not load scaler: {e}")
            print(" Using raw values without scaling")
            X_new_scaled = X_new
    else:
        print(" No scaler found, using raw values")
        X_new_scaled = X_new
    
    # Check if we need CNN reshape
    print("\n Checking model format...")
    import tensorflow as tf
    
    if os.path.exists('models/model.keras'):
        try:
            model = tf.keras.models.load_model('models/model.keras', compile=False)
            if len(model.input_shape) == 3:  # CNN expects (samples, features, 1)
                X_new_cnn = X_new_scaled.reshape(X_new_scaled.shape[0], X_new_scaled.shape[1], 1)
                print(f" Reshaped for CNN: {X_new_cnn.shape}")
            else:
                X_new_cnn = X_new_scaled
                print(f" Using raw shape: {X_new_cnn.shape}")
        except:
            # Default to CNN format
            X_new_cnn = X_new_scaled.reshape(X_new_scaled.shape[0], X_new_scaled.shape[1], 1)
            print(f" Default reshape: {X_new_cnn.shape}")
    else:
        # Assume CNN format
        X_new_cnn = X_new_scaled.reshape(X_new_scaled.shape[0], X_new_scaled.shape[1], 1)
        print(f" Reshaped for CNN: {X_new_cnn.shape}")
    
    # Save preprocessed data
    print("\n Saving preprocessed data...")
    os.makedirs('artifacts/data', exist_ok=True)
    np.save('artifacts/data/X_new.npy', X_new_cnn)
    np.save('artifacts/data/y_new.npy', y_new)
    print(f" Saved to artifacts/data/X_new.npy (shape: {X_new_cnn.shape})")
    print(f" Saved to artifacts/data/y_new.npy (shape: {y_new.shape})")
    
    # Also save to root for compatibility
    np.save('artifacts/X_new_cnn.npy', X_new_cnn)
    np.save('artifacts/y_new.npy', y_new)
    print(f" Saved to artifacts/X_new_cnn.npy for compatibility")
    
    # Combine with training data for retraining
    print("\n Preparing combined dataset...")
    if os.path.exists('train/train.csv'):
        train_df = pd.read_csv('train/train.csv')
        print(f" Original training: {len(train_df)} rows")
        
        # Make sure new data has same columns as training data
        for col in train_df.columns:
            if col not in df.columns and col != 'y':
                df[col] = 0
        
        # Ensure target column matches
        if 'y' in train_df.columns and 'y' not in df.columns:
            if 'target' in df.columns:
                df['y'] = df['target']
                df = df.drop('target', axis=1)
        
        # Combine
        combined_df = pd.concat([train_df, df], ignore_index=True)
        combined_df = combined_df.dropna()
        print(f" Combined: {len(combined_df)} rows")
        
        # Save combined
        combined_df.to_csv('train/train_combined.csv', index=False)
        
        # Backup and replace
        import shutil
        if os.path.exists('train/train.csv'):
            shutil.copy('train/train.csv', 'train/train_backup.csv')
        shutil.copy('train/train_combined.csv', 'train/train.csv')
        print(f" Updated train/train.csv with new data")
    
    print("\n" + "="*70)
    print(" Preprocessing complete!")
    print("="*70)
    print("\nNext steps:")
    print("   python src/monitor.py")
    print("   dvc repro")
    
    return True

if __name__ == "__main__":
    try:
        success = preprocess_new_data()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)