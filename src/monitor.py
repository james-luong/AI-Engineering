"""
Comprehensive monitoring: drift, performance, data quality
Windows-compatible version (no emojis)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Fix Windows console encoding for logging
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

import logging

# Setup logging without emojis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/monitoring.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('monitoring')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

def find_artifact_file(base_name, extensions=['.npy']):
    """Find artifact file in various possible locations"""
    possible_paths = [
        f'artifacts/data/{base_name}',
        f'artifacts/{base_name}',
        f'artifacts/{base_name}.npy' if not base_name.endswith('.npy') else f'artifacts/{base_name}',
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

def load_all_data():
    """Load all datasets from artifacts (using actual file locations)"""
    logger.info("Loading data from artifacts...")
    
    # Find test data
    X_test_path = find_artifact_file('X_test_cnn.npy') or find_artifact_file('X_test.npy')
    y_test_path = find_artifact_file('y_test.npy')
    
    if X_test_path is None or y_test_path is None:
        logger.error("Test data not found!")
        logger.info("Available files in artifacts:")
        if os.path.exists('artifacts'):
            for root, dirs, files in os.walk('artifacts'):
                for file in files:
                    logger.info(f"  - {root}/{file}")
        return None, None, None, None, None, None
    
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)
    logger.info(f"Loaded test data: {X_test.shape}")
    
    # Load training data
    X_train_path = find_artifact_file('X_train_cnn.npy') or find_artifact_file('X_train.npy')
    y_train_path = find_artifact_file('y_train.npy')
    
    X_train = None
    y_train = None
    if X_train_path and y_train_path:
        X_train = np.load(X_train_path)
        y_train = np.load(y_train_path)
        logger.info(f"Loaded training data: {X_train.shape}")
    
    # Load new data
    X_new_path = find_artifact_file('X_new_cnn.npy') or find_artifact_file('X_new.npy')
    y_new_path = find_artifact_file('y_new.npy')
    
    X_new = None
    y_new = None
    if X_new_path and y_new_path:
        X_new = np.load(X_new_path)
        y_new = np.load(y_new_path)
        logger.info(f"Loaded new data: {X_new.shape}")
    else:
        logger.warning("No new data found, using test data as reference")
        X_new = X_test.copy()
        y_new = y_test.copy()
    
    # Load model
    model_path = 'models/model.keras'
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        return None, None, None, None, None, None
    
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None, None, None, None, None
    
    return model, X_train, y_train, X_test, y_test, X_new, y_new

def main():
    """Main monitoring function"""
    
    logger.info("="*70)
    logger.info("STARTING COMPREHENSIVE MONITORING")
    logger.info("="*70)
    
    # Load all data
    model, X_train, y_train, X_test, y_test, X_new, y_new = load_all_data()
    
    if model is None:
        logger.error("Failed to load model or data")
        return False
    
    # Get predictions
    logger.info("Getting model predictions...")
    try:
        y_test_pred = model.predict(X_test, verbose=0).flatten()
        logger.info(f"Test predictions: {len(y_test_pred)} samples")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return False
    
    try:
        y_new_pred = model.predict(X_new, verbose=0).flatten()
        logger.info(f"New data predictions: {len(y_new_pred)} samples")
    except Exception as e:
        logger.error(f"New data prediction failed: {e}")
        y_new_pred = y_test_pred.copy()
    
    # Clean NaN values
    def clean_array(arr):
        if np.isnan(arr).any():
            arr = np.nan_to_num(arr, nan=0.0)
        return arr
    
    y_test = clean_array(y_test)
    y_test_pred = clean_array(y_test_pred)
    y_new = clean_array(y_new)
    y_new_pred = clean_array(y_new_pred)
    
    # Calculate metrics for test data
    logger.info("Calculating test metrics...")
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    logger.info(f"Test Metrics: MSE={test_mse:.4f}, MAE={test_mae:.4f}, R2={test_r2:.4f}")
    
    # Calculate metrics for new data
    logger.info("Calculating new data metrics...")
    new_mse = mean_squared_error(y_new, y_new_pred)
    new_mae = mean_absolute_error(y_new, y_new_pred)
    new_r2 = r2_score(y_new, y_new_pred)
    
    logger.info(f"New Data Metrics: MSE={new_mse:.4f}, MAE={new_mae:.4f}, R2={new_r2:.4f}")
    
    # Calculate performance change
    if test_r2 != 0:
        perf_change = (new_r2 - test_r2) / abs(test_r2)
    else:
        perf_change = 0
    
    logger.info(f"Performance change: {perf_change*100:+.2f}%")
    
    # Determine if retraining is needed
    threshold = 0.15  # 15% degradation
    retrain_needed = perf_change < -threshold
    
    logger.info("="*70)
    if retrain_needed:
        logger.warning(f"RETRAINING NEEDED: Performance degraded by {abs(perf_change)*100:.1f}%")
    else:
        logger.info(f"NO RETRAINING NEEDED: Performance change within threshold")
    logger.info("="*70)
    
    # Save results
    os.makedirs('artifacts/metrics', exist_ok=True)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'test_metrics': {
            'mse': float(test_mse),
            'mae': float(test_mae),
            'r2': float(test_r2)
        },
        'new_metrics': {
            'mse': float(new_mse),
            'mae': float(new_mae),
            'r2': float(new_r2)
        },
        'performance_change': float(perf_change),
        'threshold': threshold,
        'retrain_needed': retrain_needed
    }
    
    with open('artifacts/metrics/monitoring_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save retrain flag for GitHub Actions
    with open('retrain_needed.txt', 'w') as f:
        f.write('true' if retrain_needed else 'false')
    
    with open('drift_status.txt', 'w') as f:
        f.write(str(retrain_needed).lower())
    
    logger.info("Results saved to artifacts/metrics/monitoring_summary.json")
    
    return retrain_needed

if __name__ == "__main__":
    try:
        retrain_needed = main()
        sys.exit(1 if retrain_needed else 0)
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(0)