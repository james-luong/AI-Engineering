import os
import sys
import io
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import json
from datetime import datetime
from io import StringIO
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from dvclive import Live

# ── Load hyperparameters ───────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

EPOCHS        = params["model"]["epochs"]
BATCH_SIZE    = params["model"]["batch_size"]
LEARNING_RATE = params["model"]["learning_rate"]
SEED          = params["data"]["random_seed"]
TEST_SIZE     = params["data"]["test_size"]
CNN_FILTERS   = params["model"]["cnn_filters"]
KERNEL_SIZE   = params["model"]["kernel_size"]
POOL_SIZE     = params["model"]["pool_size"]
DENSE_1       = params["model"]["dense_units_1"]
DENSE_2       = params["model"]["dense_units_2"]
DENSE_3       = params["model"]["dense_units_3"]
DROPOUT_1     = params["model"]["dropout_1"]
DROPOUT_2     = params["model"]["dropout_2"]
ES_PATIENCE   = params["callbacks"]["early_stopping_patience"]
LR_PATIENCE   = params["callbacks"]["reduce_lr_patience"]
LR_FACTOR     = params["callbacks"]["reduce_lr_factor"]
LR_MIN        = params["callbacks"]["reduce_lr_min_lr"]

# ── Directories ────────────────────────────────────────────
artifacts_dir = "artifacts"
os.makedirs(artifacts_dir, exist_ok=True)
os.makedirs(f"{artifacts_dir}/preprocessing", exist_ok=True)
os.makedirs(f"{artifacts_dir}/data", exist_ok=True)
os.makedirs(f"{artifacts_dir}/metrics", exist_ok=True)
os.makedirs(f"{artifacts_dir}/metadata", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("=" * 50)
print("STARTING CNN REGRESSION MODEL — TRAINING")
print("=" * 50)

# ── Load data ──────────────────────────────────────────────
for path in ["train/train.csv", "test/test.csv"]:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found!")
        print("CWD:", os.getcwd())
        print("Files:", os.listdir('.'))
        sys.exit(1)

print("\nLoading data...")
data  = pd.read_csv("train/train.csv")
dtest = pd.read_csv("test/test.csv")
print(f"Train shape: {data.shape}  Test shape: {dtest.shape}")

# ── Missing values report ──────────────────────────────────
print(f"Missing — train: {data.isnull().any().sum()}  test: {dtest.isnull().any().sum()}")

train_test_data = [data, dtest]
for dataset in train_test_data:
    num_vars = [v for v in dataset.columns if dataset[v].dtype != 'O']
    print(f"Numerical variables: {len(num_vars)}")

# ── Drop constant columns ──────────────────────────────────
suspiciousData = [col for col in data.columns if data[col].nunique() == 1]
if suspiciousData:
    print(f"Dropping {len(suspiciousData)} constant columns")
    for dataset in train_test_data:
        dataset.drop(suspiciousData, axis=1, inplace=True)
else:
    print("No constant columns found")

# ── Encode categorical variables ───────────────────────────
cat_vars = [v for v in data.columns if data[v].dtype == 'O' and v not in ['ID', 'y']]
print(f"Categorical variables: {len(cat_vars)}")

if cat_vars:
    for var in cat_vars:
        freq = data[var].value_counts().to_dict()
        data[f"{var}_freq"]  = data[var].map(freq)
        dtest[f"{var}_freq"] = dtest[var].map(freq).fillna(0)
    data  = data.drop(cat_vars, axis=1)
    dtest = dtest.drop(cat_vars, axis=1)
    print("Categorical variables encoded")

# ── Features and target ────────────────────────────────────
if 'ID' in data.columns:
    data = data.drop("ID", axis=1)

if 'y' not in data.columns:
    print("ERROR: 'y' column not found! Columns:", data.columns.tolist())
    sys.exit(1)

X = data.drop("y", axis=1).apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.mean()).fillna(0).values
y = data["y"].values
print(f"X: {X.shape}  y: {y.shape}")

# ============================================
#  Save feature column names (for monitor.py)
# ============================================
feature_columns = list(data.drop("y", axis=1).columns)
with open(f"{artifacts_dir}/preprocessing/feature_columns.json", "w", encoding='utf-8') as f:
    json.dump(feature_columns, f, indent=4)
print(f" Saved: {artifacts_dir}/preprocessing/feature_columns.json ({len(feature_columns)} features)")

# Save original feature columns to root (backward compatibility)
with open(f"{artifacts_dir}/feature_columns.json", "w", encoding='utf-8') as f:
    json.dump(feature_columns, f, indent=4)
print(f" Saved: {artifacts_dir}/feature_columns.json (backward compat)")

# Save target column info
with open(f"{artifacts_dir}/preprocessing/target_column.json", "w", encoding='utf-8') as f:
    json.dump({"target_column": "y"}, f)

# ── Train / test split ─────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED
)
print(f"X_train: {X_train.shape}  X_test: {X_test.shape}")

# ============================================
#  Create and save scaler
# ============================================
print("\n📊 Creating and saving scaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f" Features scaled: mean≈0, std≈1")

# Save scaler
joblib.dump(scaler, f"{artifacts_dir}/preprocessing/scaler.pkl")
print(f" Saved scaler to {artifacts_dir}/preprocessing/scaler.pkl")

# ============================================
#  Save scaled data (as backup)
# ============================================
np.save(f"{artifacts_dir}/data/X_train_scaled.npy", X_train_scaled)
np.save(f"{artifacts_dir}/data/X_test_scaled.npy", X_test_scaled)
np.save(f"{artifacts_dir}/data/y_train.npy", y_train)
np.save(f"{artifacts_dir}/data/y_test.npy", y_test)
print(f" Saved scaled data to {artifacts_dir}/data/")

# ── Reshape for 1D CNN: (samples, features, 1) ────────────
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn  = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
print(f"CNN shapes — train: {X_train_cnn.shape}  test: {X_test_cnn.shape}")

# ── Save split data for evaluate.py ───────────────────────
# evaluate.py loads these instead of re-splitting, ensuring
# both scripts use the identical train/test partition.
np.save(f"{artifacts_dir}/X_test_cnn.npy",  X_test_cnn)
np.save(f"{artifacts_dir}/y_test.npy",      y_test)
np.save(f"{artifacts_dir}/data/X_test_cnn.npy", X_test_cnn)
np.save(f"{artifacts_dir}/data/X_train_cnn.npy", X_train_cnn)
print(" Saved: artifacts/X_test_cnn.npy, artifacts/y_test.npy")
print(" Saved: artifacts/data/X_test_cnn.npy, artifacts/data/X_train_cnn.npy")

# ── Custom R2 metric ───────────────────────────────────────
def r2_metric(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())

# ── Build model ────────────────────────────────────────────
tf.random.set_seed(SEED)

model = Sequential([
    Conv1D(CNN_FILTERS[0], kernel_size=KERNEL_SIZE, activation='relu',
           input_shape=(X_train_cnn.shape[1], 1), padding='same'),
    MaxPooling1D(pool_size=POOL_SIZE),

    Conv1D(CNN_FILTERS[1], kernel_size=KERNEL_SIZE, activation='relu', padding='same'),
    MaxPooling1D(pool_size=POOL_SIZE),

    Conv1D(CNN_FILTERS[2], kernel_size=KERNEL_SIZE, activation='relu', padding='same'),
    MaxPooling1D(pool_size=POOL_SIZE),

    Flatten(),
    Dense(DENSE_1, activation='relu'),
    Dropout(DROPOUT_1),
    Dense(DENSE_2, activation='relu'),
    Dropout(DROPOUT_2),
    Dense(DENSE_3, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(
    loss='mean_squared_error',
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=['mae', r2_metric]
)

model.summary()

# Capture summary to a string
stream = StringIO()
model.summary(print_fn=lambda x: stream.write(x + '\n'))
summary_str = stream.getvalue()

with open('model_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary_str)
print("Saved: model_summary.txt")

# ── Callbacks ──────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_loss', patience=ES_PATIENCE,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=LR_FACTOR,
                      patience=LR_PATIENCE, min_lr=LR_MIN, verbose=1)
]

# ── Train ──────────────────────────────────────────────────
print("\nTraining model...")
with Live(dir="dvclive", report="html") as live:
    live.log_param("epochs",      EPOCHS)
    live.log_param("batch_size",  BATCH_SIZE)
    live.log_param("lr",          LEARNING_RATE)
    live.log_param("cnn_filters", str(CNN_FILTERS))

    history = model.fit(
        X_train_cnn, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test_cnn, y_test),
        callbacks=callbacks,
        verbose=1
    )

    for i in range(len(history.history['loss'])):
        live.log_metric("train_loss", history.history['loss'][i])
        live.log_metric("val_loss",   history.history['val_loss'][i])
        live.log_metric("train_mae",  history.history['mae'][i])
        live.log_metric("val_mae",    history.history['val_mae'][i])
        if 'r2_metric' in history.history:
            live.log_metric("train_r2", history.history['r2_metric'][i])
            live.log_metric("val_r2",   history.history['val_r2_metric'][i])
        live.next_step()

print("Training completed!")

# ── Save model ─────────────────────────────────────────────
model.save("models/model.keras")
model.save(f"{artifacts_dir}/cnn_regression_model.h5")
print(" Saved: models/model.keras")
print(f" Saved: {artifacts_dir}/cnn_regression_model.h5")

# ── Training history plots ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(history.history['loss'],     label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Model Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].legend()
axes[0].grid(True)

if 'r2_metric' in history.history:
    axes[1].plot(history.history['r2_metric'],     label='Train R2')
    axes[1].plot(history.history['val_r2_metric'], label='Val R2')
    axes[1].set_title('Model R2 Score')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('R2')
    axes[1].legend()
    axes[1].grid(True)

plt.tight_layout()
plt.savefig('model_results.png',                  dpi=300, bbox_inches='tight')
plt.savefig(f'{artifacts_dir}/model_results.png', dpi=300, bbox_inches='tight')
plt.close()
print(" Saved: model_results.png")
print(f" Saved: {artifacts_dir}/model_results.png")

# ── Save training history for evaluate.py ─────────────────
history_dict = {
    "loss":     [float(v) for v in history.history['loss']],
    "val_loss": [float(v) for v in history.history['val_loss']],
    "mae":      [float(v) for v in history.history['mae']],
    "val_mae":  [float(v) for v in history.history['val_mae']],
}
if 'r2_metric' in history.history:
    history_dict["r2_metric"]     = [float(v) for v in history.history['r2_metric']]
    history_dict["val_r2_metric"] = [float(v) for v in history.history['val_r2_metric']]

with open(f"{artifacts_dir}/training_history.json", "w", encoding='utf-8') as f:
    json.dump(history_dict, f, indent=4)
with open(f"{artifacts_dir}/metrics/training_history.json", "w", encoding='utf-8') as f:
    json.dump(history_dict, f, indent=4)
print(" Saved: artifacts/training_history.json")
print(" Saved: artifacts/metrics/training_history.json")

# ============================================
#  Save test metrics
# ============================================
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Get predictions on test set
y_pred = model.predict(X_test_cnn, verbose=0).flatten()

test_metrics = {
    'mse': float(mean_squared_error(y_test, y_pred)),
    'mae': float(mean_absolute_error(y_test, y_pred)),
    'r2': float(r2_score(y_test, y_pred)),
    'timestamp': datetime.now().isoformat()
}

with open(f"{artifacts_dir}/metrics/test_metrics.json", "w", encoding='utf-8') as f:
    json.dump(test_metrics, f, indent=4)
print(" Saved: artifacts/metrics/test_metrics.json")

# ── Save data info ─────────────────────────────────────────
data_info = {
    "train_samples":               int(X_train.shape[0]),
    "test_samples":                int(X_test.shape[0]),
    "features_count":              int(X.shape[1]),
    "categorical_vars_original":   len(cat_vars),
    "constant_features_dropped":   len(suspiciousData),
    "target_mean": float(y.mean()),
    "target_std":  float(y.std()),
    "target_min":  float(y.min()),
    "target_max":  float(y.max())
}
with open('data_info.json', 'w', encoding='utf-8') as f:
    json.dump(data_info, f, indent=4)
with open(f"{artifacts_dir}/metadata/data_info.json", 'w', encoding='utf-8') as f:
    json.dump(data_info, f, indent=4)
print(" Saved: data_info.json")
print(" Saved: artifacts/metadata/data_info.json")

# ============================================
#  Save model metadata
# ============================================
model_metadata = {
    'model_type': 'CNN_Regression',
    'input_shape': X_train_cnn.shape[1:],
    'num_features': len(feature_columns),
    'num_training_samples': len(y_train),
    'num_test_samples': len(y_test),
    'feature_columns_preview': feature_columns[:10],  # First 10 for preview
    'target_column': 'y',
    'training_completed': datetime.now().isoformat(),
    'hyperparameters': {
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'cnn_filters': CNN_FILTERS,
        'kernel_size': KERNEL_SIZE,
        'pool_size': POOL_SIZE,
        'dense_units': [DENSE_1, DENSE_2, DENSE_3],
        'dropout_rates': [DROPOUT_1, DROPOUT_2]
    },
    'test_performance': test_metrics
}

with open(f"{artifacts_dir}/metadata/model_info.json", "w", encoding='utf-8') as f:
    json.dump(model_metadata, f, indent=4)
print(" Saved: artifacts/metadata/model_info.json")

# Save number of features as text file for easy reading
with open(f"{artifacts_dir}/metadata/num_features.txt", "w") as f:
    f.write(str(len(feature_columns)))

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*60)
print(" TRAINING COMPLETE - ALL ARTIFACTS GENERATED")
print("="*60)
print("\n Generated artifacts structure:")
print("   artifacts/preprocessing/")
print("   ├── feature_columns.json")
print("   ├── scaler.pkl")
print("   └── target_column.json")
print("   artifacts/data/")
print("   ├── X_train_scaled.npy")
print("   ├── X_test_scaled.npy")
print("   ├── X_train_cnn.npy")
print("   ├── X_test_cnn.npy")
print("   ├── y_train.npy")
print("   └── y_test.npy")
print("   artifacts/metrics/")
print("   ├── test_metrics.json")
print("   └── training_history.json")
print("   artifacts/metadata/")
print("   ├── data_info.json")
print("   ├── model_info.json")
print("   └── num_features.txt")
print("   models/model.keras")
print("\n Next steps:")
print("   1. Run: dvc add artifacts/ models/")
print("   2. Run: dvc push")
print("   3. python src/preprocess_new_data.py")
print("   4. python src/monitor.py")
print("="*60)

print("\n model.py completed successfully! Ready for monitoring and retraining.")