import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import json
from datetime import datetime

import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Load hyperparameters ───────────────────────────────────
with open("params.yaml") as f:
    params = yaml.safe_load(f)

TEST_SIZE    = params["data"]["test_size"]
BATCH_SIZE   = params["model"]["batch_size"]
EPOCHS       = params["model"]["epochs"]
METRICS_PATH = params["evaluate"]["metrics_path"]

artifacts_dir = "artifacts"

# ── Load test data saved by model.py ──────────────────────
# model.py saves the exact split it trained on so evaluate.py
# uses the identical partition — no risk of data leakage from re-splitting.
for path in ["artifacts/X_test_cnn.npy", "artifacts/y_test.npy",
             "artifacts/training_history.json", "models/model.keras"]:
    if not os.path.exists(path):
        print(f"ERROR: {path} not found — run model.py first")
        sys.exit(1)

print("Loading test data and model...")
X_test_cnn = np.load("artifacts/X_test_cnn.npy")
y_test     = np.load("artifacts/y_test.npy")
print(f"X_test_cnn: {X_test_cnn.shape}  y_test: {y_test.shape}")

with open("artifacts/training_history.json") as f:
    history_dict = json.load(f)

# ── Custom R2 metric — must match definition used in model.py ──
def r2_metric(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - SS_res / (SS_tot + tf.keras.backend.epsilon())

# ── Load model ─────────────────────────────────────────────
model = tf.keras.models.load_model(
    "models/model.keras",
    custom_objects={"r2_metric": r2_metric}
)
print("Model loaded successfully")

# ── Evaluate ───────────────────────────────────────────────
score = model.evaluate(X_test_cnn, y_test, verbose=0)
print(f"\nTest Loss (MSE): {score[0]:.4f}")
print(f"Test MAE:        {score[1]:.4f}")
print(f"Test R2:         {score[2]:.4f}")

# ── Predictions ────────────────────────────────────────────
preds = model.predict(X_test_cnn, verbose=0).flatten()

# ── Sklearn metrics ────────────────────────────────────────
mse_val  = mean_squared_error(y_test, preds)
mae_val  = mean_absolute_error(y_test, preds)
r2_val   = r2_score(y_test, preds)
rmse_val = np.sqrt(mse_val)

print(f"\nMSE:  {mse_val:.4f}")
print(f"RMSE: {rmse_val:.4f}")
print(f"MAE:  {mae_val:.4f}")
print(f"R2:   {r2_val:.4f}")

# ── Plot: predictions vs actual ────────────────────────────
plt.figure(figsize=(10, 6))
plt.scatter(y_test, preds, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Predictions vs True Values')
plt.grid(True, alpha=0.3)
plt.savefig('predictions_vs_actual.png',                  dpi=300, bbox_inches='tight')
plt.savefig(f'{artifacts_dir}/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: predictions_vs_actual.png")

# ── Plot: residuals ────────────────────────────────────────
residuals = y_test - preds
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
axes[0].set_xlabel('Residuals')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Residuals Distribution')
axes[0].grid(True, alpha=0.3)

axes[1].scatter(preds, residuals, alpha=0.5)
axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predictions')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residuals vs Predictions')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('residuals_analysis.png',                  dpi=300, bbox_inches='tight')
plt.savefig(f'{artifacts_dir}/residuals_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved: residuals_analysis.png")

# ── metrics.txt ───────────────────────────────────────────
with open('metrics.txt', 'w') as f:
    f.write("=" * 50 + "\n")
    f.write("MODEL PERFORMANCE METRICS\n")
    f.write("=" * 50 + "\n")
    f.write(f"MSE:  {mse_val:.4f}\n")
    f.write(f"RMSE: {rmse_val:.4f}\n")
    f.write(f"MAE:  {mae_val:.4f}\n")
    f.write(f"R2:   {r2_val:.4f}\n")
    f.write("=" * 50 + "\n\n")
    f.write("TRAINING HISTORY\n")
    f.write("=" * 50 + "\n")
    f.write(f"Final Train Loss: {history_dict['loss'][-1]:.4f}\n")
    f.write(f"Final Val Loss:   {history_dict['val_loss'][-1]:.4f}\n")
    if 'r2_metric' in history_dict:
        f.write(f"Final Train R2:   {history_dict['r2_metric'][-1]:.4f}\n")
        f.write(f"Final Val R2:     {history_dict['val_r2_metric'][-1]:.4f}\n")
print("Saved: metrics.txt")

# ── metrics.json — read by DVC ─────────────────────────────
metrics_out = {
    "timestamp":   datetime.now().isoformat(),
    "model_type":  "1D CNN",
    "test_size":   TEST_SIZE,
    "batch_size":  BATCH_SIZE,
    "epochs":      EPOCHS,
    "final_epoch": len(history_dict['loss']),
    "metrics": {
        "mean_squared_error":      float(mse_val),
        "root_mean_squared_error": float(rmse_val),
        "mean_absolute_error":     float(mae_val),
        "r2_score":                float(r2_val)
    },
    "training_history": {
        "final_train_loss": float(history_dict['loss'][-1]),
        "final_val_loss":   float(history_dict['val_loss'][-1])
    }
}
if 'r2_metric' in history_dict:
    metrics_out["training_history"]["final_train_r2"] = float(history_dict['r2_metric'][-1])
    metrics_out["training_history"]["final_val_r2"]   = float(history_dict['val_r2_metric'][-1])

with open(METRICS_PATH, 'w', encoding='utf-8') as f:
    json.dump(metrics_out, f, indent=4)
print(f"Saved: {METRICS_PATH}")

# ── Submission file ────────────────────────────────────────
if not os.path.exists("test/test.csv"):
    print("WARNING: test/test.csv not found — skipping submission")
    sys.exit(0)

if not os.path.exists("artifacts/feature_columns.json"):
    print("ERROR: artifacts/feature_columns.json not found — run model.py first")
    sys.exit(1)

# Load the exact feature columns the model was trained on
with open("artifacts/feature_columns.json", encoding='utf-8') as f:
    feature_columns = json.load(f)

dtest     = pd.read_csv("test/test.csv")
test_data = dtest.copy()

# Drop ID if present
if 'ID' in test_data.columns:
    test_data = test_data.drop("ID", axis=1)

# Apply same frequency encoding as training
# (cat_vars not available here so handle any object columns generically)
obj_cols = [c for c in test_data.columns if test_data[c].dtype == 'O']
for var in obj_cols:
    freq = test_data[var].value_counts().to_dict()
    test_data[f"{var}_freq"] = test_data[var].map(freq).fillna(0)
test_data = test_data.drop(obj_cols, axis=1)

# Convert to numeric
test_data = test_data.apply(pd.to_numeric, errors='coerce')

# FIX: align columns to exactly what the model was trained on.
# Add any missing columns as 0, drop any extra columns.
for col in feature_columns:
    if col not in test_data.columns:
        test_data[col] = 0.0          # missing column — fill with 0
test_data = test_data[feature_columns]  # keep only training columns, in same order

test_data = test_data.fillna(test_data.mean()).fillna(0)

print(f"Aligned test data shape: {test_data.shape}")
print(f"Expected features: {len(feature_columns)}  Got: {test_data.shape[1]}")

X_sub           = test_data.values.reshape(test_data.shape[0], test_data.shape[1], 1)
predictions_sub = model.predict(X_sub, verbose=0).flatten()

submission = pd.DataFrame({
    "ID": dtest["ID"] if 'ID' in dtest.columns else range(len(predictions_sub)),
    "y":  predictions_sub
})
submission.to_csv('submission_5.csv', index=False)
submission.to_csv(f'{artifacts_dir}/submission_5.csv', index=False)
print("Saved: submission_5.csv")