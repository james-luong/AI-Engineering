"""
src/model.py
============
MLOps retraining script — Tetuan City Power Consumption (TPC) project.

Dataset  : Tetuan City Power Consumption
Target   : Zone 1 Power Consumption → binned into Low / Medium / High
           using equal-frequency (quantile) binning (3 classes)
Model    : SVM (RBF kernel, C=10, gamma=0.1) — best result from Set 3
Features : Temperature, Humidity, Wind Speed, general diffuse flows,
           diffuse flows, Hour, DayOfWeek, Month  (Set 3: Scaled + Engineered)
Scalers  : StandardScaler for Gaussian features (Temperature, Humidity)
           RobustScaler  for outlier-prone features (Wind Speed, flows)

Artifacts saved
---------------
  artifacts/models/model.pkl              SVM model (joblib)
  artifacts/preprocessing/scaler_gaussian.pkl
  artifacts/preprocessing/scaler_outlier.pkl
  artifacts/preprocessing/encoder.pkl     LabelEncoder
  artifacts/preprocessing/bin_edges.json  Quantile bin boundaries
  artifacts/preprocessing/feature_columns.json
  artifacts/data/X_train.npy, y_train.npy, X_test.npy, y_test.npy
  artifacts/metrics/training_history.json
  artifacts/metadata/model_version.txt, data_version.txt, last_retrain.txt
"""

import json
import logging
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler
from sklearn.svm import SVC

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)

# ── Directory setup ───────────────────────────────────────────────────────────
for d in [
    "artifacts/models", "artifacts/data", "artifacts/preprocessing",
    "artifacts/metrics", "artifacts/metadata", "logs",
]:
    os.makedirs(d, exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
GAUSSIAN_FEATURES = ["Temperature", "Humidity"]
OUTLIER_FEATURES  = ["Wind Speed", "general diffuse flows", "diffuse flows"]
TEMPORAL_FEATURES = ["Hour", "DayOfWeek", "Month"]
ALL_FEATURES      = GAUSSIAN_FEATURES + OUTLIER_FEATURES + TEMPORAL_FEATURES
TARGET_RAW        = "Zone 1 Power Consumption"
TARGET_CLASS      = "Target_Class"
N_BINS            = 3
BIN_LABELS        = ["Low", "Medium", "High"]


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_data() -> tuple:
    """
    Load train/train.csv (optionally merging data/new_data.csv) and test/test.csv.
    Returns (train_df, test_df).
    """
    logger.info("Loading training data from train/train.csv")
    train_df = pd.read_csv("train/train.csv")

    logger.info("Loading test data from test/test.csv")
    test_df = pd.read_csv("test/test.csv")

    new_data_path = "data/new_data.csv"
    if os.path.exists(new_data_path):
        logger.info(f"New data detected at {new_data_path} — merging into training set")
        new_df   = pd.read_csv(new_data_path)
        before   = len(train_df)
        train_df = pd.concat([train_df, new_df], ignore_index=True)
        logger.info(f"Training set expanded: {before} → {len(train_df)} rows (+{len(new_df)})")
    else:
        logger.info("No new_data.csv found — using base training set only")

    return train_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from the DateTime column."""
    df = df.copy()
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df["Hour"]      = df["DateTime"].dt.hour
    df["DayOfWeek"] = df["DateTime"].dt.dayofweek
    df["Month"]     = df["DateTime"].dt.month
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  TARGET BINNING
# ─────────────────────────────────────────────────────────────────────────────
def bin_target(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """
    Equal-frequency (quantile) binning of Zone 1 Power Consumption into
    Low / Medium / High on the TRAINING set only.
    Bin edges are saved and applied to the test set (no leakage).
    Returns (train_df_with_class, test_df_with_class, bin_edges).
    """
    train_df = train_df.copy()
    test_df  = test_df.copy()

    train_df[TARGET_CLASS], bin_edges = pd.qcut(
        train_df[TARGET_RAW], q=N_BINS, labels=BIN_LABELS, retbins=True
    )

    # Apply same edges to test set
    test_df[TARGET_CLASS] = pd.cut(
        test_df[TARGET_RAW], bins=bin_edges, labels=BIN_LABELS, include_lowest=True
    )

    # Drop rows whose test target falls outside training bin range (edge cases)
    test_df = test_df.dropna(subset=[TARGET_CLASS])

    logger.info(f"Train class distribution:\n{train_df[TARGET_CLASS].value_counts().to_string()}")
    logger.info(f"Test  class distribution:\n{test_df[TARGET_CLASS].value_counts().to_string()}")

    # Save bin edges for use on future new data
    with open("artifacts/preprocessing/bin_edges.json", "w") as fh:
        json.dump({"bin_edges": bin_edges.tolist(), "labels": BIN_LABELS}, fh, indent=2)

    return train_df, test_df, bin_edges


# ─────────────────────────────────────────────────────────────────────────────
# 4.  PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
    """
    Apply dual-scaler preprocessing (same strategy as original model.py Set 3):
      - StandardScaler → Gaussian features (Temperature, Humidity)
      - RobustScaler   → outlier-prone features (Wind Speed, flows)
    Encode Target_Class with LabelEncoder.
    Save all preprocessors and feature lists to artifacts/preprocessing/.
    """
    logger.info("Preprocessing features …")

    # Save feature schema
    with open("artifacts/preprocessing/feature_columns.json", "w") as fh:
        json.dump({
            "features":         ALL_FEATURES,
            "gaussian_features": GAUSSIAN_FEATURES,
            "outlier_features":  OUTLIER_FEATURES,
            "temporal_features": TEMPORAL_FEATURES,
            "target":           TARGET_CLASS,
        }, fh, indent=2)

    X_train_raw = train_df[ALL_FEATURES].copy()
    X_test_raw  = test_df[ALL_FEATURES].copy()

    # StandardScaler for Gaussian features
    scaler_gaussian = StandardScaler()
    X_train_raw[GAUSSIAN_FEATURES] = scaler_gaussian.fit_transform(
        X_train_raw[GAUSSIAN_FEATURES]
    )
    X_test_raw[GAUSSIAN_FEATURES] = scaler_gaussian.transform(
        X_test_raw[GAUSSIAN_FEATURES]
    )
    joblib.dump(scaler_gaussian, "artifacts/preprocessing/scaler_gaussian.pkl")

    # RobustScaler for outlier-prone features
    scaler_outlier = RobustScaler()
    X_train_raw[OUTLIER_FEATURES] = scaler_outlier.fit_transform(
        X_train_raw[OUTLIER_FEATURES]
    )
    X_test_raw[OUTLIER_FEATURES] = scaler_outlier.transform(
        X_test_raw[OUTLIER_FEATURES]
    )
    joblib.dump(scaler_outlier, "artifacts/preprocessing/scaler_outlier.pkl")

    # Encode labels
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_df[TARGET_CLASS].astype(str))
    y_test  = encoder.transform(test_df[TARGET_CLASS].astype(str))
    joblib.dump(encoder, "artifacts/preprocessing/encoder.pkl")

    X_train = X_train_raw.values.astype(np.float32)
    X_test  = X_test_raw.values.astype(np.float32)

    # Persist arrays
    np.save("artifacts/data/X_train.npy", X_train)
    np.save("artifacts/data/y_train.npy", y_train)
    np.save("artifacts/data/X_test.npy",  X_test)
    np.save("artifacts/data/y_test.npy",  y_test)

    logger.info(
        f"Preprocessing done | train={X_train.shape} | test={X_test.shape} "
        f"| classes={list(encoder.classes_)}"
    )
    return X_train, y_train, X_test, y_test, encoder


# ─────────────────────────────────────────────────────────────────────────────
# 5.  MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    encoder,
) -> dict:
    """
    Train SVM with best hyperparameters from Set 3 tuning:
      kernel='rbf', C=10, gamma=0.1
    Saves model to artifacts/models/model.pkl.
    """
    logger.info("Training SVM (C=10, gamma=0.1, kernel=rbf) …")
    model = SVC(kernel="rbf", C=10, gamma=0.1, probability=True, random_state=42)
    model.fit(X_train, y_train)

    # Training set accuracy
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc  = accuracy_score(y_test,  model.predict(X_test))
    logger.info(f"Train accuracy = {train_acc:.4f}")
    logger.info(f"Test  accuracy = {test_acc:.4f}")

    # Save model
    joblib.dump(model, "artifacts/models/model.pkl")
    logger.info("Model saved → artifacts/models/model.pkl")

    history = {
        "model":          "SVC",
        "kernel":         "rbf",
        "C":              10,
        "gamma":          0.1,
        "train_accuracy": float(train_acc),
        "test_accuracy":  float(test_acc),
        "n_support_vectors": int(model.n_support_.sum()),
        "classes":        list(encoder.classes_),
        "timestamp":      datetime.utcnow().isoformat() + "Z",
    }
    with open("artifacts/metrics/training_history.json", "w") as fh:
        json.dump(history, fh, indent=2)

    return history


# ─────────────────────────────────────────────────────────────────────────────
# 6.  METADATA
# ─────────────────────────────────────────────────────────────────────────────
def save_metadata() -> None:
    ts      = datetime.utcnow().isoformat() + "Z"
    version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    with open("artifacts/metadata/last_retrain.txt",  "w") as fh: fh.write(ts)
    with open("artifacts/metadata/model_version.txt", "w") as fh: fh.write(f"v_{version}")
    with open("artifacts/metadata/data_version.txt",  "w") as fh: fh.write(f"v_{version}")
    logger.info(f"Metadata saved | version={version}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  MLOps Pipeline — TPC SVM Model Training")
    logger.info("=" * 60)

    train_df, test_df = load_data()
    train_df = engineer_features(train_df)
    test_df  = engineer_features(test_df)
    train_df, test_df, bin_edges = bin_target(train_df, test_df)
    X_train, y_train, X_test, y_test, encoder = preprocess_data(train_df, test_df)
    history = train_model(X_train, y_train, X_test, y_test, encoder)
    save_metadata()

    logger.info("=" * 60)
    logger.info(f"  Training complete | test accuracy = {history['test_accuracy']:.4f}")
    logger.info("=" * 60)
