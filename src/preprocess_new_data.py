"""
src/preprocess_new_data.py
===========================
Preprocess new incoming TPC data for the MLOps pipeline.

Expected input format  : same columns as original dataset
  DateTime, Temperature, Humidity, Wind Speed,
  general diffuse flows, diffuse flows,
  Zone 1 Power Consumption, Zone 2 Power Consumption, Zone 3 Power Consumption

Steps
-----
1. Load data/new_data.csv
2. Validate columns
3. Engineer temporal features (Hour, DayOfWeek, Month from DateTime)
4. Apply EXISTING scalers (do NOT refit — would invalidate the feature space)
   - StandardScaler → Temperature, Humidity
   - RobustScaler   → Wind Speed, general diffuse flows, diffuse flows
5. Save scaled features → artifacts/data/X_new.npy   (used by monitor.py)
6. If Zone 1 Power Consumption is present:
   - Apply saved bin edges to create Target_Class
   - Back up train.csv and append new rows (drop duplicates)
"""

import json
import logging
import os
import shutil
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

# ── Directories ───────────────────────────────────────────────────────────────
for d in ["artifacts/data", "logs"]:
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

# ── Constants (must match model.py) ──────────────────────────────────────────
GAUSSIAN_FEATURES = ["Temperature", "Humidity"]
OUTLIER_FEATURES  = ["Wind Speed", "general diffuse flows", "diffuse flows"]
TEMPORAL_FEATURES = ["Hour", "DayOfWeek", "Month"]
ALL_FEATURES      = GAUSSIAN_FEATURES + OUTLIER_FEATURES + TEMPORAL_FEATURES
TARGET_RAW        = "Zone 1 Power Consumption"
TARGET_CLASS      = "Target_Class"
BIN_LABELS        = ["Low", "Medium", "High"]


# ─────────────────────────────────────────────────────────────────────────────
def _check_prerequisites():
    required = [
        "artifacts/preprocessing/feature_columns.json",
        "artifacts/preprocessing/scaler_gaussian.pkl",
        "artifacts/preprocessing/scaler_outlier.pkl",
        "artifacts/preprocessing/bin_edges.json",
        "data/new_data.csv",
        "train/train.csv",
    ]
    missing = [p for p in required if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing required files:\n" + "\n".join(f"  {p}" for p in missing)
        )


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal features from DateTime."""
    df = df.copy()
    df["DateTime"]  = pd.to_datetime(df["DateTime"])
    df["Hour"]      = df["DateTime"].dt.hour
    df["DayOfWeek"] = df["DateTime"].dt.dayofweek
    df["Month"]     = df["DateTime"].dt.month
    return df


def validate_columns(df: pd.DataFrame) -> None:
    raw_required = GAUSSIAN_FEATURES + OUTLIER_FEATURES + ["DateTime"]
    missing = [c for c in raw_required if c not in df.columns]
    if missing:
        raise ValueError(
            f"new_data.csv is missing required columns: {missing}\n"
            f"Available: {list(df.columns)}"
        )
    logger.info("Column validation passed ✓")


def apply_preprocessing(df: pd.DataFrame) -> np.ndarray:
    """Apply existing scalers (no refit)."""
    scaler_gaussian = joblib.load("artifacts/preprocessing/scaler_gaussian.pkl")
    scaler_outlier  = joblib.load("artifacts/preprocessing/scaler_outlier.pkl")

    X = df[ALL_FEATURES].copy()

    # Impute any NaNs before scaling
    for col in ALL_FEATURES:
        if X[col].isna().any():
            fill = X[col].mean()
            X[col].fillna(fill, inplace=True)
            logger.warning(f"Imputed NaNs in '{col}' with column mean ({fill:.4f})")

    X[GAUSSIAN_FEATURES] = scaler_gaussian.transform(X[GAUSSIAN_FEATURES])
    X[OUTLIER_FEATURES]  = scaler_outlier.transform(X[OUTLIER_FEATURES])

    X_scaled = X.values.astype(np.float32)
    logger.info(f"Scaling applied | output shape: {X_scaled.shape}")
    return X_scaled


def apply_binning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply saved quantile bin edges to create Target_Class column."""
    with open("artifacts/preprocessing/bin_edges.json") as fh:
        bin_info = json.load(fh)
    bin_edges = bin_info["bin_edges"]

    df = df.copy()
    df[TARGET_CLASS] = pd.cut(
        df[TARGET_RAW],
        bins=bin_edges,
        labels=BIN_LABELS,
        include_lowest=True,
    )
    n_null = df[TARGET_CLASS].isna().sum()
    if n_null > 0:
        logger.warning(
            f"{n_null} rows had {TARGET_RAW} outside training bin range — dropped"
        )
        df = df.dropna(subset=[TARGET_CLASS])
    return df


def save_new_artifact(X_new: np.ndarray) -> None:
    path = "artifacts/data/X_new.npy"
    np.save(path, X_new)
    logger.info(f"Saved X_new.npy → {path} (shape={X_new.shape})")


def merge_into_training(df: pd.DataFrame) -> None:
    """Back up train.csv then append new labelled rows (drop duplicates)."""
    train_path  = "train/train.csv"
    backup_path = f"train/train_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    shutil.copy(train_path, backup_path)
    logger.info(f"Backed up train.csv → {backup_path}")

    train_df  = pd.read_csv(train_path)
    before    = len(train_df)

    # Keep only columns present in train.csv
    new_rows  = df[[c for c in train_df.columns if c in df.columns]].copy()
    merged    = pd.concat([train_df, new_rows], ignore_index=True).drop_duplicates()
    added     = len(merged) - before

    merged.to_csv(train_path, index=False)
    logger.info(
        f"train.csv updated: {before} → {len(merged)} rows "
        f"(+{added} new, {len(new_rows) - added} duplicates dropped)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("  MLOps Pipeline — Preprocess New TPC Data")
    logger.info("=" * 60)

    _check_prerequisites()

    new_df = pd.read_csv("data/new_data.csv")
    logger.info(f"Loaded {len(new_df)} rows from data/new_data.csv")

    validate_columns(new_df)
    new_df  = engineer_features(new_df)
    X_new   = apply_preprocessing(new_df)
    save_new_artifact(X_new)

    # If target column present → bin it and merge into training set
    has_target = TARGET_RAW in new_df.columns
    if has_target:
        new_df = apply_binning(new_df)
        merge_into_training(new_df)
    else:
        logger.info(
            f"'{TARGET_RAW}' column absent — treating as unlabelled. "
            "Skipping training-set merge."
        )

    logger.info("=" * 60)
    logger.info("  Preprocessing complete")
    logger.info("=" * 60)
