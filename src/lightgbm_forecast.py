import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error

# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = Path(
    "data/processed/nyc_demand_features.parquet"
)

MODEL_PATH = Path(
    "models/lightgbm_demand_model.pkl"
)

PREDICTION_PATH = Path(
    "data/processed/lightgbm_predictions.parquet"
)

# Make sure models folder exists
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("NYC TAXI DEMAND - LIGHTGBM FORECAST MODEL")
print("=" * 70)

# ============================================================
# LOAD FEATURE DATA
# ============================================================

print("\nLoading ML feature dataset...")

df = pd.read_parquet(DATA_PATH)

df = df.sort_values(
    ["pickup_hour", "LocationID"]
).reset_index(drop=True)

print("Dataset loaded successfully!")
print(f"Rows: {len(df):,}")

print(
    f"Date range: {df['pickup_hour'].min()} "
    f"to {df['pickup_hour'].max()}"
)

# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "LocationID",
    "hour",
    "day_of_week_num",
    "day_of_month",
    "is_weekend",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
    "rolling_mean_168",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos"
]

TARGET = "demand"

print("\nFeatures used:")

for feature in FEATURES:
    print(f"- {feature}")

# ============================================================
# TIME-BASED TRAIN / TEST SPLIT
# ============================================================

# Train:
# January 8 through January 24
#
# Test:
# January 25 through January 31
#
# IMPORTANT:
# We do NOT randomly split time-series data.

TEST_START = pd.Timestamp("2025-01-25 00:00:00")

train_df = df[
    df["pickup_hour"] < TEST_START
].copy()

test_df = df[
    df["pickup_hour"] >= TEST_START
].copy()

print("\n" + "=" * 70)
print("TIME-BASED TRAIN / TEST SPLIT")
print("=" * 70)

print(f"Training rows: {len(train_df):,}")
print(f"Testing rows:  {len(test_df):,}")

print(
    f"\nTraining period: "
    f"{train_df['pickup_hour'].min()} "
    f"to {train_df['pickup_hour'].max()}"
)

print(
    f"Testing period:  "
    f"{test_df['pickup_hour'].min()} "
    f"to {test_df['pickup_hour'].max()}"
)

# ============================================================
# PREPARE X AND y
# ============================================================

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

# ============================================================
# CREATE LIGHTGBM MODEL
# ============================================================

print("\nCreating LightGBM model...")

model = lgb.LGBMRegressor(
    objective="regression_l1",
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

# ============================================================
# TRAIN MODEL
# ============================================================

print("Training model...")

model.fit(
    X_train,
    y_train
)

print("Model training complete!")

# ============================================================
# MAKE PREDICTIONS
# ============================================================

print("\nGenerating predictions...")

predictions = model.predict(X_test)

# Demand cannot be negative
predictions = np.maximum(predictions, 0)

test_df["lightgbm_forecast"] = predictions

# ============================================================
# LIGHTGBM METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

absolute_error = np.abs(
    y_test.to_numpy() - predictions
)

total_actual = y_test.sum()

if total_actual > 0:
    wape = (
        absolute_error.sum() /
        total_actual
    ) * 100
else:
    wape = np.nan

# ============================================================
# BASELINE ON EXACT SAME TEST PERIOD
# ============================================================

baseline_predictions = test_df["lag_168"].to_numpy()

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_absolute_error = np.abs(
    y_test.to_numpy() - baseline_predictions
)

if total_actual > 0:
    baseline_wape = (
        baseline_absolute_error.sum() /
        total_actual
    ) * 100
else:
    baseline_wape = np.nan

# ============================================================
# IMPROVEMENT
# ============================================================

mae_improvement = (
    (baseline_mae - mae) /
    baseline_mae
) * 100

wape_improvement = (
    (baseline_wape - wape) /
    baseline_wape
) * 100

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MODEL PERFORMANCE - JANUARY 25 TO JANUARY 31")
print("=" * 70)

print("\nSame-hour-last-week baseline:")
print(f"Baseline MAE:  {baseline_mae:.2f}")
print(f"Baseline WAPE: {baseline_wape:.2f}%")

print("\nLightGBM:")
print(f"LightGBM MAE:  {mae:.2f}")
print(f"LightGBM WAPE: {wape:.2f}%")

print("\nImprovement over baseline:")
print(f"MAE improvement:  {mae_improvement:.2f}%")
print(f"WAPE improvement: {wape_improvement:.2f}%")

print("\nDemand totals:")

print(
    f"Actual demand:   "
    f"{y_test.sum():,.0f}"
)

print(
    f"Baseline demand: "
    f"{baseline_predictions.sum():,.0f}"
)

print(
    f"LightGBM demand: "
    f"{predictions.sum():,.0f}"
)

# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE LIGHTGBM PREDICTIONS")
print("=" * 70)

sample = test_df[
    test_df["demand"] > 0
][
    [
        "pickup_hour",
        "LocationID",
        "Borough",
        "Zone",
        "demand",
        "lag_168",
        "lightgbm_forecast"
    ]
].head(15)

print(sample.to_string(index=False))

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance_df = pd.DataFrame(
    {
        "feature": FEATURES,
        "importance": model.feature_importances_
    }
).sort_values(
    "importance",
    ascending=False
)

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance_df.to_string(index=False)
)

# ============================================================
# SAVE MODEL
# ============================================================

print("\nSaving trained model...")

joblib.dump(
    model,
    MODEL_PATH
)

print("Model saved successfully.")
print(f"Model file: {MODEL_PATH}")

# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_df.to_parquet(
    PREDICTION_PATH,
    index=False
)

print("\nPredictions saved successfully.")
print(f"Prediction file: {PREDICTION_PATH}")

# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("LIGHTGBM DEMAND FORECASTING COMPLETE")
print("=" * 70)