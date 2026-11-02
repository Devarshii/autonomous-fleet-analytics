import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = Path(
    "data/processed/nyc_hourly_demand_complete.parquet"
)

FEATURE_PATH = Path(
    "data/processed/nyc_demand_features.parquet"
)

print("=" * 70)
print("NYC TAXI DEMAND - ML FEATURE ENGINEERING")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading hourly demand dataset...")

df = pd.read_parquet(DATA_PATH)

print("Dataset loaded successfully!")
print(f"Rows: {len(df):,}")

# ============================================================
# SORT DATA
# ============================================================

df = df.sort_values(
    ["LocationID", "pickup_hour"]
).reset_index(drop=True)

# ============================================================
# CALENDAR FEATURES
# ============================================================

print("\nCreating calendar features...")

df["hour"] = df["pickup_hour"].dt.hour
df["day_of_week_num"] = df["pickup_hour"].dt.dayofweek
df["day_of_month"] = df["pickup_hour"].dt.day
df["is_weekend"] = (
    df["day_of_week_num"] >= 5
).astype(int)

# ============================================================
# LAG FEATURES
# ============================================================

print("Creating lag features...")

grouped = df.groupby("LocationID")["demand"]

# Previous hour
df["lag_1"] = grouped.shift(1)

# Same hour yesterday
df["lag_24"] = grouped.shift(24)

# Same hour one week ago
df["lag_168"] = grouped.shift(168)

# ============================================================
# ROLLING FEATURES
# ============================================================

print("Creating rolling demand features...")

# IMPORTANT:
# shift(1) ensures the current hour's demand is NOT used
# to predict itself.

df["rolling_mean_24"] = (
    df.groupby("LocationID")["demand"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=24,
            min_periods=1
        ).mean()
    )
)

df["rolling_mean_168"] = (
    df.groupby("LocationID")["demand"]
    .transform(
        lambda x: x.shift(1).rolling(
            window=168,
            min_periods=1
        ).mean()
    )
)

# ============================================================
# CYCLICAL TIME FEATURES
# ============================================================

print("Creating cyclical time features...")

df["hour_sin"] = np.sin(
    2 * np.pi * df["hour"] / 24
)

df["hour_cos"] = np.cos(
    2 * np.pi * df["hour"] / 24
)

df["dow_sin"] = np.sin(
    2 * np.pi * df["day_of_week_num"] / 7
)

df["dow_cos"] = np.cos(
    2 * np.pi * df["day_of_week_num"] / 7
)

# ============================================================
# FEATURE VALIDATION
# ============================================================

feature_columns = [
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

print("\n" + "=" * 70)
print("FEATURE VALIDATION")
print("=" * 70)

print("\nMissing values by feature:")

print(
    df[feature_columns]
    .isna()
    .sum()
)

# ============================================================
# MODEL-READY ROWS
# ============================================================

# lag_168 requires one full week of history.
# Remove rows where it is unavailable.

model_df = df.dropna(
    subset=["lag_168"]
).copy()

print("\n" + "=" * 70)
print("MODEL DATASET")
print("=" * 70)

print(f"Original rows:    {len(df):,}")
print(f"Model-ready rows: {len(model_df):,}")

print("\nDate range:")

print(
    model_df["pickup_hour"].min(),
    "to",
    model_df["pickup_hour"].max()
)

# ============================================================
# SAMPLE
# ============================================================

print("\nSample feature rows:")

sample_columns = [
    "pickup_hour",
    "LocationID",
    "Zone",
    "demand",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
    "rolling_mean_168"
]

print(
    model_df[sample_columns]
    .head(10)
    .to_string(index=False)
)

# ============================================================
# SAVE FEATURE DATASET
# ============================================================

print("\nSaving ML feature dataset...")

model_df.to_parquet(
    FEATURE_PATH,
    index=False
)

print("Feature dataset saved successfully.")

print("\nOutput file:")
print(FEATURE_PATH)

print("\n" + "=" * 70)
print("ML FEATURE ENGINEERING COMPLETE")
print("=" * 70)