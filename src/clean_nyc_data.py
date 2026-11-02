import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# FILE PATHS
# ============================================================

INPUT_PATH = Path("data/raw/yellow_tripdata_2025-01.parquet")
OUTPUT_PATH = Path(
    "data/processed/yellow_tripdata_2025-01_clean.parquet"
)

print("=" * 65)
print("NYC YELLOW TAXI DATA - CLEANING PIPELINE")
print("=" * 65)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading raw dataset...")

df = pd.read_parquet(INPUT_PATH)

original_rows = len(df)

print("Dataset loaded successfully.")
print(f"Original trips: {original_rows:,}")

# ============================================================
# CREATE DERIVED VARIABLES
# ============================================================

df["trip_duration_minutes"] = (
    df["tpep_dropoff_datetime"]
    - df["tpep_pickup_datetime"]
).dt.total_seconds() / 60


df["average_speed_mph"] = np.where(
    df["trip_duration_minutes"] > 0,
    df["trip_distance"]
    / (df["trip_duration_minutes"] / 60),
    np.nan
)

# ============================================================
# DATA QUALITY RULES
# ============================================================

valid_date = (
    (df["tpep_pickup_datetime"] >= "2025-01-01")
    &
    (df["tpep_pickup_datetime"] < "2025-02-01")
)

valid_distance = (
    df["trip_distance"] > 0
)

valid_duration = (
    (df["trip_duration_minutes"] > 0)
    &
    (df["trip_duration_minutes"] <= 180)
)

valid_speed = (
    (df["average_speed_mph"] > 0)
    &
    (df["average_speed_mph"] <= 80)
)

valid_pickup = (
    df["PULocationID"].notna()
    &
    (df["PULocationID"] > 0)
)

valid_dropoff = (
    df["DOLocationID"].notna()
    &
    (df["DOLocationID"] > 0)
)

valid_fare = (
    df["fare_amount"] >= 0
)

# ============================================================
# COMBINE ALL CLEANING RULES
# ============================================================

valid_rows = (
    valid_date
    &
    valid_distance
    &
    valid_duration
    &
    valid_speed
    &
    valid_pickup
    &
    valid_dropoff
    &
    valid_fare
)

# ============================================================
# CREATE CLEAN DATASET
# ============================================================

clean_df = df.loc[valid_rows].copy()

clean_rows = len(clean_df)
removed_rows = original_rows - clean_rows

removed_percentage = (
    removed_rows / original_rows
) * 100

retained_percentage = (
    clean_rows / original_rows
) * 100

# ============================================================
# DISPLAY CLEANING RESULTS
# ============================================================

print("\n" + "=" * 65)
print("CLEANING RESULTS")
print("=" * 65)

print(f"\nOriginal trips : {original_rows:,}")
print(f"Clean trips    : {clean_rows:,}")
print(f"Removed trips  : {removed_rows:,}")

print(
    f"Removed        : {removed_percentage:.2f}%"
)

print(
    f"Retained       : {retained_percentage:.2f}%"
)

# ============================================================
# VERIFY CLEAN DATA
# ============================================================

print("\n" + "=" * 65)
print("VALIDATION CHECK")
print("=" * 65)

print(
    "Invalid distances remaining:",
    (clean_df["trip_distance"] <= 0).sum()
)

print(
    "Invalid durations remaining:",
    (
        (clean_df["trip_duration_minutes"] <= 0)
        |
        (clean_df["trip_duration_minutes"] > 180)
    ).sum()
)

print(
    "Speeds above 80 mph remaining:",
    (clean_df["average_speed_mph"] > 80).sum()
)

print(
    "Negative fares remaining:",
    (clean_df["fare_amount"] < 0).sum()
)

print(
    "Trips outside January remaining:",
    (
        (clean_df["tpep_pickup_datetime"] < "2025-01-01")
        |
        (clean_df["tpep_pickup_datetime"] >= "2025-02-01")
    ).sum()
)

# ============================================================
# SAVE CLEAN DATASET
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

print("\nSaving cleaned dataset...")

clean_df.to_parquet(
    OUTPUT_PATH,
    index=False
)

print("Cleaned dataset saved successfully.")

print(f"\nOutput file:")
print(OUTPUT_PATH)

print("\n" + "=" * 65)
print("CLEANING PIPELINE COMPLETE")
print("=" * 65)