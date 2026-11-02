import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = Path("data/processed/nyc_hourly_demand_complete.parquet")
OUTPUT_PATH = Path("data/processed/nyc_baseline_forecast.parquet")

print("=" * 70)
print("NYC TAXI DEMAND - SAME HOUR LAST WEEK BASELINE")
print("=" * 70)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading complete hourly demand dataset...")

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
# CREATE SAME-HOUR-LAST-WEEK FORECAST
# ============================================================

print("\nCreating same-hour-last-week baseline forecast...")

df["baseline_forecast"] = (
    df.groupby("LocationID")["demand"]
    .shift(168)
)

# ============================================================
# CHECK FORECAST AVAILABILITY
# ============================================================

available_predictions = df["baseline_forecast"].notna().sum()
missing_predictions = df["baseline_forecast"].isna().sum()

print("\n" + "=" * 70)
print("BASELINE FORECAST AVAILABILITY")
print("=" * 70)

print(f"Predictions available:   {available_predictions:,}")
print(f"Predictions unavailable: {missing_predictions:,}")

# ============================================================
# CREATE EVALUATION DATASET
# ============================================================

evaluation_df = df[
    df["baseline_forecast"].notna()
].copy()

print(f"\nRows used for evaluation: {len(evaluation_df):,}")

# ============================================================
# CALCULATE ABSOLUTE ERROR
# ============================================================

evaluation_df["absolute_error"] = np.abs(
    evaluation_df["demand"] -
    evaluation_df["baseline_forecast"]
)

# ============================================================
# MAE
# Mean Absolute Error
# ============================================================

mae = evaluation_df["absolute_error"].mean()

# ============================================================
# WAPE
# Weighted Absolute Percentage Error
# ============================================================

total_actual_demand = evaluation_df["demand"].sum()
total_absolute_error = evaluation_df["absolute_error"].sum()

if total_actual_demand > 0:
    wape = (
        total_absolute_error /
        total_actual_demand
    ) * 100
else:
    wape = np.nan

# ============================================================
# ADDITIONAL INFORMATION
# ============================================================

actual_total = evaluation_df["demand"].sum()
forecast_total = evaluation_df["baseline_forecast"].sum()

# ============================================================
# DISPLAY METRICS
# ============================================================

print("\n" + "=" * 70)
print("BASELINE MODEL PERFORMANCE")
print("=" * 70)

print(f"MAE:  {mae:.2f} rides per zone-hour")
print(f"WAPE: {wape:.2f}%")

print(f"\nActual demand in evaluation period:   {actual_total:,.0f}")
print(f"Forecast demand in evaluation period: {forecast_total:,.0f}")

# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("SAMPLE FORECASTS")
print("=" * 70)

sample = evaluation_df[
    evaluation_df["demand"] > 0
][
    [
        "pickup_hour",
        "LocationID",
        "Borough",
        "Zone",
        "demand",
        "baseline_forecast",
        "absolute_error"
    ]
].head(15)

print(sample.to_string(index=False))

# ============================================================
# SAVE BASELINE FORECAST DATASET
# ============================================================

print("\nSaving baseline forecast dataset...")

df.to_parquet(
    OUTPUT_PATH,
    index=False
)

print("Baseline forecast dataset saved successfully.")

print("\nOutput file:")
print(OUTPUT_PATH)

# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("BASELINE FORECAST EVALUATION COMPLETE")
print("=" * 70)