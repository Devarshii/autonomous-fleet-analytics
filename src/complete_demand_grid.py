import pandas as pd
from pathlib import Path

# ============================================================
# FILE PATHS
# ============================================================

INPUT_PATH = Path(
    "data/processed/nyc_hourly_demand.parquet"
)

LOOKUP_PATH = Path(
    "data/external/taxi_zone_lookup.csv"
)

OUTPUT_PATH = Path(
    "data/processed/nyc_hourly_demand_complete.parquet"
)

print("=" * 65)
print("NYC COMPLETE HOURLY DEMAND GRID")
print("=" * 65)

# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading hourly demand data...")

demand = pd.read_parquet(INPUT_PATH)

print("Hourly demand data loaded.")
print(f"Existing rows: {len(demand):,}")

print("\nLoading taxi zone lookup...")

zones = pd.read_csv(LOOKUP_PATH)

print("Taxi zone lookup loaded.")

# ============================================================
# GET ACTIVE PICKUP ZONES
# ============================================================

# ============================================================
# REMOVE NON-GEOGRAPHIC / UNKNOWN ZONES
# ============================================================

UNKNOWN_LOCATION_IDS = [264]

unknown_demand = demand.loc[
    demand["LocationID"].isin(UNKNOWN_LOCATION_IDS),
    "demand"
].sum()

print(
    f"\nDemand excluded from unknown locations: "
    f"{unknown_demand:,}"
)

demand = demand[
    ~demand["LocationID"].isin(UNKNOWN_LOCATION_IDS)
].copy()

# ============================================================
# GET ACTIVE GEOGRAPHIC PICKUP ZONES
# ============================================================

active_zone_ids = sorted(
    demand["LocationID"].unique()
)

print(
    f"\nActive pickup zones: "
    f"{len(active_zone_ids):,}"
)

# ============================================================
# CREATE COMPLETE HOURLY RANGE
# ============================================================

all_hours = pd.date_range(
    start="2025-01-01 00:00:00",
    end="2025-01-31 23:00:00",
    freq="h"
)

print(
    f"Hours in January: "
    f"{len(all_hours):,}"
)

# ============================================================
# CREATE EVERY ZONE × HOUR COMBINATION
# ============================================================

print("\nCreating complete zone-hour grid...")

complete_index = pd.MultiIndex.from_product(
    [
        all_hours,
        active_zone_ids
    ],
    names=[
        "pickup_hour",
        "LocationID"
    ]
)

complete_grid = (
    complete_index
    .to_frame(index=False)
)

print(
    f"Complete grid rows: "
    f"{len(complete_grid):,}"
)

# ============================================================
# MERGE EXISTING DEMAND
# ============================================================

print("\nAdding observed demand...")

observed_demand = demand[
    [
        "pickup_hour",
        "LocationID",
        "demand"
    ]
].copy()

complete_grid = complete_grid.merge(
    observed_demand,
    on=[
        "pickup_hour",
        "LocationID"
    ],
    how="left"
)

# ============================================================
# FILL MISSING DEMAND WITH ZERO
# ============================================================

missing_before = (
    complete_grid["demand"]
    .isna()
    .sum()
)

complete_grid["demand"] = (
    complete_grid["demand"]
    .fillna(0)
    .astype(int)
)

print(
    f"Missing zone-hours filled with zero: "
    f"{missing_before:,}"
)

# ============================================================
# ADD ZONE INFORMATION
# ============================================================

complete_grid = complete_grid.merge(
    zones[
        [
            "LocationID",
            "Borough",
            "Zone",
            "service_zone"
        ]
    ],
    on="LocationID",
    how="left"
)

# ============================================================
# ADD TIME FEATURES
# ============================================================

complete_grid["date"] = (
    complete_grid["pickup_hour"].dt.date
)

complete_grid["hour"] = (
    complete_grid["pickup_hour"].dt.hour
)

complete_grid["day_of_week"] = (
    complete_grid["pickup_hour"]
    .dt.day_name()
)

complete_grid["day_of_week_num"] = (
    complete_grid["pickup_hour"]
    .dt.dayofweek
)

complete_grid["is_weekend"] = (
    complete_grid["day_of_week_num"] >= 5
).astype(int)

# ============================================================
# VALIDATION
# ============================================================

expected_rows = (
    len(active_zone_ids)
    * len(all_hours)
)

actual_rows = len(complete_grid)

total_demand = (
    complete_grid["demand"].sum()
)

zero_demand_rows = (
    complete_grid["demand"] == 0
).sum()

print("\n" + "=" * 65)
print("VALIDATION RESULTS")
print("=" * 65)

print(
    f"\nExpected rows: "
    f"{expected_rows:,}"
)

print(
    f"Actual rows:   "
    f"{actual_rows:,}"
)

print(
    f"Grid complete: "
    f"{expected_rows == actual_rows}"
)

print(
    f"\nTotal demand: "
    f"{total_demand:,}"
)

print(
    f"Zero-demand zone-hours: "
    f"{zero_demand_rows:,}"
)

print(
    "Missing demand values:",
    complete_grid["demand"].isna().sum()
)

print(
    "Missing zone names:",
    complete_grid["Zone"].isna().sum()
)

# ============================================================
# SAVE COMPLETE DATASET
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

print("\nSaving complete demand dataset...")

complete_grid.to_parquet(
    OUTPUT_PATH,
    index=False
)

print("Complete demand dataset saved successfully.")

print("\nOutput file:")
print(OUTPUT_PATH)

print("\n" + "=" * 65)
print("COMPLETE DEMAND GRID FINISHED")
print("=" * 65)