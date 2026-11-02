import pandas as pd
from pathlib import Path

# ============================================================
# FILE PATHS
# ============================================================

TRIP_PATH = Path(
    "data/processed/yellow_tripdata_2025-01_clean.parquet"
)

LOOKUP_PATH = Path(
    "data/external/taxi_zone_lookup.csv"
)

OUTPUT_PATH = Path(
    "data/processed/nyc_hourly_demand.parquet"
)

print("=" * 65)
print("NYC HOURLY RIDE DEMAND - DATA BUILDER")
print("=" * 65)

# ============================================================
# LOAD CLEAN TRIP DATA
# ============================================================

print("\nLoading cleaned trip data...")

df = pd.read_parquet(TRIP_PATH)

print("Trip data loaded successfully.")
print(f"Number of trips: {len(df):,}")

# ============================================================
# LOAD TAXI ZONE LOOKUP
# ============================================================

print("\nLoading taxi zone lookup...")

zones = pd.read_csv(LOOKUP_PATH)

print("Taxi zone lookup loaded successfully.")
print(f"Number of zones in lookup: {len(zones):,}")

# ============================================================
# CREATE HOURLY TIME BUCKET
# ============================================================

print("\nCreating hourly time buckets...")

df["pickup_hour"] = (
    df["tpep_pickup_datetime"]
    .dt.floor("h")
)

# ============================================================
# AGGREGATE TRIPS INTO HOURLY DEMAND
# ============================================================

print("Aggregating trips by hour and pickup zone...")

hourly_demand = (
    df.groupby(
        ["pickup_hour", "PULocationID"]
    )
    .size()
    .reset_index(name="demand")
)

print("Hourly demand aggregation complete.")

# ============================================================
# RENAME LOCATION ID
# ============================================================

hourly_demand = hourly_demand.rename(
    columns={"PULocationID": "LocationID"}
)

# ============================================================
# ADD ZONE AND BOROUGH NAMES
# ============================================================

hourly_demand = hourly_demand.merge(
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

hourly_demand["date"] = (
    hourly_demand["pickup_hour"].dt.date
)

hourly_demand["hour"] = (
    hourly_demand["pickup_hour"].dt.hour
)

hourly_demand["day_of_week"] = (
    hourly_demand["pickup_hour"].dt.day_name()
)

hourly_demand["day_of_week_num"] = (
    hourly_demand["pickup_hour"].dt.dayofweek
)

hourly_demand["is_weekend"] = (
    hourly_demand["day_of_week_num"] >= 5
).astype(int)

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 65)
print("HOURLY DEMAND DATASET")
print("=" * 65)

print(
    f"\nNumber of hourly zone records: "
    f"{len(hourly_demand):,}"
)

print(
    f"Unique pickup zones: "
    f"{hourly_demand['LocationID'].nunique():,}"
)

print(
    f"Total demand represented: "
    f"{hourly_demand['demand'].sum():,}"
)

print("\nFirst 10 rows:")

print(
    hourly_demand[
        [
            "pickup_hour",
            "LocationID",
            "Borough",
            "Zone",
            "demand"
        ]
    ].head(10)
)

# ============================================================
# DEMAND STATISTICS
# ============================================================

print("\n" + "=" * 65)
print("DEMAND STATISTICS")
print("=" * 65)

print(hourly_demand["demand"].describe())

# ============================================================
# TOP DEMAND RECORDS
# ============================================================

print("\n" + "=" * 65)
print("TOP 10 HIGHEST-DEMAND ZONE-HOURS")
print("=" * 65)

top_demand = hourly_demand.nlargest(
    10,
    "demand"
)

print(
    top_demand[
        [
            "pickup_hour",
            "Borough",
            "Zone",
            "demand"
        ]
    ].to_string(index=False)
)

# ============================================================
# SAVE DATASET
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

print("\nSaving hourly demand dataset...")

hourly_demand.to_parquet(
    OUTPUT_PATH,
    index=False
)

print("Hourly demand dataset saved successfully.")

print("\nOutput file:")
print(OUTPUT_PATH)

print("\n" + "=" * 65)
print("DEMAND DATA BUILDER COMPLETE")
print("=" * 65)