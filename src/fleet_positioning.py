import pandas as pd
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PREDICTION_PATH = Path("data/processed/lightgbm_predictions.parquet")

TOTAL_FLEET_SIZE = 1000


# ============================================================
# LOAD FORECAST DATA
# ============================================================

print("=" * 70)
print("AUTONOMOUS VEHICLE FLEET POSITIONING")
print("=" * 70)

if not PREDICTION_PATH.exists():
    raise FileNotFoundError(
        f"Prediction dataset not found: {PREDICTION_PATH}"
    )

print("\nLoading LightGBM demand forecasts...")

df = pd.read_parquet(PREDICTION_PATH)

print("Forecast dataset loaded successfully!")

print(f"\nRows: {len(df):,}")
print(f"Columns: {len(df.columns)}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "pickup_hour",
    "LocationID",
    "Zone",
    "Borough",
    "lightgbm_forecast",
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\nRequired columns found successfully.")


# ============================================================
# BASIC DATA CHECK
# ============================================================

print("\nAvailable forecast period:")

print("Start:", df["pickup_hour"].min())
print("End:  ", df["pickup_hour"].max())

print("\nFleet size:")
print(f"{TOTAL_FLEET_SIZE:,} autonomous vehicles")


# ============================================================
# SELECT ONE FORECAST HOUR
# ============================================================

target_hour = df["pickup_hour"].max()

hourly_forecast = df[
    df["pickup_hour"] == target_hour
].copy()

hourly_forecast["lightgbm_forecast"] = (
    hourly_forecast["lightgbm_forecast"]
    .clip(lower=0)
)

hourly_forecast = hourly_forecast.sort_values(
    "lightgbm_forecast",
    ascending=False
)


# ============================================================
# DISPLAY TOP DEMAND ZONES
# ============================================================

print("\n" + "=" * 70)
print("TARGET FLEET-POSITIONING HOUR")
print("=" * 70)

print(target_hour)

print("\nTop 10 zones by predicted demand:")

print(
    hourly_forecast[
        [
            "LocationID",
            "Borough",
            "Zone",
            "lightgbm_forecast",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# SUMMARY
# ============================================================

total_predicted_demand = (
    hourly_forecast["lightgbm_forecast"].sum()
)

print("\n" + "=" * 70)
print("FLEET POSITIONING INPUT SUMMARY")
print("=" * 70)

print(
    f"Total predicted rides for target hour: "
    f"{total_predicted_demand:,.0f}"
)

print(
    f"Available autonomous vehicles: "
    f"{TOTAL_FLEET_SIZE:,}"
)

print(
    f"Taxi zones considered: "
    f"{len(hourly_forecast):,}"
)

print("\nFleet-positioning input preparation complete.")
# ============================================================
# PROPORTIONAL FLEET ALLOCATION
# ============================================================

print("\n" + "=" * 70)
print("PROPORTIONAL FLEET ALLOCATION")
print("=" * 70)

# Calculate each zone's share of predicted demand
if total_predicted_demand <= 0:
    raise ValueError("Total predicted demand must be greater than zero.")

hourly_forecast["demand_share"] = (
    hourly_forecast["lightgbm_forecast"]
    / total_predicted_demand
)

# Initial whole-vehicle allocation
hourly_forecast["allocated_vehicles"] = (
    hourly_forecast["demand_share"]
    * TOTAL_FLEET_SIZE
).astype(int)

# Because converting to integers rounds down, some vehicles may remain
vehicles_allocated = hourly_forecast["allocated_vehicles"].sum()

vehicles_remaining = (
    TOTAL_FLEET_SIZE - vehicles_allocated
)

# Calculate fractional remainder for fair distribution
hourly_forecast["allocation_fraction"] = (
    hourly_forecast["demand_share"]
    * TOTAL_FLEET_SIZE
    - hourly_forecast["allocated_vehicles"]
)

# Give remaining vehicles to zones with the largest fractional remainder
if vehicles_remaining > 0:
    remainder_indices = (
        hourly_forecast
        .sort_values(
            "allocation_fraction",
            ascending=False
        )
        .head(vehicles_remaining)
        .index
    )

    hourly_forecast.loc[
        remainder_indices,
        "allocated_vehicles"
    ] += 1


# ============================================================
# VERIFY ALLOCATION
# ============================================================

final_vehicle_count = (
    hourly_forecast["allocated_vehicles"].sum()
)

print(f"\nTotal fleet available: {TOTAL_FLEET_SIZE:,}")

print(
    f"Total vehicles allocated: "
    f"{final_vehicle_count:,}"
)

print(
    f"Unallocated vehicles: "
    f"{TOTAL_FLEET_SIZE - final_vehicle_count:,}"
)


# ============================================================
# DISPLAY TOP VEHICLE ALLOCATIONS
# ============================================================

print("\nTop 15 recommended fleet positions:\n")

display_columns = [
    "LocationID",
    "Borough",
    "Zone",
    "lightgbm_forecast",
    "demand_share",
    "allocated_vehicles",
]

print(
    hourly_forecast[
        display_columns
    ]
    .sort_values(
        "allocated_vehicles",
        ascending=False
    )
    .head(15)
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("FLEET ALLOCATION COMPLETE")
print("=" * 70)
# ============================================================
# SAVE FLEET POSITIONING RESULTS
# ============================================================

OUTPUT_PATH = Path(
    "data/processed/fleet_positioning_recommendations.parquet"
)

output_columns = [
    "pickup_hour",
    "LocationID",
    "Borough",
    "Zone",
    "lightgbm_forecast",
    "demand_share",
    "allocated_vehicles",
]

fleet_recommendations = hourly_forecast[
    output_columns
].copy()

fleet_recommendations = fleet_recommendations.sort_values(
    "allocated_vehicles",
    ascending=False
)

print("\nSaving fleet-positioning recommendations...")

fleet_recommendations.to_parquet(
    OUTPUT_PATH,
    index=False
)

print("Fleet-positioning recommendations saved successfully!")

print("\nOutput file:")
print(OUTPUT_PATH)

print(
    f"\nSaved zones: "
    f"{len(fleet_recommendations):,}"
)

print(
    f"Saved vehicle allocation: "
    f"{fleet_recommendations['allocated_vehicles'].sum():,}"
)

print("\n" + "=" * 70)
print("FLEET POSITIONING RESULTS SAVED")
print("=" * 70)