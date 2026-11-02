import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# AUTONOMOUS VEHICLE MAINTENANCE & CHARGING ANALYTICS
# ============================================================

print("=" * 65)
print("AUTONOMOUS VEHICLE MAINTENANCE & CHARGING ANALYTICS")
print("=" * 65)

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

FLEET_SIZE = 1000
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

print(f"\nCreating simulated fleet of {FLEET_SIZE:,} autonomous vehicles...")

# ------------------------------------------------------------
# CREATE VEHICLE IDs
# ------------------------------------------------------------

fleet = pd.DataFrame({
    "vehicle_id": [f"AV_{i:04d}" for i in range(1, FLEET_SIZE + 1)]
})

print("\nFleet created successfully!")

print("\nFirst 10 vehicles:")
print(fleet.head(10))

print("\n" + "=" * 65)
print("FLEET SETUP COMPLETE")
print("=" * 65)
# ============================================================
# SIMULATE VEHICLE HEALTH DATA
# ============================================================

print("\nGenerating vehicle health and battery data...")

fleet["battery_pct"] = np.random.uniform(10, 100, FLEET_SIZE).round(1)

fleet["mileage"] = np.random.randint(
    5000,
    150000,
    FLEET_SIZE
)

fleet["battery_health_pct"] = np.random.uniform(
    70,
    100,
    FLEET_SIZE
).round(1)

fleet["tire_health_pct"] = np.random.uniform(
    40,
    100,
    FLEET_SIZE
).round(1)

fleet["brake_health_pct"] = np.random.uniform(
    45,
    100,
    FLEET_SIZE
).round(1)

fleet["motor_temperature_c"] = np.random.normal(
    70,
    10,
    FLEET_SIZE
).round(1)

fleet["days_since_service"] = np.random.randint(
    0,
    181,
    FLEET_SIZE
)

print("\nVehicle health data generated successfully!")

print("\nSample vehicle health data:")

print(
    fleet[
        [
            "vehicle_id",
            "battery_pct",
            "mileage",
            "battery_health_pct",
            "tire_health_pct",
            "brake_health_pct",
            "motor_temperature_c",
            "days_since_service"
        ]
    ].head(10)
)

print("\n" + "=" * 65)
print("VEHICLE HEALTH SIMULATION COMPLETE")
print("=" * 65)

# ============================================================
# IDENTIFY VEHICLES REQUIRING MAINTENANCE
# ============================================================

print("\nAnalyzing fleet maintenance requirements...")

# Individual maintenance warning conditions
fleet["battery_warning"] = fleet["battery_health_pct"] < 80
fleet["tire_warning"] = fleet["tire_health_pct"] < 55
fleet["brake_warning"] = fleet["brake_health_pct"] < 55
fleet["temperature_warning"] = fleet["motor_temperature_c"] > 90
fleet["service_warning"] = fleet["days_since_service"] > 150

# Count the number of maintenance warnings for each vehicle
warning_columns = [
    "battery_warning",
    "tire_warning",
    "brake_warning",
    "temperature_warning",
    "service_warning"
]

fleet["maintenance_warning_count"] = fleet[warning_columns].sum(axis=1)

# Determine maintenance priority
fleet["maintenance_priority"] = np.select(
    [
        fleet["maintenance_warning_count"] >= 3,
        fleet["maintenance_warning_count"] == 2,
        fleet["maintenance_warning_count"] == 1
    ],
    [
        "HIGH",
        "MEDIUM",
        "LOW"
    ],
    default="NORMAL"
)

print("\nMaintenance analysis complete!")

print("\nMaintenance priority distribution:")
print(fleet["maintenance_priority"].value_counts())

print("\nTop vehicles requiring maintenance:")

maintenance_vehicles = fleet[
    fleet["maintenance_priority"].isin(["HIGH", "MEDIUM"])
].copy()

maintenance_vehicles = maintenance_vehicles.sort_values(
    by="maintenance_warning_count",
    ascending=False
)

print(
    maintenance_vehicles[
        [
            "vehicle_id",
            "battery_health_pct",
            "tire_health_pct",
            "brake_health_pct",
            "motor_temperature_c",
            "days_since_service",
            "maintenance_warning_count",
            "maintenance_priority"
        ]
    ].head(15)
)

print("\n" + "=" * 65)
print("MAINTENANCE RISK IDENTIFICATION COMPLETE")
print("=" * 65)

# ============================================================
# CHARGING REQUIREMENT ANALYSIS
# ============================================================

print("\nAnalyzing vehicle charging requirements...")

fleet["charging_status"] = np.select(
    [
        fleet["battery_pct"] < 20,
        fleet["battery_pct"] < 35,
        fleet["battery_pct"] < 50
    ],
    [
        "CRITICAL",
        "CHARGE_SOON",
        "MONITOR"
    ],
    default="READY"
)

print("\nCharging status distribution:")
print(fleet["charging_status"].value_counts())

critical_charging = fleet[
    fleet["charging_status"] == "CRITICAL"
].copy()

print(f"\nVehicles requiring immediate charging: {len(critical_charging):,}")

print("\nSample vehicles requiring immediate charging:")

print(
    critical_charging[
        [
            "vehicle_id",
            "battery_pct",
            "battery_health_pct",
            "mileage",
            "maintenance_priority",
            "charging_status"
        ]
    ]
    .sort_values("battery_pct")
    .head(15)
)

print("\n" + "=" * 65)
print("CHARGING REQUIREMENT ANALYSIS COMPLETE")
print("=" * 65)
# ============================================================
# DETERMINE VEHICLE OPERATIONAL STATUS
# ============================================================

print("\nDetermining vehicle operational status...")

conditions = [
    fleet["maintenance_priority"] == "HIGH",
    fleet["maintenance_priority"] == "MEDIUM",
    fleet["charging_status"] == "CRITICAL"
]

choices = [
    "URGENT_MAINTENANCE",
    "MAINTENANCE",
    "CHARGING"
]

fleet["operational_status"] = np.select(
    conditions,
    choices,
    default="AVAILABLE"
)

print("\nOperational fleet status:")
print(fleet["operational_status"].value_counts())

available_count = (fleet["operational_status"] == "AVAILABLE").sum()
charging_count = (fleet["operational_status"] == "CHARGING").sum()
maintenance_count = (fleet["operational_status"] == "MAINTENANCE").sum()
urgent_count = (fleet["operational_status"] == "URGENT_MAINTENANCE").sum()

print("\nFleet availability summary:")
print(f"Total fleet:                 {len(fleet):,}")
print(f"Available vehicles:          {available_count:,}")
print(f"Charging vehicles:           {charging_count:,}")
print(f"Maintenance vehicles:        {maintenance_count:,}")
print(f"Urgent maintenance vehicles: {urgent_count:,}")

availability_rate = available_count / len(fleet) * 100

print(f"\nOperational availability rate: {availability_rate:.2f}%")

print("\n" + "=" * 65)
print("FLEET OPERATIONAL STATUS COMPLETE")
print("=" * 65)

# ============================================================
# MAINTENANCE RISK SCORE
# ============================================================

print("\nCalculating maintenance risk scores...")

# Convert health indicators into risk components
battery_risk = (100 - fleet["battery_health_pct"]) / 30
tire_risk = (100 - fleet["tire_health_pct"]) / 60
brake_risk = (100 - fleet["brake_health_pct"]) / 55

temperature_risk = np.clip(
    (fleet["motor_temperature_c"] - 60) / 40,
    0,
    1
)

service_risk = np.clip(
    fleet["days_since_service"] / 180,
    0,
    1
)

mileage_risk = np.clip(
    fleet["mileage"] / 150000,
    0,
    1
)

# Weighted maintenance risk score
fleet["maintenance_risk_score"] = (
    battery_risk * 20
    + tire_risk * 20
    + brake_risk * 20
    + temperature_risk * 15
    + service_risk * 15
    + mileage_risk * 10
)

fleet["maintenance_risk_score"] = np.clip(
    fleet["maintenance_risk_score"],
    0,
    100
).round(2)

# Rank vehicles from highest risk to lowest risk
fleet["maintenance_risk_rank"] = (
    fleet["maintenance_risk_score"]
    .rank(method="first", ascending=False)
    .astype(int)
)

print("\nTop 15 highest-risk vehicles:")

print(
    fleet[
        [
            "vehicle_id",
            "maintenance_risk_score",
            "maintenance_risk_rank",
            "maintenance_priority",
            "battery_health_pct",
            "tire_health_pct",
            "brake_health_pct",
            "days_since_service"
        ]
    ]
    .sort_values("maintenance_risk_score", ascending=False)
    .head(15)
)

print("\nAverage fleet maintenance risk:")
print(f"{fleet['maintenance_risk_score'].mean():.2f} / 100")

print("\nHighest maintenance risk:")
print(f"{fleet['maintenance_risk_score'].max():.2f} / 100")

print("\n" + "=" * 65)
print("MAINTENANCE RISK SCORING COMPLETE")
print("=" * 65)

# ============================================================
# MAINTENANCE SCHEDULING USING LOW-DEMAND WINDOWS
# ============================================================
print("\n" + "=" * 60)
print("MAINTENANCE SCHEDULING")
print("=" * 60)

# Load complete NYC hourly demand data
demand_path = Path("data/processed/nyc_hourly_demand_complete.parquet")

print("\nLoading hourly demand data...")
demand_df = pd.read_parquet(demand_path)

# Calculate total NYC demand for every hour
hourly_demand = (
    demand_df.groupby("pickup_hour")["demand"]
    .sum()
    .reset_index()
    .sort_values("demand")
)

print("Hourly demand data loaded successfully!")

print("\nTop 10 lowest-demand maintenance windows:")
print(hourly_demand.head(10).to_string(index=False))

print("\n" + "=" * 60)
print("LOW-DEMAND WINDOWS IDENTIFIED")
print("=" * 60)

# ============================================================
# SELECT VEHICLES FOR MAINTENANCE SCHEDULING
# ============================================================

print("\n" + "=" * 60)
print("SELECTING VEHICLES FOR MAINTENANCE")
print("=" * 60)

# Select vehicles currently requiring maintenance
maintenance_vehicles = fleet[
    fleet["operational_status"].isin(
        ["MAINTENANCE", "URGENT_MAINTENANCE"]
    )
].copy()

# Highest-risk vehicles should receive priority
maintenance_vehicles = maintenance_vehicles.sort_values(
    "maintenance_risk_score",
    ascending=False
).reset_index(drop=True)

print(f"\nVehicles requiring maintenance: {len(maintenance_vehicles):,}")

print("\nHighest-priority maintenance vehicles:")
print(
    maintenance_vehicles[
        [
            "vehicle_id",
            "maintenance_risk_score",
            "maintenance_priority",
            "operational_status"
        ]
    ].head(15).to_string(index=False)
)

print("\n" + "=" * 60)
print("MAINTENANCE VEHICLES SELECTED")
print("=" * 60)

# ============================================================
# ASSIGN VEHICLES TO LOW-DEMAND MAINTENANCE WINDOWS
# ============================================================

print("\n" + "=" * 60)
print("ASSIGNING MAINTENANCE WINDOWS")
print("=" * 60)

# Use the 15 lowest-demand hours as maintenance windows
maintenance_windows = hourly_demand.head(15).copy().reset_index(drop=True)

# Maximum number of vehicles that can be serviced in one window
MAINTENANCE_CAPACITY_PER_WINDOW = 20

schedule_records = []

vehicle_index = 0

for _, window in maintenance_windows.iterrows():

    for _ in range(MAINTENANCE_CAPACITY_PER_WINDOW):

        if vehicle_index >= len(maintenance_vehicles):
            break

        vehicle = maintenance_vehicles.iloc[vehicle_index]

        schedule_records.append({
            "vehicle_id": vehicle["vehicle_id"],
            "maintenance_risk_score": vehicle["maintenance_risk_score"],
            "maintenance_priority": vehicle["maintenance_priority"],
            "operational_status": vehicle["operational_status"],
            "scheduled_maintenance_hour": window["pickup_hour"],
            "city_demand_during_maintenance": window["demand"]
        })

        vehicle_index += 1

    if vehicle_index >= len(maintenance_vehicles):
        break

maintenance_schedule = pd.DataFrame(schedule_records)

print(f"\nVehicles scheduled: {len(maintenance_schedule):,}")

print("\nFirst 20 scheduled vehicles:")
print(maintenance_schedule.head(20).to_string(index=False))

print("\n" + "=" * 60)
print("MAINTENANCE WINDOW ASSIGNMENT COMPLETE")
print("=" * 60)

# ============================================================
# SAVE MAINTENANCE ANALYTICS OUTPUTS
# ============================================================

output_dir = Path("outputs/maintenance")
output_dir.mkdir(parents=True, exist_ok=True)

schedule_path = output_dir / "maintenance_schedule.csv"
fleet_path = output_dir / "fleet_health_status.csv"

maintenance_schedule.to_csv(schedule_path, index=False)
fleet.to_csv(fleet_path, index=False)

print("\n" + "=" * 60)
print("MAINTENANCE ANALYTICS RESULTS SAVED")
print("=" * 60)

print(f"\nMaintenance schedule: {schedule_path}")
print(f"Fleet health status: {fleet_path}")

print(f"\nMaintenance records saved: {len(maintenance_schedule):,}")
print(f"Fleet records saved: {len(fleet):,}")

print("\n" + "=" * 60)
print("MAINTENANCE ANALYTICS COMPLETE")
print("=" * 60)