# ============================================================
# AUTONOMOUS VEHICLE STUCK-RISK ANALYTICS
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

DEMAND_PATH = Path(
    "data/processed/nyc_hourly_demand_complete.parquet"
)

OUTPUT_DIR = Path("outputs/stuck_risk")


# ============================================================
# SETUP
# ============================================================

print("=" * 60)
print("AUTONOMOUS VEHICLE STUCK-RISK ANALYTICS")
print("=" * 60)

# Reproducible simulated AV conditions
np.random.seed(42)

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Check required demand dataset
if not DEMAND_PATH.exists():
    raise FileNotFoundError(
        f"Demand dataset not found: {DEMAND_PATH}"
    )

print("\nRequired demand dataset found!")
print(f"Dataset: {DEMAND_PATH}")

print("\nStuck-risk analytics environment ready.")

print("\n" + "=" * 60)
print("STUCK-RISK SETUP COMPLETE")
print("=" * 60)

# ============================================================
# LOAD NYC ZONE-LEVEL DEMAND DATA
# ============================================================

print("\nLoading NYC zone-level demand data...")

demand_df = pd.read_parquet(DEMAND_PATH)

print("NYC demand data loaded successfully!")

print(f"\nRows: {len(demand_df):,}")
print(f"Columns: {len(demand_df.columns)}")

print("\nAvailable columns:")
print(demand_df.columns.tolist())

print("\nSample data:")
print(demand_df.head())

print("\n" + "=" * 60)
print("NYC ZONE DATA LOADED")
print("=" * 60)

# ============================================================
# CREATE CONGESTION RISK SCORE
# ============================================================

print("\nCreating congestion risk scores...")

# Calculate demand percentile across the dataset
demand_df["congestion_risk_score"] = (
    demand_df["demand"].rank(pct=True) * 100
).round(2)

# Categorize congestion risk
def congestion_category(score):
    if score >= 90:
        return "VERY_HIGH"
    elif score >= 75:
        return "HIGH"
    elif score >= 50:
        return "MEDIUM"
    else:
        return "LOW"


demand_df["congestion_risk"] = demand_df[
    "congestion_risk_score"
].apply(congestion_category)

print("\nCongestion risk distribution:")
print(demand_df["congestion_risk"].value_counts())

print("\nTop 10 highest congestion-risk records:")

print(
    demand_df[
        [
            "pickup_hour",
            "LocationID",
            "Borough",
            "Zone",
            "demand",
            "congestion_risk_score",
            "congestion_risk",
        ]
    ]
    .sort_values("congestion_risk_score", ascending=False)
    .head(10)
    .to_string(index=False)
)

print("\n" + "=" * 60)
print("CONGESTION RISK ANALYSIS COMPLETE")
print("=" * 60)
# ============================================================
# CREATE TIME-OF-DAY RISK
# ============================================================

print("\nCreating time-of-day risk scores...")


def calculate_time_risk(hour):
    # Morning rush hour
    if 7 <= hour <= 9:
        return 80

    # Evening rush hour
    elif 16 <= hour <= 19:
        return 90

    # Late night
    elif hour >= 22 or hour <= 2:
        return 70

    # Normal daytime
    else:
        return 40


demand_df["time_risk_score"] = demand_df["hour"].apply(
    calculate_time_risk
)


def time_risk_category(score):
    if score >= 85:
        return "VERY_HIGH"
    elif score >= 70:
        return "HIGH"
    elif score >= 50:
        return "MEDIUM"
    else:
        return "LOW"


demand_df["time_risk"] = demand_df[
    "time_risk_score"
].apply(time_risk_category)


print("\nTime-risk distribution:")
print(demand_df["time_risk"].value_counts())

print("\nAverage risk by hour:")

hourly_time_risk = (
    demand_df.groupby("hour")["time_risk_score"]
    .mean()
    .reset_index()
)

print(hourly_time_risk.to_string(index=False))

print("\n" + "=" * 60)
print("TIME-OF-DAY RISK ANALYSIS COMPLETE")
print("=" * 60)

# ============================================================
# CREATE ZONE COMPLEXITY RISK
# ============================================================

print("\nCreating zone complexity risk scores...")


def calculate_zone_complexity(row):

    borough = row["Borough"]
    service_zone = row["service_zone"]

    if borough == "Manhattan":
        score = 85

    elif borough == "Brooklyn":
        score = 65

    elif borough == "Queens":
        score = 55

    elif borough == "Bronx":
        score = 55

    elif borough == "Staten Island":
        score = 35

    else:
        score = 40

    # Airports have additional operational complexity
    if service_zone == "Airports":
        score += 10

    return min(score, 100)


demand_df["zone_complexity_score"] = demand_df.apply(
    calculate_zone_complexity,
    axis=1
)


def complexity_category(score):
    if score >= 80:
        return "VERY_HIGH"
    elif score >= 60:
        return "HIGH"
    elif score >= 40:
        return "MEDIUM"
    else:
        return "LOW"


demand_df["zone_complexity_risk"] = demand_df[
    "zone_complexity_score"
].apply(complexity_category)


print("\nAverage zone complexity by borough:")

print(
    demand_df.groupby("Borough")["zone_complexity_score"]
    .mean()
    .sort_values(ascending=False)
)

print("\nZone complexity distribution:")

print(
    demand_df["zone_complexity_risk"]
    .value_counts()
)

print("\n" + "=" * 60)
print("ZONE COMPLEXITY RISK ANALYSIS COMPLETE")
print("=" * 60)
# ============================================================
# CALCULATE OVERALL AV STUCK-RISK SCORE
# ============================================================

print("\nCalculating overall autonomous vehicle stuck-risk score...")

# Weighted risk model
demand_df["stuck_risk_score"] = (
    0.45 * demand_df["congestion_risk_score"]
    + 0.30 * demand_df["time_risk_score"]
    + 0.25 * demand_df["zone_complexity_score"]
).round(2)


def stuck_risk_category(score):
    if score >= 80:
        return "CRITICAL"
    elif score >= 65:
        return "HIGH"
    elif score >= 45:
        return "MEDIUM"
    else:
        return "LOW"


demand_df["stuck_risk_level"] = demand_df[
    "stuck_risk_score"
].apply(stuck_risk_category)


print("\nStuck-risk distribution:")

print(
    demand_df["stuck_risk_level"]
    .value_counts()
)


print("\nTop 15 highest stuck-risk zone-hours:")

top_stuck_risk = (
    demand_df[
        [
            "pickup_hour",
            "LocationID",
            "Borough",
            "Zone",
            "demand",
            "congestion_risk_score",
            "time_risk_score",
            "zone_complexity_score",
            "stuck_risk_score",
            "stuck_risk_level",
        ]
    ]
    .sort_values("stuck_risk_score", ascending=False)
    .head(15)
)

print(top_stuck_risk.to_string(index=False))


print("\nAverage stuck-risk score:")
print(round(demand_df["stuck_risk_score"].mean(), 2))

print("\nHighest stuck-risk score:")
print(demand_df["stuck_risk_score"].max())


print("\n" + "=" * 60)
print("OVERALL STUCK-RISK SCORING COMPLETE")
print("=" * 60)

# ============================================================
# CREATE STUCK-RISK OPERATIONAL RECOMMENDATIONS
# ============================================================

print("\nCreating autonomous vehicle operational recommendations...")


def operational_recommendation(row):

    if row["stuck_risk_level"] == "CRITICAL":
        return "AVOID_IDLE_POSITIONING"

    elif row["stuck_risk_level"] == "HIGH":
        return "LIMIT_FLEET_EXPOSURE"

    elif row["stuck_risk_level"] == "MEDIUM":
        return "MONITOR"

    else:
        return "NORMAL_OPERATION"


demand_df["av_recommendation"] = demand_df.apply(
    operational_recommendation,
    axis=1
)


print("\nOperational recommendation distribution:")

print(
    demand_df["av_recommendation"]
    .value_counts()
)


print("\nTop critical zones where idle AV positioning should be avoided:")

critical_zones = (
    demand_df[
        demand_df["stuck_risk_level"] == "CRITICAL"
    ][
        [
            "pickup_hour",
            "LocationID",
            "Borough",
            "Zone",
            "demand",
            "stuck_risk_score",
            "av_recommendation",
        ]
    ]
    .sort_values("stuck_risk_score", ascending=False)
    .head(15)
)

print(critical_zones.to_string(index=False))


print("\n" + "=" * 60)
print("AV OPERATIONAL RECOMMENDATIONS COMPLETE")
print("=" * 60)
# ============================================================
# SAVE STUCK-RISK ANALYTICS RESULTS
# ============================================================

from pathlib import Path

output_dir = Path("outputs/stuck_risk")
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "stuck_risk_analysis.csv"

columns_to_save = [
    "pickup_hour",
    "LocationID",
    "Borough",
    "Zone",
    "demand",
    "congestion_risk_score",
    "time_risk_score",
    "zone_complexity_score",
    "stuck_risk_score",
    "stuck_risk_level",
    "av_recommendation",
]

demand_df[columns_to_save].to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 60)
print("STUCK-RISK RESULTS SAVED")
print("=" * 60)

print(f"\nOutput file: {output_file}")
print(f"Records saved: {len(demand_df):,}")

print("\n" + "=" * 60)
print("STUCK-RISK ANALYTICS COMPLETE")
print("=" * 60)