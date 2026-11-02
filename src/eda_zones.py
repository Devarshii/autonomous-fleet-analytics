import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = Path(
    "data/processed/nyc_hourly_demand_complete.parquet"
)

OUTPUT_DIR = Path("outputs/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "top_15_demand_zones.png"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("NYC TAXI ZONE DEMAND ANALYSIS")
print("=" * 70)

print("\nLoading complete hourly demand dataset...")

df = pd.read_parquet(DATA_PATH)

print("Dataset loaded successfully.")
print(f"Rows: {len(df):,}")


# ============================================================
# BASIC VALIDATION
# ============================================================

required_columns = [
    "pickup_hour",
    "LocationID",
    "Borough",
    "Zone",
    "demand",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("\nRequired columns verified.")


# ============================================================
# CREATE DATE COLUMN
# ============================================================

df["date"] = df["pickup_hour"].dt.date


# ============================================================
# CALCULATE DAILY DEMAND FOR EACH TAXI ZONE
# ============================================================

print("\nCalculating daily demand by taxi zone...")

daily_zone_demand = (
    df.groupby(
        [
            "date",
            "LocationID",
            "Borough",
            "Zone",
        ],
        as_index=False
    )["demand"]
    .sum()
)


# ============================================================
# CALCULATE AVERAGE DAILY DEMAND
# ============================================================

average_zone_demand = (
    daily_zone_demand.groupby(
        [
            "LocationID",
            "Borough",
            "Zone",
        ],
        as_index=False
    )["demand"]
    .mean()
)

average_zone_demand.rename(
    columns={
        "demand": "average_daily_demand"
    },
    inplace=True
)


# ============================================================
# GET TOP 15 TAXI ZONES
# ============================================================

top_15_zones = (
    average_zone_demand
    .sort_values(
        "average_daily_demand",
        ascending=False
    )
    .head(15)
    .copy()
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 70)
print("TOP 15 NYC TAXI ZONES BY AVERAGE DAILY DEMAND")
print("=" * 70)

display_columns = [
    "LocationID",
    "Borough",
    "Zone",
    "average_daily_demand",
]

print(
    top_15_zones[display_columns]
    .to_string(
        index=False,
        formatters={
            "average_daily_demand":
                lambda x: f"{x:,.0f}"
        }
    )
)


# ============================================================
# HIGHEST-DEMAND TAXI ZONE
# ============================================================

highest_zone = top_15_zones.iloc[0]

print("\n" + "=" * 70)
print("HIGHEST-DEMAND TAXI ZONE")
print("=" * 70)

print(
    f"Zone: {highest_zone['Zone']}"
)

print(
    f"Borough: {highest_zone['Borough']}"
)

print(
    f"LocationID: {highest_zone['LocationID']}"
)

print(
    "Average daily pickups: "
    f"{highest_zone['average_daily_demand']:,.0f}"
)


# ============================================================
# PREPARE LABELS FOR CHART
# ============================================================

top_15_zones["label"] = (
    top_15_zones["Zone"]
    + " ("
    + top_15_zones["Borough"]
    + ")"
)


# Sort ascending so the largest bar appears at the top
plot_data = top_15_zones.sort_values(
    "average_daily_demand",
    ascending=True
)


# ============================================================
# CREATE CHART
# ============================================================

plt.figure(figsize=(13, 9))

plt.barh(
    plot_data["label"],
    plot_data["average_daily_demand"]
)

plt.title(
    "Top 15 NYC Yellow Taxi Zones by Average Daily Demand\n"
    "January 2025",
    fontsize=18
)

plt.xlabel(
    "Average Number of Pickups per Day",
    fontsize=13
)

plt.ylabel(
    "Taxi Zone",
    fontsize=13
)

plt.grid(
    axis="x",
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# SAVE CHART
# ============================================================

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\nChart saved successfully.")

print("Output:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("TAXI ZONE DEMAND ANALYSIS COMPLETE")
print("=" * 70)