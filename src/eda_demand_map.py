import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# FILE PATHS
# ============================================================

DEMAND_PATH = Path(
    "data/processed/nyc_hourly_demand_complete.parquet"
)

SHAPEFILE_PATH = Path(
    "data/external/taxi_zones/taxi_zones/taxi_zones.shp"
)

OUTPUT_DIR = Path("outputs/eda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "nyc_demand_hotspot_map.png"


# ============================================================
# LOAD DEMAND DATA
# ============================================================

print("=" * 70)
print("NYC GEOGRAPHIC DEMAND HOTSPOT ANALYSIS")
print("=" * 70)

print("\nLoading demand dataset...")

df = pd.read_parquet(DEMAND_PATH)

print("Demand dataset loaded successfully.")
print(f"Demand rows: {len(df):,}")


# ============================================================
# LOAD NYC TAXI ZONE SHAPEFILE
# ============================================================

print("\nLoading NYC taxi zone shapefile...")

zones = gpd.read_file(SHAPEFILE_PATH)

print("Taxi zone shapefile loaded successfully.")
print(f"Number of map zones: {len(zones):,}")


# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

required_demand_columns = [
    "pickup_hour",
    "LocationID",
    "Borough",
    "Zone",
    "demand",
]

missing_demand_columns = [
    column
    for column in required_demand_columns
    if column not in df.columns
]

if missing_demand_columns:
    raise ValueError(
        f"Missing demand columns: {missing_demand_columns}"
    )

required_map_columns = [
    "LocationID",
    "zone",
    "borough",
    "geometry",
]

missing_map_columns = [
    column
    for column in required_map_columns
    if column not in zones.columns
]

if missing_map_columns:
    raise ValueError(
        f"Missing shapefile columns: {missing_map_columns}"
    )

print("Required columns verified.")


# ============================================================
# REMOVE NON-NYC / UNKNOWN ZONES
# ============================================================

valid_boroughs = [
    "Manhattan",
    "Queens",
    "Brooklyn",
    "Bronx",
    "Staten Island",
]

zones = zones[
    zones["borough"].isin(valid_boroughs)
].copy()

print(
    f"NYC borough taxi zones retained: {len(zones):,}"
)


# ============================================================
# CREATE DATE COLUMN
# ============================================================

df["date"] = df["pickup_hour"].dt.date


# ============================================================
# CALCULATE DAILY DEMAND BY LOCATION
# ============================================================

print("\nCalculating daily demand by taxi zone...")

daily_zone_demand = (
    df.groupby(
        [
            "date",
            "LocationID",
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
        "LocationID",
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
# MERGE DEMAND WITH GEOGRAPHIC DATA
# ============================================================

print("\nMerging demand with taxi zone geography...")

map_data = zones.merge(
    average_zone_demand,
    on="LocationID",
    how="left"
)

map_data["average_daily_demand"] = (
    map_data["average_daily_demand"]
    .fillna(0)
)

print("Merge completed successfully.")


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("MAP DATA VALIDATION")
print("=" * 70)

print(
    f"Mapped taxi zones: {len(map_data):,}"
)

print(
    "Zones with positive demand: "
    f"{(map_data['average_daily_demand'] > 0).sum():,}"
)

print(
    "Zones with zero demand: "
    f"{(map_data['average_daily_demand'] == 0).sum():,}"
)


# ============================================================
# FIND HIGHEST-DEMAND GEOGRAPHIC ZONE
# ============================================================

highest_zone = map_data.loc[
    map_data["average_daily_demand"].idxmax()
]

print("\n" + "=" * 70)
print("HIGHEST-DEMAND GEOGRAPHIC ZONE")
print("=" * 70)

print(
    f"Zone: {highest_zone['zone']}"
)

print(
    f"Borough: {highest_zone['borough']}"
)

print(
    f"LocationID: {highest_zone['LocationID']}"
)

print(
    "Average daily pickups: "
    f"{highest_zone['average_daily_demand']:,.0f}"
)


# ============================================================
# TOP 10 GEOGRAPHIC DEMAND ZONES
# ============================================================

top_10 = (
    map_data[
        [
            "LocationID",
            "borough",
            "zone",
            "average_daily_demand",
        ]
    ]
    .sort_values(
        "average_daily_demand",
        ascending=False
    )
    .head(10)
)

print("\n" + "=" * 70)
print("TOP 10 GEOGRAPHIC DEMAND HOTSPOTS")
print("=" * 70)

print(
    top_10.to_string(
        index=False,
        formatters={
            "average_daily_demand":
                lambda x: f"{x:,.0f}"
        }
    )
)


# ============================================================
# CREATE MAP
# ============================================================

print("\nCreating NYC demand hotspot map...")

fig, ax = plt.subplots(
    figsize=(13, 13)
)

map_data.plot(
    column="average_daily_demand",
    ax=ax,
    legend=True,
    cmap="YlOrRd",
    edgecolor="white",
    linewidth=0.25,
    legend_kwds={
        "label": "Average Daily Yellow Taxi Pickups",
        "shrink": 0.65,
    },
    missing_kwds={
        "color": "lightgrey",
        "label": "No data",
    },
)


# ============================================================
# ADD BOROUGH BOUNDARIES
# ============================================================

borough_boundaries = (
    map_data.dissolve(
        by="borough"
    )
)

borough_boundaries.boundary.plot(
    ax=ax,
    linewidth=1.0,
    edgecolor="black"
)


# ============================================================
# ADD LABELS FOR TOP 5 HOTSPOTS
# ============================================================

top_5_ids = (
    map_data
    .nlargest(
        5,
        "average_daily_demand"
    )["LocationID"]
    .tolist()
)

top_5_map = map_data[
    map_data["LocationID"].isin(top_5_ids)
].copy()

label_points = (
    top_5_map.geometry
    .representative_point()
)

for (_, row), point in zip(
    top_5_map.iterrows(),
    label_points
):

    ax.annotate(
        row["zone"],
        xy=(point.x, point.y),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=7,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "alpha": 0.75,
            "edgecolor": "none",
        },
    )


# ============================================================
# TITLE
# ============================================================

ax.set_title(
    "NYC Yellow Taxi Demand Hotspots\n"
    "Average Daily Pickups — January 2025",
    fontsize=18,
    pad=18
)

ax.set_axis_off()


# ============================================================
# NOTE
# ============================================================

fig.text(
    0.5,
    0.04,
    "Demand is aggregated using official NYC TLC Taxi Zone LocationIDs.",
    ha="center",
    fontsize=9
)


# ============================================================
# FINAL LAYOUT
# ============================================================

plt.tight_layout()


# ============================================================
# SAVE MAP
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

print("\nMap saved successfully.")

print("Output:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("NYC GEOGRAPHIC DEMAND HOTSPOT ANALYSIS COMPLETE")
print("=" * 70)