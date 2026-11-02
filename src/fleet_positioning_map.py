import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================================================
# FILE PATHS
# =========================================================

FLEET_PATH = Path(
    "data/processed/fleet_positioning_recommendations.parquet"
)

TAXI_ZONE_PATH = Path(
    "data/external/taxi_zones/taxi_zones/taxi_zones.shp"
)

OUTPUT_DIR = Path("outputs/fleet_positioning")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "fleet_positioning_map.png"


# =========================================================
# START
# =========================================================

print("=" * 70)
print("AUTONOMOUS FLEET POSITIONING MAP")
print("=" * 70)


# =========================================================
# CHECK FILES
# =========================================================

if not FLEET_PATH.exists():
    raise FileNotFoundError(
        f"Fleet positioning file not found:\n{FLEET_PATH}"
    )

if not TAXI_ZONE_PATH.exists():
    raise FileNotFoundError(
        f"Taxi zone shapefile not found:\n{TAXI_ZONE_PATH}"
    )


# =========================================================
# LOAD FLEET RECOMMENDATIONS
# =========================================================

print("\nLoading fleet-positioning recommendations...")

fleet = pd.read_parquet(FLEET_PATH)

print("Fleet recommendations loaded successfully!")
print(f"Zones available: {len(fleet):,}")
print(
    f"Total allocated vehicles: "
    f"{fleet['allocated_vehicles'].sum():,.0f}"
)


# =========================================================
# LOAD TAXI ZONES
# =========================================================

print("\nLoading NYC Taxi Zone shapefile...")

zones = gpd.read_file(TAXI_ZONE_PATH)

print("Taxi zones loaded successfully!")
print(f"Taxi zone polygons: {len(zones):,}")


# =========================================================
# PREPARE LOCATION IDS
# =========================================================

zones["LocationID"] = pd.to_numeric(
    zones["LocationID"],
    errors="coerce"
)

fleet["LocationID"] = pd.to_numeric(
    fleet["LocationID"],
    errors="coerce"
)


# =========================================================
# MERGE GEOGRAPHY AND FLEET ALLOCATION
# =========================================================

print("\nJoining fleet allocation with taxi-zone geography...")

map_data = zones.merge(
    fleet[
        [
            "LocationID",
            "allocated_vehicles",
            "lightgbm_forecast"
        ]
    ],
    on="LocationID",
    how="left"
)

map_data["allocated_vehicles"] = (
    map_data["allocated_vehicles"].fillna(0)
)

map_data["lightgbm_forecast"] = (
    map_data["lightgbm_forecast"].fillna(0)
)

print("Geographic merge complete!")

mapped_total = int(
    map_data["allocated_vehicles"].sum()
)

print(
    f"Vehicles represented on map: {mapped_total:,}"
)


# =========================================================
# TOP 5 POSITIONING ZONES
# =========================================================

top_zones = (
    fleet
    .sort_values(
        "allocated_vehicles",
        ascending=False
    )
    .head(5)
    .copy()
)

print("\nTop 5 fleet-positioning zones:")

print(
    top_zones[
        [
            "LocationID",
            "Borough",
            "Zone",
            "allocated_vehicles"
        ]
    ].to_string(index=False)
)


# =========================================================
# CREATE FIGURE
# =========================================================

print("\nCreating cleaned fleet-positioning map...")

fig, ax = plt.subplots(
    figsize=(14, 11)
)


# =========================================================
# DRAW NYC MAP
# =========================================================

map_data.plot(
    column="allocated_vehicles",
    cmap="YlGnBu",
    linewidth=0.30,
    edgecolor="gray",
    legend=True,
    ax=ax,
    legend_kwds={
        "label": "Recommended Number of Autonomous Vehicles",
        "shrink": 0.70
    }
)


# =========================================================
# HIGHLIGHT TOP 5 ZONES
# =========================================================

top_ids = top_zones["LocationID"].tolist()

top_shapes = map_data[
    map_data["LocationID"].isin(top_ids)
]

top_shapes.boundary.plot(
    ax=ax,
    linewidth=1.5,
    edgecolor="black"
)


# =========================================================
# ADD NUMBER MARKERS
# =========================================================

for rank, (_, row) in enumerate(
    top_zones.iterrows(),
    start=1
):

    zone_shape = map_data[
        map_data["LocationID"]
        == row["LocationID"]
    ]

    if zone_shape.empty:
        continue

    point = (
        zone_shape
        .geometry
        .representative_point()
        .iloc[0]
    )

    ax.annotate(
        str(rank),
        xy=(point.x, point.y),
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        bbox=dict(
            boxstyle="circle,pad=0.25",
            facecolor="white",
            edgecolor="black",
            linewidth=0.8
        )
    )


# =========================================================
# TITLE
# =========================================================

ax.set_title(
    "Recommended Autonomous Vehicle Positioning\n"
    "Forecast-Driven Fleet Allocation — NYC",
    fontsize=16,
    pad=18
)

ax.set_axis_off()


# =========================================================
# TOP ZONE SUMMARY PANEL
# =========================================================

summary_lines = [
    "TOP 5 POSITIONING ZONES",
    ""
]

for rank, (_, row) in enumerate(
    top_zones.iterrows(),
    start=1
):
    summary_lines.append(
        f"{rank}. {row['Zone']} ({row['Borough']})"
    )

    summary_lines.append(
        f"   {int(row['allocated_vehicles'])} vehicles"
    )

summary_text = "\n".join(summary_lines)


fig.text(
    0.77,
    0.53,
    summary_text,
    fontsize=10,
    va="center",
    ha="left",
    bbox=dict(
        boxstyle="round,pad=0.7",
        facecolor="white",
        edgecolor="gray",
        alpha=0.95
    )
)


# =========================================================
# BUSINESS EXPLANATION
# =========================================================

fig.text(
    0.5,
    0.035,
    (
        "A simulated fleet of 1,000 autonomous vehicles is "
        "distributed across NYC TLC Taxi Zones in proportion "
        "to LightGBM-predicted ride demand."
    ),
    ha="center",
    fontsize=9
)


# =========================================================
# SAVE MAP
# =========================================================

plt.tight_layout(
    rect=[0, 0.06, 0.88, 1]
)

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# FINAL OUTPUT
# =========================================================

print("\nMap saved successfully!")
print(f"Output: {OUTPUT_PATH}")

print("\nTop 5 recommended positioning zones:")

for rank, (_, row) in enumerate(
    top_zones.iterrows(),
    start=1
):
    print(
        f"{rank}. {row['Zone']} "
        f"({row['Borough']}): "
        f"{int(row['allocated_vehicles'])} vehicles"
    )

print("\n" + "=" * 70)
print("CLEAN FLEET POSITIONING MAP COMPLETE")
print("=" * 70)