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

OUTPUT_FILE = OUTPUT_DIR / "zone_hour_demand_heatmap.png"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("NYC TAXI ZONE × HOUR DEMAND ANALYSIS")
print("=" * 70)

print("\nLoading demand dataset...")

df = pd.read_parquet(DATA_PATH)

print("Dataset loaded successfully.")
print(f"Rows: {len(df):,}")


# ============================================================
# VALIDATE COLUMNS
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

print("Required columns verified.")


# ============================================================
# CREATE DATE AND HOUR COLUMNS
# ============================================================

df["date"] = df["pickup_hour"].dt.date
df["hour"] = df["pickup_hour"].dt.hour


# ============================================================
# FIND TOP 15 ZONES USING AVERAGE DAILY DEMAND
# ============================================================

print("\nIdentifying top 15 demand zones...")

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

average_daily_zone_demand = (
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

top_15 = (
    average_daily_zone_demand
    .sort_values(
        "demand",
        ascending=False
    )
    .head(15)
    .copy()
)

top_zone_ids = top_15["LocationID"].tolist()

print("Top 15 zones identified successfully.")


# ============================================================
# FILTER TO TOP 15 ZONES
# ============================================================

top_df = df[
    df["LocationID"].isin(top_zone_ids)
].copy()


# ============================================================
# CALCULATE AVERAGE DEMAND BY ZONE AND HOUR
# ============================================================

print("\nCalculating average hourly demand...")

zone_hour_daily = (
    top_df.groupby(
        [
            "date",
            "LocationID",
            "Borough",
            "Zone",
            "hour",
        ],
        as_index=False
    )["demand"]
    .sum()
)

zone_hour_average = (
    zone_hour_daily.groupby(
        [
            "LocationID",
            "Borough",
            "Zone",
            "hour",
        ],
        as_index=False
    )["demand"]
    .mean()
)

zone_hour_average.rename(
    columns={
        "demand": "average_hourly_demand"
    },
    inplace=True
)


# ============================================================
# CREATE ZONE LABEL
# ============================================================

zone_hour_average["zone_label"] = (
    zone_hour_average["Zone"]
    + " ("
    + zone_hour_average["Borough"]
    + ")"
)


# ============================================================
# CREATE HEATMAP TABLE
# ============================================================

heatmap_data = zone_hour_average.pivot(
    index="zone_label",
    columns="hour",
    values="average_hourly_demand"
)

heatmap_data = heatmap_data.reindex(
    columns=range(24),
    fill_value=0
)

heatmap_data = heatmap_data.fillna(0)


# ============================================================
# ORDER ZONES BY OVERALL DEMAND
# ============================================================

top_15["zone_label"] = (
    top_15["Zone"]
    + " ("
    + top_15["Borough"]
    + ")"
)

zone_order = top_15["zone_label"].tolist()

heatmap_data = heatmap_data.reindex(zone_order)


# ============================================================
# FIND BUSIEST ZONE-HOUR COMBINATION
# ============================================================

busiest_row = zone_hour_average.loc[
    zone_hour_average[
        "average_hourly_demand"
    ].idxmax()
]

print("\n" + "=" * 70)
print("BUSIEST ZONE-HOUR COMBINATION")
print("=" * 70)

print(
    f"Zone: {busiest_row['Zone']}"
)

print(
    f"Borough: {busiest_row['Borough']}"
)

print(
    f"Hour: {int(busiest_row['hour']):02d}:00"
)

print(
    "Average pickups during this hour: "
    f"{busiest_row['average_hourly_demand']:,.0f}"
)


# ============================================================
# PRINT PEAK HOUR FOR EACH TOP ZONE
# ============================================================

peak_indices = (
    zone_hour_average
    .groupby("LocationID")[
        "average_hourly_demand"
    ]
    .idxmax()
)

peak_hours = (
    zone_hour_average
    .loc[
        peak_indices,
        [
            "Zone",
            "Borough",
            "hour",
            "average_hourly_demand",
        ]
    ]
    .sort_values(
        "average_hourly_demand",
        ascending=False
    )
)

print("\n" + "=" * 70)
print("PEAK HOUR FOR EACH TOP TAXI ZONE")
print("=" * 70)

for _, row in peak_hours.iterrows():

    print(
        f"{row['Zone']} ({row['Borough']}): "
        f"{int(row['hour']):02d}:00 "
        f"-> {row['average_hourly_demand']:,.0f} "
        "average pickups"
    )


# ============================================================
# CREATE HEATMAP
# ============================================================

print("\nCreating heatmap...")

fig, ax = plt.subplots(
    figsize=(16, 10)
)

image = ax.imshow(
    heatmap_data.values,
    aspect="auto"
)


# ============================================================
# X AXIS
# ============================================================

ax.set_xticks(
    range(24)
)

ax.set_xticklabels(
    [
        f"{hour:02d}"
        for hour in range(24)
    ]
)

ax.set_xlabel(
    "Hour of Day",
    fontsize=13
)


# ============================================================
# Y AXIS
# ============================================================

ax.set_yticks(
    range(len(heatmap_data.index))
)

ax.set_yticklabels(
    heatmap_data.index,
    fontsize=9
)

ax.set_ylabel(
    "Taxi Zone",
    fontsize=13
)


# ============================================================
# TITLE
# ============================================================

ax.set_title(
    "Average Hourly Yellow Taxi Demand in NYC's Top 15 Zones\n"
    "January 2025",
    fontsize=18,
    pad=15
)


# ============================================================
# COLOR BAR
# ============================================================

colorbar = fig.colorbar(
    image,
    ax=ax
)

colorbar.set_label(
    "Average Number of Pickups",
    fontsize=12
)


# ============================================================
# FINAL LAYOUT
# ============================================================

plt.tight_layout()


# ============================================================
# SAVE
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

print("\nHeatmap saved successfully.")

print("Output:")
print(OUTPUT_FILE)

print("\n" + "=" * 70)
print("ZONE × HOUR DEMAND ANALYSIS COMPLETE")
print("=" * 70)