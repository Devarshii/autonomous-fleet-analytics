import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# FILE PATHS
# ============================================================

INPUT_PATH = Path(
    "data/processed/fleet_positioning_recommendations.parquet"
)

OUTPUT_DIR = Path("outputs/fleet_positioning")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "recommended_vehicle_allocation.png"


# ============================================================
# LOAD FLEET POSITIONING DATA
# ============================================================

print("=" * 70)
print("FLEET POSITIONING VISUALIZATION")
print("=" * 70)

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Fleet positioning file not found: {INPUT_PATH}"
    )

print("\nLoading fleet-positioning recommendations...")

df = pd.read_parquet(INPUT_PATH)

print("Fleet-positioning data loaded successfully!")

print(f"\nZones available: {len(df):,}")

print(
    f"Total vehicles allocated: "
    f"{df['allocated_vehicles'].sum():,}"
)


# ============================================================
# SELECT TOP 15 ZONES
# ============================================================

top_zones = (
    df.sort_values(
        "allocated_vehicles",
        ascending=False
    )
    .head(15)
    .copy()
)

top_zones["zone_label"] = (
    top_zones["Zone"]
    + " ("
    + top_zones["Borough"]
    + ")"
)

print("\nTop 15 fleet-positioning zones:\n")

print(
    top_zones[
        [
            "Zone",
            "Borough",
            "lightgbm_forecast",
            "allocated_vehicles",
        ]
    ].to_string(index=False)
)


# ============================================================
# CREATE CHART
# ============================================================

plot_data = top_zones.sort_values(
    "allocated_vehicles",
    ascending=True
)

plt.figure(figsize=(12, 8))

bars = plt.barh(
    plot_data["zone_label"],
    plot_data["allocated_vehicles"]
)

plt.title(
    "Recommended Autonomous Vehicle Positioning\n"
    "Top 15 NYC Taxi Zones"
)

plt.xlabel("Recommended Number of Vehicles")
plt.ylabel("Taxi Zone")

plt.grid(
    axis="x",
    alpha=0.3
)


# ============================================================
# ADD VEHICLE COUNTS
# ============================================================

for bar in bars:
    width = bar.get_width()

    plt.text(
        width + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f"{int(width)}",
        va="center"
    )


# ============================================================
# SAVE CHART
# ============================================================

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\nChart saved successfully!")

print("\nOutput:")
print(OUTPUT_PATH)

print("\n" + "=" * 70)
print("FLEET POSITIONING VISUALIZATION COMPLETE")
print("=" * 70)