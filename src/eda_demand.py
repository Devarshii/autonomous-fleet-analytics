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

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("NYC RIDE DEMAND - EXPLORATORY DATA ANALYSIS")
print("=" * 65)

print("\nLoading demand dataset...")

df = pd.read_parquet(DATA_PATH)

print("Dataset loaded successfully.")
print(f"Rows: {len(df):,}")
print(f"Total demand: {df['demand'].sum():,}")

# ============================================================
# DEMAND BY HOUR OF DAY
# ============================================================

print("\nCalculating demand by hour of day...")

hourly_pattern = (
    df.groupby("hour")["demand"]
    .sum()
    .reset_index()
)

print("\nDemand by hour:")

print(
    hourly_pattern.to_string(index=False)
)

# ============================================================
# FIND PEAK HOUR
# ============================================================

peak_row = hourly_pattern.loc[
    hourly_pattern["demand"].idxmax()
]

peak_hour = int(peak_row["hour"])
peak_demand = int(peak_row["demand"])

print("\n" + "=" * 65)
print("PEAK DEMAND")
print("=" * 65)

print(f"Peak hour: {peak_hour}:00")
print(f"Total rides during this hour across January: {peak_demand:,}")

# ============================================================
# CREATE CHART
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    hourly_pattern["hour"],
    hourly_pattern["demand"],
    marker="o",
    linewidth=2
)

plt.title(
    "Average NYC Yellow Taxi Demand by Day of Week\nJanuary 2025",
    fontsize=16
)

plt.xlabel(
    "Hour of Day",
    fontsize=12
)

plt.ylabel(
    "Average Pickups per Day",
    fontsize=12
)

plt.xticks(
    range(0, 24)
)

plt.grid(
    alpha=0.3
)

plt.tight_layout()

# ============================================================
# SAVE CHART
# ============================================================

OUTPUT_FILE = (
    OUTPUT_DIR
    / "demand_by_hour.png"
)

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nChart saved successfully.")

print("Output:")
print(OUTPUT_FILE)

print("\n" + "=" * 65)
print("EDA STEP COMPLETE")
print("=" * 65)

# ============================================================
# DEMAND BY DAY OF WEEK
# ============================================================

print("\nCalculating demand by day of week...")

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]

# Create calendar date from hourly timestamp
df["date"] = df["pickup_hour"].dt.date

# Calculate total demand for each individual date
daily_totals = (
    df.groupby(["date", "day_of_week"])["demand"]
    .sum()
    .reset_index()
)

# Calculate average daily demand for each weekday
daily_pattern = (
    daily_totals.groupby("day_of_week")["demand"]
    .mean()
    .reindex(day_order)
    .reset_index()
)

daily_pattern["demand"] = (
    daily_pattern["demand"]
    .round()
    .astype(int)
)

print("\nAverage demand by day of week:")
print(daily_pattern.to_string(index=False))


# ============================================================
# FIND HIGHEST-DEMAND DAY
# ============================================================

peak_day_row = daily_pattern.loc[
    daily_pattern["demand"].idxmax()
]

peak_day = peak_day_row["day_of_week"]
peak_day_demand = int(peak_day_row["demand"])

print("\n" + "=" * 65)
print("HIGHEST-DEMAND DAY")
print("=" * 65)

print(f"Highest-demand day: {peak_day}")

print(
    f"Average pickups on {peak_day}s: "
    f"{peak_day_demand:,}"
)


# ============================================================
# CREATE DAY-OF-WEEK CHART
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    daily_pattern["day_of_week"],
    daily_pattern["demand"]
)

plt.title(
    "Average NYC Yellow Taxi Demand by Day of Week\nJanuary 2025",
    fontsize=16
)

plt.xlabel(
    "Day of Week",
    fontsize=12
)

plt.ylabel(
    "Average Pickups per Day",
    fontsize=12
)

plt.xticks(
    rotation=30
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()


# ============================================================
# SAVE DAY-OF-WEEK CHART
# ============================================================

DAY_OUTPUT_FILE = (
    OUTPUT_DIR
    / "demand_by_day_of_week.png"
)

plt.savefig(
    DAY_OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nDay-of-week chart saved successfully.")

print("Output:")
print(DAY_OUTPUT_FILE)

# ============================================================
# FIND HIGHEST-DEMAND DAY
# ============================================================

peak_day_row = daily_pattern.loc[
    daily_pattern["demand"].idxmax()
]

peak_day = peak_day_row["day_of_week"]
peak_day_demand = int(peak_day_row["demand"])

print("\n" + "=" * 65)
print("HIGHEST-DEMAND DAY")
print("=" * 65)

print(f"Highest-demand day: {peak_day}")

print(
    f"Average pickups on {peak_day}s: "
    f"{peak_day_demand:,}"
)

# ============================================================
# CREATE DAY-OF-WEEK CHART
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    daily_pattern["day_of_week"],
    daily_pattern["demand"]
)

plt.title(
    "NYC Yellow Taxi Demand by Day of Week\nJanuary 2025",
    fontsize=16
)

plt.xlabel(
    "Day of Week",
    fontsize=12
)

plt.ylabel(
    "Total Number of Pickups",
    fontsize=12
)

plt.xticks(
    rotation=30
)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

# ============================================================
# SAVE DAY-OF-WEEK CHART
# ============================================================

DAY_OUTPUT_FILE = (
    OUTPUT_DIR
    / "demand_by_day_of_week.png"
)

plt.savefig(
    DAY_OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nDay-of-week chart saved successfully.")

print("Output:")
print(DAY_OUTPUT_FILE)