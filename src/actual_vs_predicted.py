import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# FILE PATHS
# ============================================================

PREDICTIONS_PATH = Path(
    "data/processed/lightgbm_predictions.parquet"
)

OUTPUT_DIR = Path("outputs/forecasting")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ACTUAL VS PREDICTED DEMAND")
print("=" * 70)

print("\nLoading LightGBM predictions...")

df = pd.read_parquet(PREDICTIONS_PATH)

print("Data loaded successfully!")
print(f"Rows: {len(df):,}")

print("\nAvailable columns:")
print(df.columns.tolist())


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "pickup_hour",
    "demand",
    "lightgbm_forecast",
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(
            f"Required column not found: {column}"
        )


# ============================================================
# AGGREGATE DEMAND ACROSS NYC
# ============================================================

print("\nAggregating demand by hour...")

hourly = (
    df.groupby("pickup_hour", as_index=False)
    .agg(
        actual_demand=("demand", "sum"),
        predicted_demand=("lightgbm_forecast", "sum"),
    )
    .sort_values("pickup_hour")
)

print("Hourly aggregation complete!")
print(f"Hours available: {len(hourly):,}")


# ============================================================
# SELECT A CLEAR 7-DAY WINDOW
# ============================================================

start_date = hourly["pickup_hour"].min()

end_date = start_date + pd.Timedelta(days=7)

plot_data = hourly[
    (hourly["pickup_hour"] >= start_date)
    & (hourly["pickup_hour"] < end_date)
].copy()

print("\nPlot period:")
print("Start:", start_date)
print("End:", end_date)

print(
    f"Actual rides in plot period: "
    f"{plot_data['actual_demand'].sum():,.0f}"
)

print(
    f"Predicted rides in plot period: "
    f"{plot_data['predicted_demand'].sum():,.0f}"
)


# ============================================================
# CREATE CHART
# ============================================================

plt.figure(figsize=(14, 6))

plt.plot(
    plot_data["pickup_hour"],
    plot_data["actual_demand"],
    label="Actual Demand",
    linewidth=2,
)

plt.plot(
    plot_data["pickup_hour"],
    plot_data["predicted_demand"],
    label="LightGBM Forecast",
    linewidth=2,
    linestyle="--",
)

plt.title(
    "NYC Yellow Taxi Demand: Actual vs LightGBM Forecast\n"
    "7-Day Evaluation Window"
)

plt.xlabel("Date and Time")
plt.ylabel("Total Pickups per Hour")

plt.legend()

plt.grid(
    alpha=0.3
)

plt.xticks(
    rotation=45
)

plt.tight_layout()


# ============================================================
# SAVE CHART
# ============================================================

output_file = (
    OUTPUT_DIR / "actual_vs_predicted_demand.png"
)

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\nChart saved successfully!")
print(f"Output: {output_file}")

print("\n" + "=" * 70)
print("ACTUAL VS PREDICTED ANALYSIS COMPLETE")
print("=" * 70)