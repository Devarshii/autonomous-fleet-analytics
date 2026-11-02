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
# LOAD PREDICTIONS
# ============================================================

print("=" * 70)
print("FORECAST MODEL COMPARISON")
print("=" * 70)

print("\nLoading prediction data...")

df = pd.read_parquet(PREDICTIONS_PATH)

print("Prediction data loaded successfully!")
print(f"Rows: {len(df):,}")

print("\nAvailable columns:")
print(df.columns.tolist())


# ============================================================
# IDENTIFY PREDICTION COLUMNS
# ============================================================

actual_col = "demand"

lightgbm_candidates = [
    "lightgbm_forecast",
    "lightgbm_prediction",
    "prediction",
    "predicted_demand",
]

baseline_candidates = [
    "baseline_forecast",
    "lag_168",
]

lightgbm_col = next(
    (col for col in lightgbm_candidates if col in df.columns),
    None,
)

baseline_col = next(
    (col for col in baseline_candidates if col in df.columns),
    None,
)

if actual_col not in df.columns:
    raise ValueError(
        f"Could not find actual demand column '{actual_col}'."
    )

if lightgbm_col is None:
    raise ValueError(
        "Could not automatically find the LightGBM prediction column."
    )

if baseline_col is None:
    raise ValueError(
        "Could not automatically find the baseline prediction column."
    )

print("\nColumns selected:")
print(f"Actual demand: {actual_col}")
print(f"Baseline forecast: {baseline_col}")
print(f"LightGBM forecast: {lightgbm_col}")


# ============================================================
# MODEL PERFORMANCE
# ============================================================

baseline_mae = (
    df[actual_col] - df[baseline_col]
).abs().mean()

lightgbm_mae = (
    df[actual_col] - df[lightgbm_col]
).abs().mean()

baseline_wape = (
    (df[actual_col] - df[baseline_col]).abs().sum()
    / df[actual_col].sum()
) * 100

lightgbm_wape = (
    (df[actual_col] - df[lightgbm_col]).abs().sum()
    / df[actual_col].sum()
) * 100

mae_improvement = (
    (baseline_mae - lightgbm_mae)
    / baseline_mae
) * 100

wape_improvement = (
    (baseline_wape - lightgbm_wape)
    / baseline_wape
) * 100


print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(f"Baseline MAE: {baseline_mae:.2f}")
print(f"LightGBM MAE: {lightgbm_mae:.2f}")

print()

print(f"Baseline WAPE: {baseline_wape:.2f}%")
print(f"LightGBM WAPE: {lightgbm_wape:.2f}%")

print()

print(f"MAE improvement: {mae_improvement:.2f}%")
print(f"WAPE improvement: {wape_improvement:.2f}%")


# ============================================================
# CREATE COMPARISON CHART
# ============================================================

models = [
    "Historical\nBaseline",
    "LightGBM",
]

mae_values = [
    baseline_mae,
    lightgbm_mae,
]

plt.figure(figsize=(8, 6))

bars = plt.bar(
    models,
    mae_values,
)

plt.title(
    "NYC Taxi Demand Forecasting Performance\n"
    "Baseline vs LightGBM"
)

plt.ylabel("Mean Absolute Error (Rides per Zone-Hour)")
plt.xlabel("Forecasting Model")

plt.grid(
    axis="y",
    alpha=0.3,
)

for bar, value in zip(bars, mae_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.05,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontsize=12,
    )

plt.text(
    0.5,
    max(mae_values) * 0.75,
    f"{mae_improvement:.1f}% lower error",
    ha="center",
    fontsize=12,
)

plt.tight_layout()

output_file = (
    OUTPUT_DIR / "baseline_vs_lightgbm_mae.png"
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
print("FORECAST COMPARISON COMPLETE")
print("=" * 70)