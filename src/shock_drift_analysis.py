# ============================================================
# SHOCK & DEMAND DRIFT ANALYSIS
# Autonomous Fleet Analytics Project
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path


# ------------------------------------------------------------
# FILE PATHS
# ------------------------------------------------------------

DEMAND_FILE = Path(
    "data/processed/nyc_hourly_demand_complete.parquet"
)

OUTPUT_DIR = Path("outputs/shock_drift")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# VERIFY REQUIRED DATA
# ------------------------------------------------------------

print("=" * 60)
print("SHOCK & DEMAND DRIFT ANALYSIS")
print("=" * 60)

if not DEMAND_FILE.exists():
    raise FileNotFoundError(
        f"Required demand dataset not found: {DEMAND_FILE}"
    )

print("\nRequired demand dataset found!")
print(f"Dataset: {DEMAND_FILE}")

print("\nShock & drift analysis environment ready.")

print("\n" + "=" * 60)
print("SHOCK & DRIFT SETUP COMPLETE")
print("=" * 60)
# ============================================================
# LOAD NYC DEMAND DATA
# ============================================================

print("\nLoading NYC demand data...")

demand_df = pd.read_parquet(DEMAND_FILE)

print("NYC demand data loaded successfully!")

print(f"\nRows: {len(demand_df):,}")
print(f"Columns: {len(demand_df.columns)}")

print("\nAvailable columns:")
print(demand_df.columns.tolist())

print("\nSample data:")
print(demand_df.head())

print("\n" + "=" * 60)
print("DEMAND DATA LOADED")
print("=" * 60)
# ============================================================
# SIMULATE SNOWSTORM DEMAND SHOCK
# ============================================================

print("\nSimulating snowstorm demand shock...")

shock_df = demand_df.copy()

# Select one day for the simulated snowstorm
snowstorm_date = pd.Timestamp("2025-01-15")

# Severe weather period: 4 PM to 11 PM
snowstorm_mask = (
    (pd.to_datetime(shock_df["date"]) == snowstorm_date)
    & (shock_df["hour"] >= 16)
    & (shock_df["hour"] <= 23)
)

# Preserve original demand
shock_df["normal_demand"] = shock_df["demand"]

# Simulate a 35% reduction in ride demand during severe weather
shock_df.loc[snowstorm_mask, "demand"] = (
    shock_df.loc[snowstorm_mask, "demand"] * 0.65
).round().astype(int)

affected_records = snowstorm_mask.sum()

normal_total = shock_df.loc[
    snowstorm_mask, "normal_demand"
].sum()

snowstorm_total = shock_df.loc[
    snowstorm_mask, "demand"
].sum()

print(f"\nSimulated snowstorm date: {snowstorm_date.date()}")
print("Shock period: 4 PM - 11 PM")

print(f"\nAffected zone-hour records: {affected_records:,}")
print(f"Normal demand: {normal_total:,}")
print(f"Snowstorm demand: {snowstorm_total:,}")

print(
    f"Demand reduction: "
    f"{((normal_total - snowstorm_total) / normal_total) * 100:.2f}%"
)

print("\n" + "=" * 60)
print("SNOWSTORM SHOCK SIMULATION COMPLETE")
print("=" * 60)

# ============================================================
# SIMULATE MAJOR EVENT DEMAND SHOCK
# ============================================================

print("\nSimulating major event demand surge...")

event_df = demand_df.copy()

event_date = pd.Timestamp("2025-01-24")

# Simulated major event in Manhattan during evening hours
event_mask = (
    (pd.to_datetime(event_df["date"]) == event_date)
    & (event_df["Borough"] == "Manhattan")
    & (event_df["hour"] >= 18)
    & (event_df["hour"] <= 23)
)

event_df["normal_demand"] = event_df["demand"]

# Simulate a 60% increase in demand
event_df.loc[event_mask, "demand"] = (
    event_df.loc[event_mask, "demand"] * 1.60
).round().astype(int)

affected_event_records = event_mask.sum()

normal_event_demand = event_df.loc[
    event_mask, "normal_demand"
].sum()

surge_event_demand = event_df.loc[
    event_mask, "demand"
].sum()

event_increase_pct = (
    (surge_event_demand - normal_event_demand)
    / normal_event_demand
) * 100

print(f"\nSimulated event date: {event_date.date()}")
print("Location: Manhattan")
print("Event period: 6 PM - 11 PM")

print(f"\nAffected zone-hour records: {affected_event_records:,}")
print(f"Normal demand: {normal_event_demand:,}")
print(f"Event demand: {surge_event_demand:,}")
print(f"Demand increase: {event_increase_pct:.2f}%")

print("\n" + "=" * 60)
print("MAJOR EVENT SHOCK SIMULATION COMPLETE")
print("=" * 60)

# ============================================================
# DEMAND DRIFT DETECTION
# ============================================================

print("\nAnalyzing demand drift...")

drift_df = demand_df.copy()

drift_df["date"] = pd.to_datetime(drift_df["date"])

# Reference period: first half of January
reference_period = drift_df[
    drift_df["date"] <= "2025-01-15"
]

# Current period: second half of January
current_period = drift_df[
    drift_df["date"] > "2025-01-15"
]

reference_avg = reference_period["demand"].mean()
current_avg = current_period["demand"].mean()

drift_pct = (
    (current_avg - reference_avg)
    / reference_avg
) * 100

print("\nReference period: January 1-15")
print("Current period: January 16-31")

print(f"\nReference average demand: {reference_avg:.2f}")
print(f"Current average demand: {current_avg:.2f}")
print(f"Demand change: {drift_pct:.2f}%")

# Simple operational drift threshold
DRIFT_THRESHOLD = 10

if abs(drift_pct) >= DRIFT_THRESHOLD:
    drift_status = "DRIFT_DETECTED"
    retraining_recommendation = "MODEL_RETRAINING_RECOMMENDED"
else:
    drift_status = "STABLE"
    retraining_recommendation = "NO_IMMEDIATE_RETRAINING"

print(f"\nDrift status: {drift_status}")
print(f"Recommendation: {retraining_recommendation}")

print("\n" + "=" * 60)
print("DEMAND DRIFT DETECTION COMPLETE")
print("=" * 60)
# ============================================================
# POPULATION STABILITY INDEX (PSI)
# ============================================================

print("\nCalculating Population Stability Index (PSI)...")

def calculate_psi(reference, current, bins=10):

    reference = np.asarray(reference)
    current = np.asarray(current)

    breakpoints = np.percentile(
        reference,
        np.linspace(0, 100, bins + 1)
    )

    breakpoints = np.unique(breakpoints)

    reference_counts, _ = np.histogram(
        reference,
        bins=breakpoints
    )

    current_counts, _ = np.histogram(
        current,
        bins=breakpoints
    )

    reference_pct = reference_counts / len(reference)
    current_pct = current_counts / len(current)

    # Prevent division by zero
    reference_pct = np.where(
        reference_pct == 0,
        0.0001,
        reference_pct
    )

    current_pct = np.where(
        current_pct == 0,
        0.0001,
        current_pct
    )

    psi = np.sum(
        (current_pct - reference_pct)
        * np.log(current_pct / reference_pct)
    )

    return psi


psi_score = calculate_psi(
    reference_period["demand"],
    current_period["demand"]
)

print(f"\nPSI score: {psi_score:.4f}")

if psi_score < 0.10:
    psi_status = "STABLE"
elif psi_score < 0.25:
    psi_status = "MODERATE_DRIFT"
else:
    psi_status = "SIGNIFICANT_DRIFT"

print(f"PSI interpretation: {psi_status}")

print("\nPSI thresholds:")
print("< 0.10  = Stable")
print("0.10-0.25 = Moderate drift")
print("> 0.25 = Significant drift")

print("\n" + "=" * 60)
print("PSI DRIFT ANALYSIS COMPLETE")
print("=" * 60)
# ============================================================
# SAVE SHOCK & DRIFT SUMMARY
# ============================================================

from pathlib import Path

output_dir = Path("outputs/shock_drift")
output_dir.mkdir(parents=True, exist_ok=True)

summary = pd.DataFrame({
    "metric": [
        "reference_average_demand",
        "current_average_demand",
        "demand_change_percent",
        "mean_drift_status",
        "psi_score",
        "psi_status"
    ],
    "value": [
        round(reference_period["demand"].mean(), 2),
        round(current_period["demand"].mean(), 2),
        round(
            (
                current_period["demand"].mean()
                - reference_period["demand"].mean()
            )
            / reference_period["demand"].mean()
            * 100,
            2
        ),
        drift_status,
        round(psi_score, 4),
        psi_status
    ]
})

output_file = output_dir / "drift_summary.csv"
summary.to_csv(output_file, index=False)

print("\n" + "=" * 60)
print("SHOCK & DRIFT RESULTS SAVED")
print("=" * 60)

print(summary.to_string(index=False))

print(f"\nOutput file: {output_file}")

print("\n" + "=" * 60)
print("SHOCK & DRIFT ANALYSIS COMPLETE")
print("=" * 60)