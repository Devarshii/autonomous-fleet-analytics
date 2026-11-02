from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

FORECAST_PATH = Path(
    "data/processed/lightgbm_predictions.parquet"
)

OUTPUT_DIR = Path("outputs/simulation")

TOTAL_FLEET = 1000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTION
# ============================================================

def allocate_integer_fleet(weights, total_vehicles):
    """
    Allocate an exact integer number of vehicles according to weights.

    Uses proportional allocation followed by the largest-remainder
    method so that the final allocation always equals total_vehicles.
    """

    weights = pd.Series(weights).fillna(0).clip(lower=0)

    if weights.sum() <= 0:
        weights = pd.Series(
            np.ones(len(weights)),
            index=weights.index
        )

    exact_allocation = (
        weights / weights.sum()
    ) * total_vehicles

    integer_allocation = np.floor(
        exact_allocation
    ).astype(int)

    vehicles_remaining = (
        total_vehicles - integer_allocation.sum()
    )

    if vehicles_remaining > 0:

        remainder = (
            exact_allocation - integer_allocation
        )

        add_indices = remainder.nlargest(
            vehicles_remaining
        ).index

        integer_allocation.loc[add_indices] += 1

    return integer_allocation


# ============================================================
# LOAD FORECAST DATA
# ============================================================

print("\n" + "=" * 70)
print("AUTONOMOUS FLEET SIMULATOR")
print("=" * 70)

print("\nLoading LightGBM forecast data...")

if not FORECAST_PATH.exists():
    raise FileNotFoundError(
        f"Forecast file not found: {FORECAST_PATH}"
    )

forecast = pd.read_parquet(FORECAST_PATH)

print("Forecast data loaded successfully!")
print(f"Forecast rows: {len(forecast):,}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "pickup_hour",
    "LocationID",
    "demand",
    "Borough",
    "Zone",
    "lag_168",
    "lightgbm_forecast",
]

missing_columns = [
    column
    for column in required_columns
    if column not in forecast.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# CHOOSE SIMULATION HOUR
# ============================================================

forecast["pickup_hour"] = pd.to_datetime(
    forecast["pickup_hour"]
)

target_hour = forecast["pickup_hour"].max()

simulation = forecast[
    forecast["pickup_hour"] == target_hour
].copy()

simulation = (
    simulation
    .sort_values("LocationID")
    .reset_index(drop=True)
)

print("\n" + "=" * 70)
print("SIMULATION SETUP")
print("=" * 70)

print(f"\nTarget simulation hour: {target_hour}")
print(f"Zones in simulation: {len(simulation):,}")
print(
    f"Actual ride demand: "
    f"{simulation['demand'].sum():,.0f}"
)
print(f"Available autonomous vehicles: {TOTAL_FLEET:,}")


# ============================================================
# POLICY A
# HISTORICAL / SAME-HOUR-LAST-WEEK POSITIONING
# ============================================================

print("\n" + "=" * 70)
print("POLICY A — HISTORICAL BASELINE")
print("=" * 70)

policy_a = simulation[
    [
        "LocationID",
        "Borough",
        "Zone",
        "demand",
        "lag_168",
    ]
].copy()

policy_a["baseline_signal"] = (
    policy_a["lag_168"]
    .fillna(0)
    .clip(lower=0)
)

policy_a["baseline_vehicles"] = allocate_integer_fleet(
    policy_a["baseline_signal"],
    TOTAL_FLEET,
)

policy_a["rides_served"] = np.minimum(
    policy_a["demand"],
    policy_a["baseline_vehicles"],
)

policy_a["unserved_demand"] = (
    policy_a["demand"]
    - policy_a["rides_served"]
)

policy_a["idle_vehicles"] = (
    policy_a["baseline_vehicles"]
    - policy_a["rides_served"]
)

policy_a_total_demand = policy_a["demand"].sum()
policy_a_total_vehicles = policy_a[
    "baseline_vehicles"
].sum()

policy_a_rides_served = policy_a[
    "rides_served"
].sum()

policy_a_unserved = policy_a[
    "unserved_demand"
].sum()

policy_a_idle = policy_a[
    "idle_vehicles"
].sum()

policy_a_service_rate = (
    policy_a_rides_served
    / policy_a_total_demand
    * 100
    if policy_a_total_demand > 0
    else 0
)

policy_a_utilization = (
    policy_a_rides_served
    / policy_a_total_vehicles
    * 100
    if policy_a_total_vehicles > 0
    else 0
)

print(
    f"\nTotal ride demand: "
    f"{policy_a_total_demand:,.0f}"
)

print(
    f"Total vehicles: "
    f"{policy_a_total_vehicles:,.0f}"
)

print(
    f"\nRides served: "
    f"{policy_a_rides_served:,.0f}"
)

print(
    f"Unserved ride requests: "
    f"{policy_a_unserved:,.0f}"
)

print(
    f"Idle vehicles: "
    f"{policy_a_idle:,.0f}"
)

print(
    f"\nDemand service rate: "
    f"{policy_a_service_rate:.2f}%"
)

print(
    f"Fleet utilization: "
    f"{policy_a_utilization:.2f}%"
)


# ============================================================
# POLICY B
# LIGHTGBM FORECAST-DRIVEN POSITIONING
# ============================================================

print("\n" + "=" * 70)
print("POLICY B — FORECAST-DRIVEN POSITIONING")
print("=" * 70)

policy_b = simulation[
    [
        "LocationID",
        "Borough",
        "Zone",
        "demand",
        "lightgbm_forecast",
    ]
].copy()

policy_b["forecast_signal"] = (
    policy_b["lightgbm_forecast"]
    .fillna(0)
    .clip(lower=0)
)

policy_b["policy_b_vehicles"] = allocate_integer_fleet(
    policy_b["forecast_signal"],
    TOTAL_FLEET,
)

policy_b["rides_served"] = np.minimum(
    policy_b["demand"],
    policy_b["policy_b_vehicles"],
)

policy_b["unserved_demand"] = (
    policy_b["demand"]
    - policy_b["rides_served"]
)

policy_b["idle_vehicles"] = (
    policy_b["policy_b_vehicles"]
    - policy_b["rides_served"]
)

policy_b_total_demand = policy_b["demand"].sum()
policy_b_total_vehicles = policy_b[
    "policy_b_vehicles"
].sum()

policy_b_rides_served = policy_b[
    "rides_served"
].sum()

policy_b_unserved = policy_b[
    "unserved_demand"
].sum()

policy_b_idle = policy_b[
    "idle_vehicles"
].sum()

policy_b_service_rate = (
    policy_b_rides_served
    / policy_b_total_demand
    * 100
    if policy_b_total_demand > 0
    else 0
)

policy_b_utilization = (
    policy_b_rides_served
    / policy_b_total_vehicles
    * 100
    if policy_b_total_vehicles > 0
    else 0
)

print(
    f"\nTotal ride demand: "
    f"{policy_b_total_demand:,.0f}"
)

print(
    f"Total vehicles: "
    f"{policy_b_total_vehicles:,.0f}"
)

print(
    f"\nRides served: "
    f"{policy_b_rides_served:,.0f}"
)

print(
    f"Unserved ride requests: "
    f"{policy_b_unserved:,.0f}"
)

print(
    f"Idle vehicles: "
    f"{policy_b_idle:,.0f}"
)

print(
    f"\nDemand service rate: "
    f"{policy_b_service_rate:.2f}%"
)

print(
    f"Fleet utilization: "
    f"{policy_b_utilization:.2f}%"
)


# ============================================================
# POLICY A VS POLICY B
# ============================================================

comparison = policy_a[
    [
        "LocationID",
        "Borough",
        "Zone",
        "demand",
        "baseline_signal",
        "baseline_vehicles",
    ]
].merge(
    policy_b[
        [
            "LocationID",
            "lightgbm_forecast",
            "policy_b_vehicles",
        ]
    ],
    on="LocationID",
    how="inner",
)

comparison["vehicle_difference"] = (
    comparison["policy_b_vehicles"]
    - comparison["baseline_vehicles"]
)

comparison["absolute_reallocation"] = (
    comparison["vehicle_difference"].abs()
)

total_reallocated = (
    comparison["absolute_reallocation"].sum()
    / 2
)


# ============================================================
# ALLOCATION ERROR
# ============================================================

comparison["baseline_allocation_error"] = (
    comparison["demand"]
    - comparison["baseline_vehicles"]
).abs()

comparison["forecast_allocation_error"] = (
    comparison["demand"]
    - comparison["policy_b_vehicles"]
).abs()

baseline_allocation_mae = (
    comparison[
        "baseline_allocation_error"
    ].mean()
)

forecast_allocation_mae = (
    comparison[
        "forecast_allocation_error"
    ].mean()
)

if baseline_allocation_mae > 0:

    allocation_error_improvement = (
        (
            baseline_allocation_mae
            - forecast_allocation_mae
        )
        / baseline_allocation_mae
        * 100
    )

else:
    allocation_error_improvement = 0


# ============================================================
# DISPLAY COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("POLICY A VS POLICY B — COMPARISON")
print("=" * 70)

print(
    f"\nVehicles moved to different zones: "
    f"{total_reallocated:,.0f}"
)

print(
    f"\nPolicy A allocation MAE: "
    f"{baseline_allocation_mae:.2f}"
)

print(
    f"Policy B allocation MAE: "
    f"{forecast_allocation_mae:.2f}"
)

print(
    f"Allocation error improvement: "
    f"{allocation_error_improvement:.2f}%"
)


# ============================================================
# BIGGEST INCREASES
# ============================================================

print(
    "\nLargest vehicle increases under "
    "forecast-driven positioning:"
)

largest_increases = (
    comparison
    .sort_values(
        "vehicle_difference",
        ascending=False
    )
    .head(10)
)

print(
    largest_increases[
        [
            "LocationID",
            "Borough",
            "Zone",
            "demand",
            "baseline_vehicles",
            "policy_b_vehicles",
            "vehicle_difference",
        ]
    ].to_string(index=False)
)


# ============================================================
# BIGGEST DECREASES
# ============================================================

print(
    "\nLargest vehicle decreases under "
    "forecast-driven positioning:"
)

largest_decreases = (
    comparison
    .sort_values(
        "vehicle_difference",
        ascending=True
    )
    .head(10)
)

print(
    largest_decreases[
        [
            "LocationID",
            "Borough",
            "Zone",
            "demand",
            "baseline_vehicles",
            "policy_b_vehicles",
            "vehicle_difference",
        ]
    ].to_string(index=False)
)


# ============================================================
# SUMMARY TABLE
# ============================================================

summary = pd.DataFrame(
    {
        "metric": [
            "Total Demand",
            "Fleet Size",
            "Rides Served",
            "Unserved Demand",
            "Idle Vehicles",
            "Service Rate (%)",
            "Fleet Utilization (%)",
        ],
        "Policy_A_Historical": [
            policy_a_total_demand,
            policy_a_total_vehicles,
            policy_a_rides_served,
            policy_a_unserved,
            policy_a_idle,
            policy_a_service_rate,
            policy_a_utilization,
        ],
        "Policy_B_Forecast": [
            policy_b_total_demand,
            policy_b_total_vehicles,
            policy_b_rides_served,
            policy_b_unserved,
            policy_b_idle,
            policy_b_service_rate,
            policy_b_utilization,
        ],
    }
)


# ============================================================
# SAVE RESULTS
# ============================================================

comparison_output = (
    OUTPUT_DIR
    / "policy_a_vs_policy_b.csv"
)

summary_output = (
    OUTPUT_DIR
    / "simulation_summary.csv"
)

comparison.to_csv(
    comparison_output,
    index=False
)

summary.to_csv(
    summary_output,
    index=False
)

print("\n" + "=" * 70)
print("SIMULATION RESULTS SAVED")
print("=" * 70)

print(
    f"\nComparison file: "
    f"{comparison_output}"
)

print(
    f"Summary file: "
    f"{summary_output}"
)

print("\n" + "=" * 70)
print("FLEET SIMULATION COMPLETE")
print("=" * 70)