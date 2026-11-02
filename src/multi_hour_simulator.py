from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

FORECAST_PATH = Path("data/processed/lightgbm_predictions.parquet")

OUTPUT_DIR = Path("outputs/simulation")
HOURLY_OUTPUT_PATH = OUTPUT_DIR / "multi_hour_simulation.csv"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "multi_hour_simulation_summary.csv"

TOTAL_FLEET = 1000


# ============================================================
# HELPER FUNCTION
# Allocate an integer fleet proportionally to a demand signal
# while guaranteeing that the final allocation equals fleet_size.
# ============================================================

def allocate_fleet(signal, fleet_size):
    signal = np.asarray(signal, dtype=float)

    # Prevent negative forecast values from affecting allocation.
    signal = np.clip(signal, 0, None)

    if signal.sum() <= 0:
        raw = np.repeat(fleet_size / len(signal), len(signal))
    else:
        raw = signal / signal.sum() * fleet_size

    allocation = np.floor(raw).astype(int)

    remaining = fleet_size - allocation.sum()

    if remaining > 0:
        fractional = raw - allocation
        order = np.argsort(-fractional)

        for idx in order[:remaining]:
            allocation[idx] += 1

    elif remaining < 0:
        order = np.argsort(-allocation)

        for idx in order[:abs(remaining)]:
            if allocation[idx] > 0:
                allocation[idx] -= 1

    return allocation


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("MULTI-HOUR AUTONOMOUS FLEET SIMULATOR")
print("=" * 70)

if not FORECAST_PATH.exists():
    raise FileNotFoundError(
        f"Forecast file not found: {FORECAST_PATH}"
    )

print("\nLoading LightGBM prediction data...")

df = pd.read_parquet(FORECAST_PATH)

df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])

required_columns = {
    "pickup_hour",
    "LocationID",
    "demand",
    "lightgbm_forecast",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

print(f"Rows loaded: {len(df):,}")
print(f"Unique hours: {df['pickup_hour'].nunique():,}")
print(
    f"Test period: {df['pickup_hour'].min()} "
    f"to {df['pickup_hour'].max()}"
)


# ============================================================
# BASELINE SIGNAL
#
# Policy A:
# Historical demand signal.
#
# We prefer lag_168 (same hour one week earlier).
# If unavailable, fall back to rolling_mean_168, then lag_24,
# then rolling_mean_24.
# ============================================================

baseline_candidates = [
    "lag_168",
    "rolling_mean_168",
    "lag_24",
    "rolling_mean_24",
]

baseline_column = None

for column in baseline_candidates:
    if column in df.columns:
        baseline_column = column
        break

if baseline_column is None:
    raise ValueError(
        "No historical baseline column was found. "
        "Expected one of: "
        + ", ".join(baseline_candidates)
    )

print(f"\nPolicy A baseline signal: {baseline_column}")
print("Policy B signal: lightgbm_forecast")
print(f"Simulated fleet size: {TOTAL_FLEET:,}")


# ============================================================
# RUN 168-HOUR SIMULATION
# ============================================================

hourly_results = []

grouped = df.groupby("pickup_hour", sort=True)

print("\nRunning hourly simulation...")

for hour, hour_df in grouped:

    hour_df = hour_df.copy()

    # --------------------------------------------------------
    # Clean signals
    # --------------------------------------------------------

    actual_demand = (
        pd.to_numeric(hour_df["demand"], errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .to_numpy(dtype=float)
    )

    baseline_signal = (
        pd.to_numeric(hour_df[baseline_column], errors="coerce")
        .fillna(0)
        .clip(lower=0)
        .to_numpy(dtype=float)
    )

    forecast_signal = (
        pd.to_numeric(
            hour_df["lightgbm_forecast"],
            errors="coerce",
        )
        .fillna(0)
        .clip(lower=0)
        .to_numpy(dtype=float)
    )

    # --------------------------------------------------------
    # Allocate fleet
    # --------------------------------------------------------

    policy_a_vehicles = allocate_fleet(
        baseline_signal,
        TOTAL_FLEET,
    )

    policy_b_vehicles = allocate_fleet(
        forecast_signal,
        TOTAL_FLEET,
    )

    # --------------------------------------------------------
    # Vehicles required if the 1,000-car fleet were allocated
    # perfectly according to the ACTUAL demand distribution.
    #
    # This gives us a fair target for comparing positioning.
    # --------------------------------------------------------

    ideal_vehicles = allocate_fleet(
        actual_demand,
        TOTAL_FLEET,
    )

    # --------------------------------------------------------
    # Allocation errors
    # --------------------------------------------------------

    policy_a_abs_error = np.abs(
        ideal_vehicles - policy_a_vehicles
    )

    policy_b_abs_error = np.abs(
        ideal_vehicles - policy_b_vehicles
    )

    policy_a_mae = policy_a_abs_error.mean()
    policy_b_mae = policy_b_abs_error.mean()

    # --------------------------------------------------------
    # Zone-level comparison
    # --------------------------------------------------------

    b_better = int(
        (policy_b_abs_error < policy_a_abs_error).sum()
    )

    a_better = int(
        (policy_a_abs_error < policy_b_abs_error).sum()
    )

    same = int(
        (policy_a_abs_error == policy_b_abs_error).sum()
    )

    # --------------------------------------------------------
    # Reallocation
    #
    # Sum of absolute differences counts both the zone losing
    # a vehicle and the zone gaining it, so divide by two.
    # --------------------------------------------------------

    vehicles_reallocated = int(
        np.abs(
            policy_b_vehicles - policy_a_vehicles
        ).sum() / 2
    )

    # --------------------------------------------------------
    # Local rides served
    #
    # Each vehicle can immediately serve at most one ride in
    # its current zone during this simplified hourly snapshot.
    #
    # This intentionally does NOT model multiple trips per hour.
    # --------------------------------------------------------

    policy_a_served = np.minimum(
        actual_demand,
        policy_a_vehicles,
    ).sum()

    policy_b_served = np.minimum(
        actual_demand,
        policy_b_vehicles,
    ).sum()

    total_demand = actual_demand.sum()

    policy_a_unserved = max(
        total_demand - policy_a_served,
        0,
    )

    policy_b_unserved = max(
        total_demand - policy_b_served,
        0,
    )

    if total_demand > 0:
        policy_a_service_rate = (
            policy_a_served / total_demand * 100
        )

        policy_b_service_rate = (
            policy_b_served / total_demand * 100
        )
    else:
        policy_a_service_rate = 0.0
        policy_b_service_rate = 0.0

    # --------------------------------------------------------
    # Store hour result
    # --------------------------------------------------------

    hourly_results.append(
        {
            "pickup_hour": hour,
            "total_demand": int(total_demand),
            "zones": len(hour_df),
            "fleet_size": TOTAL_FLEET,

            "policy_a_allocation_mae": policy_a_mae,
            "policy_b_allocation_mae": policy_b_mae,

            "allocation_mae_improvement": (
                policy_a_mae - policy_b_mae
            ),

            "allocation_improvement_pct": (
                (
                    policy_a_mae - policy_b_mae
                )
                / policy_a_mae
                * 100
                if policy_a_mae > 0
                else 0
            ),

            "policy_b_better_zones": b_better,
            "policy_a_better_zones": a_better,
            "same_zones": same,

            "vehicles_reallocated": vehicles_reallocated,

            "policy_a_rides_served": int(
                policy_a_served
            ),

            "policy_b_rides_served": int(
                policy_b_served
            ),

            "additional_rides_served": int(
                policy_b_served - policy_a_served
            ),

            "policy_a_unserved": int(
                policy_a_unserved
            ),

            "policy_b_unserved": int(
                policy_b_unserved
            ),

            "policy_a_service_rate": (
                policy_a_service_rate
            ),

            "policy_b_service_rate": (
                policy_b_service_rate
            ),
        }
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame(hourly_results)

if results.empty:
    raise ValueError("Simulation produced no results.")


# ============================================================
# OVERALL METRICS
# ============================================================

hours_simulated = len(results)

policy_b_mae_better_hours = int(
    (
        results["policy_b_allocation_mae"]
        < results["policy_a_allocation_mae"]
    ).sum()
)

policy_a_mae_better_hours = int(
    (
        results["policy_a_allocation_mae"]
        < results["policy_b_allocation_mae"]
    ).sum()
)

equal_mae_hours = int(
    (
        results["policy_a_allocation_mae"]
        == results["policy_b_allocation_mae"]
    ).sum()
)

avg_policy_a_mae = (
    results["policy_a_allocation_mae"].mean()
)

avg_policy_b_mae = (
    results["policy_b_allocation_mae"].mean()
)

if avg_policy_a_mae > 0:
    overall_mae_improvement_pct = (
        (
            avg_policy_a_mae
            - avg_policy_b_mae
        )
        / avg_policy_a_mae
        * 100
    )
else:
    overall_mae_improvement_pct = 0.0


total_a_served = int(
    results["policy_a_rides_served"].sum()
)

total_b_served = int(
    results["policy_b_rides_served"].sum()
)

additional_rides = (
    total_b_served - total_a_served
)

total_demand_all_hours = int(
    results["total_demand"].sum()
)

if total_demand_all_hours > 0:
    overall_a_service_rate = (
        total_a_served
        / total_demand_all_hours
        * 100
    )

    overall_b_service_rate = (
        total_b_served
        / total_demand_all_hours
        * 100
    )
else:
    overall_a_service_rate = 0.0
    overall_b_service_rate = 0.0


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("7-DAY SIMULATION RESULTS")
print("=" * 70)

print(f"\nHours simulated: {hours_simulated:,}")

print(
    f"Period: "
    f"{results['pickup_hour'].min()} "
    f"to {results['pickup_hour'].max()}"
)

print(f"\nTotal ride demand: {total_demand_all_hours:,}")
print(f"Fleet available per hour: {TOTAL_FLEET:,}")

print("\n" + "-" * 70)
print("ALLOCATION ACCURACY")
print("-" * 70)

print(
    f"Policy A average allocation MAE: "
    f"{avg_policy_a_mae:.2f}"
)

print(
    f"Policy B average allocation MAE: "
    f"{avg_policy_b_mae:.2f}"
)

print(
    f"Overall allocation MAE improvement: "
    f"{overall_mae_improvement_pct:.2f}%"
)

print(
    f"\nPolicy B better hours: "
    f"{policy_b_mae_better_hours}"
)

print(
    f"Policy A better hours: "
    f"{policy_a_mae_better_hours}"
)

print(
    f"Equal hours: "
    f"{equal_mae_hours}"
)


print("\n" + "-" * 70)
print("RIDE-SERVICE COMPARISON")
print("-" * 70)

print(
    f"Policy A rides served: "
    f"{total_a_served:,}"
)

print(
    f"Policy B rides served: "
    f"{total_b_served:,}"
)

print(
    f"Additional rides served by Policy B: "
    f"{additional_rides:,}"
)

print(
    f"\nPolicy A overall service rate: "
    f"{overall_a_service_rate:.2f}%"
)

print(
    f"Policy B overall service rate: "
    f"{overall_b_service_rate:.2f}%"
)

print(
    f"Service-rate improvement: "
    f"{overall_b_service_rate - overall_a_service_rate:.2f} "
    f"percentage points"
)


print("\n" + "-" * 70)
print("REPOSITIONING")
print("-" * 70)

print(
    f"Average vehicles repositioned per hour: "
    f"{results['vehicles_reallocated'].mean():.2f}"
)

print(
    f"Maximum vehicles repositioned in one hour: "
    f"{results['vehicles_reallocated'].max():,}"
)


# ============================================================
# BEST / WORST HOURS
# ============================================================

best_hour = results.loc[
    results["allocation_improvement_pct"].idxmax()
]

worst_hour = results.loc[
    results["allocation_improvement_pct"].idxmin()
]

print("\n" + "-" * 70)
print("BEST FORECAST-POSITIONING HOUR")
print("-" * 70)

print(
    f"Hour: {best_hour['pickup_hour']}"
)

print(
    f"Allocation improvement: "
    f"{best_hour['allocation_improvement_pct']:.2f}%"
)

print(
    f"Additional rides served: "
    f"{int(best_hour['additional_rides_served']):,}"
)


print("\n" + "-" * 70)
print("WORST FORECAST-POSITIONING HOUR")
print("-" * 70)

print(
    f"Hour: {worst_hour['pickup_hour']}"
)

print(
    f"Allocation improvement: "
    f"{worst_hour['allocation_improvement_pct']:.2f}%"
)

print(
    f"Additional rides served: "
    f"{int(worst_hour['additional_rides_served']):,}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

results.to_csv(
    HOURLY_OUTPUT_PATH,
    index=False,
)

summary = pd.DataFrame(
    [
        {
            "hours_simulated": hours_simulated,
            "total_demand": total_demand_all_hours,
            "fleet_size_per_hour": TOTAL_FLEET,

            "policy_a_avg_allocation_mae":
                avg_policy_a_mae,

            "policy_b_avg_allocation_mae":
                avg_policy_b_mae,

            "allocation_mae_improvement_pct":
                overall_mae_improvement_pct,

            "policy_b_better_hours":
                policy_b_mae_better_hours,

            "policy_a_better_hours":
                policy_a_mae_better_hours,

            "equal_hours":
                equal_mae_hours,

            "policy_a_rides_served":
                total_a_served,

            "policy_b_rides_served":
                total_b_served,

            "additional_rides_served":
                additional_rides,

            "policy_a_service_rate":
                overall_a_service_rate,

            "policy_b_service_rate":
                overall_b_service_rate,

            "service_rate_improvement_pp":
                (
                    overall_b_service_rate
                    - overall_a_service_rate
                ),

            "avg_vehicles_repositioned_per_hour":
                results[
                    "vehicles_reallocated"
                ].mean(),

            "max_vehicles_repositioned":
                results[
                    "vehicles_reallocated"
                ].max(),
        }
    ]
)

summary.to_csv(
    SUMMARY_OUTPUT_PATH,
    index=False,
)


print("\n" + "=" * 70)
print("MULTI-HOUR SIMULATION SAVED")
print("=" * 70)

print(f"\nHourly results:")
print(HOURLY_OUTPUT_PATH)

print(f"\nSummary:")
print(SUMMARY_OUTPUT_PATH)

print("\n" + "=" * 70)
print("MULTI-HOUR FLEET SIMULATION COMPLETE")
print("=" * 70)