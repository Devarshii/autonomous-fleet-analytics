
"""Evaluate fixed fleet strategies on February 1–7, 2025.

Uses the existing January-developed simulation policies.
Loads February trips and February forecasts.
Does not retrain, retune, or overwrite January results.

Run:
    python src/evaluate_february_fleet.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

import geographic_trip_simulator as base
import geographic_dispatch_simulator as dispatch


# ============================================================
# FEBRUARY INPUTS AND OUTPUTS
# ============================================================

FEB_TRIPS = Path(
    "data/processed/nyc_trip_replay_feb2025.parquet"
)

FEB_FORECAST = Path(
    "data/processed/lightgbm_predictions_feb2025.parquet"
)

OUTPUT_DIR = Path(
    "outputs/validation/february_fleet"
)

HOURLY_OUTPUT = OUTPUT_DIR / "february_fleet_hourly.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "february_fleet_summary.csv"
REPORT_OUTPUT = OUTPUT_DIR / "february_fleet_report.json"

EXPECTED_REQUESTS = 847348
EXPECTED_ZONES = 257
EXPECTED_HOURS = 168

POLICIES = (
    "no_repositioning",
    "forecast_repositioning",
    "arrival_aware_repositioning",
)


def main():

    print("=" * 70)
    print("FEBRUARY 2025 - OUT-OF-TIME FLEET EVALUATION")
    print("=" * 70)

    # ========================================================
    # VERIFY INPUTS
    # ========================================================

    for path in (FEB_TRIPS, FEB_FORECAST):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing February input: {path}"
            )

    print("\nUsing the existing fleet simulation policies.")
    print("No model retraining or algorithm tuning.")
    print("January data and results remain unchanged.")

    # ========================================================
    # CONFIGURE THE EXISTING INPUT LOADER
    # ========================================================

    # Change module variables only in this Python process.
    # No existing source files are edited.

    original_trips_path = base.TRIPS_FILE
    original_forecast_path = base.FORECAST_FILE

    try:
        base.TRIPS_FILE = FEB_TRIPS
        base.FORECAST_FILE = FEB_FORECAST

        trips, hours, ids, coords, forecast, historical = (
            base.load_inputs()
        )

    finally:
        base.TRIPS_FILE = original_trips_path
        base.FORECAST_FILE = original_forecast_path

    # ========================================================
    # VERIFY FEBRUARY EVALUATION DATA
    # ========================================================

    print("\n=== INPUT VALIDATION ===")

    print(f"Trip requests: {len(trips):,}")
    print(f"Taxi zones: {len(ids)}")
    print(f"Evaluation hours: {len(hours)}")
    print(f"Fleet size: {base.FLEET_SIZE:,}")

    print(f"First hour: {hours[0]}")
    print(f"Last hour: {hours[-1]}")

    if len(trips) != EXPECTED_REQUESTS:
        raise ValueError(
            f"Expected {EXPECTED_REQUESTS:,} requests, "
            f"found {len(trips):,}"
        )

    if len(ids) != EXPECTED_ZONES:
        raise ValueError(
            f"Expected {EXPECTED_ZONES} zones, "
            f"found {len(ids)}"
        )

    if len(hours) != EXPECTED_HOURS:
        raise ValueError(
            f"Expected {EXPECTED_HOURS} hours, "
            f"found {len(hours)}"
        )

    if pd.Timestamp(hours[0]) != pd.Timestamp(
        "2025-02-01 00:00:00"
    ):
        raise ValueError(
            "Evaluation must begin February 1."
        )

    if pd.Timestamp(hours[-1]) != pd.Timestamp(
        "2025-02-07 23:00:00"
    ):
        raise ValueError(
            "Evaluation must end February 7."
        )

    if not np.isfinite(forecast).all():
        raise ValueError("Nonfinite demand forecasts found")

    if not np.isfinite(historical).all():
        raise ValueError("Nonfinite historical predictions found")

    # ========================================================
    # BUILD SAME GEOGRAPHY FOR ALL POLICIES
    # ========================================================

    print("\nPreparing geographic travel estimates...")

    miles, reposition_minutes, pickup_minutes, nearest = (
        dispatch.build_pickup_matrix(ids, coords)
    )

    # ========================================================
    # EVALUATE THE THREE UNCHANGED POLICIES
    # ========================================================

    print("\n=== POLICY EVALUATION ===")

    all_hourly = []
    summaries = []

    for policy in POLICIES:

        print(f"\nRunning: {policy}", flush=True)

        hourly, summary = dispatch.simulate_dispatch(
            policy,
            trips,
            hours,
            ids,
            coords,
            forecast,
            historical,
            miles,
            reposition_minutes,
            pickup_minutes,
            nearest,
        )

        # Verify that every policy processed the same trips.
        if int(hourly["requests"].sum()) != len(trips):
            raise AssertionError(
                f"Request totals mismatch for {policy}"
            )

        if len(hourly) != EXPECTED_HOURS:
            raise AssertionError(
                f"Hourly result count mismatch for {policy}"
            )

        if int(hourly["served"].sum()) != int(
            summary["served"]
        ):
            raise AssertionError(
                f"Served-ride totals mismatch for {policy}"
            )

        if int(summary["requests"]) != len(trips):
            raise AssertionError(
                f"Summary request mismatch for {policy}"
            )

        if int((hourly["requests"] == 0).sum()) > 0:
            print(
                "WARNING: At least one hour "
                "has zero recorded requests."
            )

        all_hourly.append(hourly)
        summaries.append(summary)

        print(
            f"Served: {summary['served']:,}",
            flush=True,
        )

        print(
            f"Service rate: "
            f"{summary['service_rate_pct']:.2f}%",
            flush=True,
        )

        print(
            f"Mean wait: "
            f"{summary['mean_wait_min_served']:.2f} min",
            flush=True,
        )

        print(
            f"Estimated empty miles: "
            f"{summary['estimated_total_empty_miles']:,.1f}",
            flush=True,
        )

    # ========================================================
    # COMBINE RESULTS
    # ========================================================

    hourly_df = pd.concat(
        all_hourly,
        ignore_index=True,
    )

    summary_df = pd.DataFrame(summaries)

    # ========================================================
    # COMPARE AGAINST NO REPOSITIONING
    # ========================================================

    baseline = summary_df.loc[
        summary_df["policy"] == "no_repositioning"
    ].iloc[0]

    baseline_rides = int(baseline["served"])
    baseline_miles = float(
        baseline["estimated_total_empty_miles"]
    )

    summary_df["additional_rides_vs_baseline"] = (
        summary_df["served"] - baseline_rides
    )

    summary_df["additional_empty_miles_vs_baseline"] = (
        summary_df["estimated_total_empty_miles"]
        - baseline_miles
    )

    summary_df["service_rate_gain_pp"] = (
        summary_df["service_rate_pct"]
        - float(baseline["service_rate_pct"])
    )

    # ========================================================
    # SAVE FEBRUARY RESULTS SEPARATELY
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    hourly_df.to_csv(
        HOURLY_OUTPUT,
        index=False,
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    report = {
        "evaluation_period": (
            "February 1–7, 2025"
        ),
        "evaluation_type": (
            "Out-of-time evaluation using "
            "January-developed policies"
        ),
        "requests": len(trips),
        "zones": len(ids),
        "hours": len(hours),
        "fleet_size": int(base.FLEET_SIZE),
        "model_retrained": False,
        "policy_parameters_changed": False,
        "trip_source": str(FEB_TRIPS),
        "forecast_source": str(FEB_FORECAST),
        "results": summary_df.replace(
            {np.nan: None}
        ).to_dict(orient="records"),
        "limitations": (
            "Simulated robotaxi fleet using observed "
            "human-driven taxi trips. Geographic travel "
            "estimates are not routed driving distances. "
            "No real AV telemetry, charging, maintenance, "
            "or rider cancellation model. The February "
            "forecast uses observed demand from completed "
            "previous hours. Simulation outcomes are not "
            "real-world performance measurements."
        ),
    }

    REPORT_OUTPUT.write_text(
        json.dumps(report, indent=2)
    )

    # ========================================================
    # DISPLAY FINAL RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("FEBRUARY FLEET RESULTS")
    print("=" * 70)

    display_columns = [
        "policy",
        "requests",
        "served",
        "service_rate_pct",
        "mean_wait_min_served",
        "estimated_total_empty_miles",
        "repositions",
        "additional_rides_vs_baseline",
        "service_rate_gain_pp",
    ]

    print(
        summary_df[display_columns].to_string(
            index=False
        )
    )

    print("\n=== FILES SAVED ===")
    print(HOURLY_OUTPUT)
    print(SUMMARY_OUTPUT)
    print(REPORT_OUTPUT)

    print(
        "\nFEBRUARY FLEET EVALUATION COMPLETE"
    )


if __name__ == "__main__":
    main()
