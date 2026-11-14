"""Build a reproducible held-out NYC taxi-trip replay dataset.

Recorded yellow-taxi trips are a proxy for robotaxi requests, NOT AV telemetry.
Reads January 2025 yellow-taxi parquet in streaming batches; keeps trips beginning
in the same held-out time window as the forecast data and with mapped pickup and
dropoff taxi zones. Does not edit the original dataset or existing simulations.

Run from the repository root: python src/build_trip_replay.py
"""
from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RAW = Path("data/raw/yellow_tripdata_2025-01.parquet")
FORECAST = Path("data/processed/lightgbm_predictions.parquet")
SHAPEFILE = Path("data/external/taxi_zones/taxi_zones/taxi_zones.shp")
OUT_DIR = Path("data/processed")
REPLAY_PATH = OUT_DIR / "nyc_trip_replay_test.parquet"
REPORT_PATH = Path("outputs/validation/trip_replay_quality.json")

COLUMNS = [
    "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "PULocationID", "DOLocationID", "trip_distance",
]
BATCH_SIZE = 100_000


def main():
    print("=== NYC REAL-TRIP REPLAY DATA PREPARATION ===")
    for path in (RAW, FORECAST, SHAPEFILE):
        if not path.exists():
            raise FileNotFoundError(f"Missing required input: {path}")

    # Use the *same* evaluation window as the existing forecasting experiment.
    forecast_hours = pd.read_parquet(FORECAST, columns=["pickup_hour", "LocationID"])
    forecast_hours["pickup_hour"] = pd.to_datetime(forecast_hours["pickup_hour"])
    hours = pd.DatetimeIndex(sorted(forecast_hours["pickup_hour"].unique()))
    if len(hours) != 168 or not (hours[1:] - hours[:-1] == pd.Timedelta(hours=1)).all():
        raise ValueError("Expected 168 consecutive forecast hours")
    start, end = hours[0], hours[-1] + pd.Timedelta(hours=1)

    # Only zones available in both the forecast and official shape geometry.
    zones = gpd.read_file(SHAPEFILE)["LocationID"].astype(int)
    eligible = set(zones) & set(forecast_hours["LocationID"].astype(int))
    print(f"Evaluation period: {start} to {end} (end exclusive)")
    print(f"Geographically matched zones: {len(eligible)}")

    parquet = pq.ParquetFile(RAW)
    missing = set(COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Raw taxi data missing columns: {sorted(missing)}")

    counters = {
        "raw_rows_scanned": 0,
        "pickups_in_evaluation_window": 0,
        "valid_after_quality_and_zone_filters": 0,
    }
    parts = []

    for i, batch in enumerate(parquet.iter_batches(batch_size=BATCH_SIZE, columns=COLUMNS), 1):
        df = batch.to_pandas()
        counters["raw_rows_scanned"] += len(df)
        df = df.loc[
            (df["tpep_pickup_datetime"] >= start)
            & (df["tpep_pickup_datetime"] < end)
        ].copy()
        counters["pickups_in_evaluation_window"] += len(df)
        if df.empty:
            continue

        df["duration_min"] = (
            df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
        ).dt.total_seconds() / 60
        df["speed_mph"] = df["trip_distance"] / (df["duration_min"] / 60)
        valid = (
            df["PULocationID"].isin(eligible)
            & df["DOLocationID"].isin(eligible)
            & df["duration_min"].between(1, 180)
            & df["trip_distance"].between(0.1, 100)
            & df["speed_mph"].between(1, 80)
            & np.isfinite(df["duration_min"])
            & np.isfinite(df["speed_mph"])
        )
        df = df.loc[valid, COLUMNS + ["duration_min"]].copy()
        counters["valid_after_quality_and_zone_filters"] += len(df)
        if not df.empty:
            parts.append(df)
        if i % 10 == 0:
            print(f"Scanned {counters['raw_rows_scanned']:,} rows...")

    if not parts:
        raise ValueError("No valid trips were found in the forecast test period")
    replay = pd.concat(parts, ignore_index=True)
    replay = replay.rename(columns={
        "tpep_pickup_datetime": "pickup_time",
        "tpep_dropoff_datetime": "dropoff_time",
        "PULocationID": "pickup_zone",
        "DOLocationID": "dropoff_zone",
        "trip_distance": "trip_distance_miles",
    })
    replay = replay.sort_values(
        ["pickup_time", "pickup_zone", "dropoff_zone", "dropoff_time"],
        kind="stable",
    ).reset_index(drop=True)
    replay.insert(0, "request_id", np.arange(len(replay), dtype=np.int64))
    replay["pickup_zone"] = replay["pickup_zone"].astype("int16")
    replay["dropoff_zone"] = replay["dropoff_zone"].astype("int16")
    replay["duration_min"] = replay["duration_min"].round(3)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    replay.to_parquet(REPLAY_PATH, index=False)

    report = {
        "source": str(RAW),
        "forecast_window_start_inclusive": str(start),
        "forecast_window_end_exclusive": str(end),
        "eligible_geographic_zones": len(eligible),
        **counters,
        "excluded_within_window": (
            counters["pickups_in_evaluation_window"] - len(replay)
        ),
        "replay_requests": len(replay),
        "median_trip_duration_min": float(replay["duration_min"].median()),
        "median_trip_distance_miles": float(replay["trip_distance_miles"].median()),
        "cross_zone_trips_pct": float(
            100 * (replay["pickup_zone"] != replay["dropoff_zone"]).mean()
        ),
        "note": (
            "This is cleaned human-driven yellow-taxi trip data, not robotaxi telemetry. "
            "Filtering alters demand totals compared with the original forecast evaluation. "
            "All compared policies must replay the same cleaned trip set. "
            "Recorded pickup times are treated as requests; recorded dropoff times "
            "provide observed durations but do not account for simulated dispatch delays."
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print("\n=== REPLAY SUMMARY ===")
    print(f"Trips in test window before filtering: {counters['pickups_in_evaluation_window']:,}")
    print(f"Valid replay requests: {len(replay):,}")
    print(f"Excluded in test window: {report['excluded_within_window']:,}")
    print(f"Median duration: {report['median_trip_duration_min']:.2f} minutes")
    print(f"Median distance: {report['median_trip_distance_miles']:.2f} miles")
    print(f"Cross-zone trips: {report['cross_zone_trips_pct']:.2f}%")
    print(f"Saved replay: {REPLAY_PATH}")
    print(f"Saved quality report: {REPORT_PATH}")
    print("IMPORTANT: Replay demand may differ from the earlier 776,144-request aggregate.")


if __name__ == "__main__":
    main()
