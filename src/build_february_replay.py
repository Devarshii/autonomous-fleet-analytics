
"""Build an independent February 1–7, 2025 NYC taxi trip replay."""

from pathlib import Path
import json

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

RAW = Path("data/raw/yellow_tripdata_2025-02.parquet")
JAN_FORECAST = Path("data/processed/lightgbm_predictions.parquet")
SHAPEFILE = Path(
    "data/external/taxi_zones/taxi_zones/taxi_zones.shp"
)

OUTPUT = Path(
    "data/processed/nyc_trip_replay_feb2025.parquet"
)
REPORT = Path(
    "outputs/validation/february_trip_replay_quality.json"
)

START = pd.Timestamp("2025-02-01 00:00:00")
END = pd.Timestamp("2025-02-08 00:00:00")

COLUMNS = [
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_distance",
]

BATCH_SIZE = 100_000


def main():
    print("=== FEBRUARY INDEPENDENT TRIP REPLAY ===")
    print(f"Period: {START} to {END} (end exclusive)")

    for path in (RAW, JAN_FORECAST, SHAPEFILE):
        if not path.exists():
            raise FileNotFoundError(f"Missing input: {path}")

    january = pd.read_parquet(
        JAN_FORECAST, columns=["LocationID"]
    )

    january_zones = set(
        january["LocationID"].dropna().astype(int)
    )

    geography = gpd.read_file(SHAPEFILE)

    geographic_zones = set(
        geography["LocationID"].dropna().astype(int)
    )

    eligible = january_zones & geographic_zones
    print(f"Eligible zones: {len(eligible)}")

    parquet = pq.ParquetFile(RAW)

    missing = set(COLUMNS) - set(parquet.schema_arrow.names)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    scanned = 0
    in_window = 0
    parts = []

    for batch in parquet.iter_batches(
        batch_size=BATCH_SIZE,
        columns=COLUMNS,
    ):
        df = batch.to_pandas()
        scanned += len(df)

        df = df.loc[
            (df["tpep_pickup_datetime"] >= START)
            & (df["tpep_pickup_datetime"] < END)
        ].copy()

        in_window += len(df)

        if df.empty:
            continue

        df["duration_min"] = (
            df["tpep_dropoff_datetime"]
            - df["tpep_pickup_datetime"]
        ).dt.total_seconds() / 60

        df["speed_mph"] = (
            df["trip_distance"]
            / (df["duration_min"] / 60)
        )

        valid = (
            df["PULocationID"].isin(eligible)
            & df["DOLocationID"].isin(eligible)
            & df["duration_min"].between(1, 180)
            & df["trip_distance"].between(0.1, 100)
            & df["speed_mph"].between(1, 80)
            & np.isfinite(df["duration_min"])
            & np.isfinite(df["speed_mph"])
        )

        cleaned = df.loc[
            valid, COLUMNS + ["duration_min"]
        ].copy()

        if not cleaned.empty:
            parts.append(cleaned)

    if not parts:
        raise ValueError("No valid February trips found")

    replay = pd.concat(parts, ignore_index=True)

    replay = replay.rename(columns={
        "tpep_pickup_datetime": "pickup_time",
        "tpep_dropoff_datetime": "dropoff_time",
        "PULocationID": "pickup_zone",
        "DOLocationID": "dropoff_zone",
        "trip_distance": "trip_distance_miles",
    })

    replay = replay.sort_values(
        [
            "pickup_time",
            "pickup_zone",
            "dropoff_zone",
            "dropoff_time",
        ],
        kind="stable",
    ).reset_index(drop=True)

    replay.insert(
        0,
        "request_id",
        np.arange(len(replay), dtype=np.int64),
    )

    replay["pickup_zone"] = (
        replay["pickup_zone"].astype("int16")
    )

    replay["dropoff_zone"] = (
        replay["dropoff_zone"].astype("int16")
    )

    replay["duration_min"] = (
        replay["duration_min"].round(3)
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)

    replay.to_parquet(OUTPUT, index=False)

    report = {
        "source": str(RAW),
        "start_inclusive": str(START),
        "end_exclusive": str(END),
        "geographically_matched_zones": len(eligible),
        "raw_rows_scanned": scanned,
        "pickups_in_window": in_window,
        "valid_replay_requests": len(replay),
        "excluded_in_window": in_window - len(replay),
        "median_duration_min": float(
            replay["duration_min"].median()
        ),
        "median_distance_miles": float(
            replay["trip_distance_miles"].median()
        ),
        "cross_zone_trips_pct": float(
            100 * (
                replay["pickup_zone"]
                != replay["dropoff_zone"]
            ).mean()
        ),
        "note": (
            "Human-driven taxi trips used as a robotaxi "
            "demand proxy. Same geographic and quality "
            "filters as January development replay. "
            "This file does not contain model forecasts."
        ),
    }

    REPORT.write_text(json.dumps(report, indent=2))

    print("\n=== FEBRUARY REPLAY SUMMARY ===")
    print(f"Rows scanned: {scanned:,}")
    print(f"Trips in evaluation week: {in_window:,}")
    print(f"Valid replay requests: {len(replay):,}")
    print(f"Excluded trips: {in_window - len(replay):,}")
    print(
        f"Median duration: "
        f"{report['median_duration_min']:.2f} min"
    )
    print(
        f"Median distance: "
        f"{report['median_distance_miles']:.2f} miles"
    )
    print(f"Saved replay: {OUTPUT}")
    print(f"Saved quality report: {REPORT}")

    assert replay["pickup_time"].min() >= START
    assert replay["pickup_time"].max() < END
    assert replay["request_id"].is_unique
    assert replay["pickup_time"].is_monotonic_increasing

    print("\nFEBRUARY TRIP PREPARATION COMPLETE")


if __name__ == "__main__":
    main()
