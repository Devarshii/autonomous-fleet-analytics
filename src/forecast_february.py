
"""February 2025 out-of-time demand forecasting evaluation.

Uses the existing January-trained LightGBM model.
Previous completed hours may supply observed demand features.
Never uses the target hour's demand as a prediction feature.
Does not retrain or overwrite the January model/results.

Run from repository root:
    python src/forecast_february.py
"""

from pathlib import Path
import json

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# ============================================================
# FILE PATHS
# ============================================================

MODEL_PATH = Path("models/lightgbm_demand_model.pkl")

JANUARY_PATH = Path(
    "data/processed/nyc_hourly_demand_complete.parquet"
)

FEBRUARY_REPLAY_PATH = Path(
    "data/processed/nyc_trip_replay_feb2025.parquet"
)

SHAPEFILE_PATH = Path(
    "data/external/taxi_zones/taxi_zones/taxi_zones.shp"
)

OUTPUT_PATH = Path(
    "data/processed/lightgbm_predictions_feb2025.parquet"
)

REPORT_PATH = Path(
    "outputs/validation/february_forecast_report.json"
)

START = pd.Timestamp("2025-02-01 00:00:00")
END = pd.Timestamp("2025-02-08 00:00:00")

FEATURES = [
    "LocationID",
    "hour",
    "day_of_week_num",
    "day_of_month",
    "is_weekend",
    "lag_1",
    "lag_24",
    "lag_168",
    "rolling_mean_24",
    "rolling_mean_168",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]


def main():

    print("=" * 70)
    print("FEBRUARY 2025 - INDEPENDENT LIGHTGBM EVALUATION")
    print("=" * 70)

    # ========================================================
    # VERIFY INPUT FILES
    # ========================================================

    for path in [
        MODEL_PATH,
        JANUARY_PATH,
        FEBRUARY_REPLAY_PATH,
        SHAPEFILE_PATH,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file missing: {path}"
            )

    # ========================================================
    # LOAD TRAINED MODEL
    # ========================================================

    print("\nLoading January-trained LightGBM model...")

    model = joblib.load(MODEL_PATH)

    saved_features = list(model.feature_name_)

    if saved_features != FEATURES:
        raise ValueError(
            "Saved model features do not match this pipeline.\n"
            f"Saved: {saved_features}\n"
            f"Expected: {FEATURES}"
        )

    print("Model loaded successfully.")
    print("No model retraining will occur.")

    # ========================================================
    # LOAD HISTORICAL DEMAND
    # ========================================================

    print("\nLoading January demand history...")

    january = pd.read_parquet(
        JANUARY_PATH,
        columns=[
            "pickup_hour",
            "LocationID",
            "demand",
        ],
    )

    january["pickup_hour"] = pd.to_datetime(
        january["pickup_hour"]
    )

    january = january[
        january["pickup_hour"] < START
    ].copy()

    # ========================================================
    # SELECT SAME GEOGRAPHIC COVERAGE
    # ========================================================

    geography = gpd.read_file(SHAPEFILE_PATH)

    geographic_ids = set(
        geography["LocationID"].dropna().astype(int)
    )

    january_ids = set(
        january["LocationID"].dropna().astype(int)
    )

    eligible_zones = sorted(
        january_ids & geographic_ids
    )

    print(
        f"Eligible taxi zones: {len(eligible_zones)}"
    )

    if len(eligible_zones) != 257:
        raise ValueError(
            "Expected 257 shared geographic zones. "
            "Check the January history and shapefile."
        )

    january = january[
        january["LocationID"].isin(eligible_zones)
    ].copy()

    # ========================================================
    # LOAD FEBRUARY REPLAY
    # ========================================================

    print("\nLoading February replay requests...")

    trips = pd.read_parquet(
        FEBRUARY_REPLAY_PATH,
        columns=[
            "pickup_time",
            "pickup_zone",
        ],
    )

    trips["pickup_time"] = pd.to_datetime(
        trips["pickup_time"]
    )

    if not (
        (trips["pickup_time"] >= START)
        & (trips["pickup_time"] < END)
    ).all():
        raise ValueError(
            "February replay contains pickups "
            "outside the evaluation week."
        )

    if not trips["pickup_zone"].isin(
        eligible_zones
    ).all():
        raise ValueError(
            "February replay contains an unexpected zone."
        )

    print(f"Replay requests: {len(trips):,}")

    # ========================================================
    # AGGREGATE FEBRUARY REQUESTS BY ZONE AND HOUR
    # ========================================================

    print("\nAggregating February hourly demand...")

    trips["pickup_hour"] = (
        trips["pickup_time"].dt.floor("h")
    )

    february = (
        trips.groupby(
            ["LocationID", "pickup_hour"]
            if "LocationID" in trips.columns
            else ["pickup_zone", "pickup_hour"]
        )
        .size()
        .reset_index(name="demand")
        .rename(
            columns={"pickup_zone": "LocationID"}
        )
    )

    # ========================================================
    # COMBINE HISTORICAL HOURS AND FEBRUARY HOURS
    # ========================================================

    print("Building complete hourly feature grid...")

    all_hours = pd.date_range(
        start=january["pickup_hour"].min(),
        end=END,
        freq="h",
        inclusive="left",
    )

    index = pd.MultiIndex.from_product(
        [eligible_zones, all_hours],
        names=["LocationID", "pickup_hour"],
    )

    observed = pd.concat(
        [january, february],
        ignore_index=True,
    )

    if observed.duplicated(
        ["LocationID", "pickup_hour"]
    ).any():
        raise ValueError(
            "Duplicate zone-hour observations found."
        )

    df = (
        observed.set_index(
            ["LocationID", "pickup_hour"]
        )
        .reindex(index)
        .reset_index()
    )

    missing_january = (
        (df["pickup_hour"] < START)
        & df["demand"].isna()
    )

    if missing_january.any():
        raise ValueError(
            "January history has missing zone-hour rows. "
            "Do not silently fill historical gaps."
        )

    # Missing February zone-hour combinations mean zero rides.
    df.loc[
        df["pickup_hour"] >= START,
        "demand",
    ] = (
        df.loc[
            df["pickup_hour"] >= START,
            "demand",
        ].fillna(0)
    )

    df["demand"] = df["demand"].astype(int)

    df = df.sort_values(
        ["LocationID", "pickup_hour"]
    ).reset_index(drop=True)

    # ========================================================
    # CALENDAR FEATURES
    # ========================================================

    df["hour"] = df["pickup_hour"].dt.hour

    df["day_of_week_num"] = (
        df["pickup_hour"].dt.dayofweek
    )

    df["day_of_month"] = (
        df["pickup_hour"].dt.day
    )

    df["is_weekend"] = (
        df["day_of_week_num"] >= 5
    ).astype(int)

    # ========================================================
    # HISTORICAL LAG FEATURES
    # ========================================================

    grouped = df.groupby(
        "LocationID"
    )["demand"]

    df["lag_1"] = grouped.shift(1)
    df["lag_24"] = grouped.shift(24)
    df["lag_168"] = grouped.shift(168)

    # ========================================================
    # ROLLING FEATURES
    # ========================================================

    print("Calculating lagged rolling features...")

    df["rolling_mean_24"] = (
        df.groupby("LocationID")["demand"]
        .transform(
            lambda x: (
                x.shift(1)
                .rolling(
                    window=24,
                    min_periods=1,
                )
                .mean()
            )
        )
    )

    df["rolling_mean_168"] = (
        df.groupby("LocationID")["demand"]
        .transform(
            lambda x: (
                x.shift(1)
                .rolling(
                    window=168,
                    min_periods=1,
                )
                .mean()
            )
        )
    )

    # ========================================================
    # CYCLICAL FEATURES
    # ========================================================

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df["dow_sin"] = np.sin(
        2 * np.pi * df["day_of_week_num"] / 7
    )

    df["dow_cos"] = np.cos(
        2 * np.pi * df["day_of_week_num"] / 7
    )

    # ========================================================
    # SELECT FEBRUARY EVALUATION PERIOD
    # ========================================================

    test = df[
        (df["pickup_hour"] >= START)
        & (df["pickup_hour"] < END)
    ].copy()

    expected_rows = 257 * 168

    if len(test) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} rows, "
            f"got {len(test):,}"
        )

    if test[FEATURES].isna().any().any():
        raise ValueError(
            "Missing February model features."
        )

    if int(test["demand"].sum()) != len(trips):
        raise ValueError(
            "February demand does not match replay requests."
        )

    # ========================================================
    # GENERATE FIXED-MODEL PREDICTIONS
    # ========================================================

    print("\nGenerating February predictions...")

    predicted = model.predict(
        test[FEATURES]
    )

    test["lightgbm_forecast"] = np.maximum(
        predicted, 0
    )

    test = test.sort_values(
        ["pickup_hour", "LocationID"]
    ).reset_index(drop=True)

    # ========================================================
    # EVALUATION METRICS
    # ========================================================

    actual = test["demand"].to_numpy()
    model_pred = test["lightgbm_forecast"].to_numpy()
    baseline_pred = test["lag_168"].to_numpy()

    model_mae = mean_absolute_error(
        actual, model_pred
    )

    baseline_mae = mean_absolute_error(
        actual, baseline_pred
    )

    total_demand = actual.sum()

    model_wape = (
        100 * np.abs(
            actual - model_pred
        ).sum() / total_demand
    )

    baseline_wape = (
        100 * np.abs(
            actual - baseline_pred
        ).sum() / total_demand
    )

    improvement_pct = (
        100 * (baseline_mae - model_mae) / baseline_mae
        if baseline_mae > 0
        else None
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    OUTPUT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    REPORT_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )

    test.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    report = {
        "evaluation_start": str(START),
        "evaluation_end_exclusive": str(END),
        "zones": len(eligible_zones),
        "hours": 168,
        "rows": len(test),
        "replay_requests": len(trips),
        "actual_demand": int(total_demand),
        "baseline_mae": float(baseline_mae),
        "lightgbm_mae": float(model_mae),
        "baseline_wape_pct": float(baseline_wape),
        "lightgbm_wape_pct": float(model_wape),
        "mae_improvement_pct": (
            float(improvement_pct)
            if improvement_pct is not None
            else None
        ),
        "model_retrained": False,
        "forecast_type": (
            "One-hour-ahead rolling evaluation using "
            "observed demand from completed prior hours."
        ),
        "model_path": str(MODEL_PATH),
        "prediction_path": str(OUTPUT_PATH),
        "note": (
            "February is an out-of-time evaluation. "
            "Current-hour targets are never features."
        ),
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2)
    )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("FEBRUARY FORECAST RESULTS")
    print("=" * 70)

    print(f"Taxi zones: {len(eligible_zones)}")
    print("Evaluation hours: 168")
    print(f"Zone-hour rows: {len(test):,}")
    print(f"Observed demand: {total_demand:,}")

    print("\nSame-hour-last-week baseline:")
    print(f"MAE: {baseline_mae:.3f}")
    print(f"WAPE: {baseline_wape:.2f}%")

    print("\nJanuary-trained LightGBM:")
    print(f"MAE: {model_mae:.3f}")
    print(f"WAPE: {model_wape:.2f}%")

    print("\nImprovement over baseline:")
    print(f"MAE improvement: {improvement_pct:.2f}%")

    print("\nModel retraining: NO")
    print("Prediction horizon: One hour ahead")
    print(f"Saved predictions: {OUTPUT_PATH}")
    print(f"Saved report: {REPORT_PATH}")

    print("\nFEBRUARY FORECAST EVALUATION COMPLETE")


if __name__ == "__main__":
    main()
