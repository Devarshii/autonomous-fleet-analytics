"""Geographic, event-driven NYC fleet simulation (research prototype).

Inputs:
  data/processed/nyc_trip_replay_test.parquet
  data/processed/lightgbm_predictions.parquet
  data/external/taxi_zones/taxi_zones/taxi_zones.shp

Compare identical trip requests with and without hourly forecast-driven repositioning.
Recorded taxi trip pickup/drop-off zones and durations are used for served rides.
Empty repositioning times/distances are *estimated* from taxi-zone centroids,
not routed road distance. No rider queues, charging, or vehicle health yet.
Results are illustrative simulations, NOT real robotaxi performance.
Run from project root: python src/geographic_trip_simulator.py
"""

from collections import deque
from heapq import heappop, heappush
from pathlib import Path
import math

import geopandas as gpd
import numpy as np
import pandas as pd

TRIPS_FILE = Path("data/processed/nyc_trip_replay_test.parquet")
FORECAST_FILE = Path("data/processed/lightgbm_predictions.parquet")
ZONES_FILE = Path("data/external/taxi_zones/taxi_zones/taxi_zones.shp")
OUTPUT_DIR = Path("outputs/simulation")

FLEET_SIZE = 1000
MAX_REPOSITIONS_PER_HOUR = 100
ASSUMED_EMPTY_SPEED_MPH = 15.0
ROAD_DISTANCE_MULTIPLIER = 1.35  # straight-line centroid miles -> estimated road miles
MIN_REPOSITION_MINUTES = 5.0
SEED = 42


def integer_allocation(weights, total):
    a = np.asarray(weights, dtype=float)
    if len(a) == 0 or not np.isfinite(a).all():
        raise ValueError("Allocation weights must be non-empty and finite")
    a = np.maximum(a, 0.0)
    if a.sum() == 0:
        a = np.ones_like(a)
    fractional = total * a / a.sum()
    output = np.floor(fractional).astype(int)
    difference = int(total - output.sum())
    order = np.argsort(-(fractional - output), kind="stable")
    output[order[:difference]] += 1
    return output


def load_inputs():
    for p in [TRIPS_FILE, FORECAST_FILE, ZONES_FILE]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}")
    trips = pd.read_parquet(TRIPS_FILE, columns=[
        "request_id", "pickup_time", "dropoff_time", "pickup_zone",
        "dropoff_zone", "trip_distance_miles", "duration_min",
    ])
    if trips.empty:
        raise ValueError("Trip replay dataset is empty")
    trips["pickup_time"] = pd.to_datetime(trips["pickup_time"])
    trips["dropoff_time"] = pd.to_datetime(trips["dropoff_time"])
    trips = trips.sort_values(["pickup_time", "request_id"], kind="stable").reset_index(drop=True)
    if trips["request_id"].duplicated().any():
        raise ValueError("Duplicate request IDs in replay dataset")
    if (trips["dropoff_time"] <= trips["pickup_time"]).any():
        raise ValueError("Invalid trip duration in replay dataset")

    geodf = gpd.read_file(ZONES_FILE)
    if geodf.crs is None:
        raise ValueError("Taxi-zone shapefile has no CRS")
    geo = geodf[["LocationID", "geometry"]].copy()
    geo["LocationID"] = geo["LocationID"].astype(int)
    geo = geo.drop_duplicates("LocationID").to_crs(epsg=4326)
    # Representative points are inside their polygons (unlike some centroids).
    points = geo.geometry.representative_point()
    coordinates = {int(z): (float(p.y), float(p.x)) for z, p in zip(geo["LocationID"], points)}

    forecast_df = pd.read_parquet(FORECAST_FILE, columns=[
        "pickup_hour", "LocationID", "lightgbm_forecast", "lag_168"
    ])
    forecast_df["pickup_hour"] = pd.to_datetime(forecast_df["pickup_hour"])
    forecast_df["LocationID"] = forecast_df["LocationID"].astype(int)
    valid_ids = sorted(set(forecast_df["LocationID"]) & set(coordinates))
    zone_to_i = {z: i for i, z in enumerate(valid_ids)}
    if not valid_ids:
        raise ValueError("No matching geographic and forecast zones")
    excluded = (~trips["pickup_zone"].isin(valid_ids) | ~trips["dropoff_zone"].isin(valid_ids))
    if excluded.any():
        print(f"Excluded {int(excluded.sum()):,} trips with zones outside forecast/geography intersection")
        trips = trips.loc[~excluded].copy()
    trips["origin_i"] = trips["pickup_zone"].map(zone_to_i).astype(np.int32)
    trips["destination_i"] = trips["dropoff_zone"].map(zone_to_i).astype(np.int32)

    hours = pd.DatetimeIndex(sorted(forecast_df["pickup_hour"].unique()))
    if len(hours) != 168 or not (hours[1:] - hours[:-1] == pd.Timedelta(hours=1)).all():
        raise ValueError("Forecast must contain 168 consecutive hourly predictions")
    interval_end = hours[-1] + pd.Timedelta(hours=1)
    if trips["pickup_time"].min() < hours[0] or trips["pickup_time"].max() >= interval_end:
        raise ValueError("Replay trips outside the held-out prediction interval")
    grid = pd.MultiIndex.from_product([hours, valid_ids], names=["pickup_hour", "LocationID"])
    grid_df = forecast_df.set_index(["pickup_hour", "LocationID"])
    if grid_df.index.duplicated().any():
        raise ValueError("Duplicate zone/hour predictions")
    grid_df = grid_df.reindex(grid)
    if grid_df[["lightgbm_forecast", "lag_168"]].isna().any().any():
        raise ValueError("Forecast grid is missing zone/hour observations")
    forecast = np.clip(grid_df["lightgbm_forecast"].to_numpy().reshape(len(hours), -1), 0, None)
    historical = np.clip(grid_df["lag_168"].to_numpy().reshape(len(hours), -1), 0, None)

    return trips, hours, valid_ids, coordinates, forecast, historical


def haversine_miles(a, b):
    lat1, lon1 = a
    lat2, lon2 = b
    r = 3958.7613
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    t = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(min(1, math.sqrt(t)))


def simulate(policy, trips, hours, ids, coords, forecast, historical):
    if policy not in ("no_repositioning", "forecast_repositioning"):
        raise ValueError("Invalid policy")
    n = len(ids)
    # Same starting placement for both policies; no knowledge of future observed requests.
    start = integer_allocation(historical[0], FLEET_SIZE)
    idle = [deque() for _ in range(n)]
    vehicle_id = 0
    for i, count in enumerate(start):
        for _ in range(int(count)):
            idle[i].append(vehicle_id)
            vehicle_id += 1
    assert vehicle_id == FLEET_SIZE
    # Events: (ready_ns, vehicle_id, destination_zone_index, reason)
    events = []
    origins = trips["origin_i"].to_numpy(dtype=np.int32)
    destinations = trips["destination_i"].to_numpy(dtype=np.int32)
    pickup_ns = trips["pickup_time"].astype("int64").to_numpy()
    dropoff_ns = trips["dropoff_time"].astype("int64").to_numpy()
    estimated_reposition_miles = 0.0
    total_moves = total_served = 0
    trip_vehicle_minutes = reposition_vehicle_minutes = 0.0
    hourly_rows = []
    cursor = 0
    hour_ns = hours.astype("int64").to_numpy()
    ns_per_minute = 60 * 1_000_000_000

    def finish_events(before_ns):
        while events and events[0][0] <= before_ns:
            _, vid, dest, _ = heappop(events)
            idle[dest].append(vid)

    for h, current_ns in enumerate(hour_ns):
        finish_events(int(current_ns))
        moves_this_hour = 0
        if policy == "forecast_repositioning" and MAX_REPOSITIONS_PER_HOUR > 0:
            target = integer_allocation(forecast[h], FLEET_SIZE)
            available = np.array([len(q) for q in idle], dtype=int)
            excess = np.maximum(available - target, 0)
            need = np.maximum(target - available, 0)
            donors = np.argsort(-excess, kind="stable")
            receivers = np.argsort(-need, kind="stable")
            d = r = 0
            while d < n and r < n and moves_this_hour < MAX_REPOSITIONS_PER_HOUR:
                source, dest = int(donors[d]), int(receivers[r])
                if excess[source] == 0:
                    d += 1
                    continue
                if need[dest] == 0:
                    r += 1
                    continue
                if source == dest:
                    r += 1
                    continue
                distance = haversine_miles(coords[ids[source]], coords[ids[dest]]) * ROAD_DISTANCE_MULTIPLIER
                duration = max(MIN_REPOSITION_MINUTES, distance / ASSUMED_EMPTY_SPEED_MPH * 60)
                vid = idle[source].pop()
                ready_ns = int(current_ns + math.ceil(duration * ns_per_minute))
                heappush(events, (ready_ns, vid, dest, "reposition"))
                estimated_reposition_miles += distance
                reposition_vehicle_minutes += duration
                moves_this_hour += 1
                excess[source] -= 1
                need[dest] -= 1
        total_moves += moves_this_hour

        next_hour_ns = int(current_ns + 3600 * 1_000_000_000)
        requests = served = 0
        while cursor < len(trips) and pickup_ns[cursor] < next_hour_ns:
            at_ns = int(pickup_ns[cursor])
            finish_events(at_ns)
            origin, destination = int(origins[cursor]), int(destinations[cursor])
            requests += 1
            if idle[origin]:
                vid = idle[origin].pop()
                end_ns = int(dropoff_ns[cursor])
                heappush(events, (end_ns, vid, destination, "trip"))
                trip_vehicle_minutes += (end_ns - at_ns) / ns_per_minute
                served += 1
            cursor += 1
        total_served += served
        hourly_rows.append({
            "pickup_hour": hours[h], "policy": policy, "requests": requests,
            "served": served, "unserved": requests - served,
            "service_rate_pct": 100 * served / requests if requests else 100.0,
            "repositions": moves_this_hour,
        })
        assert sum(map(len, idle)) + len(events) == FLEET_SIZE, "Vehicle conservation failed"
    if cursor != len(trips):
        raise AssertionError(f"Unprocessed requests: {len(trips) - cursor}")
    horizon_minutes = len(hours) * 60
    result = {
        "policy": policy, "hours": len(hours), "fleet_size": FLEET_SIZE,
        "requests": len(trips), "served": total_served, "unserved": len(trips) - total_served,
        "service_rate_pct": round(100 * total_served / len(trips), 3),
        "repositions": total_moves,
        "estimated_empty_reposition_miles": round(estimated_reposition_miles, 2),
        "trip_utilization_pct": round(100 * trip_vehicle_minutes / (FLEET_SIZE * horizon_minutes), 3),
        "reposition_utilization_pct": round(100 * reposition_vehicle_minutes / (FLEET_SIZE * horizon_minutes), 3),
    }
    return pd.DataFrame(hourly_rows), result


def main():
    print("=== GEOGRAPHIC NYC TRIP-REPLAY SIMULATOR ===")
    trips, hours, ids, coords, forecast, historical = load_inputs()
    print(f"Requests: {len(trips):,} | Zones: {len(ids)} | Hours: {len(hours)} | Fleet: {FLEET_SIZE:,}")
    output_rows, summaries = [], []
    for policy in ("no_repositioning", "forecast_repositioning"):
        hourly, result = simulate(policy, trips, hours, ids, coords, forecast, historical)
        output_rows.append(hourly)
        summaries.append(result)
        print(f"{policy}: served={result['served']:,}, service={result['service_rate_pct']:.2f}%, moves={result['repositions']:,}, estimated empty miles={result['estimated_empty_reposition_miles']:,.1f}")
    difference = summaries[1]["served"] - summaries[0]["served"]
    print(f"Difference in simulated served requests: {difference:+,}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(output_rows, ignore_index=True).to_csv(OUTPUT_DIR / "geographic_fleet_hourly.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUTPUT_DIR / "geographic_fleet_summary.csv", index=False)
    print(f"Saved: {OUTPUT_DIR / 'geographic_fleet_hourly.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'geographic_fleet_summary.csv'}")
    print("LIMITATIONS: same-zone instant dispatch, unserved requests dropped, no waiting queue/charging/maintenance; road distances approximated from zone points.")
    print("Results do not establish real-world AV service improvements.")


if __name__ == "__main__":
    main()
