"""Event-based NYC taxi-zone fleet pilot (illustrative assumptions, not AV telemetry).

Input: held-out LightGBM predictions, one row per zone/hour.
Policies: no repositioning vs forecast-based idle-vehicle repositioning.
Limitations: 15-minute arrival buckets; every ride lasts 30 minutes and returns to
pickup zone; cross-zone repositioning takes 30 minutes regardless of distance;
requests not served immediately are dropped. No wait time or deadhead miles.
"""
from collections import deque
from heapq import heappop, heappush
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("data/processed/lightgbm_predictions.parquet")
OUTPUT = Path("outputs/simulation")
FLEET_SIZE = 1000
SLOTS_PER_HOUR = 4
TRIP_SLOTS = 2            # 30 minutes
REPOSITION_SLOTS = 2      # illustrative 30 minutes, independent of distance
MAX_MOVES_PER_HOUR = 100  # operational cap
SEED = 42


def proportional_allocation(weights, total):
    weights = np.maximum(np.asarray(weights, dtype=float), 0)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    desired = total * weights / weights.sum()
    allocated = np.floor(desired).astype(int)
    remaining = total - int(allocated.sum())
    if remaining:
        order = np.argsort(-(desired - allocated), kind="stable")
        allocated[order[:remaining]] += 1
    assert int(allocated.sum()) == total
    return allocated


def prepare_data():
    if not INPUT.exists():
        raise FileNotFoundError(f"Cannot find {INPUT}. Run the forecast pipeline first.")
    df = pd.read_parquet(INPUT)
    required = {"pickup_hour", "LocationID", "demand", "lightgbm_forecast", "lag_168"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df = df.copy()
    df["pickup_hour"] = pd.to_datetime(df["pickup_hour"])
    if df.duplicated(["pickup_hour", "LocationID"]).any():
        raise ValueError("Duplicate taxi-zone/hour observations")
    hours = pd.DatetimeIndex(sorted(df["pickup_hour"].unique()))
    zones = sorted(df["LocationID"].unique())
    if len(hours) != 168:
        raise ValueError(f"Expected 168 held-out hours, got {len(hours)}")
    if not (hours[1:] - hours[:-1] == pd.Timedelta(hours=1)).all():
        raise ValueError("Test hours must be consecutive")
    index = pd.MultiIndex.from_product([hours, zones], names=["pickup_hour", "LocationID"])
    frame = df.set_index(["pickup_hour", "LocationID"]).reindex(index)
    if frame[["demand", "lightgbm_forecast", "lag_168"]].isna().any().any():
        raise ValueError("Missing zone/hour inputs; check the complete grid")
    for name in ["demand", "lightgbm_forecast", "lag_168"]:
        if not np.isfinite(frame[name].to_numpy()).all():
            raise ValueError(f"Nonfinite values in {name}")
    demand = frame["demand"].to_numpy().reshape(len(hours), len(zones))
    if (demand < 0).any() or not np.allclose(demand, np.rint(demand)):
        raise ValueError("Demand must contain nonnegative integer request counts")
    forecast = np.maximum(frame["lightgbm_forecast"].to_numpy().reshape(len(hours), -1), 0)
    historical = np.maximum(frame["lag_168"].to_numpy().reshape(len(hours), -1), 0)
    return hours, zones, demand.astype(int), forecast, historical


def simulate(name, hours, zones, demand, forecast, historical):
    n_zones = len(zones)
    rng = np.random.default_rng(SEED)
    initial = proportional_allocation(historical[0], FLEET_SIZE)
    idle = [deque() for _ in zones]
    next_id = 0
    for zone, count in enumerate(initial):
        for _ in range(count):
            idle[zone].append(next_id)
            next_id += 1
    events = []  # (ready_slot, vehicle_id, destination_zone, reason)
    rows = []
    total_served = total_requests = total_moves = 0
    trip_vehicle_minutes = reposition_vehicle_minutes = 0

    for hour_i, hour in enumerate(hours):
        hourly_served = hourly_requests = hourly_moves = 0
        # Arrival schedule: exact aggregate demand, spread across four slots.
        per_slot, remainder = np.divmod(demand[hour_i], SLOTS_PER_HOUR)
        slot_requests = np.repeat(per_slot[None, :], SLOTS_PER_HOUR, axis=0)
        for z in range(n_zones):
            if remainder[z]:
                start = int(rng.integers(0, SLOTS_PER_HOUR))
                for j in range(int(remainder[z])):
                    slot_requests[(start + j) % SLOTS_PER_HOUR, z] += 1

        for quarter in range(SLOTS_PER_HOUR):
            slot = hour_i * SLOTS_PER_HOUR + quarter
            while events and events[0][0] <= slot:
                _, vehicle_id, zone, _ = heappop(events)
                idle[zone].append(vehicle_id)

            # Reposition only at the start of the hour, using the forecast for
            # this hour. Busy vehicles stay busy; moving cars cannot serve rides.
            if quarter == 0 and name == "forecast_repositioning":
                target = proportional_allocation(forecast[hour_i], FLEET_SIZE)
                available = np.array([len(pool) for pool in idle], dtype=int)
                # Do not attempt to move vehicles that are already busy.
                surplus = np.maximum(available - target, 0)
                deficit = np.maximum(target - available, 0)
                donors = list(np.argsort(-surplus, kind="stable"))
                receivers = list(np.argsort(-deficit, kind="stable"))
                a = b = 0
                while a < len(donors) and b < len(receivers) and hourly_moves < MAX_MOVES_PER_HOUR:
                    origin, dest = donors[a], receivers[b]
                    if surplus[origin] == 0:
                        a += 1
                        continue
                    if deficit[dest] == 0:
                        b += 1
                        continue
                    vehicle_id = idle[origin].pop()
                    heappush(events, (slot + REPOSITION_SLOTS, vehicle_id, int(dest), "reposition"))
                    surplus[origin] -= 1
                    deficit[dest] -= 1
                    hourly_moves += 1
                    reposition_vehicle_minutes += REPOSITION_SLOTS * 15

            for z in range(n_zones):
                requests = int(slot_requests[quarter, z])
                hourly_requests += requests
                served = min(requests, len(idle[z]))
                hourly_served += served
                for _ in range(served):
                    vehicle_id = idle[z].pop()
                    heappush(events, (slot + TRIP_SLOTS, vehicle_id, z, "trip"))
                trip_vehicle_minutes += served * TRIP_SLOTS * 15

        total_served += hourly_served
        total_requests += hourly_requests
        total_moves += hourly_moves
        rows.append({
            "pickup_hour": hour,
            "policy": name,
            "requests": hourly_requests,
            "served": hourly_served,
            "unserved": hourly_requests - hourly_served,
            "service_rate_pct": 100 * hourly_served / hourly_requests if hourly_requests else 100.0,
            "repositions": hourly_moves,
        })
        # Each vehicle must be idle or in exactly one scheduled event.
        assert sum(map(len, idle)) + len(events) == FLEET_SIZE, "Vehicle conservation failed"

    horizon_minutes = len(hours) * 60
    summary = {
        "policy": name,
        "hours": len(hours),
        "fleet_size": FLEET_SIZE,
        "requests": total_requests,
        "served": total_served,
        "unserved": total_requests - total_served,
        "service_rate_pct": round(100 * total_served / total_requests, 3),
        "repositions": total_moves,
        "trip_utilization_pct": round(100 * trip_vehicle_minutes / (FLEET_SIZE * horizon_minutes), 3),
        "reposition_utilization_pct": round(100 * reposition_vehicle_minutes / (FLEET_SIZE * horizon_minutes), 3),
    }
    return pd.DataFrame(rows), summary


def main():
    print("=== ADVANCED FLEET SIMULATION (ILLUSTRATIVE PILOT) ===")
    hours, zones, demand, forecast, historical = prepare_data()
    print(f"Loaded {len(hours)} hours, {len(zones)} zones, {demand.sum():,} ride requests")
    hourly_parts, summaries = [], []
    for policy in ("no_repositioning", "forecast_repositioning"):
        hourly, summary = simulate(policy, hours, zones, demand, forecast, historical)
        hourly_parts.append(hourly)
        summaries.append(summary)
        print(f"{policy}: served={summary['served']:,}, service rate={summary['service_rate_pct']:.2f}%, moves={summary['repositions']:,}")
    summary_df = pd.DataFrame(summaries)
    delta = summaries[1]["served"] - summaries[0]["served"]
    print(f"Additional simulated requests served: {delta:+,} (may be negative)")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.concat(hourly_parts, ignore_index=True).to_csv(OUTPUT / "advanced_fleet_hourly.csv", index=False)
    summary_df.to_csv(OUTPUT / "advanced_fleet_summary.csv", index=False)
    print(f"Saved outputs to {OUTPUT}")
    print("LIMITATIONS: Synthetic 30-minute trips and cross-zone moves; same-zone trip returns;")
    print("15-minute demand buckets; instant dispatch/no waiting; no real distances or charging.")


if __name__ == "__main__":
    main()
