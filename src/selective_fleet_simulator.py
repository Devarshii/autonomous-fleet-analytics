"""Selective geographic fleet repositioning experiment.

Place beside geographic_trip_simulator.py in src/ and run from repository root:
    python src/selective_fleet_simulator.py

Reuses precisely the same held-out trips, starting fleet, geographic assumptions,
and forecast inputs as the existing simulator. Compares two original policies with
an additional conservative, cost-aware, forecast-only repositioning heuristic.

This is NOT mathematical optimization or real AV deployment. Results are
simulation-only, with unserved requests dropped and no pickup waiting queue.
"""
from collections import deque
from heapq import heappop, heappush
from pathlib import Path
import math

import numpy as np
import pandas as pd

import geographic_trip_simulator as base

OUTPUT = Path("outputs/simulation")
# Fixed ahead of evaluating the held-out week; do not tune on this test set.
PLANNING_WINDOW_MIN = 15
MAX_MOVE_MIN = 30
RESERVE_IDLE = 1
MOVE_TIME_PENALTY = 0.35
MIN_MOVE_SCORE = 0.15


def make_move_candidates(ids, coords):
    """Precompute estimated repositioning minutes and miles for all zone pairs."""
    n = len(ids)
    miles = np.zeros((n, n), dtype=np.float32)
    minutes = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            mi = base.haversine_miles(coords[ids[i]], coords[ids[j]]) * base.ROAD_DISTANCE_MULTIPLIER
            mn = max(base.MIN_REPOSITION_MINUTES, mi / base.ASSUMED_EMPTY_SPEED_MPH * 60)
            miles[i, j] = miles[j, i] = mi
            minutes[i, j] = minutes[j, i] = mn
    return miles, minutes


def selective_moves(idle, forecast_hour, miles, minutes):
    """Choose moves using ONLY forecast, currently idle cars, and geographic cost.

    Reserve enough idle cars for forecasted near-term demand in each source zone.
    Fill near-term shortages in destination zones, preferring short moves.
    No future observed requests are used in the decision rule.
    """
    available = np.fromiter((len(q) for q in idle), dtype=np.int32, count=len(idle))
    expected_15_min = np.maximum(0, np.asarray(forecast_hour, dtype=float)) * (PLANNING_WINDOW_MIN / 60)
    required = np.ceil(expected_15_min).astype(np.int32)
    surplus = np.maximum(available - required - RESERVE_IDLE, 0)
    shortfall = np.maximum(required - available, 0)

    donors = np.flatnonzero(surplus > 0)
    receivers = np.flatnonzero(shortfall > 0)
    if not len(donors) or not len(receivers):
        return []

    # Candidate utility: shortage fraction minus estimated time/opportunity cost.
    shortage_fraction = np.minimum(1.0, shortfall[receivers] / np.maximum(1.0, required[receivers]))
    scores = shortage_fraction[None, :] - MOVE_TIME_PENALTY * (minutes[np.ix_(donors, receivers)] / MAX_MOVE_MIN)
    eligible = (minutes[np.ix_(donors, receivers)] <= MAX_MOVE_MIN) & (scores >= MIN_MOVE_SCORE)
    r_idx, c_idx = np.nonzero(eligible)
    if not len(r_idx):
        return []
    order = np.argsort(-scores[r_idx, c_idx], kind="stable")
    decisions = []
    for k in order:
        i, j = int(donors[r_idx[k]]), int(receivers[c_idx[k]])
        if surplus[i] <= 0 or shortfall[j] <= 0:
            continue
        # A fixed upper bound preserves the same operating move cap as the old policy.
        amount = min(int(surplus[i]), int(shortfall[j]), base.MAX_REPOSITIONS_PER_HOUR - len(decisions))
        for _ in range(max(0, amount)):
            decisions.append((i, j, float(miles[i, j]), float(minutes[i, j])))
        surplus[i] -= amount
        shortfall[j] -= amount
        if len(decisions) >= base.MAX_REPOSITIONS_PER_HOUR:
            break
    return decisions


def simulate_selective(trips, hours, ids, coords, forecast, historical):
    n = len(ids)
    initial = base.integer_allocation(historical[0], base.FLEET_SIZE)
    idle = [deque() for _ in range(n)]
    vid = 0
    for z, count in enumerate(initial):
        for _ in range(int(count)):
            idle[z].append(vid)
            vid += 1
    assert vid == base.FLEET_SIZE
    miles, minutes = make_move_candidates(ids, coords)

    origins = trips["origin_i"].to_numpy(dtype=np.int32)
    dests = trips["destination_i"].to_numpy(dtype=np.int32)
    pickup = trips["pickup_time"].astype("int64").to_numpy()
    dropoff = trips["dropoff_time"].astype("int64").to_numpy()
    hour_ns = hours.astype("int64").to_numpy()
    minute_ns = 60 * 1_000_000_000
    events = []
    cursor = total_served = total_moves = 0
    empty_miles = trip_minutes = reposition_minutes = 0.0
    rows = []

    def finish_events(now):
        while events and events[0][0] <= now:
            _, vehicle, destination = heappop(events)
            idle[destination].append(vehicle)

    for h, current_ns in enumerate(hour_ns):
        start_ns = int(current_ns)
        finish_events(start_ns)
        moves = selective_moves(idle, forecast[h], miles, minutes)
        for source, destination, distance, duration in moves:
            vehicle = idle[source].pop()
            heappush(events, (start_ns + math.ceil(duration * minute_ns), vehicle, destination))
            empty_miles += distance
            reposition_minutes += duration
        total_moves += len(moves)

        requests = served = 0
        stop_ns = start_ns + 3600 * 1_000_000_000
        while cursor < len(trips) and pickup[cursor] < stop_ns:
            at = int(pickup[cursor])
            finish_events(at)
            source, destination = int(origins[cursor]), int(dests[cursor])
            requests += 1
            if idle[source]:
                vehicle = idle[source].pop()
                end = int(dropoff[cursor])
                heappush(events, (end, vehicle, destination))
                trip_minutes += (end - at) / minute_ns
                served += 1
            cursor += 1
        total_served += served
        rows.append({
            "pickup_hour": hours[h], "policy": "selective_repositioning",
            "requests": requests, "served": served, "unserved": requests - served,
            "service_rate_pct": (100 * served / requests if requests else 100.0),
            "repositions": len(moves),
        })
        assert sum(map(len, idle)) + len(events) == base.FLEET_SIZE, "Vehicle conservation failed"
    assert cursor == len(trips), "Not all requests were processed"
    total_requests = len(trips)
    total_time = base.FLEET_SIZE * len(hours) * 60
    summary = {
        "policy": "selective_repositioning", "hours": len(hours),
        "fleet_size": base.FLEET_SIZE, "requests": total_requests,
        "served": total_served, "unserved": total_requests - total_served,
        "service_rate_pct": round(100 * total_served / total_requests, 3),
        "repositions": total_moves,
        "estimated_empty_reposition_miles": round(empty_miles, 2),
        "trip_utilization_pct": round(100 * trip_minutes / total_time, 3),
        "reposition_utilization_pct": round(100 * reposition_minutes / total_time, 3),
    }
    return pd.DataFrame(rows), summary


def main():
    print("=== SELECTIVE REPOSITIONING: THREE-POLICY NYC REPLAY ===")
    trips, hours, ids, coords, forecast, historical = base.load_inputs()
    print(f"Inputs: {len(trips):,} requests; {len(ids)} zones; {len(hours)} hours; {base.FLEET_SIZE:,} vehicles")
    print("Policies share identical requests, forecasts, and initial vehicle placement.")
    all_hourly, summaries = [], []
    for policy in ("no_repositioning", "forecast_repositioning"):
        hourly, summary = base.simulate(policy, trips, hours, ids, coords, forecast, historical)
        all_hourly.append(hourly)
        summaries.append(summary)
        print(f"{policy}: served={summary['served']:,}; moves={summary['repositions']:,}; estimated empty miles={summary['estimated_empty_reposition_miles']:,.1f}")
    hourly, summary = simulate_selective(trips, hours, ids, coords, forecast, historical)
    all_hourly.append(hourly)
    summaries.append(summary)
    print(f"selective_repositioning: served={summary['served']:,}; moves={summary['repositions']:,}; estimated empty miles={summary['estimated_empty_reposition_miles']:,.1f}")
    print(f"Selective versus no-repositioning: {summary['served'] - summaries[0]['served']:+,} rides")
    print(f"Selective versus proportional forecast: {summary['served'] - summaries[1]['served']:+,} rides")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.concat(all_hourly, ignore_index=True).to_csv(OUTPUT / "selective_fleet_hourly.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUTPUT / "selective_fleet_summary.csv", index=False)
    print("Saved outputs/simulation/selective_fleet_hourly.csv")
    print("Saved outputs/simulation/selective_fleet_summary.csv")
    print("LIMITATIONS: illustrative replay; immediate same-zone dispatch, no rider queue, no charging or maintenance; estimated repositioning distances.")
    print("This is an uncalibrated heuristic, not a proven optimizer. Results may be worse than either baseline.")


if __name__ == "__main__":
    main()
