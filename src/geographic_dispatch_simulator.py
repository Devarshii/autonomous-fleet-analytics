
"""Geographic rider dispatch simulation for the NYC fleet analytics pilot.

Run from the repository root:
    python src/geographic_dispatch_simulator.py

Recorded taxi pickups are used as illustrative request times. The replay uses
actual trip durations and drop-off taxi zones. Pickup travel and repositioning
miles are estimated from zone representative points, not real routed distances.
This is a research prototype, not real AV operations or a validated improvement.
"""

from collections import deque
from heapq import heappop, heappush
from pathlib import Path
import math

import numpy as np
import pandas as pd

import geographic_trip_simulator as base
import selective_fleet_simulator as selective
import arrival_aware_fleet_optimizer as arrival

OUTPUT = Path("outputs/simulation")
MAX_PICKUP_WAIT_MIN = 12.0
SAME_ZONE_PICKUP_MIN = 2.0
MAX_NEARBY_ZONES = 20


def to_ns(values):
    """Convert datetime values to an explicit, consistent nanosecond unit."""
    return pd.to_datetime(values).to_numpy(dtype="datetime64[ns]").astype("int64")


def build_pickup_matrix(ids, coords):
    miles, reposition_minutes = selective.make_move_candidates(ids, coords)
    pickup_minutes = np.maximum(reposition_minutes.astype(float), SAME_ZONE_PICKUP_MIN)
    np.fill_diagonal(pickup_minutes, SAME_ZONE_PICKUP_MIN)
    nearest = np.argsort(pickup_minutes, axis=1, kind="stable")[:, :MAX_NEARBY_ZONES]
    return miles, reposition_minutes, pickup_minutes, nearest


def simulate_dispatch(policy, trips, hours, ids, coords, forecast, historical,
                      miles, reposition_minutes, pickup_minutes, nearest):
    if policy not in {"no_repositioning", "forecast_repositioning",
                      "arrival_aware_repositioning"}:
        raise ValueError(f"Unrecognized policy: {policy}")

    n_zones = len(ids)
    starting = base.integer_allocation(historical[0], base.FLEET_SIZE)
    idle = [deque() for _ in range(n_zones)]
    vehicle_id = 0

    for zone, count in enumerate(starting):
        for _ in range(int(count)):
            idle[zone].append(vehicle_id)
            vehicle_id += 1

    assert vehicle_id == base.FLEET_SIZE

    origins = trips["origin_i"].to_numpy(dtype=np.int32)
    destinations = trips["destination_i"].to_numpy(dtype=np.int32)

    request_ns = to_ns(trips["pickup_time"])
    dropoff_ns = to_ns(trips["dropoff_time"])
    recorded_trip_ns = dropoff_ns - request_ns

    hour_ns = to_ns(hours)
    minute_ns = 60 * 1_000_000_000
    hour_ns_const = 60 * minute_ns
    horizon_end_ns = int(hour_ns[-1] + hour_ns_const)

    if not len(request_ns) or not len(hour_ns):
        raise ValueError("No requests or evaluation hours")

    if not np.all(recorded_trip_ns > 0):
        raise ValueError("Trip durations must be positive")

    if request_ns.min() < hour_ns[0] or request_ns.max() >= horizon_end_ns:
        raise ValueError("Pickup timestamps fall outside the evaluation period")

    if np.any(np.diff(request_ns) < 0):
        raise ValueError("Replay requests must be sorted chronologically")

    if np.any(np.diff(hour_ns) != hour_ns_const):
        raise ValueError("Evaluation hours must be consecutive")

    events = []
    cursor = 0
    served_total = move_total = 0
    waits = []

    pickup_miles_total = reposition_miles_total = 0.0
    trip_vehicle_minutes = 0.0
    pickup_vehicle_minutes = 0.0
    reposition_vehicle_minutes = 0.0

    hourly_rows = []

    def finish_events(up_to_ns):
        while events and events[0][0] <= up_to_ns:
            _ready, vehicle, zone = heappop(events)
            idle[zone].append(vehicle)

    for h, hour_start in enumerate(hour_ns):
        now_ns = int(hour_start)
        finish_events(now_ns)

        moves_this_hour = 0
        reposition_miles_this_hour = 0.0

        if policy == "forecast_repositioning":
            target = base.integer_allocation(forecast[h], base.FLEET_SIZE)
            available = np.array([len(q) for q in idle], dtype=int)

            excess = np.maximum(available - target, 0)
            need = np.maximum(target - available, 0)

            donors = np.argsort(-excess, kind="stable")
            receivers = np.argsort(-need, kind="stable")

            d = r = 0

            while (
                d < n_zones
                and r < n_zones
                and moves_this_hour < base.MAX_REPOSITIONS_PER_HOUR
            ):
                source = int(donors[d])
                dest = int(receivers[r])

                if excess[source] <= 0:
                    d += 1
                    continue

                if need[dest] <= 0:
                    r += 1
                    continue

                if source == dest:
                    r += 1
                    continue

                vehicle = idle[source].pop()
                travel_min = float(reposition_minutes[source, dest])

                ready_ns = now_ns + math.ceil(travel_min * minute_ns)
                heappush(events, (ready_ns, vehicle, dest))

                dist = float(miles[source, dest])
                reposition_miles_total += dist
                reposition_miles_this_hour += dist

                reposition_vehicle_minutes += max(
                    0.0,
                    min(travel_min, (horizon_end_ns - now_ns) / minute_ns)
                )

                moves_this_hour += 1
                excess[source] -= 1
                need[dest] -= 1

        elif policy == "arrival_aware_repositioning":
            decisions = arrival.choose_arrival_aware_moves(
                idle, events, forecast[h], miles,
                reposition_minutes, now_ns, minute_ns
            )

            for source, dest, dist, travel_min in decisions:
                vehicle = idle[source].pop()
                ready_ns = now_ns + math.ceil(travel_min * minute_ns)

                heappush(events, (ready_ns, vehicle, dest))

                reposition_miles_total += dist
                reposition_miles_this_hour += dist

                reposition_vehicle_minutes += max(
                    0.0,
                    min(travel_min, (horizon_end_ns - now_ns) / minute_ns)
                )

                moves_this_hour += 1

        move_total += moves_this_hour

        hour_end = now_ns + hour_ns_const
        requests_this_hour = served_this_hour = 0

        waits_this_hour = []
        pickup_miles_this_hour = 0.0

        while cursor < len(trips) and request_ns[cursor] < hour_end:
            at_ns = int(request_ns[cursor])

            if at_ns < now_ns:
                raise AssertionError("Request processed in the wrong hour")

            finish_events(at_ns)

            origin = int(origins[cursor])
            destination = int(destinations[cursor])

            requests_this_hour += 1

            selected_zone = -1
            selected_wait = 0.0

            for candidate in nearest[origin]:
                z = int(candidate)
                travel_min = float(pickup_minutes[origin, z])

                if travel_min > MAX_PICKUP_WAIT_MIN:
                    break

                if idle[z]:
                    selected_zone = z
                    selected_wait = travel_min
                    break

            if selected_zone >= 0:
                vehicle = idle[selected_zone].pop()

                trip_start_ns = at_ns + math.ceil(
                    selected_wait * minute_ns
                )

                ready_ns = (
                    trip_start_ns + int(recorded_trip_ns[cursor])
                )

                heappush(
                    events,
                    (ready_ns, vehicle, destination)
                )

                served_this_hour += 1
                waits.append(selected_wait)
                waits_this_hour.append(selected_wait)

                dist = float(miles[selected_zone, origin])

                pickup_miles_total += dist
                pickup_miles_this_hour += dist

                pickup_vehicle_minutes += max(
                    0.0,
                    min(
                        selected_wait,
                        (horizon_end_ns - at_ns) / minute_ns
                    )
                )

                trip_vehicle_minutes += max(
                    0,
                    min(
                        int(recorded_trip_ns[cursor]),
                        max(0, horizon_end_ns - trip_start_ns)
                    )
                ) / minute_ns

            cursor += 1

        served_total += served_this_hour

        hourly_rows.append({
            "pickup_hour": hours[h],
            "policy": policy,
            "requests": requests_this_hour,
            "served": served_this_hour,
            "unserved": requests_this_hour - served_this_hour,
            "service_rate_pct": (
                100 * served_this_hour / requests_this_hour
                if requests_this_hour else 100.0
            ),
            "average_wait_min": (
                float(np.mean(waits_this_hour))
                if waits_this_hour else np.nan
            ),
            "pickup_empty_miles": round(pickup_miles_this_hour, 3),
            "reposition_empty_miles": round(
                reposition_miles_this_hour, 3
            ),
            "repositions": moves_this_hour,
        })

        if sum(map(len, idle)) + len(events) != base.FLEET_SIZE:
            raise AssertionError("Vehicle conservation failed")

    if cursor != len(trips):
        raise AssertionError(
            f"Unprocessed requests: {len(trips) - cursor}"
        )

    hourly = pd.DataFrame(hourly_rows)

    if len(hourly) != len(hours):
        raise AssertionError("Incorrect number of hourly records")

    if hourly["requests"].sum() != len(trips):
        raise AssertionError(
            "Hourly request totals do not match the input"
        )

    total_capacity = base.FLEET_SIZE * len(hours) * 60

    summary = {
        "policy": policy,
        "hours": len(hours),
        "fleet_size": base.FLEET_SIZE,
        "requests": len(trips),
        "served": served_total,
        "unserved": len(trips) - served_total,
        "service_rate_pct": round(
            100 * served_total / len(trips), 3
        ),
        "mean_wait_min_served": (
            round(float(np.mean(waits)), 3)
            if waits else np.nan
        ),
        "p90_wait_min_served": (
            round(float(np.percentile(waits, 90)), 3)
            if waits else np.nan
        ),
        "repositions": move_total,
        "estimated_pickup_empty_miles": round(
            pickup_miles_total, 2
        ),
        "estimated_reposition_empty_miles": round(
            reposition_miles_total, 2
        ),
        "estimated_total_empty_miles": round(
            pickup_miles_total + reposition_miles_total, 2
        ),
        "trip_utilization_pct": round(
            100 * trip_vehicle_minutes / total_capacity, 3
        ),
        "pickup_utilization_pct": round(
            100 * pickup_vehicle_minutes / total_capacity, 3
        ),
        "reposition_utilization_pct": round(
            100 * reposition_vehicle_minutes / total_capacity, 3
        ),
    }

    if int(hourly["served"].sum()) != served_total:
        raise AssertionError(
            "Hourly served count does not match summary"
        )

    return hourly, summary


def main():
    print(
        "=== GEOGRAPHIC RIDER DISPATCH EXPERIMENT ===",
        flush=True
    )

    trips, hours, ids, coords, forecast, historical = (
        base.load_inputs()
    )

    print(
        f"Requests: {len(trips):,} | "
        f"zones: {len(ids)} | "
        f"hours: {len(hours)} | "
        f"vehicles: {base.FLEET_SIZE:,}",
        flush=True
    )

    miles, reposition_minutes, pickup_minutes, nearest = (
        build_pickup_matrix(ids, coords)
    )

    summaries = []
    hourly_results = []

    for policy in (
        "no_repositioning",
        "forecast_repositioning",
        "arrival_aware_repositioning",
    ):
        print(f"Running {policy}...", flush=True)

        hourly, summary = simulate_dispatch(
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

        hourly_results.append(hourly)
        summaries.append(summary)

        print(
            f"  served={summary['served']:,} "
            f"({summary['service_rate_pct']:.2f}%), "
            f"mean wait="
            f"{summary['mean_wait_min_served']:.2f} min, "
            f"empty miles="
            f"{summary['estimated_total_empty_miles']:,.1f}",
            flush=True
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    pd.concat(
        hourly_results,
        ignore_index=True
    ).to_csv(
        OUTPUT / "geographic_dispatch_hourly.csv",
        index=False
    )

    pd.DataFrame(summaries).to_csv(
        OUTPUT / "geographic_dispatch_summary.csv",
        index=False
    )

    print(
        "Saved: outputs/simulation/geographic_dispatch_hourly.csv",
        flush=True
    )
    print(
        "Saved: outputs/simulation/geographic_dispatch_summary.csv",
        flush=True
    )

    print(
        "LIMITATIONS: zone-point travel estimates, "
        "12-minute max pickup wait, only idle cars dispatchable, "
        "no queue or charging, and recorded pickup time treated "
        "as request time. Simulation only.",
        flush=True
    )


if __name__ == "__main__":
    main()
