"""Arrival-aware NYC fleet repositioning experiment.

Run from the repository root:
    python src/arrival_aware_fleet_optimizer.py

Uses the SAME held-out trips, forecasts, starting allocation, geographic travel
assumptions, and immediate same-zone dispatch rule as geographic_trip_simulator.
Compares no repositioning, proportional forecast repositioning, selective
repositioning, and an experimental arrival-aware policy.

IMPORTANT: This is a heuristic evaluated on the already-inspected Jan 25–31,
2025 test week. It is not an independently validated optimizer, and does NOT
establish actual robotaxi benefits. No rider queues, charging, risk, or
maintenance are modeled. Estimated empty miles are not routed road miles.
"""
from collections import deque
from heapq import heappop, heappush
from pathlib import Path
import math

import numpy as np
import pandas as pd

import geographic_trip_simulator as base
import selective_fleet_simulator as selective

OUTPUT = Path('outputs/simulation')

# Fixed, illustrative operating parameters. Do not tune them to maximize
# results on the Jan 25–31 evaluation week; use a fresh holdout later.
MAX_TRAVEL_MIN = 30.0
MAX_MOVES_PER_HOUR = 100
MIN_SOURCE_RESERVE = 1
DEMAND_LOOKAHEAD_MIN = 60.0
MIN_NET_BENEFIT = 0.50
MILE_COST_PER_MILE = 0.15  # demand-equivalent heuristic penalty, not dollars


def choose_arrival_aware_moves(idle, events, forecast_hour, miles, minutes,
                               now_ns, minute_ns):
    """Greedy move proposals based only on forecasts and current vehicle state.

    A proposed car must leave forecast-based short-term source capacity,
    reach a destination in <= MAX_TRAVEL_MIN, and cover predicted remaining
    demand after arriving. Scheduled vehicle returns reduce destination need.
    Neither observed future trip demand nor future trip endpoints are used.
    """
    n = len(idle)
    available = np.array([len(pool) for pool in idle], dtype=np.int32)
    predicted = np.maximum(np.asarray(forecast_hour, dtype=float), 0.0)

    # Count scheduled *vehicle* returns within this hour (trips and moves).
    incoming = np.zeros(n, dtype=np.int32)
    end_ns = now_ns + int(DEMAND_LOOKAHEAD_MIN * minute_ns)
    for ready_ns, _vid, dest, *_ in events:
        if now_ns < ready_ns <= end_ns:
            incoming[int(dest)] += 1

    # Protect enough cars to cover predicted demand during the early portion
    # of the hour; at least one idle car stays in a donor zone.
    early_demand = np.ceil(predicted * (MAX_TRAVEL_MIN / 60.0)).astype(int)
    source_reserve = np.maximum(MIN_SOURCE_RESERVE, early_demand)
    surplus = np.maximum(available - source_reserve, 0)

    # Remaining demand beyond the earliest plausible arrival. Incoming cars
    # may already cover it. This deliberately favors fewer, justified moves.
    shortfall = np.maximum(np.ceil(predicted * (DEMAND_LOOKAHEAD_MIN / 60.0))
                           - available - incoming, 0).astype(int)

    # Candidate scoring discounts demand by time lost in transit. A move is
    # not worthwhile just because its destination has a higher forecast.
    candidates = []
    donors = np.flatnonzero(surplus > 0)
    receivers = np.flatnonzero(shortfall > 0)
    for i in donors:
        for j in receivers:
            if i == j:
                continue
            duration = float(minutes[i, j])
            if not (0 < duration <= MAX_TRAVEL_MIN):
                continue
            distance = float(miles[i, j])
            remaining_fraction = max(0.0, (DEMAND_LOOKAHEAD_MIN - duration) /
                                      DEMAND_LOOKAHEAD_MIN)
            # Effective shortage expected after arrival, bounded to the
            # marginal benefit of one extra vehicle.
            arrival_demand = predicted[j] * remaining_fraction
            arrival_supply = available[j] + incoming[j]
            potential_benefit = min(1.0, max(0.0, arrival_demand - arrival_supply))
            # Source demand represents the vehicle's opportunity cost.
            source_demand = predicted[i] * (duration / 60.0)
            source_capacity = max(0.0, available[i] - 1.0)
            source_penalty = min(1.0, max(0.0, source_demand - source_capacity))
            score = (potential_benefit - source_penalty
                     - MILE_COST_PER_MILE * distance)
            if score >= MIN_NET_BENEFIT:
                candidates.append((-score, duration, distance, int(i), int(j)))

    candidates.sort()  # highest benefit, then shorter travel, then shorter miles
    moves = []
    for neg_score, duration, distance, i, j in candidates:
        if not surplus[i] or not shortfall[j]:
            continue
        if len(moves) >= MAX_MOVES_PER_HOUR:
            break
        moves.append((i, j, distance, duration))
        surplus[i] -= 1
        shortfall[j] -= 1
        available[i] -= 1
        available[j] += 1  # reserve incoming capacity against further moves
    return moves


def simulate_arrival_aware(trips, hours, ids, coords, forecast, historical):
    n = len(ids)
    initial = base.integer_allocation(historical[0], base.FLEET_SIZE)
    idle = [deque() for _ in range(n)]
    vid = 0
    for zone_i, count in enumerate(initial):
        for _ in range(int(count)):
            idle[zone_i].append(vid)
            vid += 1
    assert vid == base.FLEET_SIZE

    miles, minutes = selective.make_move_candidates(ids, coords)
    origin = trips['origin_i'].to_numpy(dtype=np.int32)
    destination = trips['destination_i'].to_numpy(dtype=np.int32)
    pickup_ns = trips['pickup_time'].astype('int64').to_numpy()
    dropoff_ns = trips['dropoff_time'].astype('int64').to_numpy()
    hour_ns = hours.astype('int64').to_numpy()
    minute_ns = 60 * 1_000_000_000
    events = []  # (ready_time, vehicle_id, destination_zone)
    cursor = total_served = total_moves = 0
    empty_miles = ride_minutes = move_minutes = 0.0
    rows = []

    def finish_events(at_ns):
        while events and events[0][0] <= at_ns:
            _ready, vehicle_id, zone = heappop(events)
            idle[zone].append(vehicle_id)

    for h, hour_start in enumerate(hour_ns):
        start_ns = int(hour_start)
        finish_events(start_ns)
        decisions = choose_arrival_aware_moves(
            idle, events, forecast[h], miles, minutes, start_ns, minute_ns
        )
        for source, dest, distance, duration in decisions:
            vehicle_id = idle[source].pop()
            ready = start_ns + math.ceil(duration * minute_ns)
            heappush(events, (ready, vehicle_id, dest))
            empty_miles += distance
            move_minutes += duration
        total_moves += len(decisions)

        requests = served = 0
        hour_end = start_ns + 3600 * 1_000_000_000
        while cursor < len(trips) and pickup_ns[cursor] < hour_end:
            at_ns = int(pickup_ns[cursor])
            finish_events(at_ns)
            z = int(origin[cursor])
            requests += 1
            if idle[z]:
                vehicle_id = idle[z].pop()
                end_ns = int(dropoff_ns[cursor])
                heappush(events, (end_ns, vehicle_id, int(destination[cursor])))
                ride_minutes += (end_ns - at_ns) / minute_ns
                served += 1
            cursor += 1

        total_served += served
        rows.append({
            'pickup_hour': hours[h], 'policy': 'arrival_aware_repositioning',
            'requests': requests, 'served': served, 'unserved': requests - served,
            'service_rate_pct': 100 * served / requests if requests else 100.0,
            'repositions': len(decisions),
        })
        assert sum(map(len, idle)) + len(events) == base.FLEET_SIZE, \
            'Vehicle conservation failed'

    assert cursor == len(trips), 'Not all replay requests were processed'
    capacity_minutes = base.FLEET_SIZE * len(hours) * 60
    summary = {
        'policy': 'arrival_aware_repositioning', 'hours': len(hours),
        'fleet_size': base.FLEET_SIZE, 'requests': len(trips),
        'served': total_served, 'unserved': len(trips) - total_served,
        'service_rate_pct': round(100 * total_served / len(trips), 3),
        'repositions': total_moves,
        'estimated_empty_reposition_miles': round(empty_miles, 2),
        'trip_utilization_pct': round(100 * ride_minutes / capacity_minutes, 3),
        'reposition_utilization_pct': round(100 * move_minutes / capacity_minutes, 3),
    }
    return pd.DataFrame(rows), summary


def main():
    print('=== ARRIVAL-AWARE NYC FLEET DECISION EXPERIMENT ===')
    trips, hours, ids, coords, forecast, historical = base.load_inputs()
    print(f'{len(trips):,} recorded requests | {len(ids)} zones | '
          f'{len(hours)} hours | {base.FLEET_SIZE:,} simulated vehicles')
    hourly_results, summaries = [], []
    for name in ('no_repositioning', 'forecast_repositioning'):
        rows, summary = base.simulate(name, trips, hours, ids, coords,
                                      forecast, historical)
        hourly_results.append(rows)
        summaries.append(summary)
    rows, summary = selective.simulate_selective(
        trips, hours, ids, coords, forecast, historical
    )
    hourly_results.append(rows)
    summaries.append(summary)
    rows, summary = simulate_arrival_aware(
        trips, hours, ids, coords, forecast, historical
    )
    hourly_results.append(rows)
    summaries.append(summary)

    result = pd.DataFrame(summaries)
    for r in summaries:
        print(f"{r['policy']}: served={r['served']:,}, "
              f"service={r['service_rate_pct']:.2f}%, "
              f"moves={r['repositions']:,}, "
              f"estimated empty miles={r['estimated_empty_reposition_miles']:,.1f}")
    baseline = summaries[0]['served']
    print(f'Arrival-aware minus no repositioning: {summary["served"] - baseline:+,} rides')
    assert result['requests'].nunique() == 1, 'Policies used different demand totals'
    assert result['fleet_size'].nunique() == 1, 'Policies used different fleet sizes'
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.concat(hourly_results, ignore_index=True).to_csv(
        OUTPUT / 'arrival_aware_fleet_hourly.csv', index=False
    )
    result.to_csv(OUTPUT / 'arrival_aware_fleet_summary.csv', index=False)
    print('Saved outputs/simulation/arrival_aware_fleet_hourly.csv')
    print('Saved outputs/simulation/arrival_aware_fleet_summary.csv')
    print('NOTE: Development-week comparison only; results are NOT independent validation.')
    print('LIMITATIONS: no rider queues, charging, maintenance, risk, or true road routing.')


if __name__ == '__main__':
    main()
