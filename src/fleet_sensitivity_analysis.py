"""Sensitivity audit for the illustrative autonomous fleet simulator.

Run from repository root:
    python src/fleet_sensitivity_analysis.py

Requires src/advanced_fleet_simulator.py and existing forecast parquet.
Does not overwrite current dashboard or advanced simulator results.
All experiments use identical held-out demand and random seed.

WARNING: Demonstrates sensitivity to synthetic trip/reposition timings and
same-zone returns, not measured AV ride waits or deadhead miles.
"""

from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

# Direct invocation from repository root: Python automatically places src/ on sys.path.
import advanced_fleet_simulator as sim

OUTPUT_DIR = Path("outputs/validation")
OUTPUT_CSV = OUTPUT_DIR / "fleet_sensitivity_results.csv"
SUMMARY_TXT = OUTPUT_DIR / "fleet_sensitivity_report.txt"

# Change one parameter from the original simulator at a time to isolate effects.
# All time durations are multiples of a 15-minute slot.
SCENARIOS = [
    ("original", 1000, 2, 2, 100),
    ("smaller_fleet", 750, 2, 2, 100),
    ("larger_fleet", 1500, 2, 2, 100),
    ("shorter_trips_15m", 1000, 1, 2, 100),
    ("longer_trips_45m", 1000, 3, 2, 100),
    ("longer_trips_60m", 1000, 4, 2, 100),
    ("moves_disabled", 1000, 2, 2, 0),
    ("moves_capped_25", 1000, 2, 2, 25),
    ("moves_capped_200", 1000, 2, 2, 200),
    ("faster_reposition_15m", 1000, 2, 1, 100),
    ("slower_reposition_60m", 1000, 2, 4, 100),
]


def main():
    original = {
        "FLEET_SIZE": sim.FLEET_SIZE,
        "TRIP_SLOTS": sim.TRIP_SLOTS,
        "REPOSITION_SLOTS": sim.REPOSITION_SLOTS,
        "MAX_MOVES_PER_HOUR": sim.MAX_MOVES_PER_HOUR,
    }
    print("=== FLEET SIMULATOR SENSITIVITY AUDIT ===", flush=True)
    hours, zones, demand, forecast, historical = sim.prepare_data()
    print(f"Demand: {int(demand.sum()):,} requests; {len(hours)} hours; {len(zones)} zones", flush=True)
    results = []
    try:
        for i, (label, fleet, trip_slots, reposition_slots, move_cap) in enumerate(SCENARIOS, 1):
            sim.FLEET_SIZE = fleet
            sim.TRIP_SLOTS = trip_slots
            sim.REPOSITION_SLOTS = reposition_slots
            sim.MAX_MOVES_PER_HOUR = move_cap
            start = time.monotonic()
            _, base = sim.simulate("no_repositioning", hours, zones, demand, forecast, historical)
            _, strategy = sim.simulate("forecast_repositioning", hours, zones, demand, forecast, historical)
            assert base["requests"] == strategy["requests"] == int(demand.sum())
            assert base["served"] <= base["requests"]
            assert strategy["served"] <= strategy["requests"]
            if move_cap == 0:
                assert strategy["repositions"] == 0
                assert strategy["served"] == base["served"], "Zero-move control must match baseline"
            delta = int(strategy["served"] - base["served"])
            row = {
                "scenario": label,
                "fleet_size": fleet,
                "trip_minutes": 15 * trip_slots,
                "reposition_minutes": 15 * reposition_slots,
                "max_moves_per_hour": move_cap,
                "requests": base["requests"],
                "baseline_served": base["served"],
                "forecast_served": strategy["served"],
                "additional_served": delta,
                "baseline_service_rate_pct": base["service_rate_pct"],
                "forecast_service_rate_pct": strategy["service_rate_pct"],
                "service_rate_change_pp": round(strategy["service_rate_pct"] - base["service_rate_pct"], 3),
                "reposition_moves": strategy["repositions"],
                "seconds_elapsed": round(time.monotonic() - start, 2),
            }
            results.append(row)
            print(
                f"[{i:02d}/{len(SCENARIOS)}] {label:<25} "
                f"baseline={base['served']:>7,} forecast={strategy['served']:>7,} "
                f"delta={delta:>+8,}  ({row['seconds_elapsed']} s)",
                flush=True,
            )
    finally:
        for name, value in original.items():
            setattr(sim, name, value)

    df = pd.DataFrame(results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    nontrivial = df.loc[df["max_moves_per_hour"] > 0]
    wins = int((nontrivial["additional_served"] > 0).sum())
    ties = int((nontrivial["additional_served"] == 0).sum())
    losses = int((nontrivial["additional_served"] < 0).sum())
    report = (
        "FLEET SIMULATOR — ASSUMPTION SENSITIVITY\n"
        f"Input hours: {len(hours)}; taxi zones: {len(zones)}; requests: {int(demand.sum()):,}\n"
        f"Scenarios: {len(df)} (one parameter changed at a time)\n"
        f"Excluding zero-move control: {wins} positive, {ties} equal, {losses} negative outcomes\n"
        f"Original-case additional served: {int(df.iloc[0]['additional_served']):+,}\n"
        f"Smallest non-control change: {int(nontrivial['additional_served'].min()):+,}\n"
        f"Largest non-control change: {int(nontrivial['additional_served'].max()):+,}\n"
        "These are simulated requests served under simplified, non-geographic assumptions.\n"
        "NOT measured: rider waiting time, actual distance, charging, cross-zone trip destinations.\n"
        "Results are not independent validation of the simulator; only a sensitivity check.\n"
        "Random seed and test dataset fixed across scenarios.\n"
    )
    SUMMARY_TXT.write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Saved: {OUTPUT_CSV}\nSaved: {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
