# Autonomous Fleet Analytics — January–February Validation

## Evaluation periods

- Development evaluation: January 25–31, 2025
- Out-of-time evaluation: February 1–7, 2025
- Fleet: 1,000 simulated vehicles
- Geography: NYC taxi zones
- Source: NYC Yellow Taxi trip records

## Fleet simulation results

| Metric | January | February |
|---|---:|---:|
| Requests | 796,566 | 847,348 |
| Baseline rides served | 230,781 | 257,515 |
| Forecast-driven rides served | 368,431 | 379,700 |
| Additional rides served | +137,650 | +122,185 |
| Baseline service rate (%) | 28.97 | 30.39 |
| Forecast-driven service rate (%) | 46.25 | 44.81 |
| Additional estimated empty miles | +252,379 | +217,782 |
| Extra empty miles per extra ride | 1.833 | 1.782 |

## Forecasting evaluation

- January: LightGBM MAE improved approximately 19.9% over the historical baseline.
- February: historical baseline MAE 3.860; January-trained LightGBM MAE 3.868.
- February LightGBM relative MAE improvement: approximately -0.20%.
- February evaluation used observed demand from completed prior hours, not a full-week forecast made in advance.

## Interpretation

Forecast-driven repositioning served more requests than no repositioning
in both simulated weeks, but required substantially more estimated
empty driving. Forecast accuracy did not consistently improve against
the historical baseline on the later week.

## Limitations

- These are simulated outcomes, not actual robotaxi performance.
- Human-driven taxi trips approximate potential robotaxi demand.
- Recorded pickup timestamps are treated as request timestamps.
- Travel distance and pickup times are approximated using taxi-zone geography.
- No charging, maintenance, rider cancellation, or full road routing is modeled.
- Results have not been independently validated against real AV operations.
- January was used during development; February is a later out-of-time check.
- Two adjacent evaluation weeks do not establish broad generalization.

## Business implication

Repositioning should be evaluated as a tradeoff between additional
simulated completed rides and incremental estimated empty-driving costs.
Financial conclusions require explicit assumptions and sensitivity analysis.
