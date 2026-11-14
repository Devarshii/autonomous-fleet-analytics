"""Read-only quality audit for the NYC autonomous fleet forecasting experiment.

Run from the project root:
    python src/fleet_validation_audit.py

Creates outputs/validation/forecast_audit.json and forecast_audit.md.
Does not modify training data, forecasts, or existing dashboard outputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / 'data/processed/lightgbm_predictions.parquet'
OUTPUT = ROOT / 'outputs/validation'
REQUIRED = ('pickup_hour', 'LocationID', 'demand', 'lag_168', 'lightgbm_forecast')
TEST_START = pd.Timestamp('2025-01-25')


def error_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    absolute = np.abs(actual - prediction)
    return {
        'MAE': float(absolute.mean()),
        'WAPE_pct': float(100.0 * absolute.sum() / actual.sum()) if actual.sum() > 0 else None,
        'total_predicted': float(prediction.sum()),
    }


def main() -> None:
    if not INPUT.is_file():
        raise SystemExit(f'Missing predictions: {INPUT}\nRun the existing forecast pipeline first.')
    df = pd.read_parquet(INPUT)
    missing = set(REQUIRED).difference(df.columns)
    if missing:
        raise SystemExit(f'Missing required columns: {sorted(missing)}')

    df['pickup_hour'] = pd.to_datetime(df['pickup_hour'], errors='coerce')
    values = df[['demand', 'lag_168', 'lightgbm_forecast']].apply(pd.to_numeric, errors='coerce')
    faults = {
        'bad_timestamps': int(df['pickup_hour'].isna().sum()),
        'duplicate_zone_hours': int(df.duplicated(['pickup_hour', 'LocationID']).sum()),
        'null_or_nonfinite_numeric_rows': int((~np.isfinite(values.to_numpy(dtype=float))).any(axis=1).sum()),
        'negative_actual_demand': int((values['demand'] < 0).sum()),
        'negative_baseline_forecast': int((values['lag_168'] < 0).sum()),
        'negative_model_forecast': int((values['lightgbm_forecast'] < 0).sum()),
        'test_rows_before_2025_01_25': int((df['pickup_hour'] < TEST_START).sum()),
    }
    valid = df['pickup_hour'].notna() & np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
    for col in values.columns:
        valid &= values[col] >= 0
    clean = df.loc[valid].copy()
    vals = values.loc[valid]
    if clean.empty:
        raise SystemExit('No valid observations available for audit.')

    y = vals['demand'].to_numpy(dtype=float)
    historical = vals['lag_168'].to_numpy(dtype=float)
    predicted = vals['lightgbm_forecast'].to_numpy(dtype=float)
    baseline = error_metrics(y, historical)
    model = error_metrics(y, predicted)
    pct_improvement = 100 * (baseline['MAE'] - model['MAE']) / baseline['MAE'] if baseline['MAE'] else None
    per_day = []
    for day, rows in clean.groupby(clean['pickup_hour'].dt.date, sort=True):
        z = vals.loc[rows.index]
        actual = z['demand'].to_numpy(dtype=float)
        b = error_metrics(actual, z['lag_168'].to_numpy(dtype=float))
        m = error_metrics(actual, z['lightgbm_forecast'].to_numpy(dtype=float))
        per_day.append({'day': str(day), 'zone_hours': len(rows), 'baseline_MAE': b['MAE'], 'model_MAE': m['MAE'], 'model_wins': m['MAE'] < b['MAE']})

    warnings = []
    if any(faults.values()):
        warnings.append('Data quality flags are present. See faults for details.')
    if len(per_day) < 7:
        warnings.append('Fewer than seven test dates present; sample may be incomplete.')
    warnings.append('IMPORTANT: This script cannot verify that lag and rolling features were created without look-ahead leakage. Review feature engineering before claiming forecast validity.')
    warnings.append('IMPORTANT: This is an hourly zone-level forecast audit, not validation of vehicle travel, wait time, or real robotaxi outcomes.')

    report = {
        'source': str(INPUT.relative_to(ROOT)),
        'rows_total': len(df),
        'rows_evaluated': len(clean),
        'unique_zones': int(clean['LocationID'].nunique()),
        'unique_hours': int(clean['pickup_hour'].nunique()),
        'test_start': str(clean['pickup_hour'].min()),
        'test_end': str(clean['pickup_hour'].max()),
        'total_observed_rides': float(y.sum()),
        'baseline_same_hour_last_week': baseline,
        'lightgbm': model,
        'MAE_improvement_pct': pct_improvement,
        'days_model_wins': sum(row['model_wins'] for row in per_day),
        'days_tested': len(per_day),
        'data_quality_faults': faults,
        'per_day': per_day,
        'caveats': warnings,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'forecast_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    def fmt(value):
        return 'N/A' if value is None else f'{value:,.2f}'
    lines = [
        '# Forecast Validation Audit', '',
        f"- Rows evaluated: **{len(clean):,}** of {len(df):,}",
        f"- Test window: **{report['test_start']}** to **{report['test_end']}**",
        f"- Unique zones: **{report['unique_zones']:,}**; unique hours: **{report['unique_hours']:,}**", '',
        '| Metric | Same-hour-last-week baseline | LightGBM |',
        '|---|---:|---:|',
        f"| MAE | {fmt(baseline['MAE'])} | {fmt(model['MAE'])} |",
        f"| WAPE (%) | {fmt(baseline['WAPE_pct'])} | {fmt(model['WAPE_pct'])} |",
        '', f"**MAE improvement:** {fmt(pct_improvement)}%", 
        f"**Daily wins:** {report['days_model_wins']}/{len(per_day)} days", '',
        '## Daily comparison', '', '| Day | Zone-hours | Baseline MAE | Model MAE | Model wins? |', '|---|---:|---:|---:|---|',
    ]
    for row in per_day:
        lines.append(f"| {row['day']} | {row['zone_hours']:,} | {row['baseline_MAE']:.2f} | {row['model_MAE']:.2f} | {'Yes' if row['model_wins'] else 'No'} |")
    lines.extend(['', '## Data quality flags', ''])
    for key, number in faults.items():
        lines.append(f'- {key}: **{number}**')
    lines.extend(['', '## Interpretation limits', ''])
    for item in warnings:
        lines.append(f'- {item}')
    (OUTPUT / 'forecast_audit.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print('\n=== FORECAST VALIDATION AUDIT ===')
    print(f"Rows: {len(clean):,} | Zones: {report['unique_zones']:,} | Hours: {report['unique_hours']:,}")
    print(f"Baseline MAE: {fmt(baseline['MAE'])} | LightGBM MAE: {fmt(model['MAE'])}")
    print(f"Improvement: {fmt(pct_improvement)}% | Daily wins: {report['days_model_wins']}/{len(per_day)}")
    print(f"Quality flags: {sum(faults.values())}")
    print(f"Reports saved to: {OUTPUT.relative_to(ROOT)}")
    if sum(faults.values()):
        print('WARNING: Review data-quality flags before publishing performance claims.')


if __name__ == '__main__':
    main()
