'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Fixed-package stress and eta-sensitivity runner (frozen design,
2026-10-06). Runs a declared perturbation grid on median-representative baseline
tasks with the selected hardware held fixed, without touching src/,
data/systemconf/, or the baseline weather archives. For each stress point the
declared baseline date directory is copied into an explicitly supplied new
output directory; perturbed weather CSV copies are written alongside, and the
frozen parameters.json is patched (temperature_files / pv_files paths,
fixed_configuration, eta_efficiency_scale) so the unmodified src.run pipeline
re-solves the controller and independently audits every point.
Grid (single-factor, frozen by author decision):
  k=1:   dT in {0, -2, -5, -10} degC
  dT=0:  k  in {1, 0.75, 0.5, 0}   (Harbin has no PV; equivalent k points skipped)
Eta variants (paired replay, original weather, same hardware):
  s in {1, 0.75, 0.5} scaling eta_i, plus uniform eta_i = 0.5.
Usage: python3 -m scripts.stress_sensitivity --output-dir <new-directory>
'''
import argparse
import csv
import json
import math
import shutil
from pathlib import Path

import pandas as pd

from src.run import execute

LEDGER = Path('output/frozen_20261004')
SCEN = ((1, 'Harbin'), (2, 'Muztagh Ata'), (3, 'Qinling'))
DT_LEVELS = (0.0, -2.0, -5.0, -10.0)
K_LEVELS = (1.0, 0.75, 0.5, 0.0)
ETA_VARIANTS = (
    ('scale1.00', 1.00),
    ('scale0.75', 0.75),
    ('scale0.50', 0.50),
    ('uniform0.50', 'uniform'),
)


def pick_baseline_dates(ledger_rows):
    """Median selected-load success date per scenario; sort key (load_wh, date)."""
    chosen = {}
    for scen, _name in SCEN:
        rows = [r for r in ledger_rows
                if int(r['scenario_type']) == scen and r['status'] == 'success']
        rows.sort(key=lambda r: (float(r['selected_load_wh']), r['date']))
        pick = rows[(len(rows) - 1) // 2]
        chosen[scen] = {
            'date': pick['date'],
            'tuple': [int(pick['selected_y']), int(pick['selected_h']),
                      int(pick['selected_g'])],
            'load_wh': float(pick['selected_load_wh']),
            'rank': (len(rows) - 1) // 2,
            'n_success': len(rows),
        }
    return chosen


def _patch_temperature_csv(source, destination, delta_c):
    frame = pd.read_csv(source)
    frame['air_temperature_c'] = frame['air_temperature_c'] + delta_c
    frame.to_csv(destination, index=False)


def _patch_pv_csv(source, destination, factor):
    frame = pd.read_csv(source)
    frame['pv_power_w'] = frame['pv_power_w'] * factor
    frame.to_csv(destination, index=False)


def _prepare_run_dir(baseline_dir, run_dir, fixed_tuple, delta_c, k_factor,
                     eta_variant):
    """Copy the declared baseline directory and patch parameters + weather."""
    shutil.copytree(baseline_dir, run_dir)
    params_path = run_dir / 'parameters.json'
    params = json.loads(params_path.read_text(encoding='utf-8'))
    scenario = params['config']['scenario']
    params['config']['fixed_configuration'] = list(fixed_tuple)
    weather_dir = run_dir / 'weather_inputs'
    weather_dir.mkdir(exist_ok=True)

    if delta_c is not None:
        original = scenario['temperature_files']
        if delta_c != 0.0:
            target = weather_dir / f'{Path(original).stem}_dT{delta_c:+.1f}.csv'
            _patch_temperature_csv(original, target, delta_c)
            scenario['temperature_files'] = str(target)
        # delta_c == 0.0 keeps the original baseline file untouched.

    if k_factor is not None:
        original = scenario['pv_files']
        if k_factor != 1.0:
            target = weather_dir / f'{Path(original).stem}_k{k_factor:.2f}.csv'
            _patch_pv_csv(original, target, k_factor)
            scenario['pv_files'] = str(target)
        # k_factor == 1.0 keeps the original baseline file untouched.

    if eta_variant is not None:
        label, factor = eta_variant
        efficiencies = params['config']['device']['efficiencies']
        if factor == 'uniform':
            params['config']['device']['efficiencies'] = [0.5] * len(efficiencies)
        else:
            params['config']['device']['efficiencies'] = [
                value * factor for value in efficiencies]
        params['eta_efficiency_scale'] = label

    params['stress_parent'] = str(baseline_dir)
    params_path.write_text(json.dumps(params, indent=2) + '\n', encoding='utf-8')


def _first_fail_step(run_dir, tuple_):
    """Return (step_index, reason) of the first infeasible controller solve."""
    path = run_dir / 'runs' / (
        f'y{tuple_[0]}_h{tuple_[1]}_g{tuple_[2]}.json')
    artifact = json.loads(path.read_text(encoding='utf-8'))
    for index, solve in enumerate(artifact['trajectory']['solves']):
        if not solve['success']:
            return index, solve['reason']
    return None, None


def _point_record(scen, date, tuple_, kind, label, run_dir):
    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    with (run_dir / 'candidates.csv').open(newline='', encoding='utf-8') as stream:
        candidate = next(csv.DictReader(stream))
    artifact = json.loads((run_dir / candidate['artifact']).read_text(
        encoding='utf-8'))
    evaluation = artifact['evaluation']
    metrics = evaluation['metrics']
    fail_step, fail_reason = _first_fail_step(run_dir, tuple_)
    return {
        'scenario_type': scen,
        'date': date,
        'configuration': tuple_,
        'stress_kind': kind,
        'stress_label': label,
        'status': summary['status'],
        'completed': evaluation['completed'],
        'failure_kind': evaluation['failure_kind'],
        'components': evaluation['components'],
        'mean_thermal_score_observed': metrics.get('mean_thermal_score_observed'),
        'service_floor_shortfall_duration_hours': metrics.get(
            'service_floor_shortfall_duration_hours'),
        'whole_body_unmet_heat_wh': metrics.get('whole_body_unmet_heat_wh'),
        'minimum_soc': metrics.get('minimum_soc'),
        'terminal_soc': metrics.get('terminal_soc'),
        'observed_end_soc': metrics.get('observed_end_soc'),
        'observed_duration_hours': metrics.get('observed_duration_hours'),
        'low_soc_duration_hours': metrics.get('low_soc_duration_hours'),
        'electrical_load_energy_wh': metrics.get('electrical_load_energy_wh'),
        'pv_used_energy_wh': metrics.get('pv_used_energy_wh'),
        'pv_curtailed_energy_wh': metrics.get('pv_curtailed_energy_wh'),
        'equipment_on_duration_hours': metrics.get('equipment_on_duration_hours'),
        'first_infeasible_step': fail_step,
        'first_infeasible_reason': fail_reason,
        'operation_seconds': summary['timings']['operation_seconds'],
        'evaluation_seconds': summary['timings']['evaluation_seconds'],
        'run_dir': str(run_dir),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=LEDGER)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='A new directory; existing archives are never overwritten.')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('--output-dir must not already exist')
    project = Path(__file__).resolve().parents[1]
    if any(args.output_dir.resolve().is_relative_to(path) for path in
           (args.root.resolve(), project / 'src', project / 'data')):
        parser.error('--output-dir must be outside the baseline archive and scientific inputs')
    with (args.root / 'date_report' / 'date_ledger.csv').open(
            newline='', encoding='utf-8') as stream:
        ledger_rows = list(csv.DictReader(stream))
    baselines = pick_baseline_dates(ledger_rows)

    manifest = {
        'design_frozen': '2026-10-06',
        'baseline_rule': 'median selected load_wh, sort key (load_wh, date), '
                         'index floor((n-1)/2)',
        'temperature_levels_c': list(DT_LEVELS),
        'pv_factor_levels': list(K_LEVELS),
        'eta_variants': [label for label, _ in ETA_VARIANTS],
        'output_dir': str(args.output_dir),
        'baselines': {str(scen): value for scen, value in baselines.items()},
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print('[stress] baselines:', {s: b['date'] for s, b in baselines.items()},
          flush=True)

    records = []
    for scen, name in SCEN:
        baseline = baselines[scen]
        date = baseline['date']
        tuple_ = baseline['tuple']
        baseline_dir = args.root / f's{scen}_{date}'
        has_pv = tuple_[1] > 0

        points = [('env', f'dT{delta:+.1f}', delta, None)
                  for delta in DT_LEVELS]
        if has_pv:
            points += [('env', f'k{factor:.2f}', None, factor)
                       for factor in K_LEVELS if factor != 1.0]
        points += [('eta', label, None, None) for label, _ in ETA_VARIANTS]

        for kind, label, delta_c, k_factor in points:
            run_dir = args.output_dir / f's{scen}_{date}' / label
            eta_variant = (next(item for item in ETA_VARIANTS
                                if item[0] == label) if kind == 'eta' else None)
            _prepare_run_dir(baseline_dir, run_dir, tuple_, delta_c, k_factor,
                             eta_variant)
            summary = execute_from_params(run_dir)
            record = _point_record(scen, date, tuple_, kind, label, run_dir)
            records.append(record)
            print(f'[stress] s{scen} {date} {label}: {summary["status"]} '
                  f'C/E/S/R={record["components"]}', flush=True)

    ledger_path = args.output_dir / 'stress_ledger.json'
    ledger_path.write_text(json.dumps(records, indent=2) + '\n',
                           encoding='utf-8')
    print(f'[stress] wrote {len(records)} points to {ledger_path}', flush=True)


def execute_from_params(run_dir):
    """Replay one patched baseline directory through the unmodified pipeline."""
    from datetime import datetime

    from src.inputs import Device, Garment, Person, ScenarioProfile, SystemConfig

    params = json.loads((run_dir / 'parameters.json').read_text(
        encoding='utf-8'))
    raw = params['config']
    raw['scenario']['start'] = datetime.fromisoformat(raw['scenario']['start'])
    config = SystemConfig(
        scenario=ScenarioProfile(**raw['scenario']),
        person=Person(**raw['person']),
        device=Device(**raw['device']),
        garments=[Garment(**item) for item in raw['garments']],
        settings=raw['settings'],
        evaluation=raw['evaluation'],
        max_battery_cells=raw['max_battery_cells'],
        max_pv_units=raw['max_pv_units'],
        priority=raw['priority'],
        fixed_configuration=raw['fixed_configuration'],
        forecast_mode=raw['forecast_mode'],
        source_files=raw['source_files'],
        comparison=raw['comparison'],
    )
    return execute(config, 'fixed', run_dir)


if __name__ == '__main__':
    main()
