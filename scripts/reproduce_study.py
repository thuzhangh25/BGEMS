'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Reproduce fixed-configuration, catalogue and indoor comparisons using
production APIs. Each experiment writes to a new directory under the output root.
'''
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import numpy as np

from src.configuration import prepare_mission, search_configuration
from src.inputs import load_system_config
from src.run import execute, public_candidate, write_json
from src.weather import read_scenario_weather
from scripts.evaluate_applications import building_state


def config(number):
    return load_system_config('subject.json', 'scenario_profiles.json', 'components.json', number)


def catalogue(root):
    base = config(2)
    weather = read_scenario_weather(base.scenario, base.settings['dt_hours'])
    folder = root / 'experiments/catalogue_scaling'
    folder.mkdir(parents=True, exist_ok=False)
    results = []
    # Synthetic variants change insulation, mass and cost, never body topology.
    for count in (4, 8, 16):
        trial = deepcopy(base)
        for variant in range(1, count // 4):
            for original in base.garments:
                trial.garments.append(replace(
                    original, id=4 * variant + original.id,
                    name=f'{original.name}_synthetic_{variant}',
                    clothing_resistance=original.clothing_resistance * (1 + .1 * variant),
                    mass_kg=original.mass_kg * (1 + .06 * variant),
                    cost=original.cost * (1 + .12 * variant)))
        write_json(folder / f'catalogue_{count}.json', asdict(trial))
        started = time.perf_counter()
        found = search_configuration(trial, weather)
        elapsed = time.perf_counter() - started
        write_json(folder / f'candidates_{count}.json', [public_candidate(c) for c in found['records']])
        results.append({'garments': count, 'candidate_count': len(found['records']),
            'admissible_count': len(found['admissible']), 'pareto_count': len(found['pareto']),
            'selected': public_candidate(found['selected']) if found['selected'] else None,
            'status': found['status'], 'elapsed_seconds': elapsed,
            'optimizer_calls': found['optimizer_calls'], 'timing': found['timing']})
    write_json(folder / 'summary.json', results)
    return {'catalogues': len(results), 'candidate_counts': [r['candidate_count'] for r in results]}


def indoor(root, fixed_fleece):
    trial = config(3)
    weather = read_scenario_weather(trial.scenario, trial.settings['dt_hours'])
    hardware = (8, 0, 1) if fixed_fleece else (2, 0, 2)
    met = prepare_mission(trial, weather)[hardware[2]]['planned_met']
    parent = root / 'experiments' / ('fixed_fleece/indoor' if fixed_fleece else 'indoor')
    parent.mkdir(parents=True, exist_ok=False)
    assumptions = json.loads(Path('data/systemconf/applications.json').read_text())
    rows, passive = [], []
    for state in (25, 20, 18, 15):
        folder = parent / f'setpoint_{state}'
        execute(trial, 'fixed', folder, print, configuration=hardware,
                forcing={'air_c': np.asarray(weather['air_c']) + state - 15,
                         'reference_pv_w': np.zeros(len(met)), 'met': met},
                experiment={'kind': 'fixed_fleece_indoor_setpoint' if fixed_fleece else 'indoor_setpoint',
                            'setpoint_c': state,
                            'waveform': 'Specified synthetic 15C profile shifted by setpoint minus 15C',
                            'building_outdoor': 'Separate adopted -20C design boundary'})
        for controller in ('rbc', 'pid', 'tacs', 'rhc'):
            path = folder / controller / 'run.json'
            record = json.loads(path.read_text())
            result = building_state(record, assumptions)
            rows.append({'controller': controller,
                         'source': str(path.relative_to(parent.parent)), **result})
        if fixed_fleece:
            demand = np.asarray(record['actual']['demand_w'])
            dt = record['settings']['dt_hours']
            duration = len(demand) * dt
            positive = demand > 0
            # Zero-demand intervals score one; positive-demand intervals score zero.
            base_grid = record['hardware']['base_power_w'] * duration / 1000 / assumptions['charging_efficiency']
            people = assumptions['building']['occupants']
            passive.append({'setpoint_c': state, 'mean_service': float(np.mean(~positive)),
                'unmet_heat_wh': float(demand.sum() * dt),
                'positive_demand_duration_h': float(positive.sum() * dt),
                'mean_demand_w': float(demand.mean()), 'peak_demand_w': float(demand.max()),
                'electronics_on_grid_kwh_per_person': base_grid,
                'electronics_off_grid_kwh_per_person': 0.,
                'building_total_electronics_on_kwh': result['hvac_kwh'] + people * base_grid,
                'building_total_electronics_off_kwh': result['hvac_kwh'],
                'reference_kwh': result['reference_kwh'],
                'interpretation': 'Same-garment zero-PWM reference calculated algebraically.'})
    if fixed_fleece:
        target = root / 'evaluation/fixed_fleece_applications.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(target)
        write_json(target, {'schema': 'bgems-fixed-garment-v1', 'configuration': list(hardware),
                           'assumptions': assumptions, 'building_states': rows, 'passive': passive})
    return {'runs': len(rows), 'completed': sum(row['mission_metrics']['completed_duration_h'] ==
            row['mission_metrics']['scheduled_duration_h'] for row in rows)}


def static(root):
    folder = root / 'experiments/static'
    folder.mkdir(parents=True, exist_ok=False)
    results = []
    for batteries in (4, 7):
        for number in (1, 2, 3):
            results.append(execute(config(number), 'fixed', folder / f'y{batteries}_s{number}',
                print, controllers=('rhc',), configuration=(batteries, 0, 0),
                experiment={'kind': 'fixed_wearable_tuple_reference',
                            'scope': 'Battery/PV/garment fixed; equipment installation follows scenario.'}))
    # Preserve the original tuple and failed prefix for the initial-reserve rejection.
    return {'cases': len(results), 'interpretation': 'Inspect per-case completion and failure kinds.'}


def equipment(root):
    """Exercise the installed equipment heater across its switching threshold."""
    trial = deepcopy(config(2))
    trial.scenario.set_phases([('Equipment-heater activation', 2.0, 1.0)])
    dt = trial.settings['dt_hours']
    times = np.arange(round(2.0 / dt)) * dt
    air = np.select([times < 0.5, times < 1.0, times < 1.5],
                    [-18.0, -20.0, -22.0], default=-18.0)
    folder = root / 'experiments/equipment_activation'
    execute(trial, 'fixed', folder, print, configuration=(7, 1, 0),
            forcing={'air_c': air, 'reference_pv_w': np.zeros(len(times)),
                     'met': np.ones(len(times))},
            experiment={'kind': 'equipment_activation',
                        'purpose': 'Threshold switching and electrical-load response',
                        'temperature_blocks_c': [-18, -20, -22, -18],
                        'block_duration_h': 0.5})
    return {'runs': 4, 'record_directory': str(folder)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True, choices=['static', 'catalogue-scaling', 'indoor', 'fixed-fleece', 'equipment-activation'])
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.experiment == 'catalogue-scaling':
        result = catalogue(args.output_root)
    elif args.experiment == 'static':
        result = static(args.output_root)
    elif args.experiment == 'equipment-activation':
        result = equipment(args.output_root)
    else:
        result = indoor(args.output_root, args.experiment == 'fixed-fleece')
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
