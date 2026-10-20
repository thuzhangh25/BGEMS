'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Specified single-factor perturbations on one specified-date design.
Controllers share hardware and realized forcing. No input CSV is modified and
no hardware search occurs inside the perturbation loop.
'''

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

from src.configuration import prepare_mission, search_configuration
from src.inputs import load_system_config
from src.run import execute, write_json
from src.weather import read_scenario_weather


def run_stress(config, output, configuration=None):
    weather = read_scenario_weather(config.scenario, config.settings['dt_hours'])
    if configuration is None:
        selection = search_configuration(config, weather)['selected']
        if selection is None:
            raise ValueError(
                'No unperturbed design; an explicit common comparison tuple is required.'
            )
        configuration = selection['configuration']
    met = prepare_mission(config, weather)[configuration[2]]['planned_met']
    arms = [('baseline', 0.0, 1.0, 1.0)]
    arms += [
        (f'temperature_{delta:g}', delta, 1.0, 1.0) for delta in (-2.0, -5.0, -10.0)
    ]
    arms += [(f'eta_{factor:g}', 0.0, factor, 1.0) for factor in (0.75, 0.5)]
    if configuration[1] > 0:
        arms += [(f'pv_{factor:g}', 0.0, 1.0, factor) for factor in (0.75, 0.5, 0.0)]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / 'plan.json',
        {
            'unperturbed_controller_runs_at_most': len(arms) * 4,
            'mc_controller_runs': 0,
            'configuration': configuration,
            'arms': arms,
            'candidate_controller_runs': 0,
        },
    )
    summaries = []
    for name, delta, eta, pv in arms:
        trial = deepcopy(config)
        trial.device.efficiencies = [
            value * eta for value in config.device.efficiencies
        ]
        forcing = {
            'air_c': np.asarray(weather['air_c']) + delta,
            'met': met,
            'reference_pv_w': np.asarray(weather['reference_pv_w']) * pv,
        }
        metadata = {
            'kind': 'single_factor',
            'arm': name,
            'temperature_delta_c': delta,
            'efficiency_scale': eta,
            'pv_scale': pv,
        }
        summaries.append(
            execute(
                trial,
                'fixed',
                output / name,
                print,
                configuration=configuration,
                forcing=forcing,
                experiment=metadata,
            )
        )
    write_json(output / 'stress.json', summaries)
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--configuration', type=int, nargs=3)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_system_config(
        'subject.json', 'scenario_profiles.json', 'components.json', args.scenario
    )
    run_stress(config, args.output_dir, args.configuration)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
