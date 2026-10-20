'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Same-hardware four-controller comparison and paired Monte Carlo simulation.
Generate perturbed inputs in memory with four independent RNG streams shared by
all controllers. Select hardware once before sampling.
'''

import argparse
from copy import deepcopy
from pathlib import Path
import time

import numpy as np

from src.configuration import prepare_mission, search_configuration
from src.inputs import load_system_config
from src.operation import CONTROLLERS
from src.run import execute, execution_plan, public_candidate, write_json
from src.weather import read_scenario_weather


def sample_forcing(weather, planned_met, parameters, seeds):
    """Generate temperature random walk, MET noise, beta PV and cold-shock events.

    Cold-shock events are Bernoulli draws per simulation step.
    Each driver uses its own RNG stream; temperature is clipped after walk and shocks.
    """
    if len(seeds) != 4:
        raise ValueError('Temperature, activity, PV and event seeds are required.')
    count = len(planned_met)
    rng = [np.random.default_rng(int(seed)) for seed in seeds]
    lo, hi = parameters['temperature_bounds_c']
    walk = np.cumsum(rng[0].normal(0.0, parameters['temp_sigma_c'], count))
    temperature = np.clip(np.asarray(weather['air_c']) + walk, lo, hi)
    met = np.clip(
        np.asarray(planned_met) + rng[1].normal(0.0, parameters['met_sigma'], count),
        *parameters['met_bounds'],
    )
    pv_scale = rng[2].beta(parameters['pv_alpha'], parameters['pv_beta'], count)
    events = rng[3].random(count) < parameters['event_probability_per_step']
    temperature = np.clip(
        temperature + np.cumsum(events) * parameters['event_drop_c'], lo, hi
    )
    return {
        'air_c': temperature,
        'met': met,
        'reference_pv_w': np.asarray(weather['reference_pv_w']) * pv_scale,
        'seeds': [int(s) for s in seeds],
        'event_steps': np.flatnonzero(events).tolist(),
    }


def run_monte_carlo(config, output, samples, *, configuration=None, progress=print):
    if config.scenario.scenario_type != 2:
        raise ValueError('The specified Monte Carlo mission is Muztagh Ata.')
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError('Sample count must be a positive integer.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    weather = read_scenario_weather(config.scenario, config.settings['dt_hours'])
    selection = None
    if configuration is None:
        search = search_configuration(config, weather)
        selection = search['selected']
        if selection is None:
            raise ValueError(
                'Unperturbed search found no design; explicitly register comparison hardware.'
            )
        configuration = selection['configuration']
    plan = execution_plan([config], mc_samples=samples, configuration=configuration)
    plan['unperturbed_controller_runs_at_most'] = 0
    plan['purpose'] = (
        'performance_pilot'
        if samples < config.monte_carlo['samples']
        else 'specified_monte_carlo'
    )
    if progress:
        progress(str(plan))
    write_json(output / 'plan.json', plan)
    trial_config = deepcopy(config)
    protocol = config.monte_carlo
    trial_config.settings.update(
        dt_hours=protocol['step_h'],
        horizon_steps=protocol['horizon_steps'],
        lambda_deg=protocol['lambda_deg'],
        lambda_soc=protocol['lambda_soc'],
    )
    base_weather = read_scenario_weather(
        trial_config.scenario, trial_config.settings['dt_hours']
    )
    prepared = prepare_mission(trial_config, base_weather)
    met = prepared[configuration[2]]['planned_met']
    master = np.random.default_rng(protocol['seed'])
    summaries = []
    for index in range(samples):
        seeds = master.integers(0, 2**31 - 1, size=4)
        forcing = sample_forcing(base_weather, met, protocol, seeds)
        metadata = {
            'kind': 'monte_carlo',
            'sample_id': index,
            'master_seed': protocol['seed'],
            'seeds': forcing['seeds'],
            'event_steps': forcing['event_steps'],
            'protocol': protocol,
            'purpose': plan['purpose'],
        }
        summary = execute(
            trial_config,
            'fixed',
            output / f'sample_{index:04d}',
            progress,
            configuration=configuration,
            forcing=forcing,
            experiment=metadata,
        )
        summaries.append(summary)
    result = {
        'schema': 'bgems-monte-carlo-v1',
        'samples': samples,
        'scheduled_controller_runs': samples * 4,
        'configuration': configuration,
        'unperturbed_selection': public_candidate(selection) if selection else None,
        'purpose': plan['purpose'],
        'protocol': protocol,
        'summaries': summaries,
        'optimizer_calls': sum(s['optimizer_calls'] for s in summaries),
        'total_seconds': time.perf_counter() - started,
    }
    write_json(output / 'monte_carlo.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenario', type=int, choices=(1, 2, 3), default=2)
    parser.add_argument(
        '--samples',
        type=int,
        default=0,
        help='0: unperturbed comparison; positive: paired MC.',
    )
    parser.add_argument('--configuration', type=int, nargs=3, metavar=('Y', 'H', 'G'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args(argv)
    config = load_system_config(
        'subject.json', 'scenario_profiles.json', 'components.json', args.scenario
    )
    if args.plan_only:
        print(
            execution_plan(
                [config], mc_samples=args.samples, configuration=args.configuration
            )
        )
        return 0
    if args.samples:
        run_monte_carlo(
            config, args.output_dir, args.samples, configuration=args.configuration
        )
    else:
        execute(
            config,
            'fixed' if args.configuration else 'search',
            args.output_dir,
            print,
            configuration=args.configuration,
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
