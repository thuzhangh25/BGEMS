'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Plan and execute specified finite missions offline. Search and
controller execution are separate stages; new outputs never overwrite inputs.
'''

import argparse
from dataclasses import asdict
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .configuration import build_candidate, prepare_mission, search_configuration
from .evaluation import evaluate_trajectory, summarize_runs
from .inputs import PROJECT_ROOT, load_system_config
from .modeling import demand_series
from .operation import CONTROLLERS, make_forecast, simulate
from .weather import read_scenario_weather


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str) and Path(value).is_absolute():
        return Path(os.path.relpath(value, PROJECT_ROOT)).as_posix()
    if isinstance(value, dict):
        return {_json_value(str(k)): _json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(
        json.dumps(_json_value(value), indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )


def _source_hashes(config, weather):
    paths = {Path(p).resolve() for p in config.source_files.values()}
    paths.update(Path(p).resolve() for p in weather['source_files'])
    paths.update((PROJECT_ROOT / 'src').glob('*.py'))
    result = {}
    for path in sorted(paths):
        key = Path(os.path.relpath(path, PROJECT_ROOT)).as_posix()
        result[key] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def public_candidate(candidate):
    return {
        k: v for k, v in candidate.items()
        if k not in ('hardware', 'actual', 'reference_load_w')
    }


def execution_plan(
    configs, controllers=CONTROLLERS, *, mc_samples=0, configuration=None
):
    if (
        not controllers
        or len(set(controllers)) != len(controllers)
        or any(c not in CONTROLLERS for c in controllers)
    ):
        raise ValueError('Controllers must be a nonempty unique list.')
    if (
        isinstance(mc_samples, bool)
        or not isinstance(mc_samples, int)
        or mc_samples < 0
    ):
        raise ValueError('MC sample count must be a nonnegative integer.')
    return {
        'missions': [
            {
                'scenario_type': c.scenario.scenario_type,
                'date': c.scenario.start.date().isoformat(),
                'duration_h': c.scenario.get_duration_hours(),
                'candidate_count': (
                    c.max_battery_units * (c.max_pv_units + 1) * len(c.garments)
                    if configuration is None
                    else 1
                ),
                'configuration': configuration,
                'controllers': list(controllers),
            }
            for c in configs
        ],
        'unperturbed_controller_runs_at_most': len(configs) * len(controllers),
        'mc_samples': mc_samples,
        'mc_controller_runs': mc_samples * len(controllers),
        'candidate_controller_runs': 0,
        'weather_acquisition': False,
    }


def execute(
    config,
    mode,
    output,
    progress=None,
    *,
    controllers=CONTROLLERS,
    configuration=None,
    forcing=None,
    experiment=None,
):
    """Search once or use an explicit fixed tuple, then run the requested policies."""
    if mode not in ('search', 'fixed') or (mode == 'fixed') != (
        configuration is not None
    ):
        raise ValueError(
            'Fixed mode requires a configuration; search mode must not supply one.'
        )
    if forcing is not None and mode != 'fixed':
        raise ValueError('Perturbation replay must hold hardware fixed.')
    plan = execution_plan([config], controllers, configuration=configuration)
    if progress:
        progress(json.dumps(plan))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    weather = read_scenario_weather(config.scenario, config.settings['dt_hours'])
    weather_seconds = time.perf_counter() - started
    provenance_started = time.perf_counter()
    sources = _source_hashes(config, weather)
    provenance_seconds = time.perf_counter() - provenance_started
    if forcing is not None:
        count = len(weather['air_c'])
        for key in ('air_c', 'reference_pv_w', 'met'):
            value = np.asarray(forcing[key], dtype=float)
            if value.shape != (count,) or not np.all(np.isfinite(value)):
                raise ValueError('Experiment forcing must align with the mission grid.')
        weather = dict(
            weather,
            air_c=np.asarray(forcing['air_c']),
            reference_pv_w=np.asarray(forcing['reference_pv_w']),
        )
    if mode == 'search':
        search = search_configuration(config, weather)
        selected = search['selected']
    else:
        tic = time.perf_counter()
        prepared = prepare_mission(config, weather)
        if forcing is not None:
            for garment in config.garments:
                prepared[garment.id]['met'] = np.asarray(forcing['met'])
                prepared[garment.id]['demand_w'] = demand_series(
                    config, garment, weather['air_c'], forcing['met']
                )
        selected = build_candidate(
            config, prepared[configuration[2]], weather, *configuration
        )
        search = {
            'records': [selected],
            'pareto': [],
            'selected': selected,
            'admissible': [selected] if selected['design_admissible'] else [],
            'status': 'fixed_configuration',
            'optimizer_calls': 0,
            'timing': {
                'demand_preparation_seconds': time.perf_counter() - tic,
                'configuration_seconds': 0.0,
            },
        }
    io_started = time.perf_counter()
    write_json(output / 'plan.json', plan)
    write_json(output / 'parameters.json', asdict(config))
    write_json(
        output / 'candidates.json', [public_candidate(r) for r in search['records']]
    )
    evaluations = []
    operation_seconds = evaluation_seconds = 0.0
    io_seconds = time.perf_counter() - io_started
    calls = 0
    if selected is not None:
        garment = next(
            g for g in config.garments if g.id == selected['configuration'][2]
        )
        hardware, actual = selected['hardware'], selected['actual']
        actual_digest = hashlib.sha256(
            json.dumps(_json_value(actual), sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        for controller in controllers:
            tic = time.perf_counter()
            run = simulate(
                actual,
                hardware,
                config.settings,
                config.scenario.initial_state['soc'],
                controller,
                config.controllers,
                make_forecast(config, garment, actual),
            )
            elapsed = time.perf_counter() - tic
            operation_seconds += elapsed
            calls += run['optimizer_calls']
            tic = time.perf_counter()
            evaluation = evaluate_trajectory(
                run,
                actual,
                config.scenario.initial_state['soc'],
                hardware,
                config.settings,
                **config.evaluation,
            )
            evaluated = time.perf_counter() - tic
            evaluation_seconds += evaluated
            evaluations.append(evaluation)
            record = {
                'schema': 'bgems-trajectory-v1',
                'scenario_type': config.scenario.scenario_type,
                'date': config.scenario.start.date().isoformat(),
                'scenario_name': config.scenario.name,
                'configuration': selected['configuration'],
                'candidate': public_candidate(selected),
                'hardware': hardware,
                'actual': actual,
                'forcing_sha256': actual_digest,
                'initial_soc': config.scenario.initial_state['soc'],
                'settings': config.settings,
                'evaluation_tolerances': config.evaluation,
                'controller_parameters': config.controllers,
                'run': run,
                'evaluation': evaluation,
                'source_hashes': sources,
                'experiment': experiment,
                'timing': {
                    'operation_seconds': elapsed,
                    'solver_seconds': run['solver_seconds'],
                    'evaluation_seconds': evaluated,
                },
                'optimizer_calls': run['optimizer_calls'],
            }
            tic = time.perf_counter()
            folder = output / controller
            folder.mkdir()
            write_json(folder / 'run.json', record)
            io_seconds += time.perf_counter() - tic
            if progress:
                progress(
                    f'{controller}: completed={evaluation["mission_completed"]}, valid={evaluation["valid"]}, '
                    f'elapsed={elapsed:.3f}s, optimizer_calls={run["optimizer_calls"]}'
                )
    summary = {
        'schema': 'bgems-mission-v1',
        'scenario_type': config.scenario.scenario_type,
        'date': config.scenario.start.date().isoformat(),
        'mode': mode,
        'candidate_count': len(search['records']),
        'design_admissible_count': len(search['admissible']),
        'pareto_configurations': [r['configuration'] for r in search['pareto']],
        'selected': public_candidate(selected) if selected else None,
        'design_status': search['status'],
        'controllers': list(controllers),
        'outcomes': summarize_runs(evaluations, len(controllers)),
        'optimizer_calls': calls,
        'configuration_optimizer_calls': 0,
        'evaluation_optimizer_calls': 0,
        'source_hashes': sources,
        'timing': dict(
            search['timing'],
            weather_seconds=weather_seconds,
            provenance_seconds=provenance_seconds,
            operation_seconds=operation_seconds,
            evaluation_seconds=evaluation_seconds,
            io_seconds=io_seconds,
            total_seconds=time.perf_counter() - started,
        ),
    }
    tic = time.perf_counter()
    write_json(output / 'summary.json', summary)
    (output / 'output_configuration.txt').write_text(
        f'Scenario: {config.scenario.name}\nDesign status: {search["status"]}\n'
        f'Selected: {selected["configuration"] if selected else None}\n'
        f'Completed controllers: {summary["outcomes"]["completed_runs"]}/{len(controllers)}\n',
        encoding='utf-8',
    )
    summary['timing']['io_seconds'] += time.perf_counter() - tic
    summary['timing']['total_seconds'] = time.perf_counter() - started
    write_json(output / 'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--subject', default='subject.json')
    parser.add_argument(
        '--scenario', type=int, choices=(1, 2, 3), nargs='+', default=[1, 2, 3]
    )
    parser.add_argument(
        '--date', help='Explicit YYYYMMDD override; otherwise specified original date.'
    )
    parser.add_argument(
        '--controller', choices=CONTROLLERS, nargs='+', default=list(CONTROLLERS)
    )
    parser.add_argument('--configuration', type=int, nargs=3, metavar=('Y', 'H', 'G'))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args(argv)
    configs = [
        load_system_config(
            args.subject, 'scenario_profiles.json', 'components.json', n, args.date
        )
        for n in args.scenario
    ]
    print(
        json.dumps(
            execution_plan(configs, args.controller, configuration=args.configuration),
            indent=2,
        )
    )
    if args.plan_only:
        return 0
    if args.output_dir is None:
        parser.error('--output-dir is required for execution.')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output_dir / 'plan.json',
        execution_plan(configs, args.controller, configuration=args.configuration),
    )
    summaries = []
    for config in configs:
        folder = (
            args.output_dir
            / f's{config.scenario.scenario_type}_{config.scenario.start:%Y%m%d}'
        )
        summaries.append(
            execute(
                config,
                'fixed' if args.configuration else 'search',
                folder,
                print,
                controllers=args.controller,
                configuration=args.configuration,
            )
        )
    write_json(args.output_dir / 'batch.json', summaries)
    return (
        0
        if all(
            s['outcomes']['completed_runs'] == len(args.controller) for s in summaries
        )
        else 1
    )


if __name__ == '__main__':
    raise SystemExit(main())
