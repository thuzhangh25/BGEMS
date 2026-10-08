'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Run the JSON-configured BGEMS configuration and operation pipeline.
Usage: python -m src.run [--subject FILE] [--scenario_type {1,2,3}]
                         [--date YYYYMMDD] [--output_dir DIRECTORY]
Exit codes: 0 successful/resolved selection, 1 resolved failure/no feasible candidate,
2 invalid input or execution error, 3 unresolved numerical evidence. Candidate
success fractions are not reliability estimates.
'''
import argparse
import csv
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
import platform
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy
import requests

from .configuration import build_candidate, search_configuration
from .evaluation import evaluate_trajectory, summarize_runs
from .inputs import SystemConfig, load_system_config
from .modeling import SERVICE_DEFINITION_ID
from .operation import build_scenario, make_forecast, simulate, validate_setup
from .weather import fetch_scenario_weather, read_scenario_weather
from pythermalcomfort.utilities import body_surface_area


def _json_value(value):
    """Encode arrays/timestamps; preserve nonfinite evidence as explicit text."""
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    return value


def _write_json(path, value):
    path.write_text(
        json.dumps(_json_value(value), indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )


def _write_csv(path, rows):
    """Flatten regional arrays to numbered columns; retain named residuals as JSON."""
    nan_to_none_fields = ('allocation_fulfillment', 'allocation_unmet_heat_w',
                          'allocation_deficit_duration_hours')
    flat = []
    for row in rows:
        copied = dict(row)
        # Uninstalled pad entries carry NaN ("not applicable", never read as
        # satisfied or failed); convert them to None before flattening so the
        # CSV cells are empty, not 'nan' text. _json_value still guards any
        # residual nonfinite scalar as explicit text.
        for key in nan_to_none_fields:
            value = copied.get(key)
            if isinstance(value, (list, tuple, np.ndarray)):
                copied[key] = [None if isinstance(item, float) and not math.isfinite(item)
                               else item for item in list(value)]
        result = {}
        for key, value in _json_value(copied).items():
            if isinstance(value, list):
                result.update({f'{key}_{index + 1}': item for index, item in enumerate(value)})
            elif isinstance(value, dict):
                result[key] = json.dumps(value, allow_nan=False)
            else:
                result[key] = value
        flat.append(result)
    fields = list(dict.fromkeys(key for row in flat for key in row)) or ['time_hours']
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat)


def _structural_verdict(candidate):
    """Represent a declared hardware-domain exclusion without fabricating a run."""
    return {
        'completed': False,
        'residuals': dict(candidate['structural_residuals']),
        'failure_kind': 'physical_infeasibility',
        'components': {'C': False, 'E': None, 'S': None, 'R': None},
        'success': False,
        'status': 'failed',
        'metrics': {},
        'issues': ['Declared structural constraints exclude this configuration.'],
    }




def _candidate_keys(config, mode):
    """Return the declared fixed tuple or finite complete search domain."""
    if mode not in ('fixed', 'search'):
        raise ValueError('Mode must be fixed or search.')
    if (isinstance(config.max_battery_cells, bool) or
            not isinstance(config.max_battery_cells, Integral) or
            config.max_battery_cells < 1):
        raise ValueError('max_battery_cells must be an integer >= 1.')
    if (isinstance(config.max_pv_units, bool) or
            not isinstance(config.max_pv_units, Integral) or config.max_pv_units < 0):
        raise ValueError('max_pv_units must be a nonnegative integer.')
    garment_ids = tuple(garment.id for garment in config.garments)
    if (not garment_ids or len(set(garment_ids)) != len(garment_ids) or
            any(isinstance(value, bool) or not isinstance(value, Integral) or value < 0
                for value in garment_ids)):
        raise ValueError('Garments must have distinct nonnegative integer IDs.')
    priority = tuple(config.priority)
    if len(priority) != 2 or set(priority) != {'mass', 'cost'}:
        raise ValueError('priority must order mass and cost exactly once.')
    fixed = tuple(config.fixed_configuration)
    if (len(fixed) != 3 or
            any(isinstance(value, bool) or not isinstance(value, Integral) for value in fixed)):
        raise ValueError('fixed_configuration must contain cell count, PV count and garment ID.')
    if not (1 <= fixed[0] <= config.max_battery_cells and
            0 <= fixed[1] <= config.max_pv_units and fixed[2] in garment_ids):
        raise ValueError('fixed_configuration lies outside the declared domain.')
    if mode == 'fixed':
        return (fixed,), garment_ids, priority
    keys = tuple(
        (cells, pv_units, garment_id)
        for cells in range(1, config.max_battery_cells + 1)
        for pv_units in range(config.max_pv_units + 1)
        for garment_id in sorted(garment_ids)
    )
    return keys, garment_ids, priority


def _audit_tolerances(config):
    names = ('balance_tolerance_wh', 'time_tolerance_hours')
    try:
        values = {name: config.evaluation[name] for name in names}
    except (KeyError, TypeError) as error:
        raise ValueError('evaluation must define both audit tolerances.') from error
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or
           not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError('Evaluation tolerances must be finite and nonnegative.')
    return {name: float(value) for name, value in values.items()}


def _operation_settings(config):
    """Translate declared JSON names and reject every unresolved execution input."""
    mapping = {'equipment_on_c': 'on_c', 'equipment_off_c': 'off_c'}
    required = (
        'dt_hours', 'horizon_steps', 'soc_min', 'soc_safe', 'lambda_deg',
        'lambda_soc', 'equipment_on_c', 'equipment_off_c', 'filter_window_h',
        'solver_tolerance', 'constraint_tolerance',
    )
    unresolved = [f'operation.{name}' for name in required
                  if config.settings.get(name) is None]
    if config.scenario.terminal_soc is None:
        unresolved.append('terminal_soc')
    if config.scenario.initial_state.get('battery_c') is None:
        unresolved.append('initial_state.battery_c')
    if any(phase[3] is None for phase in config.scenario.get_phases()):
        unresolved.append('phases[].service_floor')
    for name in ('voltage_v', 'heat_capacity_j_per_k', 'thermal_resistance_k_per_w',
                 'max_discharge_w', 'max_charge_w', 'bus_limit_w'):
        if getattr(config.device, name) is None:
            unresolved.append(f'components.battery/equipment.{name}')
    if config.forecast_mode is None:
        unresolved.append('operation.forecast_mode')
    if unresolved:
        raise ValueError('Unresolved execution parameters: ' + ', '.join(unresolved))
    settings = {
        mapping.get(name, name): config.settings[name]
        for name in required if name != 'filter_window_h'
    }
    window_steps = config.settings['filter_window_h'] / config.settings['dt_hours']
    if not float(window_steps).is_integer() or window_steps < 1:
        raise ValueError(
            'operation.filter_window_h must be a positive exact multiple of step_h.')
    settings['filter_window'] = int(window_steps)
    settings['terminal_soc'] = float(config.scenario.terminal_soc)
    settings['priority_mode'] = 'lexicographic'
    return settings


def _source_hashes(config, weather_data):
    sources = set(weather_data['source_files'])
    sources.update(Path(__file__).parent.glob('*.py'))
    sources.update(Path(value) for value in config.source_files.values())
    hashes = {}
    for path in sorted(Path(value).resolve() for value in sources):
        with path.open('rb') as stream:
            hashes[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
    return hashes

def _write_requested_outputs(output, summary, selected, config):
    """Write the two user-facing artifacts required by the command-line contract."""
    person = config.person
    height_m = person.height_cm / 100
    bmi = person.weight_kg / height_m ** 2
    bsa = body_surface_area(person.weight_kg, height_m)
    lines = [
        f"status: {summary['status']}",
        f"subject_height_cm: {person.height_cm:.0f}",
        f"subject_weight_kg: {person.weight_kg:.1f}",
        f"subject_age_years: {person.age_years:.0f}",
        f"subject_bmi: {bmi:.2f}",
        f"subject_bsa_m2: {bsa:.3f}",
    ]
    if selected is None:
        lines.append('selected_configuration: none')
    else:
        cells, pv_units, garment_id = selected['configuration']
        garment = next(item for item in config.garments if item.id == garment_id)
        lines.extend((
            f'garment_type: {garment.name}',
            f'backpack_included: {str(config.scenario.backpack_included).lower()}',
            f'battery_units: {cells}',
            f'pv_units: {pv_units}',
            f'system_mass_kg: {selected["objectives"][0]:.6g}',
            f'system_cost_usd: {selected["objectives"][1]:.6g}',
        ))
    (output / 'output_configuration.txt').write_text(
        '\n'.join(lines) + '\n', encoding='utf-8')

    has_pv = selected is not None and selected['configuration'][1] > 0
    n_plots = 4 if has_pv else 3
    figure, axes = plt.subplots(n_plots, 1, figsize=(11, 8.5), sharex=True)
    if selected is None:
        for axis in axes:
            axis.set_axis_off()
        axes[1].text(0.5, 0.5, f'No accepted configuration: {summary["status"]}',
                     ha='center', va='center', transform=axes[1].transAxes)
    else:
        artifact = output / selected['operation']['artifact']
        evidence = json.loads(artifact.read_text(encoding='utf-8'))
        steps = evidence['trajectory']['steps']
        states = evidence['trajectory']['states']
        # Convert elapsed hours to local time for display.
        scenario_start = config.scenario.start
        step_time = [(scenario_start + timedelta(hours=item['time_hours'])).strftime('%H:%M')
                     for item in steps]
        state_time = [(scenario_start + timedelta(hours=item['time_hours'])).strftime('%H:%M')
                      for item in states]
        step_x = [item['time_hours'] for item in steps]
        state_x = [item['time_hours'] for item in states]
        # Extend data to scenario end so the last phase is fully visible.
        scenario_end = config.scenario.get_duration_hours()
        step_x_ext = step_x + [scenario_end]
        state_x_ext = state_x + [scenario_end]
        air_ext = [item['air_c'] for item in steps] + [steps[-1]['air_c']]
        soc_ext = [item['soc'] for item in states] + [states[-1]['soc']]
        thermal_ext = [item['thermal_score'] for item in steps] + [steps[-1]['thermal_score']]
        axes[0].plot(step_x_ext, air_ext)
        axes[0].set_ylabel('$T_a$ (°C)')
        axes[1].plot(state_x_ext, soc_ext)
        axes[1].set_ylabel('SOC')
        if has_pv:
            pv_ext = [item['pv_available_w'] for item in steps] + [steps[-1]['pv_available_w']]
            axes[2].plot(step_x_ext, pv_ext)
            axes[2].set_ylabel('PV (W)')
        axes[n_plots-1].plot(step_x_ext, thermal_ext)
        axes[n_plots-1].set_ylabel('$U_{thermal}$ (whole-body)')
        boundaries, labels, cursor = [0.0], [], 0.0
        for label, duration, _, _ in config.scenario.get_phases():
            labels.append(label)
            cursor += duration
            boundaries.append(cursor)
        # Scenario phase boundaries as yellow dashed lines (including start and end).
        for axis in axes:
            for boundary in boundaries:
                axis.axvline(boundary, color='gold', linestyle='--', linewidth=1.0)
            axis.grid(True, alpha=0.25)
        # Phase labels at the center of each phase.
        centers = [(left + right) / 2 for left, right in zip(boundaries, boundaries[1:])]
        for idx, (center, label) in enumerate(zip(centers, labels)):
            # Alternate label positions to avoid overlap.
            y_pos = 0.95 if idx % 2 == 0 else 0.05
            axes[n_plots-1].text(center, y_pos, label,
                                 transform=axes[n_plots-1].get_xaxis_transform(),
                                 ha='center', va='top' if idx % 2 == 0 else 'bottom',
                                 fontsize=8, rotation=90, color='dimgray')
        # Time labels at phase boundaries (including start and end).
        boundary_times = [(scenario_start + timedelta(hours=h)).strftime('%H:%M')
                          for h in boundaries]
        axes[n_plots-1].set_xticks(boundaries, boundary_times, rotation=0, ha='center')
    axes[-1].set_xlabel(f'Local time ({config.scenario.timezone} {config.scenario.start.strftime("%Y-%m-%d")})')
    figure.suptitle(f'{config.scenario.name}: operation status {summary["status"]}')
    figure.tight_layout()
    figure.savefig(output / 'output_operation.pdf')
    plt.close(figure)


def execute(config, mode, output, progress=None):
    """Run a SystemConfig and persist independently evaluated candidate evidence."""
    if not isinstance(config, SystemConfig):
        raise TypeError('execute requires a SystemConfig instance.')
    if progress is not None and not callable(progress):
        raise TypeError('Progress reporter must be callable.')
    report = progress or (lambda message: None)
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc)
    output = Path(output).expanduser().resolve()
    keys, garment_ids, priority = _candidate_keys(config, mode)
    positions = {key: index for index, key in enumerate(keys, start=1)}
    report(f'[search] Preparing {len(keys)} candidates')
    audit_tolerances = _audit_tolerances(config)
    operation_settings = _operation_settings(config)
    if config.forecast_mode not in ('persistence', 'perfect-preview'):
        raise ValueError("forecast_mode must be 'persistence' or 'perfect-preview'.")
    weather_data = read_scenario_weather(
        config.scenario, operation_settings['dt_hours'])
    garments = {garment.id: garment for garment in config.garments}
    active_garment_ids = {garment_id for _, _, garment_id in keys}
    scenarios = {
        identifier: build_scenario(config, garments[identifier], weather_data)
        for identifier in active_garment_ids
    }
    candidates = {}
    for cells, pv_units, garment_id in keys:
        value = build_candidate(
            config,
            scenarios[garment_id],
            weather_data,
            cells,
            pv_units,
            garment_id,
        )
        validate_setup(
            value['hardware'], operation_settings, config.scenario.initial_state)
        candidates[(cells, pv_units, garment_id)] = value
    report(f'[search] Prepared {len(keys)} candidates')
    preparation_seconds = time.perf_counter() - started

    output.mkdir(parents=True, exist_ok=True)
    runs_dir = output / 'runs'
    runs_dir.mkdir(exist_ok=True)
    for pattern in ('y*_h*_g*.json', 'y*_h*_g*_states.csv', 'y*_h*_g*_steps.csv'):
        for stale in runs_dir.glob(pattern):
            stale.unlink()
    hashes = _source_hashes(config, weather_data)
    _write_json(output / 'parameters.json', {
        'mode': mode,
        'config': asdict(config),
        'config_sources': config.source_files,
        'started_utc': started_utc,
        'source_sha256': hashes,
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'forecast_mode': config.forecast_mode,
        'service_definition': SERVICE_DEFINITION_ID,
        'priority': priority,
        'weather_semantics': weather_data['semantics'],
        'weather_sample_hours': weather_data['sample_hours'],
    })
    evaluations = []
    candidate_rows = []
    timings = {
        'preparation_seconds': preparation_seconds,
        'operation_seconds': 0.0,
        'evaluation_seconds': 0.0,
        'priority_lp_seconds': 0.0,
        'lp_seconds': 0.0,
        'priority_audit_seconds': 0.0,
        'selection_seconds': 0.0,
    }

    def candidate(cells, pv_units, garment_id):
        return candidates[(cells, pv_units, garment_id)]

    def evaluate(cells, pv_units, garment_id, value):
        position = positions[(cells, pv_units, garment_id)]
        candidate_started = time.perf_counter()
        report(
            f'[search] Candidate {position}/{len(keys)}: '
            f'y{cells}_h{pv_units}_g{garment_id}')
        before = time.perf_counter()
        scenario_data = scenarios[garment_id]
        run = simulate(
            value['actual'],
            make_forecast(config, garments[garment_id], scenario_data, value['actual']),
            config.forecast_mode,
            scenario_data['service_floor'],
            config.scenario.initial_state,
            value['hardware'],
            operation_settings,
        )
        timings['operation_seconds'] += time.perf_counter() - before
        timings['priority_lp_seconds'] += sum(
            stage['seconds'] for solve in run['solves']
            for stage in solve.get('priority_stages', ()))
        timings['lp_seconds'] += sum(solve.get('lp_seconds', 0.) for solve in run['solves'])
        before = time.perf_counter()
        verdict = evaluate_trajectory(
            run,
            value['actual'],
            scenario_data['service_floor'],
            config.scenario.initial_state,
            value['hardware'],
            operation_settings,
            **audit_tolerances,
        )
        timings['evaluation_seconds'] += time.perf_counter() - before
        timings['priority_audit_seconds'] += verdict['metrics']['priority_audit_seconds']
        name = f'y{cells}_h{pv_units}_g{garment_id}'
        artifact = runs_dir / f'{name}.json'
        _write_json(artifact, {
            'configuration': (cells, pv_units, garment_id),
            'service_definition': SERVICE_DEFINITION_ID,
            'hardware': value['hardware'],
            'actual': value['actual'],
            'evaluation': verdict,
            'trajectory': run,
        })
        origin = weather_data['times'][0]
        for kind in ('states', 'steps'):
            rows = []
            for row in run[kind]:
                copied = dict(row)
                copied['timestamp'] = (
                    origin.astimezone(timezone.utc) +
                    timedelta(hours=float(row['time_hours']))
                ).astimezone(origin.tzinfo)
                if kind == 'steps' and 'soc' in copied:
                    copied['soc_end'] = copied.pop('soc')
                rows.append(copied)
            _write_csv(runs_dir / f'{name}_{kind}.csv', rows)
        report(
            f'[search] Candidate {position}/{len(keys)}: {verdict["status"]} '
            f'({time.perf_counter() - candidate_started:.1f}s)')
        return {**verdict, 'artifact': str(artifact.relative_to(output))}

    if mode == 'fixed':
        cells, pv_units, garment_id = keys[0]
        value = candidate(cells, pv_units, garment_id)
        verdict = (
            evaluate(cells, pv_units, garment_id, value)
            if all(residual <= 0 for residual in value['structural_residuals'].values())
            else _structural_verdict(value)
        )
        records = [{
            'configuration': (cells, pv_units, garment_id),
            'objectives': value['objectives'],
            'candidate': value,
            'operation': verdict,
            'status': 'feasible' if verdict['success'] else verdict['failure_kind'],
        }]
        selected = records[0] if verdict['success'] else None
        pareto = []
    else:
        search = search_configuration(
            config.max_battery_cells,
            config.max_pv_units,
            garment_ids,
            candidate,
            evaluate,
            priority,
        )
        records, selected, pareto = search['records'], search['selected'], search['pareto']
        timings['selection_seconds'] = search['selection_seconds']

    for record in records:
        verdict = record['operation']
        if verdict is None:
            verdict = _structural_verdict(record['candidate'])
        evaluations.append(verdict)
        cells, pv_units, garment_id = record['configuration']
        candidate_rows.append({
            'battery_cells': cells,
            'pv_units': pv_units,
            'garment_id': garment_id,
            'service_definition': SERVICE_DEFINITION_ID,
            'capability_diagnostic': record['candidate']['capability'],
            'mass_kg': record['objectives'][0],
            'acquisition_cost': record['objectives'][1],
            'status': record['status'],
            'outcome': verdict['status'],
            'components': verdict['components'],
            'diagnostics': record['candidate']['diagnostics'],
            'structural_residuals': record['candidate']['structural_residuals'],
            'residuals': verdict['residuals'],
            'issues': verdict['issues'],
            'metrics': verdict['metrics'],
            'artifact': verdict.get('artifact'),
        })
    counts = summarize_runs(evaluations, scheduled_runs=len(records))
    unresolved = any(item['status'] == 'unresolved' for item in evaluations)
    status = 'unresolved' if unresolved else 'success' if selected else 'failed'
    _write_csv(output / 'candidates.csv', candidate_rows)
    timings['elapsed_before_summary_seconds'] = time.perf_counter() - started
    summary = {
        'mode': mode,
        'status': status,
        'service_definition': SERVICE_DEFINITION_ID,
        'priority': priority,
        'forecast_mode': config.forecast_mode,
        'counts': counts,
        'timings': timings,
        'selected': selected['configuration'] if selected else None,
        'pareto': [item['configuration'] for item in pareto],
        'selection_scope': (
            'resolved_feasible_candidates' if unresolved else
            'declared_domain' if mode == 'search' else
            'fixed_configuration'
        ),
        'note': 'Design-domain candidate counts are not a reliability sample.',
    }
    report(f'[search] Completed {len(records)} candidates: {status}')
    _write_requested_outputs(output, summary, selected, config)
    timings['total_seconds'] = time.perf_counter() - started
    _write_json(output / 'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--subject', default='subject.json',
        help='Subject JSON filename under data/systemconf or an explicit path.',
    )
    parser.add_argument(
        '--scenario_type', type=int, choices=(1, 2, 3), default=2,
        help='1=Harbin skiing, 2=Muztagh Ata climbing, 3=Qinling expedition.',
    )
    parser.add_argument(
        '--date', default=date.today().strftime('%Y%m%d'),
        help='Scenario date in YYYYMMDD format; default is today.',
    )
    parser.add_argument(
        '--output_dir', type=Path, default=Path('./output'),
        help='Output directory; created when missing and reused when present.',
    )
    args = parser.parse_args(argv)
    def report(message):
        print(message, flush=True)

    report('[config] Loading system configuration')
    try:
        config = load_system_config(
            args.subject, 'scenario_profiles.json', 'components.json',
            args.scenario_type, args.date)
        person = config.person
        height_m = person.height_cm / 100
        bmi = person.weight_kg / height_m ** 2
        bsa = body_surface_area(person.weight_kg, height_m)
        report(f'[subject] height={person.height_cm:.0f}cm weight={person.weight_kg:.1f}kg '
               f'age={person.age_years:.0f}yr BMI={bmi:.2f} BSA={bsa:.3f}m²')
        if config.comparison['enabled']:
            from scripts.compare import compare, frozen_reference
            frozen_reference(config.comparison['design_manifest'], config, args.date)
        fetch_scenario_weather(config.scenario, progress=report)
        if config.comparison['enabled']:
            paired = compare(config, args.date, config.comparison['design_manifest'],
                             args.output_dir)
            print('comparison_evidence:', Path(args.output_dir) / 'comparison.json')
            return (3 if not paired['comparison_resolved'] else
                    1 if any(arm['date_status'] == 'failed'
                             for arm in paired['arms'].values()) else 0)
        summary = execute(config, 'search', args.output_dir, progress=report)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError,
            FloatingPointError, requests.RequestException) as error:
        print(f'Run error: {error}', file=sys.stderr)
        return 2
    print((Path(args.output_dir) / 'output_configuration.txt').read_text(
        encoding='utf-8').rstrip())
    print('operation_status:', summary['status'])
    print('operation_figure:', Path(args.output_dir) / 'output_operation.pdf')
    return {'success': 0, 'failed': 1, 'unresolved': 3}[summary['status']]


if __name__ == '__main__':
    raise SystemExit(main())
