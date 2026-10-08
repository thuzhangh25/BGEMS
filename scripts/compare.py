'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Run predeclared independent-design hardware and same-hardware
RHC/myopic comparisons. Paths in the design manifest are relative to its parent;
each design output must be a resolved search with the same model settings and
executable source hashes.
Design manifest: {"scenario_type": 1, "subject_id": "...", "design_days":
  [{"date": "YYYYMMDD", "directory": "path/to/search/output"}, ...]}.
Usage: python3 -m scripts.compare --design_manifest design.json --scenario_type 1
       --date 20251201 --subject subject.json --output_dir output/comparison
'''
import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path

from src.configuration import build_candidate
from src.evaluation import evaluate_trajectory
from src.inputs import PROJECT_ROOT, load_system_config
from src.operation import build_scenario, make_forecast, simulate, validate_setup
from src.run import (_audit_tolerances, _candidate_keys, _json_value,
                     _operation_settings, execute)
from src.weather import read_scenario_weather


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_reference(manifest_path, config, evaluation_date):
    """Select minimum-mass-then-cost hardware feasible on *all* declared design days."""
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if (set(manifest) != {'scenario_type', 'subject_id', 'design_days'} or
            manifest['scenario_type'] != config.scenario.scenario_type or
            manifest['subject_id'] != config.person.subject_id or
            not isinstance(manifest['design_days'], list) or
            not manifest['design_days']):
        raise ValueError('Design manifest must declare scenario, subject and design days.')
    source_hashes = {name: _digest(file) for name, file in config.source_files.items()
                     if name in ('subject', 'components')}
    days, sources, common = set(), [], None
    for entry in manifest['design_days']:
        if set(entry) != {'date', 'directory'}:
            raise ValueError('Every design day must declare date and directory.')
        day = entry['date']
        try:
            datetime.strptime(day, '%Y%m%d')
        except (TypeError, ValueError) as error:
            raise ValueError('Design dates must use YYYYMMDD.') from error
        if day == evaluation_date or day in days:
            raise ValueError('Design dates must be distinct from one another and evaluation date.')
        days.add(day)
        folder = (path.parent / entry['directory']).resolve()
        if folder == path.parent or not folder.is_dir():
            raise ValueError(f'Design result directory does not exist: {folder}')
        summary_file, parameters_file = folder / 'summary.json', folder / 'parameters.json'
        candidates_file = folder / 'candidates.csv'
        summary = json.loads(summary_file.read_text(encoding='utf-8'))
        parameters = json.loads(parameters_file.read_text(encoding='utf-8'))
        if (summary['mode'] != 'search' or summary['status'] != 'success' or
                summary['counts']['unresolved'] != 0 or
                parameters['config']['scenario']['start'][:10].replace('-', '') != day or
                parameters['config']['person']['subject_id'] != config.person.subject_id or
                parameters['config']['scenario']['scenario_type'] != config.scenario.scenario_type):
            raise ValueError(f'Design day {day} is not a resolved matching search.')
        recorded = parameters['source_sha256']
        if any(recorded.get(str(Path(file).resolve())) != source_hashes[name]
               for name, file in config.source_files.items() if name in source_hashes):
            raise ValueError(f'Design day {day} has different subject/component hashes.')
        current = _json_value(asdict(config))
        designed = parameters['config']
        for values in (current, designed):
            values.pop('source_files', None)
            values.pop('comparison', None)
            values['scenario'].pop('start', None)
            values['scenario'].pop('temperature_files', None)
            values['scenario'].pop('pv_files', None)
            values.pop('fixed_configuration', None)
        if current != designed:
            raise ValueError(f'Design day {day} uses different scientific settings.')
        for source in (PROJECT_ROOT / 'src').glob('*.py'):
            if recorded.get(str(source.resolve())) != _digest(source):
                raise ValueError(f'Design day {day} uses different executable source: {source.name}')
        weather_sources = [(Path(name), digest) for name, digest in recorded.items()
                           if Path(name).suffix == '.csv']
        if len(weather_sources) < 2 or any(
                not source.is_file() or _digest(source) != digest
                for source, digest in weather_sources):
            raise ValueError(f'Design day {day} has missing or changed weather evidence.')
        with candidates_file.open(newline='', encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != summary['counts']['scheduled']:
            raise ValueError(f'Design day {day} has incomplete candidate records.')
        feasible = {}
        for row in rows:
            if row['status'] == 'feasible' and row['outcome'] == 'success':
                key = tuple(int(row[name]) for name in
                            ('battery_cells', 'pv_units', 'garment_id'))
                artifact = (folder / row['artifact']).resolve()
                if not artifact.is_relative_to((folder / 'runs').resolve()):
                    raise ValueError('Design candidate artifact must reside under runs/.')
                evidence = json.loads(artifact.read_text(encoding='utf-8'))
                verdict = evidence['evaluation']
                if (tuple(evidence['configuration']) != key or
                        verdict['status'] != 'success' or
                        any(verdict['components'][component] is not True
                            for component in 'CESR')):
                    raise ValueError(f'Design day {day} has unsupported accepted candidate.')
                feasible[key] = (float(row['mass_kg']), float(row['acquisition_cost']))
        common = feasible if common is None else {
            key: value for key, value in common.items()
            if key in feasible and value == feasible[key]}
        sources.append({'date': day, 'directory': str(folder),
                        'summary_sha256': _digest(summary_file),
                        'parameters_sha256': _digest(parameters_file),
                        'candidates_sha256': _digest(candidates_file)})
    if not common:
        raise ValueError('No legal hardware is independently feasible across all design dates.')
    selected = min(common, key=lambda key: (common[key], key))
    return selected, {'manifest_sha256': _digest(path), 'design_days': sources,
                      'reference_objectives': common[selected]}


def compare(config, evaluation_date, design_manifest, output):
    """Persist date-specific hardware and same-hardware controller arm evidence."""
    if config.scenario.start.strftime('%Y%m%d') != evaluation_date:
        raise ValueError('Evaluation date must match the configured scenario start.')
    _candidate_keys(config, 'search')
    settings = _operation_settings(config)
    if config.forecast_mode not in ('persistence', 'perfect-preview'):
        raise ValueError('Comparison requires a declared persistence/perfect-preview information mode.')
    frozen, provenance = frozen_reference(design_manifest, config, evaluation_date)
    weather = read_scenario_weather(config.scenario, settings['dt_hours'])
    output = Path(output).resolve()
    adaptive = execute(config, 'search', output / 'adaptive_search')
    reference_config = replace(config, fixed_configuration=frozen)
    reference = execute(reference_config, 'fixed', output / 'frozen_hardware')
    arms = {}
    for name, summary in (('adaptive_rhc', adaptive), ('frozen_rhc', reference)):
        key = summary['selected'] if name == 'adaptive_rhc' else frozen
        folder = output / ('adaptive_search' if name == 'adaptive_rhc' else 'frozen_hardware')
        artifact = folder / 'runs' / ('y%d_h%d_g%d.json' % tuple(key)) if key else None
        verdict = json.loads(artifact.read_text(encoding='utf-8'))['evaluation'] if artifact and artifact.exists() else None
        arms[name] = {'configuration': key, 'date_status': summary['status'],
                      'evaluation': verdict, 'artifact': str(artifact.relative_to(output)) if verdict else None}
    for name, key in (('adaptive_myopic', adaptive['selected']), ('frozen_myopic', frozen)):
        if key is None:
            arms[name] = {'configuration': None, 'date_status': adaptive['status'],
                          'evaluation': None, 'artifact': None}
            continue
        garment = next(item for item in config.garments if item.id == key[2])
        scenario = build_scenario(config, garment, weather)
        candidate = build_candidate(config, scenario, weather, *key)
        validate_setup(candidate['hardware'], settings, config.scenario.initial_state)
        if any(value > 0 for value in candidate['structural_residuals'].values()):
            arms[name] = {'configuration': key, 'date_status': 'failed',
                          'evaluation': None, 'artifact': None, 'reason': 'Structural exclusion'}
            continue
        run = simulate(candidate['actual'],
                       make_forecast(config, garment, scenario, candidate['actual']),
                       config.forecast_mode, scenario['service_floor'],
                       config.scenario.initial_state, candidate['hardware'], settings,
                       strategy='myopic_qp')
        verdict = evaluate_trajectory(run, candidate['actual'], scenario['service_floor'],
                                      config.scenario.initial_state, candidate['hardware'],
                                      settings, **_audit_tolerances(config))
        artifact = output / (name + '.json')
        artifact.write_text(json.dumps(_json_value({
            'configuration': key, 'hardware': candidate['hardware'],
            'actual': candidate['actual'], 'trajectory': run, 'evaluation': verdict}),
            indent=2) + '\n', encoding='utf-8')
        arms[name] = {'configuration': key, 'date_status': verdict['status'],
                      'evaluation': verdict, 'artifact': artifact.name}
    result = {'scenario_type': config.scenario.scenario_type, 'date': evaluation_date,
              'comparison_script_sha256': _digest(Path(__file__)),
              'comparison_resolved': adaptive['status'] != 'unresolved' and
              reference['status'] != 'unresolved' and
              all(arm['date_status'] != 'unresolved' for arm in arms.values()),
              'controller_neutral_feasibility': {
                  'adaptive_hardware': arms['adaptive_myopic']['date_status'] == 'success',
                  'frozen_hardware': arms['frozen_myopic']['date_status'] == 'success'},
              'subject_id': config.person.subject_id,
              'frozen_configuration': frozen, 'reference_provenance': provenance,
              'forecast_mode': config.forecast_mode, 'controller_contract':
              'Same hardware, inputs, floors, initial state, priority, and solver; '
              'myopic_QP limits prediction to the observed interval, with the '
              'actual end-of-mission terminal constraint only on the last interval.',
              'arms': arms}
    candidates_by_arm = {}
    for design, folder in (('adaptive', 'adaptive_search'),
                           ('frozen', 'frozen_hardware')):
        with (output / folder / 'candidates.csv').open(
                newline='', encoding='utf-8') as stream:
            candidates_by_arm[design] = {
                (int(row['battery_cells']), int(row['pv_units']), int(row['garment_id'])): row
                for row in csv.DictReader(stream)}
    rows = []
    for name, arm in arms.items():
        design = 'adaptive' if name.startswith('adaptive') else 'frozen'
        hardware_row = candidates_by_arm[design].get(
            tuple(arm['configuration'])) if arm['configuration'] is not None else None
        evaluation = arm['evaluation'] or {}
        metrics = evaluation.get('metrics', {})
        rows.append({
            'arm': name, 'configuration': json.dumps(arm['configuration']),
            'status': arm['date_status'], 'artifact': arm['artifact'],
            'mass_kg': hardware_row['mass_kg'] if hardware_row else None,
            'acquisition_cost': hardware_row['acquisition_cost'] if hardware_row else None,
            **{key: evaluation.get('components', {}).get(key) for key in 'CESR'},
            **{key: metrics.get(key) for key in (
                'observed_duration_hours', 'electrical_load_energy_wh',
                'mean_thermal_score_observed', 'minimum_soc', 'terminal_soc',
                'priority_audit_seconds')},
        })
    with (output / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'comparison.json').write_text(
        json.dumps(_json_value(result), indent=2) + '\n', encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design_manifest', type=Path, required=True)
    parser.add_argument('--scenario_type', type=int, choices=(1, 2, 3), required=True)
    parser.add_argument('--date', required=True)
    parser.add_argument('--subject', default='subject.json')
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_system_config(args.subject, 'scenario_profiles.json',
                                'components.json', args.scenario_type, args.date)
    result = compare(config, args.date, args.design_manifest, args.output_dir)
    print(json.dumps({name: value['date_status'] for name, value in result['arms'].items()}))
    return 3 if any(value['date_status'] == 'unresolved' for value in result['arms'].values()) else 0


if __name__ == '__main__':
    raise SystemExit(main())
