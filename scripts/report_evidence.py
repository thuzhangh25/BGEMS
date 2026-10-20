'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Independently recompute finite-task metrics and export self-contained
plot data. This entry never fetches weather or invokes a controller/optimizer.
'''

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from src.evaluation import evaluate_trajectory, summarize_runs


def _verified_record(path, common_code_hashes, cohorts):
    """Share corruption and paired-input checks between both report formats."""
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('schema') != 'bgems-trajectory-v1':
        raise ValueError(f'Not a current finite-task trajectory: {path}')
    code = {k: v for k, v in record['source_hashes'].items() if k.startswith('src/')}
    if common_code_hashes is not None and code != common_code_hashes:
        raise ValueError('Input records mix different scientific source versions.')
    digest = hashlib.sha256(json.dumps(
        record['actual'], sort_keys=True, allow_nan=False).encode()).hexdigest()
    if digest != record['forcing_sha256']:
        raise ValueError(f'Forcing identity mismatch: {path}')
    signature = json.dumps({key: record[key] for key in (
        'configuration', 'hardware', 'settings', 'initial_soc', 'forcing_sha256',
        'controller_parameters', 'experiment')}, sort_keys=True, allow_nan=False)
    expected, seen = cohorts.setdefault(path.parent.parent, (signature, set()))
    controller = record['run']['controller']
    if signature != expected or controller in seen:
        raise ValueError(f'Unpaired or duplicate controller evidence: {path.parent.parent}')
    seen.add(controller)
    evaluation = evaluate_trajectory(
        record['run'], record['actual'], record['initial_soc'], record['hardware'],
        record['settings'], **record['evaluation_tolerances'])
    if not evaluation['valid']:
        raise ValueError(f'Invalid trajectory {path}: {evaluation["issues"]}')
    record['evaluation'] = evaluation
    return record, code


def report(data_dir, output_dir):
    source, output = Path(data_dir), Path(output_dir)
    records = []
    identities = {}
    common_code_hashes = None
    cohorts = {}
    for path in sorted(source.rglob('run.json')):
        record, common_code_hashes = _verified_record(path, common_code_hashes, cohorts)
        identities[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({
            **{key: record[key] for key in (
                'scenario_type', 'scenario_name', 'date', 'configuration', 'experiment',
                'forcing_sha256', 'evaluation', 'run', 'settings', 'candidate')},
            'controller': record['run']['controller'],
            'source': str(path.relative_to(source)),
        })
    plan_path = source / 'plan.json'
    if plan_path.is_file():
        plan = json.loads(plan_path.read_text())
        scheduled = plan['unperturbed_controller_runs_at_most'] + plan['mc_controller_runs']
    else:
        summaries = [json.loads(p.read_text()) for p in source.rglob('summary.json')]
        scheduled = sum(
            len(s['controllers'])
            for s in summaries
            if s.get('schema') == 'bgems-mission-v1'
        )
    if scheduled == 0:
        raise ValueError('No specified finite-task runs found.')
    outcome = summarize_runs([r['evaluation'] for r in records], scheduled)
    output.mkdir(parents=True, exist_ok=False)
    evidence = {
        'schema': 'bgems-evidence-v1',
        'records': records,
        'outcomes': outcome,
        'source_files_sha256': identities,
        'scientific_source_hashes': common_code_hashes,
        'evaluation_optimizer_calls': 0,
    }
    (output / 'evidence.json').write_text(
        json.dumps(evidence, indent=2, allow_nan=False) + '\n'
    )
    rows = []
    for record in records:
        rows.append(
            {
                'source': record['source'],
                'scenario_type': record['scenario_type'],
                'date': record['date'],
                'controller': record['controller'],
                'mission_completed': record['evaluation']['mission_completed'],
                'failure_kind': (record['evaluation']['failure'] or {}).get('kind'),
                **record['evaluation']['metrics'],
            }
        )
    if rows:
        with (output / 'metrics.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return evidence


def study_report(data_dir, output_dir):
    """Audit the released study layout without private research directories."""
    root = Path(data_dir)
    roots = {'unperturbed': root / 'results'}
    roots.update({batch: root / 'experiments' / batch for batch in
                  ('pilot_mc', 'formal_mc', 'stress', 'static', 'indoor', 'fixed_fleece')})
    rows, traces, identities, cohorts = [], {}, {}, {}
    code = None
    priority_gap, split_steps = 0., 0
    for batch, folder in roots.items():
        for path in sorted(folder.rglob('run.json')):
            record, code = _verified_record(path, code, cohorts)
            prefix = 'results' if batch == 'unperturbed' else f'experiments/{batch}'
            relative = str(path.relative_to(folder))
            identity = f'{prefix}/{relative}'
            identities[identity] = hashlib.sha256(path.read_bytes()).hexdigest()
            row = {
                'source': identity, 'batch': batch, 'case': str(Path(relative).parent.parent),
                'controller': record['run']['controller'], 'scenario_type': record['scenario_type'],
                'configuration': record['configuration'], 'mass_kg': record['candidate']['mass_kg'],
                'cost_usd': record['candidate']['cost_usd'],
                'completed': record['evaluation']['mission_completed'],
                'failure_kind': (record['evaluation']['failure'] or {}).get('kind'),
                **record['evaluation']['metrics'],
            }
            rows.append(row)
            if batch not in ('formal_mc', 'pilot_mc'):
                traces[identity] = record
            weights = np.asarray(record['hardware']['weights'])
            for action in record['run']['actions']:
                values = []
                for target in action['priority_targets']:
                    ix = target['indices']
                    value = np.dot(weights[ix], np.asarray(action['regional_fulfillment'])[ix]) / weights[ix].sum()
                    priority_gap = max(priority_gap, abs(float(value) - target['target']))
                    values.append(value)
                split_steps += len(values) > 1 and max(values) - min(values) > 1e-4
    expected = {'unperturbed': 12, 'pilot_mc': 40, 'formal_mc': 4000, 'stress': 84,
                'static': 6, 'indoor': 16, 'fixed_fleece': 16}
    counts = {batch: {'runs': sum(r['batch'] == batch for r in rows),
                      'completed': sum(r['batch'] == batch and r['completed'] for r in rows)}
              for batch in expected}
    if {batch: value['runs'] for batch, value in counts.items()} != expected or sum(expected.values()) != len(rows):
        raise ValueError(f'Incomplete or unexpected specified study: {counts}')
    for case, (_, controllers) in cohorts.items():
        required = {'rhc'} if case.parent.name == 'static' else {'rbc', 'pid', 'tacs', 'rhc'}
        if controllers != required:
            raise ValueError(f'Incomplete controller cohort: {case}')
    controllers = ('rbc', 'pid', 'tacs', 'rhc')
    metrics = ('mean_service_full_mission', 'load_energy_wh', 'terminal_soc',
               'unmet_regional_heat_wh', 'reserve_limited_duration_h')
    mc = {c: sorted((r for r in rows if r['batch'] == 'formal_mc' and r['controller'] == c),
                    key=lambda r: r['case']) for c in controllers}
    statistics, paired = {}, {}
    for controller, records in mc.items():
        complete = [r for r in records if r['completed']]
        statistics[controller] = {
            'completed': len(complete), 'scheduled': len(records),
            'service_below_0.9_count': sum(r['mean_service_full_mission'] < .9 for r in complete),
            'metrics': {key: {'mean': float(np.mean([r[key] for r in complete])),
                              'q025': float(np.quantile([r[key] for r in complete], .025)),
                              'q975': float(np.quantile([r[key] for r in complete], .975))}
                        for key in metrics} if complete else {},
        }
        if controller == 'rhc':
            continue
        pairs = [(a, b) for a, b in zip(mc['rhc'], records) if a['completed'] and b['completed']]
        if any(a['case'] != b['case'] for a, b in pairs):
            raise ValueError('MC sample identities differ between controllers.')
        paired[controller] = {}
        for key in metrics[:3]:
            values = np.asarray([a[key] - b[key] for a, b in pairs])
            mean = float(values.mean()) if len(values) else None
            half = float(1.96 * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None
            paired[controller][key] = {'n': len(values), 'mean': mean,
                'mean_difference_95mc_interval': [mean - half, mean + half] if half is not None else None}

    def load(path, identity):
        identities[identity] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    unperturbed = []
    for folder in sorted(roots['unperturbed'].iterdir()):
        if not folder.is_dir():
            continue
        unperturbed.append({'summary': load(folder / 'summary.json', f'results/{folder.name}/summary.json'),
                        'candidates': load(folder / 'candidates.json', f'results/{folder.name}/candidates.json')})
    evidence = {
        'schema': 'bgems-study-results-v1', 'rows': rows, 'trajectories': traces,
        'counts': counts, 'unperturbed': unperturbed, 'monte_carlo': statistics, 'paired': paired,
        'priority_max_lock_gap': priority_gap, 'different_group_steps': int(split_steps),
        'applications': load(root / 'evaluation/applications/applications.json', 'evaluation/applications/applications.json'),
        'fixed_fleece': load(root / 'evaluation/fixed_fleece_applications.json', 'evaluation/fixed_fleece_applications.json'),
        'catalogue_scaling': load(root / 'experiments/catalogue_scaling/summary.json', 'experiments/catalogue_scaling/summary.json'),
        'design_checks': load(root / 'evaluation/design_checks.json', 'evaluation/design_checks.json'),
        'source_files_sha256': identities, 'scientific_source_hashes': code,
        'outcomes': {'executed_runs': len(rows), 'completed_runs': sum(r['completed'] for r in rows),
                     'independently_valid_runs': len(rows), 'evaluation_optimizer_calls': 0},
        'mean_service_reference': {'value': .8, 'status': 'retrospective illustrative mean-score reference'},
    }
    # Application values must derive from the same independently recomputed trajectory metrics.
    indexed = {r['source']: r for r in rows}
    for state in evidence['fixed_fleece']['building_states']:
        row = indexed['experiments/fixed_fleece/' + state['source']]
        if any(row[k] != v for k, v in state['mission_metrics'].items()):
            raise ValueError('Supplemental application metrics differ from trajectory audit.')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'evidence.json').write_text(json.dumps(evidence, indent=2, allow_nan=False) + '\n')
    with (output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--study', action='store_true', help='Export the complete specified finite study.')
    args = parser.parse_args(argv)
    if args.study:
        evidence = study_report(args.data_dir, args.output_dir)
    else:
        evidence = report(args.data_dir, args.output_dir)
    print(json.dumps(evidence['outcomes'], indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
