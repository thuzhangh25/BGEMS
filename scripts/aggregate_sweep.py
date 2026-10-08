'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Aggregate planned scenario-date runs without imputing missing or failed
days. Outputs date_ledger.csv and date_summary.json over the predeclared 317
planned (scenario, date) rows. A directory that exists but lacks summary.json is
recorded as status=error with reason visible in the ledger; a missing directory
is status=not_run. Folders whose parameters.json lacks the whole-body service
definition (all pre-refactor outputs) are rejected as status=error, never
silently aggregated. Candidate counts are reported separately and are never
confused with the scenario-date denominator.
Usage: python3 -m scripts.aggregate_sweep --input_root output --output_dir output/date_report
'''
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from src.modeling import SERVICE_DEFINITION_ID


RANGES = {
    1: ([f'202512{d:02d}' for d in range(1, 32)] +
        [f'202601{d:02d}' for d in range(1, 32)] +
        [f'202602{d:02d}' for d in range(1, 29)]),
    2: ([f'202606{d:02d}' for d in range(15, 31)] +
        [f'202607{d:02d}' for d in range(1, 31)]),
    3: ([f'202511{d:02d}' for d in range(1, 31)] +
        [f'202512{d:02d}' for d in range(1, 32)] +
        [f'202601{d:02d}' for d in range(1, 32)] +
        [f'202602{d:02d}' for d in range(1, 29)] +
        [f'202603{d:02d}' for d in range(1, 32)] +
        [f'202604{d:02d}' for d in range(1, 31)]),
}


def aggregate(input_root):
    """One immutable ledger entry per planned date; candidate counts stay separate."""
    root = Path(input_root)
    rows, contract = [], None
    for scenario, dates in RANGES.items():
        for day in dates:
            folder = root / f's{scenario}_{day}'
            path = folder / 'summary.json'
            row = {'scenario_type': scenario, 'date': day,
                   'status': 'error' if folder.exists() else 'not_run',
                   'selected_y': None, 'selected_h': None, 'selected_g': None,
                   'selected_mass_kg': None, 'selected_cost': None,
                   'selected_load_wh': None, 'selected_min_soc': None,
                   'selected_terminal_soc': None, 'selected_artifact': None,
                   'candidate_success': None, 'candidate_failed': None,
                   'candidate_unresolved': None, 'causes': None,
                   'has_provisional_candidate': False, 'subject_id': None,
                   'reason': 'Missing summary.json' if folder.exists() else None}
            if path.exists():
                try:
                    summary = json.loads(path.read_text(encoding='utf-8'))
                    parameters = json.loads((folder / 'parameters.json').read_text(encoding='utf-8'))
                    service_def = parameters.get('service_definition')
                    if service_def != SERVICE_DEFINITION_ID:
                        raise ValueError('Legacy or missing service definition; '
                                         'pre-refactor outputs cannot be aggregated '
                                         'with the whole-body model')
                    with (folder / 'candidates.csv').open(newline='', encoding='utf-8') as stream:
                        candidates = list(csv.DictReader(stream))
                    counts = summary['counts']
                    status, selected = summary['status'], summary['selected']
                    if (summary['mode'] != 'search' or
                            status not in ('success', 'failed', 'unresolved') or
                            parameters['priority'] not in (['mass', 'cost'], ['cost', 'mass']) or
                            parameters['config']['scenario']['scenario_type'] != scenario or
                            parameters['config']['scenario']['start'][:10].replace('-', '') != day or
                            parameters['config']['comparison']['enabled'] or
                            len(candidates) != counts['scheduled'] or
                            sum(counts[key] for key in ('success', 'failed', 'unresolved')) !=
                            counts['scheduled'] or
                            (status == 'success') !=
                            (isinstance(selected, list) and len(selected) == 3 and
                             counts['unresolved'] == 0) or
                            (status == 'failed' and (selected is not None or
                                                     counts['success'] != 0)) or
                            (status == 'unresolved' and counts['unresolved'] == 0)):
                        raise ValueError('Search status, source contract or candidate ledger disagree')
                    causes = set()
                    for candidate in candidates:
                        if candidate['status'] != 'feasible':
                            causes.add(candidate['status'])
                        components = json.loads(candidate['components'])
                        for component in ('C', 'E', 'S', 'R'):
                            if components[component] is False:
                                causes.add(component + '_failed')
                    row.update(status=status, reason=None,
                               candidate_success=int(counts['success']),
                               candidate_failed=int(counts['failed']),
                               candidate_unresolved=int(counts['unresolved']),
                               has_provisional_candidate=status == 'unresolved' and
                               counts['success'] > 0,
                               causes=json.dumps(sorted(causes)))
                    if status == 'success':
                        matches = [candidate for candidate in candidates
                                   if [int(candidate[key]) for key in
                                       ('battery_cells', 'pv_units', 'garment_id')] == selected]
                        if (len(matches) != 1 or matches[0]['status'] != 'feasible' or
                                matches[0]['outcome'] != 'success'):
                            raise ValueError('Selected configuration lacks accepted candidate evidence')
                        candidate = matches[0]
                        artifact = folder / candidate['artifact']
                        verdict = json.loads(artifact.read_text(encoding='utf-8'))['evaluation']
                        if (verdict['status'] != 'success' or
                                any(verdict['components'][key] is not True for key in 'CESR')):
                            raise ValueError('Selected artifact fails independent C/E/S/R audit')
                        metrics = verdict['metrics']
                        row.update(selected_y=selected[0], selected_h=selected[1],
                                   selected_g=selected[2],
                                   selected_mass_kg=float(candidate['mass_kg']),
                                   selected_cost=float(candidate['acquisition_cost']),
                                   selected_load_wh=metrics['electrical_load_energy_wh'],
                                   selected_min_soc=metrics['minimum_soc'],
                                   selected_terminal_soc=metrics['terminal_soc'],
                                   selected_artifact=candidate['artifact'])
                    frozen_hashes = {
                        Path(name).name: digest
                        for name, digest in parameters['source_sha256'].items()
                        if (Path(name).parent.name == 'src' or
                            name in parameters['config_sources'].values())}
                    if (not frozen_hashes or
                            len(frozen_hashes) < 10):
                        raise ValueError('Missing executable and configuration hashes')
                    fingerprint = (parameters['config']['person']['subject_id'],
                                   service_def, frozen_hashes)
                    if contract is not None and fingerprint != contract:
                        raise ValueError('Mixed subject or executable/configuration versions')
                    contract = fingerprint
                    row['subject_id'] = fingerprint[0]
                except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
                    row.update(status='error', selected_y=None, selected_h=None,
                               selected_g=None, selected_mass_kg=None,
                               selected_cost=None, selected_load_wh=None,
                               selected_min_soc=None, selected_terminal_soc=None,
                               selected_artifact=None, causes=None,
                               has_provisional_candidate=False, reason=str(error))
            rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_root', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args(argv)
    rows = aggregate(args.input_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'date_ledger.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for scenario, dates in RANGES.items():
        subset = [row for row in rows if row['scenario_type'] == scenario]
        statuses = Counter(row['status'] for row in subset)
        successes = [row for row in subset if row['status'] == 'success']
        coverage = Counter(cause for row in subset if row['causes'] is not None
                           for cause in json.loads(row['causes']))
        summary[str(scenario)] = {
            'service_definition': SERVICE_DEFINITION_ID,
            'planned_dates': len(dates), 'statuses': dict(statuses),
            'confirmed_selection_fraction': len(successes) / len(dates),
            'successful_date_distribution': {
                'denominator': len(successes),
                'battery_units': dict(Counter(str(row['selected_y']) for row in successes)),
                'pv_units': dict(Counter(str(row['selected_h']) for row in successes)),
                'garment_ids': dict(Counter(str(row['selected_g']) for row in successes)),
                'joint_configuration': dict(Counter(json.dumps([
                    row['selected_y'], row['selected_h'], row['selected_g']])
                    for row in successes)),
            },
            'failure_cause_date_coverage': dict(coverage),
        }
    (args.output_dir / 'date_summary.json').write_text(
        json.dumps({'planned_dates': len(rows), 'by_scenario': summary},
                   indent=2) + '\n', encoding='utf-8')
    print(f'{len(rows)} planned dates: ' +
          str({key: value['statuses'] for key, value in summary.items()}))


if __name__ == '__main__':
    main()
