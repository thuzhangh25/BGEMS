'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Verify frozen records and export self-contained numerical evidence.
No weather acquisition, solver execution, plotting or prose rewriting occurs.
Usage: python3 -m scripts.report_evidence --output-dir <new-directory>
Optional --tex-dir exports only the two live author-side tables/macros.
'''
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import mean

import numpy as np

from scripts.aggregate_sweep import RANGES
from src.inputs import load_system_config
from src.operation import equipment_switch
from src.weather import read_scenario_weather

PROJECT = Path(__file__).resolve().parents[1]
STATUSES = ('success', 'failed', 'unresolved', 'error', 'not_run')
INPUTS = {
    'components': 'data/systemconf/components.json',
    'subject': 'data/systemconf/subject.json',
    'scenario_profiles': 'data/systemconf/scenario_profiles.json',
}
# A recorded analysis-version transition, not a replacement frozen-source hash.
# This exact post-freeze weather version changes acquisition/plotting. Every
# dated offline output is additionally checked against its retained run below.
WEATHER_SOURCE_TRANSITION = (
    '27cb601d5855f6a6aaa2d6d7e3ee56974af2fcd275b1296b241d7d3414cdbcdb',
    '76da3229a6dd8fe9b94a6631f77524ac00a106b2b0bc92a1f351659d83d7dd0d',
)
# Only the module-docstring delimiters changed; the frozen and current ASTs
# are identical. Keep both identities rather than rewriting historical hashes.
OPERATION_SOURCE_TRANSITION = (
    '89b4ea321cf36ebd084418a8ebeea414050837f419048c63ac50710348585834',
    '4189be246a479b6cc9a48550260784b819999c2cacc06f2414306cd63017b34d',
)


def read_csv(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'Nonfinite evidence: {value}')
    return value


def derive_selected(folder, row, params, candidates):
    selected = [int(row[f'selected_{key}']) for key in ('y', 'h', 'g')]
    matches = [c for c in candidates if [int(c[k]) for k in
               ('battery_cells', 'pv_units', 'garment_id')] == selected]
    if len(matches) != 1 or matches[0]['outcome'] != 'success':
        raise ValueError(f'{folder}: selected candidate missing/ambiguous/unaccepted')
    candidate = matches[0]
    if candidate['artifact'] != row['selected_artifact']:
        raise ValueError(f'{folder}: selected artifact mismatch')
    artifact = json.loads((folder / candidate['artifact']).read_text())
    metrics = artifact['evaluation']['metrics']
    trajectory = artifact['trajectory']
    if not artifact['evaluation']['success'] or artifact['configuration'] != selected:
        raise ValueError(f'{folder}: selected audit/configuration mismatch')
    steps, states = trajectory['steps'], trajectory['states']
    settings, scenario = params['config']['settings'], params['config']['scenario']
    dt = number(settings['dt_hours'])
    full_duration = sum(number(phase[1]) for phase in scenario['phases'])
    if (len(steps) != metrics['scheduled_steps'] or len(states) != len(steps) + 1
            or not math.isclose(len(steps)*dt, full_duration, abs_tol=1e-8)):
        raise ValueError(f'{folder}: selected trajectory is incomplete')
    floor_wh = full_wh = heater_wh = 0.0
    for i, step in enumerate(steps):
        t = number(step['time_hours'])
        if not (math.isclose(t, i*dt, abs_tol=1e-8)
                and math.isclose(number(states[i+1]['time_hours']), (i+1)*dt, abs_tol=1e-8)):
            raise ValueError(f'{folder}: interval/state alignment mismatch')
        end = 0.0
        for phase in scenario['phases']:
            end += number(phase[1])
            if t < end - 1e-9:
                alpha = number(phase[3])
                break
        else:
            raise ValueError(f'{folder}: no task phase at {t}')
        demand = number(step['demand_w'])
        supplied = number(step['total_effective_heat_w'])
        floor_wh += max(alpha*demand-supplied, 0.0)*dt
        full_wh += max(demand-supplied, 0.0)*dt
        heater_wh += (number(step['equipment_on'])*number(artifact['hardware']['equipment_power_w'])
                      *int(artifact['hardware']['equipment_available'])*dt)
    if not math.isclose(floor_wh, number(metrics['whole_body_unmet_heat_wh']), abs_tol=1e-7):
        raise ValueError(f'{folder}: historical floor-deficit field does not match reconstruction')
    solves = trajectory['solves']
    decision_times = [number(s['decision_seconds']) for s in solves]
    if len(solves) != len(steps) or not all(s['success'] for s in solves):
        raise ValueError(f'{folder}: selected solve trace mismatch')
    allocation_hours = metrics['allocation_deficit_duration_hours']
    installed = artifact['hardware']['mask']
    if len(allocation_hours) != len(installed) or any(
            (value is None) != (not bool(mask)) for value, mask in zip(allocation_hours, installed)):
        raise ValueError(f'{folder}: missing installed-node allocation evidence')
    node_hours = sum(number(value) for value, mask in zip(allocation_hours, installed) if mask)
    result = {
        'scenario': int(row['scenario_type']), 'date': row['date'],
        'artifact': str((folder / candidate['artifact']).as_posix()),
        'battery_units': selected[0], 'pv_units': selected[1], 'garment_id': selected[2],
        'mass_kg': number(candidate['mass_kg']), 'cost_usd': number(candidate['acquisition_cost']),
        'floor_deficit_wh': floor_wh, 'full_demand_deficit_wh': full_wh,
        'allocation_deficit_node_hours': node_hours,
        'equipment_energy_wh': heater_wh,
        'selected_decision_seconds_total': sum(decision_times),
        'selected_decision_seconds_mean': mean(decision_times),
        'selected_decision_seconds_max': max(decision_times),
    }
    for key in ('observed_duration_hours', 'mean_thermal_score_observed',
                'service_floor_shortfall_duration_hours', 'electrical_load_energy_wh',
                'pv_used_energy_wh', 'pv_curtailed_energy_wh', 'minimum_soc', 'terminal_soc',
                'low_soc_duration_hours', 'equipment_on_duration_hours', 'priority_audit_seconds'):
        result[key] = number(metrics[key])
    return result


def collect_primary(root):
    ledger = read_csv(root / 'date_report/date_ledger.csv')
    keys = [(r['scenario_type'], r['date']) for r in ledger]
    if len(set(keys)) != len(keys) or any(r['status'] not in STATUSES for r in ledger):
        raise ValueError('Duplicate date keys or unrecognized status in ledger')
    selected, failures, search_times, sources = [], [], [], {}
    for row in ledger:
        scenario, date = int(row['scenario_type']), row['date']
        folder = root / f's{scenario}_{date}'
        if row['status'] not in ('success', 'failed'):
            continue  # Retained explicitly in the date denominator and status table.
        params = json.loads((folder / 'parameters.json').read_text())
        summary = json.loads((folder / 'summary.json').read_text())
        candidates = read_csv(folder / 'candidates.csv')
        config = params['config']
        expected = len(config['garments'])*config['max_battery_cells']*(config['max_pv_units']+1)
        candidate_keys = {(c['battery_cells'], c['pv_units'], c['garment_id']) for c in candidates}
        if len(candidates) != expected or len(candidate_keys) != expected:
            raise ValueError(f'{folder}: incomplete/duplicate candidate domain')
        for file in (folder/'parameters.json', folder/'summary.json', folder/'candidates.csv'):
            sources[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
        times = {'scenario': scenario, 'date': date, 'status': row['status'], 'candidate_count': expected}
        times.update({k: number(v) for k,v in summary['timings'].items()})
        search_times.append(times)
        if row['status'] == 'success':
            if summary['selected'] != [int(row[f'selected_{k}']) for k in ('y','h','g')]:
                raise ValueError(f'{folder}: ledger/summary selection mismatch')
            item = derive_selected(folder, row, params, candidates)
            selected.append(item)
            file = Path(item['artifact'])
            sources[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
        else:
            if any(c['outcome'] != 'failed' for c in candidates):
                raise ValueError(f'{folder}: resolved failure has non-failed candidates')
            observed = [number(json.loads(c['metrics'])['observed_duration_hours']) for c in candidates]
            representative = min(candidates, key=lambda c: (
                -number(json.loads(c['metrics'])['observed_duration_hours']), number(c['mass_kg']),
                number(c['acquisition_cost']), int(c['battery_cells']), int(c['pv_units']), int(c['garment_id'])))
            m = json.loads(representative['metrics'])
            failures.append({'scenario': scenario, 'date': date, 'candidate_count': len(candidates),
                'min_observed_hours': min(observed), 'max_observed_hours': max(observed),
                'candidates_at_max': sum(math.isclose(v,max(observed),abs_tol=1e-8) for v in observed),
                'all_completion_failed': all(json.loads(c['components'])['C'] is False for c in candidates),
                'representative_tuple': [int(representative[k]) for k in ('battery_cells','pv_units','garment_id')],
                'representative_end_soc': number(m['observed_end_soc']),
                'representative_artifact': str(folder / representative['artifact'])})
    if len(selected) != sum(r['status']=='success' for r in ledger):
        raise ValueError('Selected denominator mismatch')
    return {'ledger':ledger, 'selected':selected, 'failed_dates':failures,
            'date_search_timings':search_times, 'source_sha256':sources}


def collect_catalogue(manifest_path):
    """Bind raw values and loader-derived masks/areas to frozen input hashes."""
    manifest_path = Path(manifest_path).resolve()
    manifest_bytes = manifest_path.read_bytes()
    parameters = json.loads(manifest_bytes)
    archived_root = Path(parameters['config_sources']['components']).parents[2]
    hashes, inputs = {}, {}
    for name, relative in INPUTS.items():
        if Path(parameters['config_sources'][name]) != archived_root / relative:
            raise ValueError(f'{manifest_path}: unexpected {name} source path')
    for relative in (*INPUTS.values(), 'src/inputs.py', 'src/configuration.py'):
        contents = (PROJECT / relative).read_bytes()
        digest = hashlib.sha256(contents).hexdigest()
        expected = parameters['source_sha256'].get(str(archived_root / relative))
        if expected is None or digest != expected:
            raise ValueError(f'{manifest_path}: frozen SHA-256 mismatch for {relative}')
        hashes[relative] = digest
        if relative in INPUTS.values():
            inputs[relative] = json.loads(contents)

    frozen = parameters['config']
    scenario = frozen['scenario']
    config = load_system_config(
        PROJECT / INPUTS['subject'], PROJECT / INPUTS['scenario_profiles'],
        PROJECT / INPUTS['components'], scenario['scenario_type'],
        datetime.fromisoformat(scenario['start']).date())
    loaded = {
        'person': asdict(config.person), 'device': asdict(config.device),
        'garments': [asdict(garment) for garment in config.garments],
    }
    # The manifest serializes tuples as arrays; compare the same JSON representation.
    for name, value in json.loads(json.dumps(loaded)).items():
        if value != frozen[name]:
            raise ValueError(f'{manifest_path}: frozen {name} catalogue mismatch')
    if (config.scenario.backpack_included != scenario['backpack_included'] or
            config.scenario.equipment_heater_required != scenario['equipment_heater_required']):
        raise ValueError(f'{manifest_path}: frozen backpack/heater flags mismatch')

    components = inputs[INPUTS['components']]
    profiles = inputs[INPUTS['scenario_profiles']]
    garments = sorted(config.garments, key=lambda garment: garment.id)
    if ([garment.id for garment in garments] != [0, 1, 2, 3] or
            not all(garment.mass_and_cost_include_heating_pads for garment in garments)):
        raise ValueError('Expected garment IDs 0--3 with pad-inclusive mass and cost')
    masks = {garment.id: garment.mask for garment in garments}
    person = config.person
    proxy_sum = sum(person.area_allocation_proxies)
    pads = {pad['location']: pad for pad in components['heating_pads']}
    areas = person.regional_areas_m2
    regional_inputs = [
        {
            'node_index': index + 1, 'region': region,
            'area_allocation_proxy': person.area_allocation_proxies[index],
            'normalized_area_share': person.area_allocation_proxies[index] / proxy_sum,
            'regional_area_m2': areas[index],
            'weight': person.weights[index],
            'reference_skin_temperature_c': person.skin_temperatures_c[index],
            'pad_id': pads[region]['id'], 'pad_type': pads[region]['pad_type'],
            'zone': pads[region]['zone'],
            'rated_power_w': config.device.rated_powers_w[index],
            'efficiency': config.device.efficiencies[index],
        }
        for index, region in enumerate(person.region_order)
    ]
    scenario_inclusion = [
        {
            'profile_key': key, 'scenario_type': profile['scenario_type'],
            'display_name': profile['display_name'],
            'backpack_included': profile['backpack']['included'],
            'equipment_heater_installed': profile['backpack']['equipment_heater_installed'],
            'max_pv_units': profile['backpack']['max_pv_units'],
        }
        for key, profile in sorted(profiles.items(), key=lambda item: item[1]['scenario_type'])
    ]
    return {
        'provenance': {
            'generator': 'scripts/report_evidence.py',
            'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'frozen_parameters': str(manifest_path),
            'frozen_parameters_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
            'verified_source_sha256': hashes,
            'verified_runtime_fields': list(loaded),
            'source_fields': {
                'subject': INPUTS['subject'],
                'catalogue': INPUTS['components'],
                'regional_inputs': INPUTS['components'] + '#/regional_model and #/heating_pads',
                'scenario_inclusion': INPUTS['scenario_profiles'] + '#/<profile>/backpack',
                'garment_masks': 'src/inputs.py:_component_objects',
                'body_area_m2': 'src/inputs.py:Person.body_area_m2',
                'regional_areas_m2': 'src/inputs.py:Person.regional_areas_m2',
                'objective_accounting': 'src/configuration.py:configuration_objectives and build_candidate',
            },
        },
        'subject': inputs[INPUTS['subject']],
        'system': components['system'],
        'battery_unit': components['battery_unit'],
        'photovoltaic_unit': components['photovoltaic_unit'],
        'garments': [
            dict(garment, mask=masks[garment['id']])
            for garment in sorted(components['garments'], key=lambda item: item['id'])
        ],
        'pad_types': components['pad_types'],
        'equipment_backpack': components['equipment_backpack'],
        'region_order': list(person.region_order),
        'regional_inputs': regional_inputs,
        'area_normalization': {
            'raw_proxy_sum': proxy_sum, 'body_area_m2': person.body_area_m2,
            'body_area_formula': '0.202 * weight_kg**0.425 * (height_cm/100)**0.725',
            'normalized_share_formula': 'area_allocation_proxy / sum(area_allocation_proxies)',
            'regional_area_formula': 'body_area_m2 * area_allocation_proxy / sum(area_allocation_proxies)',
            'interpretation': 'Adopted allocation proxies; not demonstrated JOS-3-derived anatomical areas.',
        },
        'scenario_inclusion': scenario_inclusion,
        'objective_accounting': {
            'mass_kg': 'base_electronics_mass_kg + garment.mass_kg + y*battery_unit.mass_kg + h*photovoltaic_unit.mass_kg',
            'cost_usd': 'base_electronics_cost_usd + garment.cost_usd + y*battery_unit.cost_usd + h*photovoltaic_unit.cost_usd',
            'y': 'battery-unit count', 'h': 'PV-unit count',
            'garment_mass_and_cost_include_heating_pads': True,
            'equipment_backpack_mass_and_cost_added_to_objectives': False,
            'separate_backpack_shell_mass_and_cost_declared': False,
            'separate_pad_cost_added_to_objectives': False,
            'interpretation': 'Declared component subtotals, not procurement validation or a complete carried-system bill of materials.',
        },
        'efficiency_interpretation': 'Assumed model heat-delivery factors; not measured body-side heat fractions.',
    }


def table(label, caption, body, notes):
    return '\n'.join([
        '% Generated by scripts.report_evidence; do not edit.',
        r'\begin{table}[pos=htbp]', r'\centering\small',
        r'\caption{' + caption + '}', r'\label{' + label + '}', body,
        r'\par\medskip', r'\begin{minipage}{\linewidth}\small', notes,
        r'\end{minipage}', r'\end{table}', '',
    ])


def render_regional_inputs(report):
    lines = [
        r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrr@{}}',
        r'\toprule',
        r'Node & $a_i$ & $a_i/\sum_j a_j$ & \shortstack{$P_i$\\(W)} & $\eta_i$ & $\omega_i$ & \shortstack{$T_{\mathrm{sk},i}^{\mathrm{ref}}$\\($^\circ$C)} \\',
        r'\midrule',
    ]
    for row in report['regional_inputs']:
        name = row['region'].replace('_', ' ').capitalize()
        lines.append(
            f"{row['node_index']}. {name} & {row['area_allocation_proxy']:g} & "
            f"{row['normalized_area_share']:.6f} & {row['rated_power_w']:g} & "
            f"{row['efficiency']:g} & {row['weight']:g} & "
            f"{row['reference_skin_temperature_c']:.2f}" + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular*}'])
    subject, area = report['subject'], report['area_normalization']
    notes = '\n'.join([
        f"The raw regional allocation weights sum to {area['raw_proxy_sum']:.12g}. "
        "They are divided by this sum to obtain unit-sum shares before allocating "
        "the whole-body surface area; thus, the modeled regional areas sum to the whole-body area.",
        r'$A=0.202\,m^{0.425}(H/100)^{0.725}$ and $A_i=Aa_i/\sum_j a_j$, with $m$ in kg and $H$ in cm.',
        f"The reference subject is {subject['sex']}, {subject['age_years']} years, "
        f"{subject['height_cm']:g} cm and {subject['weight_kg']:g} kg, giving $A\\approx {area['body_area_m2']:.6f}$ m$^2$.",
        r'The $a_i$ are adopted allocation proxies, not demonstrated JOS-3-derived anatomical areas.',
        r'The weights $\omega_i$ are separate from the normalized area shares; $P_i$ denotes pad electrical rating, and $\eta_i$ is an assumed model heat-delivery factor, not a measured body-side heat fraction.',
        r'Displayed shares are rounded to six decimals; the accompanying \texttt{model\_catalogue.json} retains unrounded loader-derived values and frozen-source hashes.',
    ])
    return table('tab:regional_inputs',
                 'Adopted regional area allocation, pad inputs and reference skin temperatures.',
                 '\n'.join(lines), notes)


def phase_floor(phases, time):
    """Use the same half-open phase convention as the frozen result collector."""
    end = 0.0
    for phase in phases:
        end += phase[1]
        if time < end - 1e-9:
            return phase[3]
    raise ValueError(f'No service phase at elapsed time {time}')


def candidate_records(folder):
    rows = read_csv(folder / 'candidates.csv')
    for row in rows:
        row['tuple'] = [int(row[k]) for k in ('battery_cells', 'pv_units', 'garment_id')]
        row['mass'] = float(row['mass_kg'])
        row['cost'] = float(row['acquisition_cost'])
    return rows


def pareto_records(rows):
    """Strict mass/cost dominance; retain distinct tuples at identical vectors."""
    accepted = [r for r in rows if r['outcome'] == 'success']
    best_cost = math.inf
    frontier = set()
    for mass, cost in sorted({(r['mass'], r['cost']) for r in accepted}):
        if cost < best_cost:
            frontier.add((mass, cost))
            best_cost = cost
    return [r for r in accepted if (r['mass'], r['cost']) in frontier]


def seasonal_comparison(domains, candidates, selected):
    """Intersect accepted tuples over every planned date, including empty days.

    References use the same retrospective inputs and policy as daily selection;
    they are not independently designed or out-of-sample static baselines.
    """
    results = []
    for scenario in sorted({day['scenario'] for day in domains}):
        days = [day for day in domains if day['scenario'] == scenario]
        common, catalogue = None, {}
        for day in days:
            rows = candidates[f"s{scenario}_{day['date']}"]
            accepted = set()
            for row in rows:
                key = tuple(row['tuple'])
                resources = (row['mass'], row['cost'])
                if key in catalogue and catalogue[key] != resources:
                    raise ValueError(f'Scenario {scenario}: resource catalogue varies by date')
                catalogue[key] = resources
                if row['outcome'] == 'success':
                    accepted.add(key)
            common = accepted if common is None else common & accepted
        daily = [row for row in selected if row['scenario'] == scenario]
        references = {}
        for ordering, columns in [('mass_first', (0, 1)), ('cost_first', (1, 0))]:
            if not common:
                references[ordering] = None
                continue
            key = min(common, key=lambda key: (
                catalogue[key][columns[0]], catalogue[key][columns[1]], key))
            mass, cost = catalogue[key]
            differences = []
            for row in daily:
                differences.append({
                    'date': row['date'],
                    'mass_saving_percent': 100 * (mass - row['mass_kg']) / mass,
                    'cost_saving_percent': 100 * (cost - row['cost_usd']) / cost})
            references[ordering] = {
                'tuple': list(key), 'mass_kg': mass, 'cost_usd': cost,
                'successful_dates': len(days), 'daily_differences': differences,
                'saving_summary': {
                    metric: {'mean': float(np.mean([row[metric] for row in differences])),
                             'quartiles': np.quantile(
                                 [row[metric] for row in differences], [.25, .5, .75]).tolist()}
                    for metric in ('mass_saving_percent', 'cost_saving_percent')}}
        ymax = max(key[0] for key in catalogue)
        hmax = max(key[1] for key in catalogue)
        results.append({
            'scenario': scenario, 'dates': [day['date'] for day in days],
            'common_accepted_count': len(common), 'references': references,
            'selected_dates': len(daily),
            'selected_mean_mass_kg': float(np.mean([r['mass_kg'] for r in daily])) if daily else None,
            'selected_mean_cost_usd': float(np.mean([r['cost_usd'] for r in daily])) if daily else None,
            'battery_limit': ymax, 'pv_limit': hmax, 'pv_enabled': hmax > 0,
            'battery_limit_count': sum(r['battery_units'] == ymax for r in daily),
            'pv_limit_count': sum(r['pv_units'] == hmax for r in daily) if hmax else None,
            'maximum_floor_deficit_wh': max((r['floor_deficit_wh'] for r in daily), default=None),
            'pv_used_to_load_ratio': (sum(r['pv_used_energy_wh'] for r in daily) /
                                      sum(r['electrical_load_energy_wh'] for r in daily)) if daily else None})
    return results


def minimum_body_load(required_heat, hardware):
    """Fractional fill by descending efficiency gives an electrical lower bound.

    Drops all regional priority, bus and energy constraints. Only use when
    the requested heat does not exceed total installed effective power.
    """
    remaining = required_heat
    load = 0.0
    nodes = sorted(zip(hardware['efficiencies'], hardware['rated_powers_w'],
                       hardware['mask']), reverse=True)
    for efficiency, power, mask in nodes:
        if not mask or efficiency <= 0:
            continue
        heat = min(remaining, efficiency * power)
        load += heat / efficiency
        remaining -= heat
    if remaining > 1e-7:
        raise ValueError('Thermal capacity violated before energy bound')
    return load


def stress_certificate(folder, record):
    """Necessary feasibility bounds on an archived solve, never a future replay."""
    config = json.loads((folder / 'parameters.json').read_text())['config']
    artifact_path, = (folder / 'runs').glob('*.json')
    artifact = json.loads(artifact_path.read_text())
    trajectory, hardware = artifact['trajectory'], artifact['hardware']
    settings, scenario = config['settings'], config['scenario']
    index = trajectory['failure_step'] if not trajectory['completed'] else 0
    forecast = trajectory['solves'][index]['forecast']
    dt = settings['dt_hours']
    floor = np.array([phase_floor(scenario['phases'], (index + j) * dt)
                      for j in range(len(forecast['demand_w']))])
    required = floor * forecast['demand_w']
    capacity = sum(e * p * m for e, p, m in zip(
        hardware['efficiencies'], hardware['rated_powers_w'], hardware['mask']))
    thermal_margin = float(np.min(capacity - required))
    result = {
        'scenario': record['scenario_type'], 'date': record['date'],
        'level': record['stress_label'], 'status': record['status'],
        'configuration': artifact['configuration'], 'solve_time_h': index * dt,
        'thermal_margin_w': thermal_margin, 'energy_margin_wh': None,
        'electrical_margin_w': None, 'energy_boundary_h': None,
        'energy_boundary_soc': None, 'certificate': 'thermal' if thermal_margin < -1e-7 else None,
        'max_heat_w': capacity, 'current_required_w': float(required[0]),
        'artifact': str(artifact_path),
    }
    if thermal_margin < -1e-7:
        return result
    # Replay only the deterministic thermostat on the stored forecast. No PWM
    # or SOC trajectory is fabricated beyond the observed failure prefix.
    history = list(scenario['initial_state']['air_history_c'])
    history.extend(artifact['actual']['air_c'][:index])
    previous = (trajectory['steps'][index - 1]['equipment_on'] if index else
                scenario['initial_state']['equipment_on'])
    loads = []
    for heat, air in zip(required, forecast['air_c']):
        history.append(air)
        previous, _ = equipment_switch(
            history, previous, hardware['equipment_available'],
            settings['equipment_on_c'], settings['equipment_off_c'],
            round(settings['filter_window_h'] / dt))
        loads.append(minimum_body_load(heat, hardware) + hardware['base_power_w']
                     + previous * hardware['equipment_power_w'])
    # Give the optimistic bound all forecast PV, even if curtailment/charging
    # limits would prevent using it. A negative bound still proves infeasibility.
    net_load = np.array(loads) - np.array(forecast['pv_w'])
    soc = trajectory['states'][index]['soc']
    lower_soc = np.full(len(loads), settings['soc_min'])
    if index + len(loads) == trajectory['scheduled_steps']:
        lower_soc[-1] = scenario['terminal_soc']
    margins = ((soc - lower_soc) * hardware['capacity_wh']
               - np.cumsum(net_load) * dt)
    worst = int(np.argmin(margins))
    electrical = float(np.min(np.minimum(
        hardware['bus_limit_w'] - np.array(loads),
        hardware['max_discharge_w'] - net_load)))
    result.update(energy_margin_wh=float(margins[worst]),
                  electrical_margin_w=electrical,
                  energy_boundary_h=(index + worst + 1) * dt,
                  energy_boundary_soc=float(lower_soc[worst]))
    result['certificate'] = ('energy' if margins[worst] < -1e-7 else
                             'electrical' if electrical < -1e-7 else 'not_identified')
    if record['status'] == 'success':
        if result['certificate'] != 'not_identified':
            raise ValueError('Necessary bound incorrectly excludes successful baseline')
        result['certificate'] = 'baseline'
    return result


def collect_evidence(root, stress_root):
    """Reuse complete-trajectory checks; then inspect candidate and solve records."""
    report = collect_primary(root)
    candidates, fronts, domains = {}, {}, []
    for row in report['ledger']:
        key = f"s{row['scenario_type']}_{row['date']}"
        rows = candidate_records(root / key)
        front = pareto_records(rows)
        candidates[key], fronts[key] = rows, front
        domains.append({'scenario': int(row['scenario_type']), 'date': row['date'],
                        'candidate_count': len(rows), 'accepted_count': sum(
                            r['outcome'] == 'success' for r in rows), 'pareto_count': len(front)})
        if row['status'] == 'success':
            chosen = min(front, key=lambda r: (r['mass'], r['cost'], r['tuple']))
            if chosen['tuple'] != [int(row[f'selected_{k}']) for k in ('y', 'h', 'g')]:
                raise ValueError(f'{key}: reconstructed resource selection differs')
    max_front = min(fronts, key=lambda key: (-len(fronts[key]), key))
    priority_rows, priority_cases = [], []
    for selected in report['selected']:
        artifact = json.loads(Path(selected['artifact']).read_text())
        tr = artifact['trajectory']
        key = f"s{selected['scenario']}_{selected['date']}"
        settings = json.loads((root / key / 'parameters.json').read_text())['config']['settings']
        safe = settings['soc_safe']
        mask = np.array(artifact['hardware']['mask'], dtype=bool)
        eligible_count = 0
        for index, (step, solve) in enumerate(zip(tr['steps'], tr['solves'])):
            targets, realized = np.array(solve['priority_targets']), np.array(solve['priority_realized'])
            pwm = np.array(step['pwm'])[mask]
            eligible = bool(step['demand_w'] > 0 and min(targets) < 1 - 1e-6
                            and np.any((pwm > 1e-6) & (pwm < 1 - 1e-6)))
            eligible_count += eligible
            shortfall = np.maximum(safe - np.array(solve['planned_start_soc']), 0)
            penalty = np.maximum.reduce([np.zeros_like(shortfall)] + [
                2 * b * shortfall - b * b for b in np.arange(1, 9) * safe / 8])
            priority_rows.append({
                'scenario': selected['scenario'], 'date': selected['date'], 'step': index,
                'time_h': step['time_hours'], 'eligible': eligible,
                'targets': targets.tolist(), 'realized': realized.tolist(),
                'lock_loss': float(np.max(targets - realized)),
                'first_action_weight': solve['deg_weights'][0],
                'forecast_below_safe': any(s < safe - 1e-8 for s in solve['planned_start_soc']),
                'minimum_planned_soc': min(solve['planned_start_soc']),
                'forecast_soc_penalty': float(settings['lambda_soc'] * penalty.sum()),
            })
        priority_cases.append((eligible_count, selected))
    representative = min(priority_cases, key=lambda item: (
        -item[0], item[1]['scenario'], item[1]['date']))[1]
    stress_ledger = json.loads((stress_root / 'stress_ledger.json').read_text())
    stress = [stress_certificate(stress_root / f"s{r['scenario_type']}_{r['date']}" /
                                 r['stress_label'], r) for r in stress_ledger]
    # The fixed [2,0,0] package is distinct from a fixed-garment strategy with
    # batteries reselected daily; preserve all 90 Harbin dates in this comparison.
    static_rows = []
    for day in domains:
        if day['scenario'] != 1:
            continue
        rows = candidates[f"s1_{day['date']}"]
        fixed = next(r for r in rows if r['tuple'] == [2, 0, 0])
        static_rows.append({'date': day['date'], 'fixed_2_0_0': fixed['outcome'],
                            'fleece_top_accepted': sum(r['tuple'][2] == 3 and
                                r['outcome'] == 'success' for r in rows)})
    evidence = {'failed_dates': report['failed_dates'], 'domains': domains, 'selected': report['selected'],
                'timings': report['date_search_timings'], 'static_harbin': static_rows,
                'fronts': {k: [{'tuple': r['tuple'], 'mass': r['mass'], 'cost': r['cost']}
                               for r in v] for k, v in fronts.items()},
                'max_front_date': max_front, 'priority_representative': representative,
                'priority_steps': len(priority_rows),
                'priority_eligible_steps': sum(r['eligible'] for r in priority_rows),
                'priority_max_lock_loss': max(r['lock_loss'] for r in priority_rows),
                'first_action_weight_counts': dict(Counter(r['first_action_weight'] for r in priority_rows)),
                'forecast_below_safe_solves': sum(r['forecast_below_safe'] for r in priority_rows),
                'stress': stress, 'source_sha256': report['source_sha256']}
    evidence['seasonal_comparison'] = seasonal_comparison(domains, candidates, report['selected'])
    for path in [stress_root / 'stress_ledger.json', root / 'date_report/date_ledger.csv']:
        evidence['source_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for case in stress:
        for path in [Path(case['artifact']), Path(case['artifact']).parents[1] / 'parameters.json']:
            evidence['source_sha256'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return evidence, candidates, priority_rows


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def mission_statistics(config, weather):
    """Summarize the reader's complete, uniform [start, end) mission grid.

    Equal-duration interval means have an arithmetic time-weighted mean. The
    min/max describe model-input intervals, not subinterval instantaneous peaks.
    Harbin's required cached PV is read by the weather API but not summarized.
    """
    scenario = config.scenario
    dt_hours = config.settings['dt_hours']
    times = weather['times']
    count = len(times)
    duration = scenario.get_duration_hours()
    if (count == 0 or count != config.settings['horizon_steps'] or
            times[0] != scenario.start or
            not np.isclose(count * dt_hours, duration, rtol=0, atol=1e-9)):
        raise ValueError('Weather grid does not cover the configured mission.')
    air = np.asarray(weather['air_c'], dtype=float)
    if air.shape != (count,) or not np.all(np.isfinite(air)):
        raise ValueError('Mission temperature must contain one finite value per interval.')
    record = {
        'scenario_type': scenario.scenario_type,
        'date': scenario.start.strftime('%Y%m%d'),
        'mission_start': times[0],
        'mission_end': times[-1] + timedelta(hours=dt_hours),
        'timezone': scenario.timezone,
        'interval_count': count,
        'dt_hours': dt_hours,
        'duration_hours': duration,
        'temperature_min_c': float(np.min(air)),
        'temperature_mean_c': float(np.mean(air)),
        'temperature_max_c': float(np.max(air)),
        'pv_unit_rating_w': None,
        'pv_reference_rating_w': None,
        'pv_scale': None,
        'pv_min_w': None,
        'pv_mean_w': None,
        'pv_max_w': None,
    }
    if scenario.scenario_type != 1:
        reference_pv = np.asarray(weather['reference_pv_w'], dtype=float)
        if (reference_pv.shape != (count,) or
                not np.all(np.isfinite(reference_pv)) or np.any(reference_pv < 0)):
            raise ValueError('Mission PV must contain one finite nonnegative value per interval.')
        scale = config.device.pv_rating_w / scenario.pv_reference_rating_w
        unit_pv = reference_pv * scale
        record.update(
            pv_unit_rating_w=config.device.pv_rating_w,
            pv_reference_rating_w=scenario.pv_reference_rating_w,
            pv_scale=scale,
            pv_min_w=float(np.min(unit_pv)),
            pv_mean_w=float(np.mean(unit_pv)),
            pv_max_w=float(np.max(unit_pv)),
        )
    return record


def verify_hashes(parameters, paths, cache, archive):
    """Map recorded project paths without changing the recorded identity."""
    archived_root = Path(parameters['config_sources']['components']).parents[2]
    for path in paths:
        path = Path(path).resolve()
        relative = path.relative_to(PROJECT)
        expected = parameters['source_sha256'].get(str(archived_root / relative))
        if expected is None:
            raise ValueError(f'{archive}: missing recorded SHA-256 for {relative}')
        if path not in cache:
            cache[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        actual = cache[path]
        transition = {'src/weather.py': WEATHER_SOURCE_TRANSITION,
                      'src/operation.py': OPERATION_SOURCE_TRANSITION}.get(relative.as_posix())
        if actual != expected and (expected, actual) != transition:
            raise ValueError(f'{archive}: frozen SHA-256 mismatch for {relative}')


def collect_forcing(root):
    """Verify all dates, immutable inputs and retained actual forcing offline."""
    rows = read_csv(root / 'date_report/date_ledger.csv')
    expected = {(scenario, day) for scenario, days in RANGES.items() for day in days}
    counts = Counter((int(row['scenario_type']), row['date']) for row in rows)
    if set(counts) != expected or any(count != 1 for count in counts.values()):
        raise ValueError('Frozen ledger date coverage is incomplete or duplicated')
    if any(row['status'] not in ('success', 'failed', 'unresolved') for row in rows):
        raise ValueError('Frozen ledger contains an unrun or invalid date')
    source_files = tuple((PROJECT / 'src').glob('*.py'))
    cache, forcing, garments, checks = {}, [], None, []
    row_by_key = {(int(row['scenario_type']), row['date']): row for row in rows}
    for scenario, days in RANGES.items():
        for day in days:
            archive = root / f's{scenario}_{day}'
            parameters = json.loads((archive / 'parameters.json').read_text())
            archived_root = Path(parameters['config_sources']['components']).parents[2]
            sources = {key: PROJECT / Path(value).relative_to(archived_root)
                       for key, value in parameters['config_sources'].items()}
            verify_hashes(parameters, (*source_files, *sources.values()), cache, archive)
            config = load_system_config(sources['subject'], sources['scenario_profiles'],
                                        sources['components'], scenario, day)
            recorded = parameters['config']
            if (recorded['scenario']['scenario_type'] != scenario or
                    recorded['scenario']['start'] != config.scenario.start.isoformat() or
                    recorded['settings']['dt_hours'] != config.settings['dt_hours']):
                raise ValueError(f'{archive}: recorded scenario/date/mission grid disagrees')
            names = [garment.name for garment in config.garments]
            if (names != [garment['name'] for garment in recorded['garments']] or
                    (garments is not None and names != garments)):
                raise ValueError(f'{archive}: frozen garment category order disagrees')
            garments = names
            # The verified loader owns the installed project's weather root;
            # archived absolute keys are mapped only when checking their hashes.
            weather = read_scenario_weather(config.scenario, config.settings['dt_hours'])
            verify_hashes(parameters, weather['source_files'], cache, archive)
            if (weather['semantics'] != parameters['weather_semantics'] or
                    weather['sample_hours'] != parameters['weather_sample_hours']):
                raise ValueError(f'{archive}: recorded weather alignment disagrees')
            row = row_by_key[(scenario, day)]
            if row['status'] == 'success':
                artifact_path = archive / row['selected_artifact']
            else:
                artifact_path = next(iter(sorted((archive / 'runs').glob('*.json'))), None)
                if artifact_path is None:
                    raise ValueError(f'{archive}: no retained forcing witness')
            artifact = json.loads(artifact_path.read_text())
            # Use the frozen multiplication order exactly, including zero-PV runs.
            pv = artifact['configuration'][1] * (weather['reference_pv_w'] * (
                config.device.pv_rating_w / config.scenario.pv_reference_rating_w))
            if (not np.array_equal(weather['air_c'], artifact['actual']['air_c']) or
                    not np.array_equal(pv, artifact['actual']['pv_w'])):
                raise ValueError(f'{archive}: offline reader differs from archived actual forcing')
            record = mission_statistics(config, weather)
            record['mission_start'] = record['mission_start'].isoformat()
            record['mission_end'] = record['mission_end'].isoformat()
            forcing.append(record)
            checks.append({'scenario': scenario, 'date': day, 'artifact': str(artifact_path),
                           'artifact_sha256': hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
                           'air_and_selected_pv_exact': True})
    return {'ledger': rows, 'garments': garments, 'forcing': forcing,
            'reader_checks': checks,
            'verified_analysis_sha256': {str(path.relative_to(PROJECT)): digest
                                         for path, digest in cache.items()},
            'weather_source_transition': {'frozen_sha256': WEATHER_SOURCE_TRANSITION[0],
                                          'analysis_sha256': WEATHER_SOURCE_TRANSITION[1]},
            'operation_source_transition': {'frozen_sha256': OPERATION_SOURCE_TRANSITION[0],
                                            'analysis_sha256': OPERATION_SOURCE_TRANSITION[1]}}


def numeric_macros(evidence):
    counts = Counter(row['scenario'] for row in evidence['selected'])
    failed = evidence['failed_dates']
    macros = {'PlannedDates': len(evidence['domains']), 'SelectedDates': len(evidence['selected']),
            'HarbinSelected': counts[1], 'MuztaghSelected': counts[2], 'QinlingSelected': counts[3],
            'FailedDates': len(failed),
            'ShortestBestPrefix': f"{min(row['max_observed_hours'] for row in failed):.0f}",
            'LongestBestPrefix': f"{max(row['max_observed_hours'] for row in failed):.0f}"}
    harbin = next(row for row in evidence['seasonal_comparison'] if row['scenario'] == 1)
    reference = harbin['references']['mass_first']
    savings = reference['saving_summary']
    macros.update(
        HarbinFixedSuccess=sum(row['fixed_2_0_0'] == 'success' for row in evidence['static_harbin']),
        HarbinSeasonMass=f"{reference['mass_kg']:.2f}",
        HarbinSeasonCost=f"{reference['cost_usd']:.0f}",
        HarbinMeanMassSaving=f"{savings['mass_saving_percent']['mean']:.2f}",
        HarbinMeanCostSaving=f"{savings['cost_saving_percent']['mean']:.2f}")
    return macros


def operation_records(evidence, root):
    selected = evidence['priority_representative']
    artifact = json.loads(Path(selected['artifact']).read_text())
    key = f"s{selected['scenario']}_{selected['date']}"
    config = json.loads((root / key / 'parameters.json').read_text())['config']
    trajectory = artifact['trajectory']
    return {'dt_hours': config['settings']['dt_hours'],
            'floor': [phase_floor(config['scenario']['phases'], step['time_hours'])
                      for step in trajectory['steps']],
            'steps': trajectory['steps'], 'states': trajectory['states'],
            'solves': [{key: solve[key] for key in
                        ('priority_group_labels', 'priority_realized', 'priority_targets', 'deg_weights')}
                       for solve in trajectory['solves']]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT / 'output/frozen_20261004')
    parser.add_argument('--stress-root', type=Path, default=PROJECT / 'output/stress_20261006')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='A new directory; existing output is never overwritten.')
    parser.add_argument('--tex-dir', type=Path, help='Optional author-only table destination.')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('--output-dir must not already exist')
    protected = (args.root.resolve(), args.stress_root.resolve(),
                 PROJECT / 'src', PROJECT / 'data')
    for destination in (args.output_dir, args.tex_dir):
        if destination and any(destination.resolve().is_relative_to(path) for path in protected):
            parser.error('Output destinations must be outside scientific inputs and frozen archives')
    forcing = collect_forcing(args.root)
    evidence, candidates, priority = collect_evidence(args.root, args.stress_root)
    catalogue = collect_catalogue(args.root / 's1_20260228/parameters.json')
    evidence.update(schema_version=1, configuration=forcing, priority_records=priority,
                    operation=operation_records(evidence, args.root))
    cases = ['s1_20251201', 's1_20260210', evidence['max_front_date']]
    evidence['candidate_points'] = {key: [{field: row[field] for field in
        ('tuple', 'mass', 'cost', 'outcome')} for row in candidates[key]] for key in cases}
    evidence['numeric_macros'] = numeric_macros(evidence)
    evidence['analysis'] = {'generator': 'scripts/report_evidence.py',
                           'sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output_dir.mkdir(parents=True)
    for filename, value in [('evidence.json', evidence), ('model_catalogue.json', catalogue)]:
        (args.output_dir / filename).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    for filename, rows in [('pareto_domains', evidence['domains']),
                           ('selected_joint_metrics', evidence['selected']),
                           ('priority_steps', priority), ('stress_certificates', evidence['stress']),
                           ('failed_dates', evidence['failed_dates']),
                           ('date_search_timings', evidence['timings'])]:
        write_csv(args.output_dir / f'{filename}.csv', rows)
    if args.tex_dir:
        args.tex_dir.mkdir(parents=True, exist_ok=True)
        macros = '% Generated by scripts.report_evidence; do not edit.\n'
        macros += ''.join(f'\\newcommand{{\\{key}}}{{{value}}}\n'
                          for key, value in evidence['numeric_macros'].items())
        (args.tex_dir / 'frozen_result_numbers.tex').write_text(macros)
        (args.tex_dir / 'regional_inputs.tex').write_text(render_regional_inputs(catalogue))
    print(f"Verified {len(forcing['forcing'])} mission grids; derived "
          f"{len(evidence['selected'])} selections; stress certificates: "
          f"{dict(Counter(row['certificate'] for row in evidence['stress']))}")
    print(f'Wrote self-contained evidence to {args.output_dir}')


if __name__ == '__main__':
    main()
