'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Evaluate rental cashflows and finite indoor/HVAC states from completed
recorded missions. Never selects hardware, simulates controllers or fetches weather.
'''

import argparse
import hashlib
from itertools import product
import json
import math
from pathlib import Path

import numpy as np

from src.evaluation import evaluate_trajectory
from src.run import write_json


def cashflow(capex, annual_net_usd, discount_rate):
    """Eqs. (21)--(22): net cashflow, NPV and within-year payback interpolation."""
    values = [float(v) for v in annual_net_usd]
    if not values or any(not math.isfinite(v) for v in [capex, discount_rate, *values]):
        raise ValueError('Cashflows must be finite and contain at least one year.')
    if capex < 0 or discount_rate < 0:
        raise ValueError('CAPEX and discount rate must be nonnegative.')
    net_balance = discounted_balance = -float(capex)
    payback = discounted_payback = 0.0 if capex == 0 else None
    years = []
    for year, net in enumerate(values, 1):
        present = net / (1 + discount_rate) ** year
        if payback is None and net > 0 and net_balance < 0 <= net_balance + net:
            payback = year - 1 - net_balance / net
        if discounted_payback is None and present > 0 and discounted_balance < 0 <= discounted_balance + present:
            discounted_payback = year - 1 - discounted_balance / present
        net_balance += net
        discounted_balance += present
        years.append({'year': year, 'net_cashflow_usd': net, 'present_value_usd': present,
                      'cumulative_net_usd': net_balance, 'cumulative_discounted_usd': discounted_balance})
    return {'npv_usd': discounted_balance, 'net_payback_years': payback,
            'discounted_payback_years': discounted_payback, 'annual_cashflows': years}


def recharge_energy_kwh(record, charging_efficiency):
    """Close the repeated-mission SOC cycle, counting utilized PV exactly once."""
    if not 0 < charging_efficiency <= 1:
        raise ValueError('Charging efficiency must lie in (0, 1].')
    audited = evaluate_trajectory(record['run'], record['actual'], record['initial_soc'],
                                  record['hardware'], record['settings'],
                                  **record['evaluation_tolerances'])
    if not audited['valid'] or not audited['mission_completed']:
        raise ValueError('Application accounting requires an independently valid complete mission.')
    metrics = audited['metrics']
    depletion_wh = metrics['battery_discharge_wh'] - metrics['battery_charge_wh']
    return max(0.0, depletion_wh) / (1000 * charging_efficiency), metrics


def economic_sensitivity(record, assumptions):
    scenario = assumptions['economics'][str(record['scenario_type'])]
    energy, metrics = recharge_energy_kwh(record, assumptions['charging_efficiency'])
    capex = record['candidate']['cost_usd']

    def one(price_factor, utilization_factor, maintenance_factor, lifetime, discount):
        price = scenario['rental_usd_per_day'] * price_factor
        days = scenario['utilization_days_per_year'] * utilization_factor
        maintenance = scenario['annual_maintenance_usd'] * maintenance_factor
        electricity = energy * days * assumptions['electricity_price_usd_per_kwh']
        revenue = price * days
        return {'price_factor': price_factor, 'utilization_factor': utilization_factor,
                'maintenance_factor': maintenance_factor, 'lifetime_years': lifetime,
                'discount_rate': discount, 'rental_usd_per_day': price,
                'utilization_days_per_year': days, 'annual_maintenance_usd': maintenance,
                'annual_electricity_usd': electricity, 'annual_revenue_usd': revenue,
                'gross_revenue_to_capex_years': capex / revenue if revenue > 0 else None,
                **cashflow(capex, [revenue - maintenance - electricity] * lifetime, discount)}

    factors = assumptions['sensitivity_factors']
    rows = [one(p, u, m, life, assumptions['discount_rate'])
            for p, u, m, life in product(factors, factors, factors, assumptions['lifetime_sensitivity_years'])]
    return {'scenario_type': record['scenario_type'], 'configuration': record['configuration'],
            'capex_usd': capex, 'mission_grid_recharge_kwh': energy,
            'mission_metrics': metrics,
            'baseline': one(1., 1., 1., assumptions['lifetime_years'], assumptions['discount_rate']),
            'joint_sensitivity': rows,
            'discount_sensitivity': [one(1., 1., 1., assumptions['lifetime_years'], rate)
                                     for rate in assumptions['discount_sensitivity']]}


def building_state(record, assumptions, *, heat_loss_scale=1., cop_scale=1.,
                   occupancy_scale=1., outdoor_c=None):
    """Eqs. (24)--(27): 24-h building energy plus mission-cycle grid recharge."""
    b = assumptions['building']
    outside = b['outdoor_c'] if outdoor_c is None else outdoor_c
    lower, upper = b['cop_valid_temperature_c']
    cop = b['cop_rated'] * (1 - b['cop_alpha_per_k2'] * max(0., b['cop_rated_temperature_c'] - outside) ** 2) * cop_scale
    if not lower <= outside <= upper or not math.isfinite(cop) or cop <= 0:
        return {'status': 'outside_cop_model_domain', 'outdoor_c': outside, 'cop': cop}
    if min(heat_loss_scale, occupancy_scale) <= 0 or b['wearable_heat_credit_fraction'] != 0:
        raise ValueError('Positive building scales and the declared zero heat-credit boundary are required.')
    recharge, metrics = recharge_energy_kwh(record, assumptions['charging_efficiency'])
    duration = metrics['scheduled_duration_h']
    if duration > b['day_hours'] or record['configuration'][1] != 0:
        raise ValueError('Building accounting requires a within-day mission and the declared no-PV package.')
    dt = record['settings']['dt_hours']
    people = b['occupants'] * occupancy_scale
    gain = people * b['metabolic_gain_w_per_person'] / 1000
    ua = b['heat_loss_kw_per_k'] * heat_loss_scale
    task_air = np.asarray(record['actual']['air_c'], dtype=float)
    # Both cases use identical occupancy and off-task policy. The actual indoor
    # waveform drives task-period heat loss; no second credit for wearable heat.
    task_hvac = float(np.maximum(0., ua * (task_air - outside) - gain).sum() * dt / cop)
    off_hvac = max(0., ua * (b['nonmission_setpoint_c'] - outside) - gain) * (b['day_hours'] - duration) / cop
    task_reference_air = task_air + b['baseline_setpoint_c'] - record['experiment']['setpoint_c']
    reference = float(np.maximum(0., ua * (task_reference_air - outside) - gain).sum() * dt / cop) + off_hvac
    wearable = people * recharge
    total = task_hvac + off_hvac + wearable
    saving = reference - total
    return {'status': 'computed', 'setpoint_c': record['experiment']['setpoint_c'],
            'outdoor_c': outside, 'cop': cop, 'occupants': people,
            'heat_loss_kw_per_k': ua, 'wearable_duration_h': duration,
            'building_duration_h': b['day_hours'], 'hvac_kwh': task_hvac + off_hvac,
            'wearable_grid_kwh': wearable, 'total_kwh': total, 'reference_kwh': reference,
            'saving_kwh': saving, 'saving_percent': 100 * saving / reference if reference > 0 else None,
            'setpoint_only_saving_kwh': reference - task_hvac - off_hvac,
            'incremental_active_heating_kwh': metrics['body_heating_energy_wh'] * people / 1000,
            'mission_metrics': metrics,
            'interpretation': 'Building-electricity assessment using mission recharge and specified indoor-temperature profiles.'}


def evaluate(unperturbed_dir, indoor_dir, assumptions_path, output_dir):
    assumptions_path = Path(assumptions_path)
    assumptions = json.loads(assumptions_path.read_text())
    if assumptions.get('schema') != 'bgems-applications-v1':
        raise ValueError('Expected the specified application assumption schema.')
    sources = {str(assumptions_path): hashlib.sha256(assumptions_path.read_bytes()).hexdigest()}

    def load(path):
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())

    economic = []
    for path in sorted(Path(unperturbed_dir).glob('*/rhc/run.json')):
        record = load(path)
        if record['scenario_type'] in (1, 2):
            economic.append(economic_sensitivity(record, assumptions))
    if sorted(r['scenario_type'] for r in economic) != [1, 2]:
        raise ValueError('Expected exactly one unperturbed record for each commercial scenario.')
    building = []
    for setpoint in assumptions['building']['setpoints_c']:
        record = load(Path(indoor_dir) / f'setpoint_{setpoint:g}' / 'rhc' / 'run.json')
        for arm in assumptions['building']['single_factor_arms']:
            kwargs = {k: v for k, v in arm.items() if k != 'name'}
            building.append({'arm': arm['name'], **building_state(record, assumptions, **kwargs)})
    result = {'schema': 'bgems-applications-v1', 'assumptions': assumptions,
              'economics': economic, 'building_states': building,
              'source_files_sha256': sources, 'optimizer_calls': 0}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'applications.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unperturbed-dir', type=Path, required=True)
    parser.add_argument('--indoor-dir', type=Path, required=True)
    parser.add_argument('--assumptions', type=Path, default=Path('data/systemconf/applications.json'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    result = evaluate(args.unperturbed_dir, args.indoor_dir, args.assumptions, args.output_dir)
    print(f"Path A: {sum(len(r['joint_sensitivity']) for r in result['economics'])} joint cases; "
          f"Path B: {len(result['building_states'])} states; optimizer calls: 0")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
