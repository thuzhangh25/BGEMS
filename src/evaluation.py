'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Reconstruct electrical/thermal metrics from realized actions and states.
No controller, optimizer, configuration search or weather acquisition is invoked.
'''

import math

import numpy as np

from .modeling import regional_fulfillment


def evaluate_trajectory(
    run,
    actual,
    initial_soc,
    hardware,
    settings,
    *,
    balance_tolerance_wh,
    time_tolerance_hours,
):
    """Check a complete trajectory or its actual prefix, without re-solving it."""
    issues = []
    actions = run.get('actions', [])
    states = run.get('states', [])
    dt = settings['dt_hours']
    n = len(actual['demand_w'])
    tol = settings['constraint_tolerance']

    def check(condition, message):
        if not condition:
            issues.append(message)

    check(run.get('scheduled_steps') == n, 'Scheduled-step mismatch.')
    check(run.get('dt_hours') == dt, 'Timestep mismatch.')
    check(
        len(actions) <= n and len(states) == len(actions) + 1,
        'Action/state lengths do not describe a prefix.',
    )
    complete = len(actions) == n
    check(
        run.get('completed') is complete,
        'Completion flag disagrees with observed intervals.',
    )
    check(
        (run.get('failure') is None) == complete,
        'Failure record disagrees with completion.',
    )
    if run.get('failure') is not None:
        failure = run['failure']
        check(
            failure.get('kind')
            in ('policy_failure', 'physical_infeasibility', 'numerical_failure',
                'reserve_domain_rejection'),
            'Unknown failure kind.',
        )
        check(
            failure.get('step') == len(actions),
            'Failure step does not terminate the prefix.',
        )
        check(
            abs(failure.get('time_h', math.inf) - len(actions) * dt)
            <= time_tolerance_hours,
            'Failure time mismatch.',
        )
    capacity = hardware['capacity_wh']
    voltage = hardware['voltage_v']
    socs = []
    scores = []
    deficits = []
    loads = []
    body_energy = []
    equipment_energy = []
    used_pv = []
    curtailed = []
    discharged = []
    charged = []
    penalties = []
    reserve_curtailment = []
    physical_curtailment = []
    reserve_margins = []
    mandatory_max = hardware['base_power_w'] + (
        hardware['equipment_power_w'] if hardware['equipment_available'] else 0.0
    )
    domain_ok = (
        (initial_soc - settings['soc_min']) * capacity + tol * capacity >= mandatory_max * n * dt
        and mandatory_max <= min(hardware['bus_limit_w'], hardware['max_discharge_w'])
    )
    rejected = (run.get('failure') or {}).get('kind') == 'reserve_domain_rejection'
    check(run.get('reserve_policy') == 'remaining_mandatory_load', 'Unknown reserve policy.')
    check(rejected == (not domain_ok), 'Reserve-domain acceptance mismatch.')
    if rejected:
        check(not actions, 'Reserve-domain rejection must precede execution.')
    for k, state in enumerate(states):
        value = state.get('soc', math.nan)
        when = state.get('time_h', math.nan)
        check(
            math.isfinite(value) and settings['soc_min'] - tol <= value <= 1 + tol,
            'SOC violates the declared interval.',
        )
        check(
            math.isfinite(when) and abs(when - k * dt) <= time_tolerance_hours,
            'State time mismatch.',
        )
        socs.append(value)
        margin = (value - settings['soc_min']) * capacity - mandatory_max * max(0, n - k) * dt
        reserve_margins.append(margin)
        if domain_ok:
            check(margin >= -tol * capacity, 'Remaining mandatory reserve violated.')
    if states:
        check(abs(socs[0] - initial_soc) <= tol, 'Initial SOC mismatch.')
    for k, row in enumerate(actions[:n]):
        if k + 1 >= len(states):
            break
        try:
            pwm = np.asarray(row['pwm'], dtype=float)
            command = np.asarray(row['command_pwm'], dtype=float)
            if any(
                v.shape != (12,)
                or not np.all(np.isfinite(v))
                or np.any((v < 0) | (v > 1))
                for v in (pwm, command)
            ):
                issues.append('Invalid executed or commanded PWM.')
                break
            check(
                np.all(np.abs(pwm[np.asarray(hardware['mask']) == 0]) <= tol),
                'Uninstalled pad was executed.',
            )
            check(
                np.all(pwm <= command + tol),
                'Actuator added body heating beyond the command.',
            )
            check(
                abs(row['time_h'] - k * dt) <= time_tolerance_hours,
                'Action time mismatch.',
            )
            demand = float(actual['demand_w'][k])
            air = float(actual['air_c'][k])
            pv = float(actual['pv_w'][k])
            for key in ('demand_w', 'air_c', 'pv_w', 'met'):
                check(
                    math.isfinite(row[key])
                    and abs(row[key] - float(actual[key][k])) <= tol,
                    f'{key} differs from realized input.',
                )
            on = int(bool(hardware['equipment_available']) and air <= -20.0)
            check(
                row['equipment_on'] == on,
                'Equipment command violates installation/threshold rule.',
            )
            body = float(
                np.dot(hardware['rated_powers_w'], pwm * np.asarray(hardware['mask']))
            )
            equipment = on * hardware['equipment_power_w']
            load = body + equipment + hardware['base_power_w']
            used = float(row['pv_used_w'])
            battery = load - used
            check(
                math.isfinite(used) and -tol <= used <= pv + tol,
                'PV use exceeds available power.',
            )
            check(load <= hardware['bus_limit_w'] + tol, 'Bus power limit violated.')
            check(
                -hardware['max_charge_w'] - tol
                <= battery
                <= hardware['max_discharge_w'] + tol,
                'Battery power limit violated.',
            )
            expected = socs[k] - battery * dt / capacity
            check(
                abs((socs[k + 1] - expected) * capacity) <= balance_tolerance_wh,
                'Electrical energy balance violated.',
            )
            reserve_after = mandatory_max * (n - k - 1) * dt
            commanded = float(np.dot(
                hardware['rated_powers_w'], command * np.asarray(hardware['mask'])
            ))
            physical_limit = max(0.0, min(
                hardware['bus_limit_w'],
                pv + min(hardware['max_discharge_w'],
                         max(0.0, (socs[k] - settings['soc_min']) * capacity / dt)),
            ) - equipment - hardware['base_power_w'])
            physical_body = min(commanded, physical_limit)
            reserve_limit = max(
                0.0, pv - equipment - hardware['base_power_w']
                + ((socs[k] - settings['soc_min']) * capacity - reserve_after) / dt,
            )
            expected_body = min(physical_body, reserve_limit)
            check(abs(body - expected_body) * dt <= balance_tolerance_wh,
                  'Common actuator physical/reserve limit mismatch.')
            for key, value in (
                ('reserve_required_after_wh', reserve_after),
                ('reserve_margin_after_wh', reserve_margins[k + 1]),
                ('physical_curtailed_body_w', commanded - physical_body),
                ('reserve_curtailed_body_w', physical_body - expected_body),
            ):
                check(abs(row.get(key, math.inf) - value) <= max(tol * capacity, balance_tolerance_wh / dt),
                      f'Reported {key} mismatch.')
            physical_curtailment.append((commanded - physical_body) * dt)
            reserve_curtailment.append((physical_body - expected_body) * dt)
            for key, value in [
                ('load_w', load),
                ('battery_w', battery),
                ('curtailed_pv_w', pv - used),
            ]:
                check(
                    math.isfinite(row[key])
                    and abs(row[key] - value) * dt <= balance_tolerance_wh,
                    f'Reported {key} mismatch.',
                )
            delivered = regional_fulfillment(
                demand,
                hardware['regional_areas_m2'],
                hardware['weights'],
                hardware['efficiencies'],
                hardware['rated_powers_w'],
                hardware['mask'],
                pwm,
            )
            check(
                abs(row['score'] - delivered['score']) <= tol,
                'Reported regional score mismatch.',
            )
            check(
                np.allclose(
                    row['regional_fulfillment'],
                    delivered['regional_fulfillment'],
                    rtol=0,
                    atol=tol,
                ),
                'Reported regional fulfillment mismatch.',
            )
            ah = max(battery, 0.0) * dt / voltage
            penalty = (1.0 if socs[k] < 0.2 else 0.5 if socs[k] < 0.5 else 0.2) * ah
            check(
                abs(row['discharged_ah'] - ah) <= tol,
                'Interval discharge charge mismatch.',
            )
            check(
                abs(row['degradation_penalty'] - penalty) <= tol,
                'Interval degradation penalty mismatch.',
            )
            scores.append(delivered['score'])
            deficits.append(float(delivered['regional_unmet_heat_w'].sum()) * dt)
            loads.append(load * dt)
            body_energy.append(body * dt)
            equipment_energy.append(equipment * dt)
            used_pv.append(used * dt)
            curtailed.append((pv - used) * dt)
            discharged.append(max(battery, 0.0) * dt)
            charged.append(max(-battery, 0.0) * dt)
            penalties.append(penalty)
        except (KeyError, ValueError, TypeError, IndexError) as error:
            issues.append(f'Malformed trajectory row {k}: {error}')
            break
    valid = not issues
    metrics = {
        'scheduled_duration_h': n * dt,
        'completed_duration_h': len(actions) * dt,
        'completion_fraction': len(actions) / n if n else 0.0,
        'mean_service_full_mission': (
            float(np.mean(scores)) if valid and complete else None
        ),
        'mean_service_completed_prefix': float(np.mean(scores)) if scores else None,
        'observed_service_integral_h': float(sum(scores) * dt),
        'unmet_regional_heat_wh': float(sum(deficits)),
        'load_energy_wh': float(sum(loads)),
        'body_heating_energy_wh': float(sum(body_energy)),
        'equipment_energy_wh': float(sum(equipment_energy)),
        'pv_used_wh': float(sum(used_pv)),
        'pv_curtailed_wh': float(sum(curtailed)),
        'battery_discharge_wh': float(sum(discharged)),
        'battery_charge_wh': float(sum(charged)),
        'discharge_penalty_ah': float(sum(penalties)),
        'min_soc': float(min(socs)) if socs else None,
        'last_observed_soc': float(socs[-1]) if socs else None,
        'terminal_soc': float(socs[-1]) if valid and complete else None,
        'reserve_domain_admissible': domain_ok,
        'min_reserve_margin_wh': float(min(reserve_margins)) if reserve_margins else None,
        'physical_curtailed_body_energy_wh': float(sum(physical_curtailment)),
        'reserve_curtailed_body_energy_wh': float(sum(reserve_curtailment)),
        'reserve_limited_duration_h': sum(v > balance_tolerance_wh for v in reserve_curtailment) * dt,
    }
    return {
        'valid': valid,
        'mission_completed': valid and complete,
        'failure': run.get('failure'),
        'issues': issues,
        'metrics': metrics,
    }


def summarize_runs(evaluations, scheduled_runs):
    """Keep unexecuted, invalid and failed members in the scheduled denominator."""
    if (
        isinstance(scheduled_runs, bool)
        or not isinstance(scheduled_runs, int)
        or scheduled_runs < len(evaluations)
    ):
        raise ValueError('Invalid scheduled run count.')
    completed = sum(bool(e['mission_completed']) for e in evaluations)
    return {
        'scheduled_runs': scheduled_runs,
        'executed_runs': len(evaluations),
        'not_run': scheduled_runs - len(evaluations),
        'completed_runs': completed,
        'failed_runs': sum(
            e['valid'] and not e['mission_completed'] for e in evaluations
        ),
        'invalid_runs': sum(not e['valid'] for e in evaluations),
        'completion_rate': completed / scheduled_runs if scheduled_runs else None,
    }
