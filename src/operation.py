'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Four controllers sharing an electrical plant and mandatory-load reserve.
RHC uses sequential LPs; equipment heating follows the ambient-temperature trigger.
'''

import math
import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack

from .inputs import REGION_GROUPS, REGION_ORDER
from .modeling import demand_series, equipment_heater_action, regional_fulfillment

CONTROLLERS = ('rbc', 'pid', 'tacs', 'rhc')


def discharge_penalty(soc, discharged_ah):
    """Eqs. (13)--(14): SOC-weighted discharged Ah for one interval."""
    if not math.isfinite(soc) or not math.isfinite(discharged_ah) or discharged_ah < 0:
        raise ValueError('SOC and nonnegative discharge must be finite.')
    return (1.0 if soc < 0.2 else 0.5 if soc < 0.5 else 0.2) * discharged_ah


def service(hardware, demand_w, pwm):
    return regional_fulfillment(
        demand_w,
        hardware['regional_areas_m2'],
        hardware['weights'],
        hardware['efficiencies'],
        hardware['rated_powers_w'],
        hardware['mask'],
        pwm,
    )


def validate_setup(hardware, settings, initial_soc):
    for key in ('capacity_wh', 'voltage_v', 'bus_limit_w'):
        if not math.isfinite(hardware[key]) or hardware[key] <= 0:
            raise ValueError(f'{key} must be finite and positive.')
    for key in ('base_power_w', 'equipment_power_w', 'max_charge_w', 'max_discharge_w'):
        if not math.isfinite(hardware[key]) or hardware[key] < 0:
            raise ValueError(f'{key} must be finite and nonnegative.')
    service(hardware, 0.0, np.zeros(12))
    if (
        not math.isfinite(initial_soc)
        or not 0 <= settings['soc_min'] <= initial_soc <= 1
        or not settings['soc_min'] <= settings['soc_safe'] <= 1
        or not math.isfinite(settings['dt_hours'])
        or settings['dt_hours'] <= 0
    ):
        raise ValueError('Invalid initial SOC, thresholds or timestep.')
    for key in ('lambda_deg', 'lambda_soc', 'solver_tolerance', 'constraint_tolerance'):
        if not math.isfinite(settings[key]) or settings[key] < 0:
            raise ValueError(f'Invalid {key}.')
    if hardware['equipment_available'] not in (False, True):
        raise ValueError('Equipment installation must be binary.')


def electrical_step(soc, pwm, equipment_on, pv_used_w, hardware, dt_hours):
    """Eqs. (16)--(17): SOC update from electrical load and utilized PV."""
    body = float(np.dot(hardware['rated_powers_w'], np.asarray(pwm) * hardware['mask']))
    load = (
        body + hardware['base_power_w'] + equipment_on * hardware['equipment_power_w']
    )
    battery = load - pv_used_w
    return {
        'load_w': load,
        'battery_w': battery,
        'soc': soc - battery * dt_hours / hardware['capacity_wh'],
        'discharged_ah': max(battery, 0.0) * dt_hours / hardware['voltage_v'],
    }


def mandatory_reserve_wh(hardware, remaining_h):
    """Eq. (19): worst-case remaining mandatory energy, without future PV credit."""
    if not math.isfinite(remaining_h) or remaining_h < 0:
        raise ValueError('Remaining mission time must be finite and nonnegative.')
    power = hardware['base_power_w'] + (
        hardware['equipment_power_w'] if hardware['equipment_available'] else 0.0
    )
    return power * remaining_h


def _reserve_domain_issue(soc, hardware, settings, remaining_h):
    mandatory = mandatory_reserve_wh(hardware, 1.0)
    if mandatory > min(hardware['bus_limit_w'], hardware['max_discharge_w']):
        return 'Mandatory worst-case power exceeds the zero-PV reserve-policy domain.'
    available = (soc - settings['soc_min']) * hardware['capacity_wh']
    if available + settings['constraint_tolerance'] * hardware['capacity_wh'] < mandatory * remaining_h:
        return 'Initial energy is below the remaining mandatory-load reserve.'
    return None


def execute_action(soc, command, ambient_c, pv_w, hardware, settings, *, remaining_h):
    """Common actuator: mandatory equipment/base first, then proportional body limit.

    Physical clipping and the shared mandatory-load reserve are recorded separately.
    Remaining time includes this interval and extends beyond any forecast horizon.
    """
    command = np.asarray(command, dtype=float)
    if (
        command.shape != (12,)
        or not np.all(np.isfinite(command))
        or np.any((command < 0) | (command > 1))
    ):
        raise ValueError('Controller PWM must contain twelve fractions.')
    if not math.isfinite(pv_w) or pv_w < 0:
        raise ValueError('Available PV must be finite and nonnegative.')
    on = equipment_heater_action(ambient_c, hardware['equipment_available'])
    fixed = hardware['base_power_w'] + on * hardware['equipment_power_w']
    dt, capacity = settings['dt_hours'], hardware['capacity_wh']
    if remaining_h < dt - 1e-10:
        raise ValueError('Remaining mission time must include the current interval.')
    reserve_after = mandatory_reserve_wh(hardware, max(0.0, remaining_h - dt))
    discharge_limit = min(
        hardware['max_discharge_w'],
        max(0.0, (soc - settings['soc_min']) * capacity / dt),
    )
    available = min(hardware['bus_limit_w'], pv_w + discharge_limit)
    if fixed > available + settings['constraint_tolerance']:
        return {
            'success': False,
            'reason': 'Mandatory base/equipment load exceeds available power or interval energy.',
        }
    pwm = command * np.asarray(hardware['mask'])
    commanded_body = float(np.dot(pwm, hardware['rated_powers_w']))
    physical_body = min(commanded_body, max(0.0, available - fixed))
    reserve_body_limit = max(
        0.0, pv_w - fixed + ((soc - settings['soc_min']) * capacity - reserve_after) / dt
    )
    executed_body = min(physical_body, reserve_body_limit)
    if commanded_body > 0:
        pwm *= executed_body / commanded_body
    load = float(np.dot(pwm, hardware['rated_powers_w'])) + fixed
    charge_limit = min(hardware['max_charge_w'], max(0.0, (1 - soc) * capacity / dt))
    used = min(pv_w, load + charge_limit)
    step = electrical_step(soc, pwm, on, used, hardware, dt)
    if (
        step['soc'] < settings['soc_min'] - settings['constraint_tolerance']
        or step['soc'] > 1 + settings['constraint_tolerance']
        or (step['soc'] - settings['soc_min']) * capacity
        < reserve_after - settings['constraint_tolerance'] * capacity
    ):
        return {
            'success': False,
            'reason': 'Unclipped electrical transition violates SOC.',
        }
    return dict(
        step,
        success=True,
        pwm=pwm,
        equipment_on=on,
        pv_used_w=used,
        curtailed_pv_w=pv_w - used,
        reserve_required_after_wh=reserve_after,
        reserve_margin_after_wh=(step['soc'] - settings['soc_min']) * capacity - reserve_after,
        physical_curtailed_body_w=commanded_body - physical_body,
        reserve_curtailed_body_w=physical_body - executed_body,
    )


def make_forecast(config, garment, actual):
    """Forecast with current air/PV observations and scheduled future activity."""
    planned_met = np.asarray(actual['planned_met'], dtype=float)

    def provider(step, horizon):
        air = np.full(horizon, actual['air_c'][step])
        met = planned_met[step : step + horizon].copy()
        met[0] = actual['met'][step]
        demand = demand_series(config, garment, air, met)
        demand[0] = actual['demand_w'][step]
        return {
            'air_c': air,
            'pv_w': np.full(horizon, actual['pv_w'][step]),
            'demand_w': demand,
        }

    return provider


def solve_rhc(soc, forecast, hardware, settings, parameters, deg_weights=None, *, remaining_h):
    """Eqs. (12)--(19): sequential regional priorities and reserve-constrained LP.

    The configured tangents approximate the quadratic low-SOC penalty.
    Discharge weights use the shifted preceding plan and remain fixed during
    each solve. Group priorities are locked before the horizon objective.
    """
    started = time.perf_counter()
    demand, pv, air = (
        np.asarray(forecast[k], dtype=float) for k in ('demand_w', 'pv_w', 'air_c')
    )
    n = len(demand)
    if (
        n < 1
        or any(a.shape != (n,) or not np.all(np.isfinite(a)) for a in (demand, pv, air))
        or np.any(demand < 0)
        or np.any(pv < 0)
    ):
        raise ValueError('Invalid forecast arrays.')
    dt, capacity = settings['dt_hours'], hardware['capacity_wh']
    if remaining_h < n * dt - 1e-10:
        raise ValueError('Forecast extends beyond the remaining mission.')
    domain_issue = _reserve_domain_issue(soc, hardware, settings, remaining_h)
    if domain_issue:
        return {
            'success': False, 'failure_kind': 'reserve_domain_rejection',
            'reason': domain_issue, 'optimizer_calls': 0, 'solver_seconds': 0.0,
        }
    rated = np.asarray(hardware['rated_powers_w']) * hardware['mask']
    effective = rated * hardware['efficiencies']
    areas = np.asarray(hardware['regional_areas_m2'])
    fractions = areas / areas.sum()
    weights = np.asarray(hardware['weights'])
    fixed = hardware['base_power_w'] + hardware['equipment_power_w'] * np.array(
        [equipment_heater_action(t, hardware['equipment_available']) for t in air]
    )
    # 12 PWM, 12 regional-fulfillment variables, next SOC, used PV, discharge W, penalty.
    width, size = 28, 28 * n
    bounds = []
    eq_r = []
    eq_c = []
    eq_v = []
    eq_b = []
    ub_r = []
    ub_c = []
    ub_v = []
    ub_b = []

    def equality(items, rhs):
        row = len(eq_b)
        for col, value in items:
            if value:
                eq_r.append(row)
                eq_c.append(col)
                eq_v.append(value)
        eq_b.append(rhs)

    def upper(items, rhs):
        row = len(ub_b)
        for col, value in items:
            if value:
                ub_r.append(row)
                ub_c.append(col)
                ub_v.append(value)
        ub_b.append(rhs)

    for k in range(n):
        b = k * width
        local = demand[k] * fractions
        maximum = np.divide(local, effective, out=np.zeros(12), where=effective > 0)
        maximum = np.minimum(maximum, np.asarray(hardware['mask'], dtype=float))
        bounds.extend((0.0, float(v)) for v in maximum)
        bounds.extend([(1.0, 1.0) if demand[k] == 0 else (0.0, 1.0) for _ in range(12)])
        bounds.extend(
            [
                (settings['soc_min'] + mandatory_reserve_wh(
                    hardware, max(0.0, remaining_h - (k + 1) * dt)
                ) / capacity, 1.0),
                (0.0, float(pv[k])),
                (0.0, hardware['max_discharge_w']),
                (0.0, None),
            ]
        )
        for j in range(12):
            if demand[k] > 0:
                upper([(b + 12 + j, local[j]), (b + j, -effective[j])], 0.0)
        items = [(b + 24, 1.0), (b + 25, -dt / capacity)] + [
            (b + j, dt * rated[j] / capacity) for j in range(12)
        ]
        if k:
            items.append((b - width + 24, -1.0))
        equality(items, (soc if k == 0 else 0.0) - fixed[k] * dt / capacity)
        upper(
            [(b + j, rated[j]) for j in range(12)] + [(b + 25, -1.0), (b + 26, -1.0)],
            -fixed[k],
        )
        upper(
            [(b + j, -rated[j]) for j in range(12)] + [(b + 25, 1.0)],
            fixed[k] + hardware['max_charge_w'],
        )
        upper(
            [(b + j, rated[j]) for j in range(12)], hardware['bus_limit_w'] - fixed[k]
        )
        for tangent in np.linspace(
            0.0, settings['soc_safe'], parameters['pwl_segments'] + 1
        )[1:]:
            upper(
                [(b + 24, -2 * tangent), (b + 27, -1.0)],
                tangent * tangent - 2 * tangent * settings['soc_safe'],
            )
    Aeq = coo_matrix((eq_v, (eq_r, eq_c)), shape=(len(eq_b), size)).tocsr()
    Aub = coo_matrix((ub_v, (ub_r, ub_c)), shape=(len(ub_b), size)).tocsr()
    rhs = np.array(ub_b)
    calls = 0
    seconds = 0.0

    def optimize(objective):
        nonlocal calls, seconds
        tic = time.perf_counter()
        calls += 1
        result = linprog(
            objective,
            A_ub=Aub,
            b_ub=rhs,
            A_eq=Aeq,
            b_eq=eq_b,
            bounds=bounds,
            method='highs',
            options={
                'time_limit': parameters['solve_time_limit_s'],
                'primal_feasibility_tolerance': max(
                    1e-10, settings['constraint_tolerance']
                ),
                'dual_feasibility_tolerance': max(1e-10, settings['solver_tolerance']),
            },
        )
        seconds += time.perf_counter() - tic
        return result

    targets = []
    for group in REGION_GROUPS:
        indices = [REGION_ORDER.index(name) for name in group]
        if (
            demand[0] == 0
            or weights[indices].sum() <= 0
            or not np.any(np.asarray(hardware['mask'])[indices])
        ):
            continue
        objective = np.zeros(size)
        objective[12 + np.array(indices)] = -weights[indices] / weights[indices].sum()
        solved = optimize(objective)
        if not solved.success:
            break
        target = -float(solved.fun)
        targets.append({'indices': indices, 'target': target})
        Aub = vstack([Aub, coo_matrix(objective.reshape(1, -1))], format='csr')
        rhs = np.r_[rhs, -target + parameters['priority_lock_tolerance']]
    else:
        if deg_weights is None:
            deg_weights = np.full(n, discharge_penalty(soc, 1.0))
        deg_weights = np.asarray(deg_weights, dtype=float)
        if (
            deg_weights.shape != (n,)
            or not np.all(np.isfinite(deg_weights))
            or np.any(deg_weights < 0)
        ):
            raise ValueError('SOC discharge weights must align with the prediction horizon.')
        objective = np.zeros(size)
        for k in range(n):
            b = k * width
            objective[b + 12 : b + 24] = -weights
            objective[b + 26] = (
                settings['lambda_deg'] * deg_weights[k] * dt / hardware['voltage_v']
            )
            objective[b + 27] = settings['lambda_soc']
        solved = optimize(objective)
    if not solved.success or solved.x is None or not np.all(np.isfinite(solved.x)):
        return {
            'success': False,
            'failure_kind': (
                'policy_failure' if solved.status == 2 else 'numerical_failure'
            ),
            'reason': f'Prediction LP status {solved.status}: {solved.message}',
            'optimizer_calls': calls,
            'solver_seconds': seconds,
        }
    x = solved.x.reshape(n, width)
    command = np.clip(x[0, :12], 0.0, np.asarray(hardware['mask'], dtype=float))
    realized = service(hardware, float(demand[0]), command)['regional_fulfillment']
    for lock in targets:
        ix = lock['indices']
        value = float(np.dot(weights[ix], realized[ix]) / weights[ix].sum())
        if (
            value
            < lock['target']
            - parameters['priority_lock_tolerance']
            - 2 * settings['solver_tolerance']
        ):
            return {
                'success': False,
                'failure_kind': 'numerical_failure',
                'reason': 'Priority lock violated after numerical projection.',
                'optimizer_calls': calls,
                'solver_seconds': seconds,
            }
    return {
        'success': True,
        'command_pwm': command,
        'planned_soc': np.r_[soc, x[:, 24]],
        'priority_targets': targets,
        'optimizer_calls': calls,
        'solver_seconds': seconds,
        'planning_seconds': time.perf_counter() - started,
    }


def _baseline_command(
    controller, score, soc, met, air, remaining_h, maximum, dt, state, p
):
    """Comparison policies, driven by the same regional feedback quantity."""
    if controller == 'rbc':
        return (
            maximum
            if score < p['cold_threshold']
            else 0.5 * maximum if score < p['comfort_threshold'] else 0.0
        )
    if controller == 'pid':
        error = p['setpoint'] - score
        if state['last_error'] is None:
            state['last_error'] = error
            return 0.0
        limit = maximum / p['ki'] if p['ki'] > 0 else 0.0
        state['integral'] = float(
            np.clip(state['integral'] + error * dt, -limit, limit)
        )
        power = (
            p['kp'] * error
            + p['ki'] * state['integral']
            + p['kd'] * (error - state['last_error']) / dt
        )
        state['last_error'] = error
        return float(np.clip(power, 0.0, maximum))
    threshold = p['base_threshold'] + soc * p['soc_factor']
    if met < 2 and remaining_h > 1 and air < -20:
        threshold *= p['after_chill_multiplier']
    if state['last_action'] > 0:
        threshold -= p['deadband']
    power = (
        (maximum if threshold - score > 0.5 else 0.7 * maximum)
        if score < threshold
        else 0.0
    )
    state['last_action'] = power
    return power


def simulate(
    actual,
    hardware,
    settings,
    initial_soc,
    controller,
    parameters,
    forecast_provider=None,
):
    """Run one specified policy; retain N actions / N+1 states or a real prefix."""
    if controller not in CONTROLLERS:
        raise ValueError('Unknown controller.')
    validate_setup(hardware, settings, initial_soc)
    required = ('demand_w', 'pv_w', 'air_c', 'met')
    values = {k: np.asarray(actual[k], dtype=float) for k in required}
    n = len(values['demand_w'])
    if n < 1 or any(
        a.shape != (n,) or not np.all(np.isfinite(a)) for a in values.values()
    ):
        raise ValueError('Realized forcing must align on a nonempty finite grid.')
    if (
        np.any(values['demand_w'] < 0)
        or np.any(values['pv_w'] < 0)
        or np.any(values['met'] < 0)
    ):
        raise ValueError('Demand, PV and MET cannot be negative.')
    if controller == 'rhc' and not callable(forecast_provider):
        raise ValueError('RHC requires a declared causal forecast provider.')
    dt = settings['dt_hours']
    soc = float(initial_soc)
    states = [{'time_h': 0.0, 'soc': soc}]
    actions = []
    domain_issue = _reserve_domain_issue(soc, hardware, settings, n * dt)
    failure = (
        {'kind': 'reserve_domain_rejection', 'reason': domain_issue, 'step': 0, 'time_h': 0.0}
        if domain_issue else None
    )
    calls = 0
    seconds = 0.0
    previous = np.zeros(12)
    previous_plan = None
    state = {'integral': 0.0, 'last_error': None, 'last_action': 0.0}
    maximum = float(np.dot(hardware['rated_powers_w'], hardware['mask']))
    for k in range(n):
        if failure is not None:
            break
        observation = {key: float(a[k]) for key, a in values.items()}
        feedback = service(hardware, observation['demand_w'], previous)['score']
        decision = {}
        if controller == 'rhc':
            horizon = min(settings['horizon_steps'], n - k)
            prediction = forecast_provider(k, horizon)
            if any(
                not math.isclose(
                    float(prediction[key][0]),
                    observation[key],
                    rel_tol=0,
                    abs_tol=1e-10,
                )
                for key in ('demand_w', 'pv_w', 'air_c')
            ):
                raise ValueError(
                    'Forecast first sample differs from current observation.'
                )
            bands = None
            if previous_plan is not None:
                guess = np.r_[previous_plan[1:], previous_plan[-1]]
                if len(guess) < horizon:
                    guess = np.pad(guess, (0, horizon - len(guess)), mode='edge')
                guess = guess[:horizon]
                guess[0] = soc
                bands = np.where(guess < 0.2, 1.0, np.where(guess < 0.5, 0.5, 0.2))
            decision = solve_rhc(
                soc, prediction, hardware, settings, parameters['rhc'], bands,
                remaining_h=(n - k) * dt,
            )
            calls += decision['optimizer_calls']
            seconds += decision['solver_seconds']
            if not decision['success']:
                failure = {
                    'kind': decision['failure_kind'],
                    'reason': decision['reason'],
                    'step': k,
                    'time_h': k * dt,
                }
                break
            command = decision['command_pwm']
            previous_plan = decision['planned_soc']
        else:
            power = _baseline_command(
                controller,
                feedback,
                soc,
                observation['met'],
                observation['air_c'],
                (n - k) * dt,
                maximum,
                dt,
                state,
                parameters[controller],
            )
            command = np.asarray(hardware['mask'], dtype=float) * (
                power / maximum if maximum > 0 else 0.0
            )
        step = execute_action(
            soc, command, observation['air_c'], observation['pv_w'], hardware, settings,
            remaining_h=(n - k) * dt,
        )
        if not step['success']:
            failure = {
                'kind': 'physical_infeasibility',
                'reason': step['reason'],
                'step': k,
                'time_h': k * dt,
            }
            break
        delivered = service(hardware, observation['demand_w'], step['pwm'])
        actions.append(
            dict(
                observation,
                time_h=k * dt,
                command_pwm=command.tolist(),
                pwm=step['pwm'].tolist(),
                equipment_on=step['equipment_on'],
                load_w=step['load_w'],
                battery_w=step['battery_w'],
                pv_used_w=step['pv_used_w'],
                curtailed_pv_w=step['curtailed_pv_w'],
                reserve_required_after_wh=step['reserve_required_after_wh'],
                reserve_margin_after_wh=step['reserve_margin_after_wh'],
                physical_curtailed_body_w=step['physical_curtailed_body_w'],
                reserve_curtailed_body_w=step['reserve_curtailed_body_w'],
                discharged_ah=step['discharged_ah'],
                degradation_penalty=discharge_penalty(soc, step['discharged_ah']),
                score=delivered['score'],
                regional_fulfillment=delivered['regional_fulfillment'].tolist(),
                feedback_score=feedback,
                priority_targets=decision.get('priority_targets', []),
            )
        )
        soc = step['soc']
        previous = step['pwm']
        states.append({'time_h': (k + 1) * dt, 'soc': soc})
    return {
        'controller': controller,
        'completed': failure is None and len(actions) == n,
        'failure': failure,
        'scheduled_steps': n,
        'dt_hours': dt,
        'actions': actions,
        'reserve_policy': 'remaining_mandatory_load',
        'states': states,
        'optimizer_calls': calls,
        'solver_seconds': seconds,
    }
