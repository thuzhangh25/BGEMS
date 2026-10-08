'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Solve continuous multi-step RHC and advance one shared scenario simulation.
All physical and solver settings are explicit. Forecasts are separate from realized
forcing; numerical failure stops a run without a substitute controller. Eq. (13)–(22)
follow the current manuscript. HiGHS solves each LP subproblem to global optimality.
The low-SOC penalty of Eq. (16) enters as its declared 8-tangent piecewise-linear
form, and the piecewise discharge weights are extrapolated one step from the
previous control step's planned SOC trajectory (no fixed-point iteration).
Service-floor rows are planned with a declared 1e-9 margin so that raw
realized acceptance stays exact at boundary solutions, and the no-waste cap
P_supp <= P_req is planned with the same declared margin (mirrored at
acceptance as the shared _RESIDUAL_TOLERANCE) so the alpha = 1 boundary where
floor and cap coincide never becomes infeasible. Group values are whole-body
allocation preferences, not service constraints. No aging validation is
claimed.
'''
import math
import time
from numbers import Integral

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .modeling import (
    allocation_reference,
    battery_temperature_step,
    heat_losses,
    heating_requirement,
    whole_body_fulfillment,
)

from .inputs import REGION_GROUPS, REGION_ORDER

# Declared planning margin applied to the whole-body service-floor rows and
# to the no-waste cap row when demand is positive. Acceptance of realized
# residuals remains raw on the deficit side (<= 0, no hidden tolerance); the
# margin only prevents the last floating-point unit of an exactly binding
# service plan from flipping the realized recomputation (a different
# arithmetic path) positive. On the cap side the same declared margin is
# mirrored at acceptance through _RESIDUAL_TOLERANCE below: without it the
# alpha = 1 boundary, where the floor row and the cap row coincide at
# P_supp = P_req, would be an infeasible exact-equality pair.
_PLAN_MARGIN = 1e-9

# Declared oversupply acceptance margin, shared with evaluation.py so the
# planner and the independent auditor use the exact same constant on the
# no-waste cap. Deficit-side ('service') acceptance stays raw (<= 0) and is
# not listed here. When demand is zero the cap is accepted at exactly 0.
_RESIDUAL_TOLERANCE = {'service_oversupply': _PLAN_MARGIN}

# Declared number of tangent breakpoints of the piecewise-linear low-SOC
# penalty: Eq. (16) as the declared 8-tangent piecewise-linear form. Tangents
# sit at b_j = j * soc_safe / 8, j = 1..8, and underestimate the quadratic
# shortfall penalty on [0, soc_safe] with a maximum gap of (soc_safe/16)^2
# (8.79e-5 at the declared soc_safe = 0.15).
_PWL_SEGMENTS = 8

# Declared per-solve wall-clock budget in seconds. Typical solves finish in
# tens of milliseconds; the budget only bounds pathological instances.
_SOLVE_TIME_LIMIT_S = 3.0

# Declared priority-lock margin. The final LP may use lock slack up to this
# margin, its own feasibility tolerance applies on the lock rows, and the
# realized group allocation preference is recomputed on a different arithmetic
# path (allocation_reference) than the LP epigraph; the acceptance check in
# solve_rhc and the independent audit in evaluation.py must therefore allow
# margin + solver tolerance + roundoff, never a tighter bound than the lock
# itself permits. 1e-6 matches the scale of the raw-row acceptance guard.
_PRIORITY_MARGIN = 1e-6


def equipment_switch(air_history_c, previous_on, available, on_c, off_c, window):
    """Eq. (22): causal trailing mean and air-sensor hysteresis.

    History includes the current sample, never future samples. At startup use all
    supplied samples up to window; equality to either threshold retains state.
    Returns (binary action, filtered air temperature in degrees C).
    """
    if previous_on not in (0, 1) or available not in (0, 1):
        raise ValueError('Equipment availability and previous state must be binary.')
    if isinstance(window, bool) or not isinstance(window, Integral) or window < 1:
        raise ValueError('Filter window must be a positive integer.')
    history = np.asarray(air_history_c, dtype=float)
    if history.ndim != 1 or not history.size or not np.all(np.isfinite(history)):
        raise ValueError('Air history must be nonempty and finite.')
    if not math.isfinite(on_c) or not math.isfinite(off_c) or on_c >= off_c:
        raise ValueError('Finite equipment thresholds must satisfy on < off.')
    filtered = float(np.mean(history[-window:]))
    action = 0 if not available else 1 if filtered < on_c else 0 if filtered > off_c else int(previous_on)
    return action, filtered


def discharge_penalty(soc, discharged_ah):
    """Eq. (15), Eq. (16): start-of-interval SOC weight times interval Ah.

    Charge is nonnegative discharged charge, not cumulative or signed charge.
    Exact 20% and 50% boundaries belong to the higher-SOC branch.
    """
    if not math.isfinite(soc) or not math.isfinite(discharged_ah) or discharged_ah < 0:
        raise ValueError('SOC and nonnegative discharge must be finite.')
    weight = 1.0 if soc < 0.2 else 0.5 if soc < 0.5 else 0.2
    return weight * discharged_ah


def electrical_step(soc, pwm, equipment_on, pv_used_w, hardware, dt_hours):
    """Eq. (18), Eq. (19): electrical balance, un-clipped next SOC and interval Ah.

    hardware: rated_powers_w, mask (12 entries); equipment_power_w,
    equipment_available; base_power_w, capacity_wh and voltage_v (constant pack
    voltage). Body heat-transfer efficiency never enters electrical demand.
    pv_used_w is bus-referred electrical power, not irradiance or gross PV.
    """
    duty = np.asarray(pwm, dtype=float)
    if duty.shape != (12,) or not np.all(np.isfinite(duty)) or np.any((duty < 0) | (duty > 1)):
        raise ValueError('PWM must contain twelve finite fractions.')
    if equipment_on not in (0, 1) or not math.isfinite(pv_used_w) or pv_used_w < 0:
        raise ValueError('Invalid equipment action or PV use.')
    if not math.isfinite(soc) or not math.isfinite(dt_hours) or dt_hours <= 0:
        raise ValueError('SOC must be finite and timestep positive.')
    load = float(np.dot(hardware['rated_powers_w'], np.asarray(hardware['mask']) * duty)
                 + hardware['equipment_power_w'] * hardware['equipment_available'] * equipment_on
                 + hardware['base_power_w'])
    battery = load - pv_used_w
    return {'load_w': load, 'battery_w': battery,
            'soc': soc - battery * dt_hours / hardware['capacity_wh'],
            'discharged_ah': max(battery, 0.0) * dt_hours / hardware['voltage_v']}


def _validate(hardware, settings):
    """Validate shared physical inputs once per public solve/run, not per trial."""
    whole_body_fulfillment(0, hardware['efficiencies'], hardware['rated_powers_w'],
                           hardware['mask'], np.zeros(12))
    allocation_reference(0, hardware['regional_areas_m2'], hardware['weights'],
                         hardware['efficiencies'], hardware['rated_powers_w'],
                         hardware['mask'], np.zeros(12))
    for name in ('capacity_wh', 'voltage_v', 'heat_capacity_j_per_k', 'thermal_resistance_k_per_w'):
        if not math.isfinite(hardware[name]) or hardware[name] <= 0:
            raise ValueError(name + ' must be finite and positive.')
    for name in ('base_power_w', 'equipment_power_w', 'max_discharge_w', 'max_charge_w', 'bus_limit_w'):
        if not math.isfinite(hardware[name]) or hardware[name] < 0:
            raise ValueError(name + ' must be finite and nonnegative.')
    equipment_switch([0], 0, hardware['equipment_available'], settings['on_c'],
                     settings['off_c'], settings['filter_window'])
    for name in ('dt_hours', 'solver_tolerance'):
        if not math.isfinite(settings[name]) or settings[name] <= 0:
            raise ValueError(name + ' must be finite and positive.')
    for name in ('lambda_deg', 'lambda_soc', 'constraint_tolerance'):
        if not math.isfinite(settings[name]) or settings[name] < 0:
            raise ValueError(name + ' must be finite and nonnegative.')
    if settings['constraint_tolerance'] < 1e-10:
        raise ValueError('HiGHS requires constraint_tolerance >= 1e-10; raw acceptance still uses <= 0.')
    for name in ('soc_min', 'soc_safe', 'terminal_soc'):
        if not math.isfinite(settings[name]) or not 0 <= settings[name] <= 1:
            raise ValueError(name + ' must be a fraction.')
    if settings['soc_safe'] < settings['soc_min'] or settings['terminal_soc'] < settings['soc_min']:
        raise ValueError('Safe/terminal SOC cannot be below the hard floor.')
    for name in ('horizon_steps',):
        if isinstance(settings[name], bool) or not isinstance(settings[name], Integral) or settings[name] < 0:
            raise ValueError(name + ' must be a nonnegative integer.')


def validate_setup(hardware, settings, initial_state):
    """Validate finite runtime inputs without running or reclassifying a scenario."""
    _validate(hardware, settings)
    try:
        soc = initial_state['soc']
        battery_c = initial_state['battery_c']
        equipment_on = initial_state['equipment_on']
        history = np.asarray(initial_state['air_history_c'], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('Initial state must define SOC, battery temperature, equipment state and history.') from error
    if (not math.isfinite(soc) or not math.isfinite(battery_c) or history.ndim != 1 or
            not np.all(np.isfinite(history))):
        raise ValueError('Initial states/history must be finite.')
    if isinstance(equipment_on, bool) or equipment_on not in (0, 1):
        raise ValueError('Initial equipment state must be integer zero or one.')


def _forcing(data):
    values = tuple(np.asarray(data[key], dtype=float) for key in ('demand_w', 'pv_w', 'air_c'))
    if any(v.ndim != 1 or not v.size or not np.all(np.isfinite(v)) for v in values):
        raise ValueError('Forcing must contain finite nonempty 1-D trajectories.')
    if any(v.shape != values[0].shape for v in values) or any(np.any(v < 0) for v in values[:2]):
        raise ValueError('Forcing trajectories must align; demand and PV are nonnegative.')
    return values


def _floors(service_floor, count):
    floors = np.asarray(service_floor, dtype=float)
    if floors.shape != (count,) or not np.all(np.isfinite(floors)) or np.any((floors < 0) | (floors > 1)):
        raise ValueError('Service floors must be a finite (steps,) scalar-alpha array in [0, 1].')
    return floors


def build_scenario(config, garment, weather_data):
    """Construct realized demand and service arrays from one SystemConfig scenario.

    Source weather is already placed on the operation grid by weather.py. Activity
    and service requirements are queried at each elapsed interval start. The E/W
    rules are explicit heuristic heat-loss inputs, not physiological validation.
    """
    times = tuple(weather_data['times'])
    air_c = np.asarray(weather_data['air_c'], dtype=float)
    if not times or air_c.shape != (len(times),) or not np.all(np.isfinite(air_c)):
        raise ValueError('Weather must provide one finite air-temperature value per scenario step.')
    dt_hours = config.settings['dt_hours']
    elapsed = np.arange(len(times), dtype=float) * dt_hours
    met = np.asarray(
        [config.scenario.get_activity(float(hour)) for hour in elapsed], dtype=float)
    if met.shape != air_c.shape or not np.all(np.isfinite(met)) or np.any(met < 0):
        raise ValueError('Scenario activity queries must return finite nonnegative MET values.')
    floors = []
    for hour in elapsed:
        value = config.scenario.get_service_requirement(float(hour))
        if np.ndim(value) > 0:
            # A 12-vector is accepted only when all twelve entries are exactly
            # equal (backward compatibility with hand-expanded configs); it is
            # then converted to that scalar. Non-uniform vectors are rejected:
            # the whole-body service definition has no per-region floors.
            array = np.asarray(value, dtype=float)
            if (array.shape != (12,) or not np.all(np.isfinite(array)) or
                    np.any((array < 0) | (array > 1)) or not np.all(array == array[0])):
                raise ValueError('Scenario service requirements must be a scalar '
                                 'whole-body alpha under the whole-body service '
                                 'definition; a non-uniform 12-vector is rejected.')
            value = float(array[0])
        value = float(value)
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Scenario service requirements must be fractions in [0, 1].')
        floors.append(value)
    service_floor = np.asarray(floors, dtype=float)
    evaporation_w = np.empty(len(times))
    mechanical_work_w = np.empty(len(times))
    demand_w = np.empty(len(times))
    body_area_m2 = config.person.body_area_m2
    skin_setpoint_c = config.person.skin_setpoint_c
    for index, (ambient_c, activity) in enumerate(zip(air_c, met)):
        evaporation_w[index], mechanical_work_w[index] = heat_losses(
            activity, ambient_c, body_area_m2)
        demand_w[index] = heating_requirement(
            ambient_c,
            activity,
            body_area_m2,
            garment.clothing_resistance,
            config.scenario.air_resistance,
            skin_setpoint_c,
            evaporation_w[index],
            mechanical_work_w[index],
        )
    return {
        'times': times,
        'air_c': air_c,
        'met': met,
        'evaporation_w': evaporation_w,
        'mechanical_work_w': mechanical_work_w,
        'demand_w': demand_w,
        'service_floor': service_floor,
    }


def make_forecast(config, garment, scenario_data, actual):
    """Return the declared causal persistence or deliberate perfect-preview policy."""
    mode = config.forecast_mode
    if mode not in ('persistence', 'perfect-preview'):
        raise ValueError("forecast_mode must be 'persistence' or 'perfect-preview'.")
    realized = {key: np.asarray(actual[key], dtype=float)
                for key in ('demand_w', 'pv_w', 'air_c')}
    if any(values.ndim != 1 for values in realized.values()):
        raise ValueError('Actual forcing must contain one-dimensional trajectories.')
    count = len(realized['demand_w'])
    if any(len(values) != count for values in realized.values()):
        raise ValueError('Actual forcing trajectories must align.')
    met = np.asarray(scenario_data['met'], dtype=float)
    if met.shape != (count,):
        raise ValueError('Scenario activity and actual forcing must align.')
    body_area_m2 = config.person.body_area_m2
    skin_setpoint_c = config.person.skin_setpoint_c

    def provider(step, observation, horizon):
        if (isinstance(step, bool) or not isinstance(step, Integral) or step < 0 or
                isinstance(horizon, bool) or not isinstance(horizon, Integral) or
                horizon < 1 or step + horizon > count):
            raise ValueError('Forecast request lies outside the scenario grid.')
        for key, values in realized.items():
            if key not in observation or not math.isfinite(observation[key]):
                raise ValueError('Forecast observation must contain finite current forcing.')
            if float(values[step]) != float(observation[key]):
                raise ValueError('Forecast observation does not match realized current forcing.')
        if mode == 'perfect-preview':
            return {
                key: values[step:step + horizon].copy()
                for key, values in realized.items()
            }
        air_c = np.full(horizon, float(observation['air_c']))
        pv_w = np.full(horizon, float(observation['pv_w']))
        demand_w = np.empty(horizon)
        for offset in range(horizon):
            evaporation_w, mechanical_work_w = heat_losses(
                met[step + offset], air_c[offset], body_area_m2)
            demand_w[offset] = heating_requirement(
                air_c[offset],
                met[step + offset],
                body_area_m2,
                garment.clothing_resistance,
                config.scenario.air_resistance,
                skin_setpoint_c,
                evaporation_w,
                mechanical_work_w,
            )
        demand_w[0] = float(observation['demand_w'])
        return {'demand_w': demand_w, 'pv_w': pv_w, 'air_c': air_c}

    return provider


def _group_indices(hardware):
    """Nonempty installed priority groups in REGION_GROUPS order.

    Returns (group_index_arrays, group_labels) covering only G ∩ I_g (labels
    'core', 'distal', 'other'). A group with no installed pad is silently
    dropped — no exception and no phantom full score. A nonempty installed
    group whose total weight is <= 0 still raises.
    """
    order = tuple(hardware.get('region_order', REGION_ORDER))
    if order != REGION_ORDER:
        raise ValueError('Hardware regional locations must match the declared order.')
    mask = np.asarray(hardware['mask'], dtype=float)
    weights = np.asarray(hardware['weights'], dtype=float)
    groups, labels = [], []
    for label, group in zip(('core', 'distal', 'other'), REGION_GROUPS):
        indices = np.array([order.index(name) for name in group
                            if mask[order.index(name)] == 1], dtype=int)
        if indices.size == 0:
            continue
        if float(weights[indices].sum()) <= 0:
            raise ValueError('Every nonempty installed priority group must have '
                             'positive total weight.')
        groups.append(indices)
        labels.append(label)
    return groups, labels


def group_service(hardware, demand_w, pwm):
    """Allocation preference per nonempty installed group, in REGION_GROUPS order.

    Each value is sum_{i in G ∩ I_g} (omega_i / sum omega) * r_i^alloc from
    allocation_reference — an allocation preference used only for the
    lexicographic priority stages, never a service constraint and never a
    decider of S. When P_req == 0 every returned value is exactly 1.0. The
    tuple may have length 0 (garment with no installed pads).
    """
    allocation = allocation_reference(
        demand_w, hardware['regional_areas_m2'], hardware['weights'],
        hardware['efficiencies'], hardware['rated_powers_w'],
        hardware['mask'], pwm)['allocation_fulfillment']
    weights = np.asarray(hardware['weights'], dtype=float)
    values = []
    for indices in _group_indices(hardware)[0]:
        group_weight = float(weights[indices].sum())
        values.append(float(weights[indices] @ allocation[indices] / group_weight))
    return tuple(values)


def _priority_rows(matrix, lower, upper, column_bounds, f_column, demand, hardware,
                   settings):
    """Maximize first-action group allocation preference on the hard-feasible set.

    demand is the allocated task matrix D (H, 12) with D[k, i] = beta_i *
    demand[k]; the group values are allocation preferences, not service
    constraints. A group whose direction vector is exactly zero (zero whole-
    body demand) records its constant target of 1.0 but adds no lock row,
    avoiding the degenerate 0 >= -1e-6 row.
    """
    weights = np.asarray(hardware['weights'], dtype=float)
    indices, _labels = _group_indices(hardware)
    ncols = matrix.shape[1]
    locked_rows, locked_lower, targets, certificates = [], [], [], []
    bounds = [(0., float(v)) for v in column_bounds]
    for index in indices:
        direction = np.zeros(ncols)
        group_weight = float(weights[index].sum())
        constant = 0.
        for i in index:
            fraction = float(weights[i] / group_weight)
            if demand[0, i] > 0:
                direction[f_column[(0, int(i))]] = fraction
            else:
                constant += fraction
        if locked_rows:
            rows = sp.vstack([matrix, *locked_rows], format='csc')
            lo = np.concatenate([lower, locked_lower])
            hi = np.concatenate([upper, np.full(len(locked_rows), np.inf)])
        else:
            rows, lo, hi = matrix, lower, upper
        A_ub = sp.vstack([rows[np.isfinite(hi)], -rows[np.isfinite(lo)]], format='csc')
        b_ub = np.concatenate([hi[np.isfinite(hi)], -lo[np.isfinite(lo)]])
        stage_started = time.perf_counter()
        outcome = linprog(
            -direction, A_ub=A_ub, b_ub=b_ub,
            bounds=bounds, method='highs',
            options={
                'time_limit': _SOLVE_TIME_LIMIT_S,
                'primal_feasibility_tolerance': max(
                    1e-10, float(settings['constraint_tolerance'])),
                'dual_feasibility_tolerance': max(
                    1e-10, float(settings['constraint_tolerance'])),
                'ipm_optimality_tolerance': float(settings['solver_tolerance']),
            })
        elapsed = time.perf_counter() - stage_started
        if outcome.status != 0 or outcome.x is None or not np.all(np.isfinite(outcome.x)):
            kind = 'policy_failure' if not targets and outcome.status == 2 else 'numerical_failure'
            return None, None, None, {'success': False, 'failure_kind': kind,
                                      'reason': f'Priority LP: {outcome.message}',
                                      'solver': 'HiGHS-LP-priority', 'solver_status': outcome.status,
                                      'priority_stages': certificates}, certificates
        multipliers = np.asarray(outcome.ineqlin.marginals)
        upper_marginals = np.asarray(outcome.upper.marginals)
        lower_marginals = np.asarray(outcome.lower.marginals)
        bound_values = np.asarray([bound[1] for bound in bounds])
        # Masked certificate products: an exactly zero marginal contributes
        # exactly zero even against a huge right-hand side or bound, which
        # keeps degenerate duals finite instead of letting 0*large or
        # inf-inf produce a NaN that would silently pass the gap check (NaN
        # comparisons are False). BLAS kernels also leak spurious FP exception
        # flags on finite data, so warnings are suppressed locally and every
        # quantity is instead checked explicitly below.
        with np.errstate(all='ignore'):
            dual = float(np.sum(np.where(multipliers != 0, b_ub * multipliers, 0.0)) +
                         np.sum(np.where(upper_marginals != 0,
                                         bound_values * upper_marginals, 0.0)))
            stationarity = float(np.max(np.abs(
                -direction - A_ub.T @ multipliers - lower_marginals - upper_marginals)))
            optimum = constant + float(direction @ outcome.x)
        gap = abs(float(outcome.fun) - dual)
        tolerance = settings['solver_tolerance']
        certificate = {'status': int(outcome.status), 'seconds': elapsed,
                       'duality_gap': gap, 'stationarity_residual': float(stationarity)}
        certificates.append(certificate)
        if (not math.isfinite(gap) or not math.isfinite(stationarity) or
                not math.isfinite(optimum) or
                gap > tolerance or stationarity > tolerance):
            return None, None, None, {'success': False, 'failure_kind': 'numerical_failure',
                                      'reason': 'Priority LP optimality certificate exceeds tolerance.',
                                      'priority_stages': certificates}, certificates
        targets.append(optimum)
        # A zero-direction group (zero whole-body demand, constant target 1.0)
        # imposes nothing on the LP: adding its 0 >= -1e-6 row would be a
        # degenerate lock, so it is skipped.
        if np.any(direction != 0):
            locked_rows.append(sp.csr_matrix(direction))
            locked_lower.append(optimum - constant - _PRIORITY_MARGIN)
    if locked_rows:
        locks = sp.vstack(locked_rows, format='csc')
    else:
        locks = sp.csr_matrix((0, ncols))
    return (locks, np.array(locked_lower),
            tuple(targets), None, certificates)


def solve_rhc(soc, previous_equipment_on, air_history_c, forecast, service_floor,
              reaches_scenario_end, hardware, settings, deg_weights=None):
    """Eq. (13)–(22): prioritize current allocation preference, then optimize horizon PWM/PV.

    forecast: demand_w, pv_w, air_c arrays; air_history_c contains samples strictly
    BEFORE the current interval. Forecast first air sample must be current measured
    air (simulate enforces all current forcing). service_floor is (H,) — one
    scalar whole-body alpha per step.
    settings: dt_hours, horizon_steps, soc_min, soc_safe, terminal_soc, lambda_deg,
    lambda_soc, on_c, off_c, filter_window, solver_tolerance, constraint_tolerance.
    No undeclared values are inferred. `initial_radius` was a legacy trust-region
    placeholder and is no longer part of the LP/PWL contract; the RHC subproblem
    carries only the declared box bounds.

    The RHC subproblem is a pure LP: linear power/SOC/service constraints, the
    whole-body service score epigraph (U_B = min(1, P_supp / P_req) via
    P_req * z <= P_supp, z <= 1), the hard whole-body service floor
    P_supp >= alpha * P_req and the no-waste cap P_supp <= P_req, and the
    low-SOC penalty of Eq. (16) as the declared 8-tangent piecewise-linear
    form. The piecewise-constant discharge weights of Eq. (15)/(16) are
    extrapolated one step from the previous control step's planned SOC
    trajectory and supplied by the caller through deg_weights; the declared
    default is the constant initial 0.2 branch. The core -> distal -> other
    priority stages maximize per-group allocation preferences (r_i^alloc from
    the beta_i allocation); these are preferences only — the sole hard service
    constraints are the whole-body floor and cap rows.
    HiGHS solves the LP to global optimality. The floor row is planned with
    the declared 1e-9 margin and the positive-demand cap row carries the same
    declared margin (mirrored at acceptance as _RESIDUAL_TOLERANCE) so the
    alpha = 1 boundary where the two rows coincide stays feasible; realized
    floor acceptance stays raw. A non-optimal solver exit is recorded as
    numerical failure, never accepted as an operating plan and never replaced
    by a substitute controller.
    """
    _validate(hardware, settings)
    demand, pv, air = _forcing(forecast)
    n = demand.size
    floors = _floors(service_floor, n)
    if n > settings['horizon_steps'] or not isinstance(reaches_scenario_end, bool):
        raise ValueError('Invalid prediction length or terminal flag.')
    if not math.isfinite(soc):
        raise ValueError('Initial SOC must be finite.')
    history = list(air_history_c)
    if not all(math.isfinite(v) for v in history):
        raise ValueError('Initial sensor history must be finite.')
    if deg_weights is None:
        branch_weights = np.full(n, 0.2)
    else:
        branch_weights = np.asarray(deg_weights, dtype=float)
        if (branch_weights.shape != (n,) or not np.all(np.isfinite(branch_weights)) or
                not np.all(np.isin(branch_weights, (0.2, 0.5, 1.0)))):
            raise ValueError('Degradation weights must be one Eq. (15)/(16) branch '
                             'value (0.2, 0.5 or 1.0) per step.')
    equipment = np.empty(n)
    previous = previous_equipment_on
    for k in range(n):
        history.append(air[k])
        previous, _ = equipment_switch(history, previous, hardware['equipment_available'],
                                        settings['on_c'], settings['off_c'], settings['filter_window'])
        equipment[k] = previous
    if not settings['soc_min'] <= soc <= 1:
        return {'success': False, 'failure_kind': 'policy_failure', 'reason': 'Initial SOC violates bounds.'}
    rated = np.asarray(hardware['rated_powers_w']) * hardware['mask']
    effective = rated * hardware['efficiencies']
    areas = np.asarray(hardware['regional_areas_m2'], dtype=float)
    installed = np.asarray(hardware['mask'], dtype=float)
    installed_area = float(np.sum(areas * installed))
    if installed_area > 0:
        beta = areas * installed / installed_area
    else:
        beta = np.zeros(12)
    alloc_demand = demand[:, None] * beta
    # A garment with no installed pads cannot meet any positive whole-body
    # floor: the base floor row sum_i effective_i * u >= alpha * demand is
    # infeasible because effective is identically zero, and with every group
    # empty no priority stage exists to classify the infeasibility. The
    # declared outcome is policy_failure — no fabricated full score.
    if not np.any(hardware['mask']) and np.any(floors * demand > 0):
        return {'success': False, 'failure_kind': 'policy_failure',
                'reason': 'No installed pads can meet the whole-body service floor.'}
    base = hardware['base_power_w']
    scale = settings['dt_hours'] / hardware['capacity_wh']
    dt_over_V = settings['dt_hours'] / hardware['voltage_v']
    fixed_load = base + equipment * hardware['equipment_power_w']

    # Variables: x = [pwm_0..pwm_11, pv] * n, then y_k, z_k, a_k for k=0..n-1,
    # then one allocation-preference epigraph variable f_{k,i} per installed
    # node with positive allocated task D_{k,i} = beta_i * demand_k (f_ki is a
    # preference diagnostic, not a service constraint; no q_i <= D_i row is
    # imposed, so a rich pad may cover a poor pad's share),
    # then one penalty epigraph variable p_k per step (Eq. (16) as the
    # declared 8-tangent piecewise-linear form).
    nv = 13 * n
    f_locations = [(k, i) for k in range(n) for i in range(12)
                   if installed[i] == 1 and alloc_demand[k, i] > 0]
    f_column = {location: nv + 3 * n + index
                for index, location in enumerate(f_locations)}
    nf = len(f_locations)
    p_column = nv + 3 * n + nf
    nv_full = p_column + n
    # HiGHS treats 1e30 as +infinity; rows carrying it are dropped from the
    # one-sided linprog form below.
    INF = 1e30

    # --- Constraint matrix A ---
    rows = []
    lbs = []
    ubs = []

    def add(A, lo, hi):
        rows.append(sp.csr_matrix(A))
        lbs.append(np.atleast_1d(lo))
        ubs.append(np.atleast_1d(hi))

    # 1. Variable bounds as constraints (we use [0, INF] var bounds and put real bounds in A).
    lb_var = np.zeros(nv)
    ub_var = np.zeros(nv)
    for k in range(n):
        for i in range(12):
            ub_var[13*k+i] = hardware['mask'][i]
        ub_var[13*k+12] = min(pv[k], hardware['bus_limit_w'])
    add(sp.eye(nv, format='csc'), lb_var, ub_var)

    # 2. Bus, discharge, charge limits per step.
    for k in range(n):
        row = np.zeros(nv); row[13*k:13*k+12] = rated
        add(row, -INF, hardware['bus_limit_w'] - fixed_load[k])
        row = np.zeros(nv); row[13*k:13*k+12] = rated; row[13*k+12] = -1
        add(row, -INF, hardware['max_discharge_w'] - fixed_load[k])
        row = np.zeros(nv); row[13*k:13*k+12] = -rated; row[13*k+12] = 1
        add(row, -INF, hardware['max_charge_w'] + fixed_load[k])

    # 3. SOC bounds: soc_min <= SOC_k <= 1.
    #    SOC_k = soc - scale * sum_{j<=k} (pwm_j@rated + base - pv_j)
    #    Let row.x = -scale*sum_{j<=k}(pwm_j@rated - pv_j).
    #    Then SOC_k = soc + row.x - scale*(k+1)*base.
    for k in range(n):
        row = np.zeros(nv)
        for j in range(k+1):
            row[13*j:13*j+12] = -scale * rated
            row[13*j+12] = scale
        add(row,
            settings['soc_min'] - soc + scale * np.sum(fixed_load[:k+1]),
            1 - soc + scale * np.sum(fixed_load[:k+1]))

    # 4. Terminal SOC.
    if reaches_scenario_end:
        row = np.zeros(nv)
        for j in range(n):
            row[13*j:13*j+12] = -scale * rated
            row[13*j+12] = scale
        add(row,
            settings['terminal_soc'] - soc + scale * np.sum(fixed_load),
            1 - soc + scale * np.sum(fixed_load))

    # 5. Whole-body service rows per step, over the total effective supplied
    #    heat P_supp = sum_i effective_i * u_{k,i}:
    #    - Floor row (only when floors[k] > 0 and demand[k] > 0):
    #      P_supp >= floors[k] * demand[k] + _PLAN_MARGIN — planned with the
    #      declared margin so the raw zero-tolerance realized recomputation
    #      (a different arithmetic path) cannot flip the last floating-point
    #      unit of an exactly binding plan positive.
    #    - Cap row (no-waste contract): when demand[k] > 0,
    #      P_supp <= demand[k] + _PLAN_MARGIN — the same declared margin is
    #      mirrored at acceptance (_RESIDUAL_TOLERANCE) so the alpha = 1
    #      exact-equality boundary, where floor and cap coincide at
    #      P_supp = P_req, never becomes infeasible. When demand[k] == 0 the
    #      exact row P_supp <= 0 forces every installed pad to u = 0.
    #    Other rows stay exact: legitimate zero-slack boundary solutions
    #    (e.g. a full battery with zero load curtailing all PV) must remain
    #    feasible.
    for k in range(n):
        if floors[k] > 0 and demand[k] > 0:
            row = np.zeros(nv)
            row[13*k:13*k+12] = effective
            add(row, floors[k] * demand[k] + _PLAN_MARGIN, INF)
        row = np.zeros(nv)
        row[13*k:13*k+12] = effective
        add(row, -INF, demand[k] + _PLAN_MARGIN if demand[k] > 0 else 0.)

    A = sp.vstack(rows).tocsc()
    lb = np.concatenate(lbs)
    ub = np.concatenate(ubs)

    # --- Objective: minimize -J = sum(-z_k + lambda_soc*p_k + lambda_deg*w_k*a_k) ---
    # The low-SOC penalty of Eq. (16) enters as the declared 8-tangent
    # piecewise-linear form lambda_soc * p_k with p_k >= y_k^2 on [0, soc_safe].
    q = np.zeros(nv_full)
    for k in range(n):
        q[nv + n + k] = -1.0  # maximize score => minimize -z
        q[nv + 2 * n + k] = settings['lambda_deg'] * branch_weights[k]
        q[p_column + k] = settings['lambda_soc']

    # Epigraph constraints for y_k, z_k, a_k, p_k.
    rows2 = []
    lb2 = []
    ub2 = []

    def add2(A, lo, hi):
        rows2.append(sp.csr_matrix(A))
        lb2.append(np.atleast_1d(lo))
        ub2.append(np.atleast_1d(hi))

    soc_safe = settings['soc_safe']
    tangents = tuple(soc_safe * j / _PWL_SEGMENTS for j in range(1, _PWL_SEGMENTS + 1))

    for k in range(n):
        yv = nv + k
        z = nv + n + k
        a = nv + 2 * n + k
        p = p_column + k

        # Eq. (17) uses start-of-interval SOC_k; only earlier actions j<k
        # contribute. Battery discharge raises the shortfall, PV lowers it.
        row = np.zeros(nv_full)
        row[yv] = 1
        for j in range(k):
            row[13*j:13*j+12] -= scale * rated
            row[13*j+12] += scale
        add2(row, soc_safe - soc + scale * np.sum(fixed_load[:k]), INF)
        # y_k >= 0
        row = np.zeros(nv_full); row[yv] = 1
        add2(row, 0, INF)

        # Allocation-preference epigraph: f_ki <= q_ki / D_ki, entered here as
        # f_ki * D_ki <= effective_i * u_ki with f_ki <= 1 as a column bound.
        # f_ki never constrains u beyond the priority lock rows (no q_i <= D_i
        # row), so cross-pad compensation stays possible.
        for i in range(12):
            if (k, i) in f_column:
                column = f_column[(k, i)]
                row = np.zeros(nv_full)
                row[column] = 1
                row[13*k+i] -= effective[i] / alloc_demand[k, i]
                add2(row, -INF, 0)
        # U_B epigraph: demand[k] * z_k <= P_supp = sum_i effective_i * u_ki
        # (z_k <= 1 is a column bound; the maximizing objective drives z_k to
        # min(1, P_supp / demand[k])). When demand[k] == 0 no row is added:
        # the [0, 1] column bound plus the maximizing objective yields
        # z_k = 1, matching U_B = 1.
        if demand[k] > 0:
            row = np.zeros(nv_full)
            row[z] = demand[k]
            row[13*k:13*k+12] -= effective
            add2(row, -INF, 0)

        # a_k >= power_k * dt / V, a_k >= 0
        # power_k = pwm_k@rated + base - pv_k
        # a_k >= (pwm_k@rated + base - pv_k) * dt/V
        # a_k - (dt/V)*pwm_k@rated + (dt/V)*pv_k >= (dt/V)*base
        row = np.zeros(nv_full); row[a] = 1
        row[13*k:13*k+12] = -dt_over_V * rated
        row[13*k+12] = dt_over_V
        add2(row, dt_over_V * fixed_load[k], INF)
        row = np.zeros(nv_full); row[a] = 1
        add2(row, 0, INF)

        # p_k >= 2*b_j*y_k - b_j^2 for every tangent b_j: the epigraph of the
        # convex piecewise-linear underestimator of y_k^2 on [0, soc_safe]
        # (Eq. (16) as the declared 8-tangent piecewise-linear form).
        for b in tangents:
            row = np.zeros(nv_full)
            row[p] = 1
            row[yv] = -2.0 * b
            add2(row, -b * b, INF)

    A2 = sp.vstack(rows2).tocsc()
    lb2 = np.concatenate(lb2)
    ub2 = np.concatenate(ub2)
    Afull = sp.hstack([A, sp.csc_matrix((A.shape[0], nv_full - nv))]).tocsc()
    Afull = sp.vstack([Afull, A2]).tocsc()
    lbfull = np.concatenate([lb, lb2])
    ubfull = np.concatenate([ub, ub2])

    # Bound the three first-action service groups before optimizing the soft
    # operating objective. The LPs are priority stages, not a fallback.
    # Finite tight column bounds are required by the priority-stage dual
    # certificates. The x-block bounds mirror the identity rows (which remain
    # the audited source); the aux bounds cap a SOC shortfall (1), the score
    # U_B (exactly 1), one interval's discharged Ah, f_ki <= 1 of the
    # allocation-preference epigraph, and the penalty p_k (soc_safe^2: y_k <=
    # soc_safe - soc_min <= soc_safe, so the tangent envelope of y_k^2 never
    # exceeds soc_safe^2).
    col_upper = np.full(nv_full, INF)
    col_upper[:nv] = ub_var
    col_upper[nv:nv + n] = 1.0
    col_upper[nv + n:nv + 2 * n] = 1.0
    col_upper[nv + 2 * n:nv + 3 * n] = (rated.sum() + fixed_load.max()) * dt_over_V
    col_upper[nv + 3*n:p_column] = 1.0
    col_upper[p_column:] = max(soc_safe * soc_safe, 1e-12)
    locks, minima, priority_targets, priority_failure, stages = _priority_rows(
        Afull, lbfull, ubfull, col_upper, f_column, alloc_demand,
        hardware, settings)
    if priority_failure is not None:
        return priority_failure
    lp_started = time.perf_counter()
    Afull = sp.vstack([Afull, locks], format='csc')
    lbfull = np.concatenate([lbfull, minima])
    ubfull = np.concatenate([ubfull, np.full(len(minima), INF)])

    # One-sided linprog form of the locked LP: finite upper-bound rows plus
    # negated finite lower-bound rows, over the full column set.
    finite_hi = np.isfinite(ubfull)
    finite_lo = np.isfinite(lbfull)
    A_ub = sp.vstack([Afull[finite_hi], -Afull[finite_lo]], format='csc')
    b_ub = np.concatenate([ubfull[finite_hi], -lbfull[finite_lo]])
    # Declared per-solve wall-clock budget: a budgeted solve that hits the
    # limit returns a non-optimal status and is recorded as a numerical
    # failure, never silently accepted.
    result = linprog(
        q, A_ub=A_ub, b_ub=b_ub,
        bounds=[(0.0, float(value)) for value in col_upper],
        method='highs',
        options={
            'time_limit': _SOLVE_TIME_LIMIT_S,
            'primal_feasibility_tolerance': max(
                1e-10, float(settings['constraint_tolerance'])),
            'dual_feasibility_tolerance': max(
                1e-10, float(settings['constraint_tolerance'])),
            'ipm_optimality_tolerance': float(settings['solver_tolerance']),
        })
    lp_seconds = time.perf_counter() - lp_started
    if result.status != 0 or result.x is None or not np.all(np.isfinite(result.x)):
        return {'success': False, 'failure_kind': 'numerical_failure',
                'reason': f'HiGHS LP status {result.status}: {result.message}',
                'solver': 'HiGHS-LP', 'solver_status': int(result.status),
                'lp_seconds': lp_seconds, 'priority_stages': stages}
    # Project the returned vertex onto the declared variable box: HiGHS
    # may overshoot a bound by its feasibility tolerance (~1e-7), which the
    # downstream strict [0, 1] PWM validation must never see.
    x = np.clip(np.asarray(result.x[:nv]), lb_var, ub_var)
    plan = x.reshape(n, 13)
    power = plan[:, :12] @ rated + fixed_load - plan[:, 12]
    start_soc = soc - scale * np.r_[0., np.cumsum(power[:-1])]
    branch = np.where(start_soc < .2, 1., np.where(start_soc < .5, .5, .2))
    group_values = group_service(hardware, float(demand[0]), plan[0, :12])
    # The check must accept everything the lock rows themselves permit: lock
    # margin + the LP's feasibility slack on the lock rows + recomputation on
    # a different arithmetic path (see _PRIORITY_MARGIN). Both sequences
    # shrink together (empty on a no-pad garment), so the plain zip is
    # empty-safe (strict=False semantics): any(...) over an empty zip is
    # trivially False.
    if any(value < target - _PRIORITY_MARGIN - settings['solver_tolerance'] - 1e-9
           for value, target in zip(group_values, priority_targets)):
        return {'success': False, 'failure_kind': 'numerical_failure',
                'reason': 'Executed action loses a prior service-group optimum.',
                'solver': 'HiGHS-LP', 'priority_targets': priority_targets,
                'priority_realized': group_values, 'lp_seconds': lp_seconds}

    # Objective decompositions. The solver-consistent reconstruction uses the
    # extrapolated weights actually handed to the LP and the declared tangent
    # envelope of the shortfall (Eq. (16) as the declared 8-tangent
    # piecewise-linear form); the realized-weights diagnostic uses the
    # executed branch weights and the exact quadratic, because the one-step
    # extrapolation and the tangent form are declared approximations.
    y_vals = np.clip(np.asarray(result.x[nv:nv + n]), 0.0, 1.0)
    pwl_penalty = np.maximum.reduce(
        [np.zeros(n)] + [2.0 * b * y_vals - b * b for b in tangents])
    scores = [whole_body_fulfillment(
        float(demand[k]), hardware['efficiencies'], hardware['rated_powers_w'],
        hardware['mask'], plan[k, :12])['score'] for k in range(n)]
    discharged_ah = np.maximum(power, 0) * dt_over_V
    objective = float(np.sum(
        scores - settings['lambda_deg'] * branch_weights * discharged_ah -
        settings['lambda_soc'] * pwl_penalty))
    objective_realized = float(np.sum(
        scores - settings['lambda_deg'] * branch * discharged_ah -
        settings['lambda_soc'] * np.maximum(soc_safe - start_soc, 0) ** 2))
    optimized_objective = -float(result.fun)
    if abs(objective - optimized_objective) > max(1e-5, settings['solver_tolerance'] * 100):
        return {'success': False, 'failure_kind': 'numerical_failure',
                'reason': 'LP objective disagrees with independently reconstructed trajectory.',
                'solver': 'HiGHS-LP', 'objective': objective,
                'optimized_objective': optimized_objective, 'lp_seconds': lp_seconds}
    # Post-solve acceptance guard: audit the returned solution against the
    # declared rows and refuse solver-level garbage. Positive residuals far
    # beyond solver tolerance are recorded as numerical failure, never
    # relabelled as a feasible plan (small strictly-positive values within
    # 1e-6 remain reported in max_constraint_residual for the evidence trail).
    row_values = A @ x
    residual = float(np.max(np.concatenate([lb - row_values, row_values - ub])))
    if not math.isfinite(residual) or residual > 1e-6:
        return {'success': False, 'failure_kind': 'numerical_failure',
                'reason': (f'LP solution violates declared rows by {residual:.3g} '
                           '(non-finite values compare False); rejecting solver output.'),
                'solver': 'HiGHS-LP', 'lp_seconds': lp_seconds,
                'max_constraint_residual': residual}
    return {'success': True, 'failure_kind': None,
            'pwm': plan[:, :12], 'pv_used_w': plan[:, 12],
            'equipment_on': equipment, 'objective': objective,
            'reason': 'HiGHS LP optimal under lexicographic allocation locks',
            'solver': 'HiGHS-LP', 'evaluations': 1,
            'priority_targets': priority_targets, 'priority_realized': group_values,
            'priority_group_labels': _group_indices(hardware)[1],
            'priority_stages': stages, 'lp_seconds': lp_seconds,
            'priority_tolerance': settings['solver_tolerance'],
            'planned_start_soc': start_soc, 'deg_weights': branch_weights,
            'objective_realized_weights': objective_realized,
            'max_constraint_residual': residual}

def simulate(actual, forecast_provider, forecast_mode, service_floor, initial_state,
             hardware, settings, *, strategy='rhc'):
    """Replay one scenario; return configuration.py's operating-evaluator fields.

    actual: full realized demand_w, pv_w, air_c arrays on a uniform interval grid.
    forecast_provider(step, observation, horizon): returns those three forecast
    arrays. observation contains current forcing, SOC, previous equipment action
    and PAST air samples only; no latent battery temperature or future truth.
    forecast_mode: explicit nonempty information-policy description; use e.g.
    'perfect-preview' only when the provider intentionally has future access.
    initial_state: soc, battery_c, equipment_on, air_history_c (strictly past).
    hardware includes the fields documented above and modeling.py's twelve-region
    parameters, heat_capacity_j_per_k, thermal_resistance_k_per_w, equipment_available,
    max_charge_w, max_discharge_w, bus_limit_w. All values are explicit.

    Real-time PV dispatch uses as much actual generation as load, charge power and
    remaining capacity permit. This electrical curtailment is not a fallback PWM
    controller. Current forecast entries must equal current measurements. Subsequent
    forecast errors affect later plans, never overwrite realized trajectory values.
    The discharge weights handed to each solve are extrapolated one step from the
    previous solve's planned SOC trajectory (declared approximation; the first
    step and any short previous plan use the initial 0.2 branch). Returns
    interval steps and Nt+1 states (partial lengths on failure), raw named
    worst residuals, completion, failure_kind, failure_step and cause. Positive
    physical residuals are never hidden by solver tolerance or SOC clipping;
    every service residual uses raw <= 0 acceptance except the no-waste
    oversupply residual, which uses the shared declared _RESIDUAL_TOLERANCE
    margin when demand is positive and exactly 0 when demand is zero.
    """
    if strategy not in ('rhc', 'myopic_qp'):
        raise ValueError('Unknown declared control strategy.')
    validate_setup(hardware, settings, initial_state)
    demand, pv, air = _forcing(actual)
    n = demand.size
    floors = _floors(service_floor, n)
    if not callable(forecast_provider) or not isinstance(forecast_mode, str) or not forecast_mode.strip():
        raise ValueError('A forecast provider and declared information mode are required.')
    soc, battery = initial_state['soc'], initial_state['battery_c']
    previous = initial_state['equipment_on']
    history = list(initial_state['air_history_c'])
    if not all(math.isfinite(v) for v in (soc, battery, *history)):
        raise ValueError('Initial states/history must be finite.')
    equipment_switch(history + [air[0]], previous, hardware['equipment_available'],
                     settings['on_c'], settings['off_c'], settings['filter_window'])
    dt = settings['dt_hours']
    states = [{'time_hours': 0., 'soc': soc, 'battery_c': battery}]
    steps, solves = [], []
    residuals = {'soc_lower': settings['soc_min'] - soc, 'soc_upper': soc - 1}
    failure, failure_step, reason = None, None, ''
    if not settings['soc_min'] <= soc <= 1:
        failure, failure_step, reason = 'policy_failure', 0, 'Initial SOC violates hard bounds.'
    # One-step weight extrapolation state: the previous control step's planned
    # start-of-interval SOC trajectory.
    prev_start_soc = None
    for t in range(n):
        if failure:
            break
        horizon = min(1 if strategy == 'myopic_qp' else settings['horizon_steps'], n-t)
        # Eq. (15)/(16) branches on the previous plan's start-of-interval SOC
        # trajectory shifted one interval forward; when the previous plan does
        # not reach one interval past the current horizon the declared
        # fallback is the initial constant 0.2 branch.
        deg = None
        if prev_start_soc is not None and len(prev_start_soc) >= horizon + 1:
            projected = np.asarray(prev_start_soc[1:horizon + 1], dtype=float)
            deg = np.where(projected < 0.2, 1.0,
                           np.where(projected < 0.5, 0.5, 0.2))
        observation = {'soc': soc, 'equipment_on': previous, 'air_history_c': tuple(history),
                       'demand_w': float(demand[t]), 'pv_w': float(pv[t]), 'air_c': float(air[t])}
        forecast = forecast_provider(t, observation, horizon)
        future = _forcing(forecast)
        if future[0].size != horizon or any(values[0] != observation[key]
                for values, key in zip(future, ('demand_w', 'pv_w', 'air_c'))):
            raise ValueError('Forecast must have the requested length and begin with observed forcing.')
        decision_started = time.perf_counter()
        solution = solve_rhc(soc, previous, history, forecast, floors[t:t+horizon],
                             t+horizon == n, hardware, settings, deg_weights=deg)
        solution['decision_seconds'] = time.perf_counter() - decision_started
        solution['forecast'] = {key: value.copy() for key, value in
                                zip(('demand_w', 'pv_w', 'air_c'), future)}
        solves.append(solution)
        if not solution['success']:
            failure, failure_step, reason = solution['failure_kind'], t, solution['reason']
            break
        prev_start_soc = solution.get('planned_start_soc')
        history.append(float(air[t]))
        on, filtered = equipment_switch(history, previous, hardware['equipment_available'],
                                        settings['on_c'], settings['off_c'], settings['filter_window'])
        pwm = solution['pwm'][0]
        before_pv = electrical_step(soc, pwm, on, 0., hardware, dt)
        charge_room_w = max(0., (1 - soc) * hardware['capacity_wh'] / dt)
        pv_used = min(float(pv[t]), before_pv['load_w'] + min(hardware['max_charge_w'], charge_room_w))
        balance = electrical_step(soc, pwm, on, pv_used, hardware, dt)
        whole = whole_body_fulfillment(float(demand[t]), hardware['efficiencies'],
                                       hardware['rated_powers_w'], hardware['mask'], pwm)
        alloc = allocation_reference(float(demand[t]), hardware['regional_areas_m2'],
                                     hardware['weights'], hardware['efficiencies'],
                                     hardware['rated_powers_w'], hardware['mask'], pwm)
        next_battery = battery_temperature_step(battery, air[t], on * hardware['equipment_available'] *
                        hardware['equipment_power_w'], hardware['heat_capacity_j_per_k'],
                        hardware['thermal_resistance_k_per_w'], dt*3600)
        total_effective_heat_w = whole['total_effective_heat_w']
        current = {'soc_lower': settings['soc_min'] - balance['soc'], 'soc_upper': balance['soc'] - 1,
                   'discharge_w': balance['battery_w'] - hardware['max_discharge_w'],
                   'charge_w': -balance['battery_w'] - hardware['max_charge_w'],
                   'bus_w': balance['load_w'] - hardware['bus_limit_w'],
                   # Raw whole-body service residuals: the deficit side is
                   # accepted at <= 0 exactly; the no-waste oversupply side is
                   # accepted at the shared declared margin when demand > 0
                   # and at exactly 0 when demand == 0 (see _RESIDUAL_TOLERANCE).
                   'service': float(floors[t] * demand[t] - total_effective_heat_w),
                   'service_oversupply': float(total_effective_heat_w - demand[t])}
        if t == n-1:
            terminal_residual = float(settings['terminal_soc'] - balance['soc'])
            # Collapse a difference within the floating-point ulp of the
            # compared magnitudes to zero: it arises from evaluation order of
            # an exactly terminal trajectory, not a physical reserve shortfall.
            # Mirrors the independent audit's rounding in evaluation.py so the
            # planner and the auditor agree on an exactly-terminal boundary.
            if 0 < terminal_residual <= 2 * math.ulp(max(abs(settings['terminal_soc']),
                                                         abs(balance['soc']), 1e-300)):
                terminal_residual = 0.0
            current['terminal_soc'] = terminal_residual
        for key, value in current.items():
            residuals[key] = max(residuals.get(key, -math.inf), value)
        steps.append({'time_hours': t*dt, 'pwm': pwm.copy(), 'equipment_on': on,
                      'filtered_air_c': filtered, 'air_c': float(air[t]), 'demand_w': float(demand[t]),
                      'pv_available_w': float(pv[t]), 'pv_used_w': pv_used,
                      'pv_curtailed_w': float(pv[t]) - pv_used, **balance,
                      'thermal_score': whole['score'],
                      'allocation_fulfillment': alloc['allocation_fulfillment'],
                      'installed': whole['installed'],
                      'total_effective_heat_w': total_effective_heat_w,
                      'unmet_heat_w': max(float(demand[t]) - total_effective_heat_w, 0.0),
                      'allocation_unmet_heat_w': np.where(
                          alloc['installed'],
                          np.maximum(alloc['allocated_task_w'] - alloc['effective_heat_w'], 0.0),
                          np.nan),
                      'discharge_penalty': discharge_penalty(soc, balance['discharged_ah']),
                      'residuals': current})
        soc, battery, previous = balance['soc'], next_battery, on
        states.append({'time_hours': (t+1)*dt, 'soc': soc, 'battery_c': battery})
        # Per-residual acceptance: every residual is raw (<= 0) except
        # 'service_oversupply', which carries the shared declared
        # _PLAN_MARGIN when demand is positive and exactly 0 when demand == 0
        # (the cap row is exact there). The margin is evaluated as the
        # planner's cap-row bound (demand + _PLAN_MARGIN) translated back to
        # residual space — the same rounding on both sides, so the declared
        # mirror is exact and the alpha = 1 coincident boundary, where the
        # plan pins at fl(demand + _PLAN_MARGIN), passes the raw audit as
        # promised; in real arithmetic the threshold equals _PLAN_MARGIN.
        oversupply_limit = (
            (float(demand[t]) + _RESIDUAL_TOLERANCE['service_oversupply'])
            - float(demand[t]) if demand[t] > 0 else 0.0)
        if not all(math.isfinite(v) for v in (soc, battery, *current.values())):
            failure, failure_step, reason = 'numerical_failure', t, 'Nonfinite realized state or residual.'
        elif any(value > (oversupply_limit if key == 'service_oversupply' else 0.0)
                 for key, value in current.items()):
            failure, failure_step, reason = 'policy_failure', t, 'Realized constraint violation.'
    return {'completed': len(steps) == n, 'failure_kind': failure, 'failure_step': failure_step,
            'reason': reason, 'residuals': residuals, 'states': states, 'steps': steps,
            'solves': solves, 'forecast_mode': forecast_mode, 'strategy': strategy,
            'scheduled_steps': n}
