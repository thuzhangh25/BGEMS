'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Independently audit operating trajectories and summarize declared run sets.
The audit reconstructs the whole-body service (Eq. (3), P_supp >= alpha * P_req with
the no-waste cap P_supp <= P_req), Eq. (4), Eq. (13), and Eq. (18)--(22) quantities
from realized forcing and executed actions; reported scores and residuals are not
evidence. Regional numbers are allocation diagnostics of the preserved regional
control structure, never per-region service failures.
'''
import math
import time
from numbers import Integral

import numpy as np

from .modeling import (allocation_reference, battery_temperature_step,
                       whole_body_fulfillment)
from .operation import _PRIORITY_MARGIN as PRIORITY_MARGIN
from .operation import _RESIDUAL_TOLERANCE, _group_indices
from .operation import electrical_step, equipment_switch, group_service, solve_rhc


_FAILURE_KINDS = (None, 'policy_failure', 'numerical_failure', 'physical_infeasibility')


def _worst(residuals, name, value):
    """Keep one named worst-case residual with acceptance at <= 0."""
    value = float(value)
    residuals[name] = max(residuals.get(name, -math.inf), value)


def _range(values):
    if not values:
        return None
    return {'minimum': float(min(values)), 'maximum': float(max(values))}


def evaluate_trajectory(run, actual, service_floor, initial_state, hardware, settings, *,
                        balance_tolerance_wh, time_tolerance_hours):
    """Reconstruct one scenario outcome from immutable forcing and executed actions.

    ``actual`` supplies realized demand and available PV in W and air temperature
    in degrees C on the uniform ``settings['dt_hours']`` grid. ``service_floor`` is
    the declared ``(N,)`` whole-body scalar-alpha array. Initial SOC, battery
    temperature, equipment state and past sensor history come only from
    ``initial_state``.

    ``balance_tolerance_wh`` applies only when recorded SOC is compared with the
    reconstructed Eq. (18) electrical state. ``time_tolerance_hours`` applies only
    to grid timestamps. Physical action, SOC, power, service, and reserve residuals
    are never relaxed. A clean partial prefix has C=False, E/S=None unless already
    violated, and R=None. Malformed or numerical evidence is unresolved, not a
    physical failure. Partial metrics describe only the auditable prefix; low-SOC
    duration follows the discrete start-of-interval state convention.

    Returns a configuration-compatible verdict with independently computed
    completion, named residuals, C/E/S/R components, metrics, and evidence issues.
    """
    for name, value in (('balance_tolerance_wh', balance_tolerance_wh),
                        ('time_tolerance_hours', time_tolerance_hours)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(name + ' must be finite and nonnegative.')

    try:
        demand, pv_available, air = (np.asarray(actual[name], dtype=float)
                                     for name in ('demand_w', 'pv_w', 'air_c'))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('Actual forcing must provide numeric demand_w, pv_w and air_c.') from exc
    if (any(values.ndim != 1 or not values.size or not np.all(np.isfinite(values))
            for values in (demand, pv_available, air)) or
            demand.shape != pv_available.shape or demand.shape != air.shape):
        raise ValueError('Actual forcing must contain aligned, finite, nonempty 1-D arrays.')
    if np.any(demand < 0) or np.any(pv_available < 0):
        raise ValueError('Demand and available PV must be nonnegative.')
    count = demand.size
    floors = np.asarray(service_floor, dtype=float)
    if (floors.shape != (count,) or not np.all(np.isfinite(floors)) or
            np.any((floors < 0) | (floors > 1))):
        raise ValueError('Service floors must be a finite (steps,) scalar-alpha '
                         'array in [0, 1].')

    try:
        dt = settings['dt_hours']
        soc_min, soc_safe, terminal_soc = (settings[name]
                                           for name in ('soc_min', 'soc_safe', 'terminal_soc'))
        soc, battery = initial_state['soc'], initial_state['battery_c']
        previous = initial_state['equipment_on']
        history = list(initial_state['air_history_c'])
        scalar_hardware = tuple(hardware[name] for name in (
            'capacity_wh', 'voltage_v', 'equipment_power_w', 'base_power_w',
            'max_discharge_w', 'max_charge_w', 'bus_limit_w',
            'heat_capacity_j_per_k', 'thermal_resistance_k_per_w'))
    except (KeyError, TypeError) as exc:
        raise ValueError('Hardware, settings and initial state are incomplete.') from exc
    if not all(math.isfinite(value) for value in
               (dt, soc_min, soc_safe, terminal_soc, soc, battery, *history, *scalar_hardware)):
        raise ValueError('Hardware, settings and initial-state scalars must be finite.')
    if dt <= 0 or hardware['capacity_wh'] <= 0 or hardware['voltage_v'] <= 0:
        raise ValueError('Timestep, capacity and voltage must be positive.')
    if hardware['heat_capacity_j_per_k'] <= 0 or hardware['thermal_resistance_k_per_w'] <= 0:
        raise ValueError('Equipment heat capacity and thermal resistance must be positive.')
    if any(hardware[name] < 0 for name in ('equipment_power_w', 'base_power_w',
                                           'max_discharge_w', 'max_charge_w', 'bus_limit_w')):
        raise ValueError('Powers and limits must be nonnegative.')
    if hardware['equipment_available'] not in (0, 1) or previous not in (0, 1):
        raise ValueError('Equipment availability and initial action must be binary.')
    if not 0 <= soc_min <= soc_safe <= 1 or not soc_min <= terminal_soc <= 1:
        raise ValueError('Invalid minimum, safe or terminal SOC declaration.')

    # Public physical functions validate the regional and equipment model inputs.
    whole_body_fulfillment(0., hardware['efficiencies'], hardware['rated_powers_w'],
                           hardware['mask'], np.zeros(12))
    allocation_reference(0., hardware['regional_areas_m2'], hardware['weights'],
                         hardware['efficiencies'], hardware['rated_powers_w'],
                         hardware['mask'], np.zeros(12))
    equipment_switch(history + [float(air[0])], previous, hardware['equipment_available'],
                     settings['on_c'], settings['off_c'], settings['filter_window'])
    battery_temperature_step(battery, float(air[0]), 0., hardware['heat_capacity_j_per_k'],
                             hardware['thermal_resistance_k_per_w'], dt * 3600)

    residuals = {
        'soc_lower': float(soc_min - soc),
        'soc_upper': float(soc - 1),
        'time_alignment_hours': float(-time_tolerance_hours),
        'electrical_balance_wh': float(-balance_tolerance_wh),
        'battery_temperature_state_c': 0.,
        'forcing_alignment': 0.,
    }
    issues = []
    malformed = False
    fatal = False
    priority_audit = settings.get('priority_mode') == 'lexicographic'
    if settings.get('priority_mode') not in (None, 'lexicographic'):
        raise ValueError('Unknown service-priority audit mode.')
    group_values = []
    priority_audit_seconds = 0.0
    policy_action_error = False

    if not isinstance(run, dict):
        steps, states, reported_failure = (), (), None
        issues.append('Run evidence is not a dictionary.')
        malformed = fatal = True
    else:
        steps, states = run.get('steps'), run.get('states')
        reported_failure = run.get('failure_kind')
        if reported_failure not in _FAILURE_KINDS:
            issues.append('Run reports an unknown failure classification.')
            malformed = True
        if not isinstance(steps, (list, tuple)) or not isinstance(states, (list, tuple)):
            steps, states = (), ()
            issues.append('Run evidence must contain step and state sequences.')
            malformed = fatal = True
        elif len(steps) > count or len(states) != len(steps) + 1:
            issues.append('Step/state counts do not form a declared-grid scenario prefix.')
            malformed = fatal = True
        reported_count = run.get('scheduled_steps', count)
        if (isinstance(reported_count, bool) or not isinstance(reported_count, Integral)
                or reported_count != count):
            issues.append('Reported scheduled-step count disagrees with actual forcing.')
            malformed = True

    def invalidate(message, stop=False, nonfinite=False):
        nonlocal malformed, fatal
        issues.append(message)
        malformed = True
        fatal = fatal or stop
        if nonfinite:
            residuals['numerical_evidence'] = math.nan

    def numbers(record, names):
        try:
            values = tuple(float(record[name]) for name in names)
        except (KeyError, TypeError, ValueError):
            return None
        return values if all(math.isfinite(value) for value in values) else None

    recorded_end_socs = (float(soc),)
    if not fatal:
        initial_record = states[0] if isinstance(states[0], dict) else None
        values = numbers(initial_record, ('time_hours', 'soc', 'battery_c')) if initial_record else None
        if values is None:
            invalidate('Initial recorded state is missing, nonnumeric or nonfinite.', True, True)
        else:
            recorded_time, recorded_soc, recorded_battery = values
            recorded_end_socs = (recorded_soc,)
            _worst(residuals, 'soc_lower', soc_min - recorded_soc)
            _worst(residuals, 'soc_upper', recorded_soc - 1)
            _worst(residuals, 'time_alignment_hours',
                   abs(recorded_time) - time_tolerance_hours)
            _worst(residuals, 'electrical_balance_wh',
                   abs(recorded_soc - soc) * hardware['capacity_wh'] - balance_tolerance_wh)
            _worst(residuals, 'battery_temperature_state_c', abs(recorded_battery - battery))
            if residuals['time_alignment_hours'] > 0:
                invalidate('Initial state time is off the declared grid.')
            if residuals['electrical_balance_wh'] > 0:
                invalidate('Initial recorded SOC disagrees with the declared initial state.')
            if residuals['battery_temperature_state_c'] > 0:
                invalidate('Initial battery temperature disagrees with the declared initial state.')

    mask = np.asarray(hardware['mask'], dtype=float)
    installed_flags = mask == 1
    deficit_duration = np.zeros(12)
    unmet_heat = np.zeros(12)
    whole_unmet_heat = 0.
    floor_shortfall_duration = 0.
    # Worst per-step excess of P_supp over its own mirrored no-waste cap bound
    # (see the S component below); the raw worst value is exported separately
    # as the 'service_oversupply' residual.
    oversupply_excess = -math.inf
    scores, observed_air, filtered_air = [], [], []
    soc_values, battery_values = [float(soc)], [float(battery)]
    load_energy = available_pv_energy = used_pv_energy = curtailed_pv_energy = 0.
    discharged_ah = low_soc_duration = equipment_on_duration = 0.
    peak_load = None
    audited = 0

    for index, step in enumerate(steps if not fatal else ()):
        next_state = states[index + 1]
        scalar_names = ('time_hours', 'equipment_on', 'filtered_air_c', 'air_c',
                        'demand_w', 'pv_available_w', 'pv_used_w', 'soc')
        step_values = numbers(step, scalar_names) if isinstance(step, dict) else None
        state_values = numbers(next_state, ('time_hours', 'soc', 'battery_c')) \
            if isinstance(next_state, dict) else None
        try:
            pwm = np.asarray(step['pwm'], dtype=float)
        except (KeyError, TypeError, ValueError):
            pwm = np.empty(0)
        if (step_values is None or state_values is None or pwm.shape != (12,) or
                not np.all(np.isfinite(pwm))):
            invalidate('Step/state %d has missing, malformed or nonfinite evidence.' % index,
                       True, True)
            break
        (step_time, equipment_value, filtered_value, recorded_air, recorded_demand,
         recorded_pv, pv_used, recorded_step_soc) = step_values
        state_time, recorded_state_soc, recorded_battery = state_values
        if equipment_value not in (0., 1.):
            invalidate('Equipment action at step %d is not binary.' % index, True, True)
            break
        equipment_on = int(equipment_value)

        time_excess = max(abs(step_time - index * dt),
                          abs(state_time - (index + 1) * dt)) - time_tolerance_hours
        _worst(residuals, 'time_alignment_hours', time_excess)
        if time_excess > 0:
            invalidate('Step %d is off the declared time grid.' % index)
        if not (recorded_air == air[index] and recorded_demand == demand[index]
                and recorded_pv == pv_available[index]):
            residuals['forcing_alignment'] = 1.
            invalidate('Step %d forcing disagrees with immutable actual forcing.' % index)

        _worst(residuals, 'pwm_lower', np.max(-pwm))
        _worst(residuals, 'pwm_mask', np.max(pwm - mask))
        _worst(residuals, 'equipment_availability',
               equipment_on - hardware['equipment_available'])
        if np.any((pwm < 0) | (pwm > 1)) or pv_used < 0:
            _worst(residuals, 'pv_lower_w', -pv_used)
            issues.append('Step %d has a physically invalid PWM or negative PV action.' % index)
            policy_action_error = True
            break
        _worst(residuals, 'pv_lower_w', -pv_used)
        _worst(residuals, 'pv_upper_w', pv_used - pv_available[index])

        history.append(float(air[index]))
        expected_on, expected_filtered = equipment_switch(
            history, previous, hardware['equipment_available'], settings['on_c'],
            settings['off_c'], settings['filter_window'])
        _worst(residuals, 'equipment_hysteresis', abs(equipment_on - expected_on))
        filter_error = abs(filtered_value - expected_filtered)
        _worst(residuals, 'sensor_filter_state_c', filter_error)
        if filter_error > 0:
            invalidate('Step %d filtered air is inconsistent with sensor history.' % index)

        if priority_audit:
            priority_started = time.perf_counter()
            try:
                original = run['solves'][index]
                declared = original['forecast']
                forecast = {name: np.asarray(declared[name], dtype=float)
                            for name in ('demand_w', 'pv_w', 'air_c')}
                horizon = len(forecast['demand_w'])
                strategy = run.get('strategy')
                expected_horizon = min(1 if strategy == 'myopic_qp'
                                       else settings['horizon_steps'], count-index)
                if (strategy not in ('rhc', 'myopic_qp') or horizon != expected_horizon or
                        any(arr.shape != (horizon,) or not np.all(np.isfinite(arr))
                            for arr in forecast.values()) or
                        any(forecast[name][0] != actual_value for name, actual_value in
                            (('demand_w', demand[index]), ('pv_w', pv_available[index]),
                             ('air_c', air[index])))):
                    raise ValueError('Declared priority forecast does not match the information contract.')
                # Repeat the optimization from the audited state and recorded forecast;
                # reported solver targets alone are not proof of a priority optimum.
                reference = solve_rhc(
                    soc, previous, history[:-1], forecast, floors[index:index+horizon],
                    index+horizon == count, hardware, settings)
                if not reference['success']:
                    raise ValueError('Independent priority solve is unresolved.')
                if list(reference['priority_group_labels']) != list(_group_indices(hardware)[1]):
                    raise ValueError('Recorded priority group labels disagree with '
                                     'the audited hardware.')
                realized_group = group_service(hardware, demand[index], pwm)
                group_values.append(realized_group)
                tol = settings['solver_tolerance']
                # Match the solver's declared lock margin, its feasibility
                # slack on the lock rows, and arithmetic round-off; the
                # published residual itself retains <= 0 acceptance. The
                # margin constant is shared with operation.py's lock rows so
                # the audit never rejects an action the locks permitted.
                # The empty default guards the no-pad garment where the
                # installed-group tuple and the targets are both empty. Use a
                # finite sentinel (no priority groups => no conformance loss)
                # so the residual stays finite and is never misread as a
                # numerical failure by the search enumeration.
                loss = max((target - achieved - PRIORITY_MARGIN - tol - 1e-9
                            for target, achieved in zip(reference['priority_targets'],
                                                        realized_group)),
                           default=-1.0)
                _worst(residuals, 'priority_conformance', loss)
                if loss > 0:
                    issues.append('Step %d sacrifices an achievable higher-priority group.' % index)
            except (KeyError, IndexError, TypeError, ValueError, FloatingPointError) as error:
                invalidate('Step %d priority evidence cannot be independently verified: %s' %
                           (index, error))
            finally:
                priority_audit_seconds += time.perf_counter() - priority_started
        electrical = electrical_step(soc, pwm, equipment_on, pv_used, hardware, dt)
        # Independent arithmetic path: rebuild P_req, P_supp and the beta_i
        # allocation reference from the executed PWM (never solver outputs).
        whole = whole_body_fulfillment(
            float(demand[index]), hardware['efficiencies'], hardware['rated_powers_w'],
            hardware['mask'], pwm)
        alloc = allocation_reference(
            float(demand[index]), hardware['regional_areas_m2'], hardware['weights'],
            hardware['efficiencies'], hardware['rated_powers_w'], hardware['mask'], pwm)
        next_battery = battery_temperature_step(
            battery, float(air[index]),
            equipment_on * hardware['equipment_available'] * hardware['equipment_power_w'],
            hardware['heat_capacity_j_per_k'], hardware['thermal_resistance_k_per_w'], dt * 3600)

        balance_error = max(abs(recorded_step_soc - electrical['soc']),
                            abs(recorded_state_soc - electrical['soc'])) * hardware['capacity_wh']
        recorded_end_socs = (recorded_step_soc, recorded_state_soc)
        thermal_error = abs(recorded_battery - next_battery)
        _worst(residuals, 'electrical_balance_wh', balance_error - balance_tolerance_wh)
        _worst(residuals, 'battery_temperature_state_c', thermal_error)
        if balance_error > balance_tolerance_wh:
            invalidate('Step %d recorded SOC fails the reconstructed Wh balance.' % index)
        if thermal_error > 0:
            invalidate('Step %d recorded battery temperature fails Eq. (4).' % index)

        for observed_soc in (electrical['soc'], recorded_step_soc, recorded_state_soc):
            _worst(residuals, 'soc_lower', soc_min - observed_soc)
            _worst(residuals, 'soc_upper', observed_soc - 1)
        _worst(residuals, 'discharge_w', electrical['battery_w'] - hardware['max_discharge_w'])
        _worst(residuals, 'charge_w', -electrical['battery_w'] - hardware['max_charge_w'])
        _worst(residuals, 'bus_w', electrical['load_w'] - hardware['bus_limit_w'])
        # Raw whole-body service residuals: P_req(t) = demand, alpha(t) the
        # scalar floor, P_supp(t) the total effective supplied heat. The
        # deficit side is accepted at <= 0 exactly; the no-waste oversupply
        # side is recorded raw and accepted per step at the mirrored planner
        # cap bound (see the S component below), shared with operation.py
        # through _RESIDUAL_TOLERANCE.
        required_heat_w = float(floors[index] * demand[index])
        supplied_heat_w = whole['total_effective_heat_w']
        _worst(residuals, 'service', required_heat_w - supplied_heat_w)
        raw_oversupply = supplied_heat_w - float(demand[index])
        if demand[index] > 0:
            # Mirror the planner's rounded cap-row bound (P_req + margin,
            # translated back to residual space) with the same rounding on
            # both sides — exactly 0 when P_req = 0; never an independent
            # tolerance.
            oversupply_limit = ((float(demand[index]) +
                                 _RESIDUAL_TOLERANCE['service_oversupply']) -
                                float(demand[index]))
        else:
            oversupply_limit = 0.
        # Export the excess over the declared bound, not the raw value, so an
        # accepted candidate's exported residual reads <= 0 under the same
        # zero-tolerance rule the search enumeration applies.
        _worst(residuals, 'service_oversupply', raw_oversupply - oversupply_limit)
        oversupply_excess = max(oversupply_excess,
                                raw_oversupply - oversupply_limit)

        # Regional allocation diagnostics: a node whose pad undersupplies its
        # beta_i allocation share produces metrics only, never an S failure.
        deficit_duration += (alloc['installed'] & (alloc['allocated_task_w'] > 0) &
                             (alloc['allocation_fulfillment'] < 1)) * dt
        unmet_heat += np.where(alloc['installed'],
                               np.maximum(alloc['allocated_task_w'] -
                                          alloc['effective_heat_w'], 0.), 0.) * dt
        # Whole-body service diagnostics (structurally zero when S passes).
        whole_unmet_heat += max(required_heat_w - supplied_heat_w, 0.) * dt
        if supplied_heat_w < required_heat_w:
            floor_shortfall_duration += dt
        scores.append(whole['score'])
        load_energy += electrical['load_w'] * dt
        available_pv_energy += pv_available[index] * dt
        used_pv_energy += pv_used * dt
        curtailed_pv_energy += (pv_available[index] - pv_used) * dt
        discharged_ah += electrical['discharged_ah']
        low_soc_duration += dt if soc < soc_safe else 0.
        equipment_on_duration += equipment_on * dt
        peak_load = electrical['load_w'] if peak_load is None else max(peak_load, electrical['load_w'])
        observed_air.append(float(air[index]))
        filtered_air.append(expected_filtered)
        soc, battery, previous = electrical['soc'], next_battery, equipment_on
        soc_values.append(float(soc))
        battery_values.append(float(battery))
        audited += 1

    complete_component = (len(steps) == count) if not malformed and not fatal \
        and not policy_action_error and audited == len(steps) else None
    if complete_component is False:
        issues.append('Only %d of %d intervals are present; unseen intervals were not imputed.' %
                      (audited, count))

    electrical_names = ('pwm_lower', 'pwm_mask', 'equipment_availability',
                        'equipment_hysteresis', 'pv_lower_w', 'pv_upper_w', 'soc_lower',
                        'soc_upper', 'discharge_w', 'charge_w', 'bus_w')
    electrical_violation = any(residuals.get(name, -math.inf) > 0
                               for name in electrical_names)
    # S accepts the whole-body service only: the deficit-side residual at
    # <= 0 raw, the no-waste oversupply residual per step at the mirrored
    # planner cap bound (oversupply_excess pairs each step's raw P_supp -
    # P_req with its own demand-dependent limit from the shared
    # _RESIDUAL_TOLERANCE constant — exactly 0 when P_req = 0, whose
    # oversupply is also caught exactly by the raw 'service' residual
    # -P_supp), and the priority allocation-preference conformance at its
    # declared guard. Regional allocation shortfalls never enter S.
    service_violation = (residuals.get('service', -math.inf) > 0 or
                         oversupply_excess > 0 or
                         residuals.get('priority_conformance', -1.0) > 0)
    electrical_component = False if electrical_violation else \
        True if complete_component is True and audited == count else None
    service_component = False if service_violation else \
        True if complete_component is True and audited == count else None
    if complete_component is True and audited == count:
        terminal_residual = float(max(
            terminal_soc - observed_soc for observed_soc in (soc, *recorded_end_socs)))
        # Collapse a difference within the floating-point ulp of the compared
        # magnitudes to zero: it arises from evaluation order of an exactly
        # terminal trajectory, not from a physical reserve shortfall. This is
        # rounding of representation error, not a tolerance on the constraint.
        if 0 < terminal_residual <= 2 * math.ulp(max(abs(terminal_soc), abs(soc), 1e-300)):
            terminal_residual = 0.0
        residuals['terminal_soc'] = terminal_residual
        reserve_component = residuals['terminal_soc'] <= 0
        terminal_metric = float(soc)
    else:
        reserve_component = terminal_metric = None
    components = {'C': complete_component, 'E': electrical_component,
                  'S': service_component, 'R': reserve_component}
    completed = complete_component is True

    if isinstance(run, dict) and isinstance(run.get('completed'), bool) and run['completed'] != completed:
        issues.append('Reported completion disagrees with the audited trace and was ignored.')
    if (complete_component is True and
            reported_failure in ('policy_failure', 'physical_infeasibility') and
            all(value is True for value in components.values())):
        invalidate('Reported failure contradicts a fully compliant trajectory.')
    if malformed or reported_failure == 'numerical_failure':
        status, failure_kind = 'unresolved', 'numerical_failure'
    elif electrical_component is False or service_component is False or reserve_component is False:
        status, failure_kind = 'failed', 'policy_failure'
    elif all(value is True for value in components.values()):
        status, failure_kind = 'success', None
    elif complete_component is False and reported_failure in ('policy_failure', 'physical_infeasibility'):
        status, failure_kind = 'failed', reported_failure
    elif policy_action_error:
        status, failure_kind = 'failed', 'policy_failure'
    else:
        status, failure_kind = 'unresolved', 'numerical_failure'
        if complete_component is False and reported_failure is None:
            issues.append('The clean truncated prefix has no failure classification.')

    metrics = {
        'scheduled_steps': int(count),
        'observed_steps': int(audited),
        'full_duration_hours': float(count * dt),
        'observed_duration_hours': float(audited * dt),
        'mean_thermal_score_observed': float(np.mean(scores)) if scores else None,
        'whole_body_unmet_heat_wh': float(whole_unmet_heat),
        'service_floor_shortfall_duration_hours': float(floor_shortfall_duration),
        'allocation_deficit_duration_hours': [
            float(value) if installed_flag else None
            for value, installed_flag in zip(deficit_duration, installed_flags)],
        'allocation_unmet_heat_wh': [
            float(value) if installed_flag else None
            for value, installed_flag in zip(unmet_heat, installed_flags)],
        'electrical_load_energy_wh': float(load_energy),
        'pv_available_energy_wh': float(available_pv_energy),
        'pv_used_energy_wh': float(used_pv_energy),
        'pv_curtailed_energy_wh': float(curtailed_pv_energy),
        'discharged_ah': float(discharged_ah),
        'minimum_soc': float(min(soc_values)),
        'observed_end_soc': float(soc),
        'terminal_soc': terminal_metric,
        'low_soc_duration_hours': float(low_soc_duration),
        'peak_load_w': float(peak_load) if peak_load is not None else None,
        'equipment_on_duration_hours': float(equipment_on_duration),
        'air_temperature_range_c': _range(observed_air),
        'filtered_air_temperature_range_c': _range(filtered_air),
        'battery_temperature_range_c': _range(battery_values),
        'priority_groups_observed': [list(values) for values in group_values],
        'priority_audit_seconds': priority_audit_seconds,
    }
    return {'completed': completed, 'residuals': residuals, 'failure_kind': failure_kind,
            'components': components, 'success': status == 'success', 'status': status,
            'metrics': metrics, 'issues': issues}


def summarize_runs(evaluations, scheduled_runs):
    """Count candidate outcomes using the declared candidate count as denominator.

    Missing and unresolved outcomes remain explicit. This function introduces no
    confidence interval, sampling assumption, or replacement for a missing run.
    """
    if (isinstance(scheduled_runs, bool) or not isinstance(scheduled_runs, Integral)
            or scheduled_runs < 0):
        raise ValueError('scheduled_runs must be a nonnegative integer.')
    records = list(evaluations)
    if len(records) > scheduled_runs:
        raise ValueError('Available evaluations cannot exceed scheduled runs.')
    statuses = [record.get('status') if isinstance(record, dict) else None for record in records]
    if any(status not in ('success', 'failed', 'unresolved') for status in statuses):
        raise ValueError('Every available evaluation must have a recognized status.')
    success, failed, unresolved = (statuses.count(name)
                                   for name in ('success', 'failed', 'unresolved'))
    missing = scheduled_runs - len(records)
    fraction = success / scheduled_runs if scheduled_runs else None
    return {
        'scheduled': int(scheduled_runs),
        'available': len(records),
        'resolved': success + failed,
        'success': success,
        'failed': failed,
        'unresolved': unresolved,
        'missing': missing,
        'unresolved_or_missing': unresolved + missing,
        'candidate_feasible_fraction': fraction,
        'candidate_denominator': int(scheduled_runs),
    }
