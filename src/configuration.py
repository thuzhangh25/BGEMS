'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Enumerate hardware configurations, evaluate objectives and select a Pareto member.
Physical inputs and the operating evaluator are required; diagnostic screens do not
certify scenario feasibility. Equation numbers follow docs/sections/2_methods.tex.
'''
import math
import time
from numbers import Integral

import numpy as np
from .modeling import whole_body_fulfillment


def thermal_capability(required_power_w, efficiencies, rated_powers_w, mask):
    """Eq. (6), Eq. (7): mean full-activation whole-body capability on uniform steps.

    required_power_w is a nonempty 1-D W trajectory for this garment under the
    shared scenario. Scores follow the whole-body total-heat service definition
    (Eq. (3) replacement) at full activation u = ones(12): each step contributes
    min(1, P_supp_full / P_req), with the zero-demand convention (score 1 when
    P_req = 0; a no-pad garment therefore scores 1.0 on zero-demand steps —
    accepted convention). The equipment heater is excluded. This is an Eq. (6)/(7)
    diagnostic, not feasibility evidence. Uniform steps cover the entire
    scenario, so dt cancels in the time average.
    """
    demand = np.asarray(required_power_w, dtype=float)
    if demand.ndim != 1 or demand.size == 0:
        raise ValueError('Demand must be a nonempty one-dimensional trajectory.')
    duty = np.ones(12)
    return float(np.mean([
        whole_body_fulfillment(power, efficiencies, rated_powers_w, mask, duty)['score']
        for power in demand
    ]))


def configuration_objectives(y, h, base_mass_kg, garment_mass_kg,
                             cell_mass_kg, pv_mass_kg, base_cost, garment_cost,
                             cell_cost, pv_cost):
    """Eq. (8), Eq. (9): return (total mass in kg, acquisition cost).

    Garment mass/cost includes integrated pads. Thermal service is checked
    through the operating trajectory before either resource is optimized.
    """
    if any(isinstance(n, bool) or not isinstance(n, Integral) for n in (y, h)) or y < 1 or h < 0:
        raise ValueError('Battery/PV counts must be integers with y >= 1 and h >= 0.')
    values = (base_mass_kg, garment_mass_kg, cell_mass_kg, pv_mass_kg,
              base_cost, garment_cost, cell_cost, pv_cost)
    if not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError('Masses and costs must be finite and nonnegative.')
    return (base_mass_kg + garment_mass_kg + y * cell_mass_kg + h * pv_mass_kg,
            base_cost + garment_cost + y * cell_cost + h * pv_cost)


def diagnostic_screens(y, h, cell_capacity_wh, initial_soc, minimum_soc,
                       pv_unit_power_w, full_load_w, effective_body_power_w,
                       required_power_w, dt_hours):
    """Eq. (10), Eq. (11): diagnostic residuals; <= 0 means screen passes.

    PV input is an already converted per-unit electrical W trajectory: do not
    multiply efficiency again. full_load_w includes all available heaters
    (including equipment) and electronics. Both screens are non-excluding.
    All trajectories share uniform dt_hours and the full scenario horizon.
    """
    pv, load, demand = (np.asarray(v, dtype=float)
                        for v in (pv_unit_power_w, full_load_w, required_power_w))
    if any(v.ndim != 1 or v.size == 0 or not np.all(np.isfinite(v)) or np.any(v < 0)
           for v in (pv, load, demand)) or pv.shape != load.shape or pv.shape != demand.shape:
        raise ValueError('Power trajectories must be finite, nonnegative and aligned.')
    if any(isinstance(n, bool) or not isinstance(n, Integral) for n in (y, h)) or y < 1 or h < 0:
        raise ValueError('Invalid component counts.')
    if not all(math.isfinite(v) for v in (cell_capacity_wh, initial_soc, minimum_soc,
                                         effective_body_power_w, dt_hours)):
        raise ValueError('Diagnostic inputs must be finite.')
    if cell_capacity_wh <= 0 or dt_hours <= 0 or effective_body_power_w < 0 or not 0 <= minimum_soc <= initial_soc <= 1:
        raise ValueError('Invalid capacity, timestep, body power or SOC bounds.')
    return {
        'energy_wh': float(load.sum() * dt_hours - cell_capacity_wh * y *
                           (initial_soc - minimum_soc) - pv.sum() * h * dt_hours),
        'body_power_w': float(demand.max() - effective_body_power_w),
    }


def _device_number(value, name, *, positive=False):
    """Read one runtime device quantity and reject unset ``None`` values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Device {name} must be explicitly set to a finite number.')
    value = float(value)
    if (positive and value <= 0) or (not positive and value < 0):
        qualifier = 'positive' if positive else 'nonnegative'
        raise ValueError(f'Device {name} must be {qualifier}.')
    return value


def _garment(config, garment_id):
    matches = [garment for garment in config.garments if garment.id == garment_id]
    if len(matches) != 1:
        raise ValueError('Candidate garment ID must identify exactly one configured garment.')
    return matches[0]


def build_candidate(config, scenario_data, weather_data, y, h, garment_id):
    """Construct one Stage 2 candidate from the five-class parameter model.

    Scenario demand is produced by operation.build_scenario; this function only
    combines it with one declared hardware tuple. ``equipment_heater_required`` is
    an acceptance requirement. It never installs or activates equipment hardware.
    """
    if (any(isinstance(value, bool) or not isinstance(value, Integral)
            for value in (y, h, garment_id)) or y < 1 or h < 0):
        raise ValueError('Candidate counts/ID must be integers with y >= 1 and h >= 0.')
    if y > config.max_battery_cells or h > config.max_pv_units:
        raise ValueError('Candidate lies outside the declared search domain.')
    garment = _garment(config, garment_id)
    device = config.device
    if not isinstance(device.equipment_available, bool):
        raise ValueError('Device equipment_available must be boolean.')
    cell_capacity_wh = _device_number(
        device.cell_capacity_wh, 'cell_capacity_wh', positive=True)
    pv_rating_w = _device_number(device.pv_rating_w, 'pv_rating_w', positive=True)
    equipment_power_w = _device_number(
        device.equipment_power_w, 'equipment_power_w')
    equipment_available = int(device.equipment_available and equipment_power_w > 0)
    initial_state = config.scenario.initial_state
    structural = {
        'equipment_unavailable': float(
            bool(config.scenario.equipment_heater_required) and not equipment_available),
        'initial_equipment_unavailable': float(
            bool(initial_state['equipment_on']) and not equipment_available),
    }
    hardware = {
        'region_order': tuple(config.person.region_order),
        'rated_powers_w': np.asarray(device.rated_powers_w, dtype=float),
        'mask': np.asarray(garment.mask, dtype=float),
        'efficiencies': np.asarray(device.efficiencies, dtype=float),
        'weights': np.asarray(config.person.weights, dtype=float),
        'regional_areas_m2': np.asarray(config.person.regional_areas_m2, dtype=float),
        'capacity_wh': y * cell_capacity_wh,
        'voltage_v': _device_number(device.voltage_v, 'voltage_v', positive=True),
        'equipment_power_w': equipment_power_w,
        'equipment_available': equipment_available,
        'base_power_w': _device_number(device.base_power_w, 'base_power_w'),
        'max_discharge_w': y * _device_number(
            device.max_discharge_w, 'max_discharge_w'),
        'max_charge_w': y * _device_number(device.max_charge_w, 'max_charge_w'),
        'bus_limit_w': _device_number(device.bus_limit_w, 'bus_limit_w'),
        'heat_capacity_j_per_k': y * _device_number(
            device.heat_capacity_j_per_k, 'heat_capacity_j_per_k', positive=True),
        'thermal_resistance_k_per_w': _device_number(
            device.thermal_resistance_k_per_w,
            'thermal_resistance_k_per_w',
            positive=True,
        ),
    }
    demand_w = np.asarray(scenario_data['demand_w'], dtype=float)
    air_c = np.asarray(scenario_data['air_c'], dtype=float)
    reference_pv_w = np.asarray(weather_data['reference_pv_w'], dtype=float)
    reference_rating_w = config.scenario.pv_reference_rating_w
    if (not isinstance(reference_rating_w, (int, float)) or
            not math.isfinite(reference_rating_w) or reference_rating_w <= 0):
        raise ValueError('Scenario pv_reference_rating_w must be finite and positive.')
    if demand_w.shape != air_c.shape or demand_w.shape != reference_pv_w.shape:
        raise ValueError('Scenario demand, temperature and PV trajectories must align.')
    pv_unit_w = reference_pv_w * (pv_rating_w / reference_rating_w)
    actual = {'demand_w': demand_w, 'pv_w': h * pv_unit_w, 'air_c': air_c}
    capability = thermal_capability(
        demand_w,
        hardware['efficiencies'],
        hardware['rated_powers_w'],
        hardware['mask'],
    )
    full_load_w = np.full(
        demand_w.shape,
        float(np.dot(hardware['rated_powers_w'], hardware['mask']) +
              equipment_available * equipment_power_w + hardware['base_power_w']),
    )
    effective_body_power_w = float(np.dot(
        hardware['rated_powers_w'] * hardware['efficiencies'], hardware['mask']))
    diagnostics = diagnostic_screens(
        y,
        h,
        cell_capacity_wh,
        initial_state['soc'],
        config.settings['soc_min'],
        pv_unit_w,
        full_load_w,
        effective_body_power_w,
        demand_w,
        config.settings['dt_hours'],
    )
    objectives = configuration_objectives(
        y, h,
        _device_number(device.base_mass_kg, 'base_mass_kg'),
        garment.mass_kg,
        _device_number(device.cell_mass_kg, 'cell_mass_kg'),
        _device_number(device.pv_mass_kg, 'pv_mass_kg'),
        _device_number(device.base_cost, 'base_cost'),
        garment.cost,
        _device_number(device.cell_cost, 'cell_cost'),
        _device_number(device.pv_cost, 'pv_cost'),
    )
    return {
        'objectives': objectives,
        'capability': capability,
        'diagnostics': diagnostics,
        'structural_residuals': structural,
        'hardware': hardware,
        'actual': actual,
        'garment_id': garment_id,
    }


def search_configuration(max_battery_cells, max_pv_units, garment_ids,
                         evaluate_candidate, evaluate_operation, priority):
    """Algorithm 1 and Eq. (12): full enumeration followed by Pareto selection.

    garment_ids: Distinct nonnegative integer catalogue IDs; no implicit catalogue.
    priority: Permutation of ('mass', 'cost'), fixed before search.
    evaluate_candidate(y, h, g): Returns a dict with 'objectives' (mass, cost),
        'diagnostics' (named finite residuals) and 'structural_residuals' (named
        finite residuals, <= 0 accepted). May carry extra hardware/scenario data.
    evaluate_operation(y, h, g, candidate): Required real scenario evaluator.
        Returns 'completed' (bool), 'residuals' (nonempty named enforced residuals),
        and 'failure_kind': None, 'physical_infeasibility', 'policy_failure', or
        'numerical_failure'. Attach traces/reasons as extra dict fields. The
        evaluator must include full-horizon service, power, SOC and reserve checks.
        Nonfinite operating residuals are numerical failure, never feasibility.

    Returns all evaluated records, feasible/Pareto subsets, selected record or
    None, status, and declared priority. Unexpected evaluator exceptions propagate;
    they are not silently reclassified as physical infeasibility. All inequalities
    use residual <= 0 without a hidden feasibility tolerance.
    """
    if not callable(evaluate_candidate) or not callable(evaluate_operation):
        raise TypeError('Both candidate and real operating evaluators are required.')
    for value, lower in ((max_battery_cells, 1), (max_pv_units, 0)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < lower:
            raise ValueError('Invalid integer search bound.')
    garments = tuple(garment_ids)
    if not garments or any(isinstance(g, bool) or not isinstance(g, Integral) or g < 0 for g in garments) or len(set(garments)) != len(garments):
        raise ValueError('Garment IDs must be distinct nonnegative integers.')
    priority = tuple(priority)
    names = ('mass', 'cost')
    if len(priority) != 2 or set(priority) != set(names):
        raise ValueError('Priority must order mass and cost exactly once.')
    order = tuple(names.index(name) for name in priority)
    records, feasible = [], []
    for y in range(1, max_battery_cells + 1):
        for h in range(max_pv_units + 1):
            for g in sorted(garments):
                candidate = evaluate_candidate(y, h, g)
                objectives = tuple(candidate['objectives'])
                if len(objectives) != 2 or not all(math.isfinite(v) for v in objectives):
                    raise ValueError('Candidate must provide two finite resource objectives.')
                for field in ('diagnostics', 'structural_residuals'):
                    if not all(math.isfinite(v) for v in candidate[field].values()):
                        raise ValueError('Candidate residuals must be finite.')
                record = {'configuration': (y, h, g), 'objectives': objectives,
                          'candidate': candidate, 'operation': None,
                          'status': 'physical_infeasibility'}
                if all(v <= 0 for v in candidate['structural_residuals'].values()):
                    run = evaluate_operation(y, h, g, candidate)
                    record['operation'] = run
                    if not isinstance(run['completed'], bool) or not run['residuals']:
                        raise ValueError('Operation must report completion and enforced residuals.')
                    kind = run['failure_kind']
                    if kind not in (None, 'physical_infeasibility', 'policy_failure', 'numerical_failure'):
                        raise ValueError('Unknown operating failure classification.')
                    residuals = tuple(run['residuals'].values())
                    if not all(math.isfinite(v) for v in residuals):
                        record['status'] = 'numerical_failure'
                    elif kind is not None:
                        record['status'] = kind
                    elif run['completed'] and all(v <= 0 for v in residuals):
                        record['status'] = 'feasible'
                        feasible.append(record)
                    else:
                        record['status'] = 'policy_failure'
                records.append(record)
    selection_started = time.perf_counter()
    # Strict dominance retains distinct hardware with identical objective vectors.
    pareto = [a for a in feasible if not any(
        all(x <= z for x, z in zip(b['objectives'], a['objectives'])) and
        any(x < z for x, z in zip(b['objectives'], a['objectives']))
        for b in feasible)]
    selected = min(pareto, key=lambda r: tuple(r['objectives'][i] for i in order)
                   + r['configuration']) if pareto else None
    return {'records': records, 'feasible': feasible, 'pareto': pareto,
            'selected': selected, 'priority': priority,
            'selection_seconds': time.perf_counter() - selection_started,
            'status': 'selected' if selected else 'no_feasible_configuration'}
