'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Enumerate the design catalogue and return feasible, Pareto and selected
configurations for subsequent controller execution.
'''

import time
from numbers import Integral

import numpy as np

from .modeling import demand_series, equipment_heater_action, regional_fulfillment


def prepare_mission(config, weather):
    """Precompute mission demand and activity arrays once per garment."""
    count = len(weather['air_c'])
    dt = config.settings['dt_hours']
    met = np.asarray(
        [config.scenario.get_activity(round(k * dt, 12)) for k in range(count)],
        dtype=float,
    )
    return {
        g.id: {
            'demand_w': demand_series(config, g, weather['air_c'], met),
            'air_c': np.asarray(weather['air_c'], dtype=float),
            'met': met,
            'planned_met': met,
        }
        for g in config.garments
    }


def _design_budget(config, hardware, actual, reference_load_w, full_activation_energy_wh):
    """Eq. (9): screen reference power in W and energy in Wh.

    Credit concurrent PV up to the reference load. The electrical plant models
    charging during operation. Full-activation energy is reported as a diagnostic.
    """
    dt = config.settings['dt_hours']
    usable = hardware['capacity_wh'] * (
        config.scenario.initial_state['soc'] - config.settings['soc_min']
    )
    deficit_w = np.maximum(reference_load_w - actual['pv_w'], 0.0)
    budget_wh = config.endurance_multiplier * float(np.sum(deficit_w) * dt)
    effective_max_w = float(np.sum(
        hardware['rated_powers_w'] * hardware['efficiencies'] * hardware['mask']
    ))
    constraints = {
        'reference_energy_residual_wh': budget_wh - usable,
        'bus_power_residual_w': float(np.max(reference_load_w)) - hardware['bus_limit_w'],
        'discharge_power_residual_w': float(np.max(deficit_w)) - hardware['max_discharge_w'],
        'thermal_capability_residual_w': float(np.max(actual['demand_w'])) - effective_max_w,
    }
    full_budget_wh = config.endurance_multiplier * max(
        0.0, full_activation_energy_wh - float(np.sum(actual['pv_w']) * dt)
    )
    return {
        'reference_energy_budget_wh': budget_wh,
        'design_constraints': constraints,
        'diagnostics': {'full_activation_energy_residual_wh': full_budget_wh - usable},
        'design_admissible': all(value <= 0 for value in constraints.values()),
    }


def build_candidate(config, scenario_data, weather_data, y, h, garment_id):
    """Build one generic electrical plant and its declared design diagnostics."""
    if any(
        isinstance(v, bool) or not isinstance(v, Integral) for v in (y, h, garment_id)
    ):
        raise ValueError('Candidate counts and garment ID must be integers.')
    if not 1 <= y <= config.max_battery_units or not 0 <= h <= config.max_pv_units:
        raise ValueError('Candidate lies outside the declared domain.')
    garments = [g for g in config.garments if g.id == garment_id]
    if len(garments) != 1:
        raise ValueError('Unknown garment ID.')
    g, d = garments[0], config.device
    installed = bool(config.scenario.equipment_heater_installed)
    if installed != d.equipment_available:
        raise ValueError('Scenario and device installation flags disagree.')
    actual = {
        key: np.asarray(value, dtype=float) for key, value in scenario_data.items()
    }
    actual['pv_w'] = np.asarray(weather_data['reference_pv_w'], dtype=float) * (
        h * d.pv_rating_w / config.scenario.pv_reference_rating_w
    )
    hardware = {
        'rated_powers_w': np.asarray(d.rated_powers_w),
        'mask': np.asarray(g.mask),
        'efficiencies': np.asarray(d.efficiencies),
        'weights': np.asarray(config.person.weights),
        'regional_areas_m2': np.asarray(config.person.regional_areas_m2),
        'region_order': tuple(config.person.region_order),
        'capacity_wh': y * d.battery_unit_capacity_wh,
        'voltage_v': d.voltage_v,
        'max_discharge_w': y * d.max_discharge_w,
        'max_charge_w': y * d.max_charge_w,
        'bus_limit_w': d.bus_limit_w,
        'base_power_w': d.base_power_w,
        'equipment_available': installed,
        'equipment_power_w': d.equipment_power_w,
    }
    dt = config.settings['dt_hours']
    full_body = float(np.dot(hardware['rated_powers_w'], hardware['mask']))
    equipment = np.asarray(
        [equipment_heater_action(t, installed) for t in actual['air_c']]
    )
    full_load = full_body + d.base_power_w + equipment * d.equipment_power_w
    full_energy_wh = float(np.sum(full_load) * dt)
    effective_rated_w = hardware['rated_powers_w'] * hardware['efficiencies']
    reference_load_w = np.empty(len(actual['demand_w']))
    values = []
    for index, required_power_w in enumerate(actual['demand_w']):
        regional = regional_fulfillment(
            required_power_w,
            hardware['regional_areas_m2'],
            hardware['weights'],
            hardware['efficiencies'],
            hardware['rated_powers_w'],
            hardware['mask'],
            np.ones(12),
        )
        # Allocate P_req,i across all twelve on-body nodes.
        # Demand-capped duty attains the full-activation score.
        reference_duty = np.minimum(
            hardware['mask'],
            np.divide(
                regional['regional_demand_w'], effective_rated_w,
                out=np.zeros(12), where=effective_rated_w > 0,
            ),
        )
        reference_load_w[index] = (
            float(np.sum(hardware['rated_powers_w'] * reference_duty))
            + d.base_power_w + equipment[index] * d.equipment_power_w
        )
        values.append(regional['score'])
    score = float(np.mean(values))
    demand_total = float(np.sum(actual['demand_w']))
    areas = hardware['regional_areas_m2']
    uncovered = (
        float(np.sum(areas[hardware['mask'] == 0]) / np.sum(areas))
        if demand_total > 0
        else 0.0
    )
    mass = (
        d.base_mass_kg
        + g.mass_kg
        + y * d.battery_unit_mass_kg
        + h * d.pv_mass_kg
        + installed * d.equipment_mass_kg
    )
    cost = (
        d.base_cost
        + g.cost
        + y * d.battery_unit_cost
        + h * d.pv_cost
        + installed * d.equipment_cost
    )
    return {
        'configuration': (int(y), int(h), int(garment_id)),
        'objectives': (-score, float(mass), float(cost)),
        'score': score,
        'mass_kg': float(mass),
        'cost_usd': float(cost),
        'garment': g.name,
        'uncovered_demand_ratio': uncovered,
        **_design_budget(config, hardware, actual, reference_load_w, full_energy_wh),
        'full_activation_energy_wh': full_energy_wh,
        'reference_load_energy_wh': float(np.sum(reference_load_w) * dt),
        'reference_load_w': reference_load_w,
        'hardware': hardware,
        'actual': actual,
    }


def pareto_front(records):
    """Strict dominance retains different configurations with identical objectives."""
    if not records:
        return []
    values = np.asarray([r['objectives'] for r in records], dtype=float)
    if values.shape != (len(records), 3) or not np.all(np.isfinite(values)):
        raise ValueError('Every candidate must have three finite objectives.')
    return [
        r
        for i, r in enumerate(records)
        if not np.any(
            np.all(values <= values[i], axis=1) & np.any(values < values[i], axis=1)
        )
    ]


def select_configuration(front, preference):
    """Scenario preferences, with explicit deterministic final ties."""
    if preference not in ('lightweight', 'comfort', 'balanced'):
        raise ValueError('Unknown selection preference.')
    if not front:
        return None
    tie = lambda r: (-r['score'], r['mass_kg'], r['cost_usd'], *r['configuration'])
    if preference == 'comfort':
        return min(front, key=tie)
    if preference == 'lightweight':
        return min(
            front,
            key=lambda r: (
                r['mass_kg'],
                -r['score'],
                r['cost_usd'],
                *r['configuration'],
            ),
        )
    best = max(r['score'] for r in front)
    pool = [r for r in front if r['score'] >= best - 0.05]
    covered = [r for r in pool if r['uncovered_demand_ratio'] <= 0.25]
    pool = covered or pool
    limits = {
        key: (min(r[key] for r in pool), max(r[key] for r in pool))
        for key in ('mass_kg', 'cost_usd')
    }

    def normalized(r):
        value = sum(
            (r[key] - lo) / (hi - lo) if hi > lo else 0.0
            for key, (lo, hi) in limits.items()
        )
        return (value, *tie(r))

    return min(pool, key=normalized)


def search_configuration(config, weather_data):
    """Algorithm 1: exhaustively evaluate the design constraints and objectives."""
    started = time.perf_counter()
    prepared = prepare_mission(config, weather_data)
    preparation_seconds = time.perf_counter() - started
    records = []
    # Reuse each garment's regional score and demand across battery/PV counts.
    # Count-dependent arrays and design screens are evaluated for each tuple.
    prototypes = {
        g.id: build_candidate(config, prepared[g.id], weather_data, 1, 0, g.id)
        for g in config.garments
    }
    for y in range(1, config.max_battery_units + 1):
        for h in range(config.max_pv_units + 1):
            for g in config.garments:
                prototype = prototypes[g.id]
                d = config.device
                installed = prototype['hardware']['equipment_available']
                hardware = dict(
                    prototype['hardware'],
                    capacity_wh=y * d.battery_unit_capacity_wh,
                    max_discharge_w=y * d.max_discharge_w,
                    max_charge_w=y * d.max_charge_w,
                )
                pv = (
                    np.asarray(weather_data['reference_pv_w'])
                    * h
                    * d.pv_rating_w
                    / config.scenario.pv_reference_rating_w
                )
                actual = dict(prototype['actual'], pv_w=pv)
                screen = _design_budget(
                    config, hardware, actual, prototype['reference_load_w'],
                    prototype['full_activation_energy_wh'],
                )
                mass = (
                    prototype['mass_kg'] + (y - 1) * d.battery_unit_mass_kg + h * d.pv_mass_kg
                )
                cost = prototype['cost_usd'] + (y - 1) * d.battery_unit_cost + h * d.pv_cost
                records.append(
                    dict(
                        prototype,
                        configuration=(y, h, g.id),
                        hardware=hardware,
                        actual=actual,
                        mass_kg=mass,
                        cost_usd=cost,
                        objectives=(-prototype['score'], mass, cost),
                        **screen,
                    )
                )
    admissible = [r for r in records if r['design_admissible']]
    front = pareto_front(admissible)
    selected = select_configuration(front, config.preference)
    return {
        'records': records,
        'admissible': admissible,
        'pareto': front,
        'selected': selected,
        'preference': config.preference,
        'status': 'selected' if selected else 'no_design_admissible_configuration',
        'timing': {
            'demand_preparation_seconds': preparation_seconds,
            'configuration_seconds': time.perf_counter()
            - started
            - preparation_seconds,
        },
        'optimizer_calls': 0,
    }
