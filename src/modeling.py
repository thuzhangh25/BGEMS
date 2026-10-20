'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Thermal demand, localized thermal-comfort score and equipment activation.
Effective heating power is electrical heating power multiplied by the regional factor.
'''
import math
import numpy as np


# Tolerance for floating-point sums of normalized weights.
WEIGHT_SUM_ATOL = 1e-12


def weighted_skin_temperature(skin_temperatures_c, weights):
    """Compute the reference skin temperature used in Eq. (2).

    Args:
        skin_temperatures_c: Twelve regional reference temperatures in degrees C.
        weights: Twelve nonnegative, normalized regional scoring weights.

    Returns:
        Weighted reference skin temperature in degrees C.
    """
    temperatures = np.asarray(skin_temperatures_c, dtype=float)
    omega = np.asarray(weights, dtype=float)
    if temperatures.shape != (12,) or omega.shape != (12,):
        raise ValueError("Skin temperatures and weights must each have 12 entries.")
    if not np.all(np.isfinite(temperatures)) or not np.all(np.isfinite(omega)):
        raise ValueError("Skin temperatures and weights must be finite.")
    if np.any(omega < 0) or not math.isclose(float(omega.sum()), 1.0, rel_tol=0, abs_tol=WEIGHT_SUM_ATOL):
        raise ValueError("Regional weights must be nonnegative and sum to one.")
    return float(omega @ temperatures)


def heat_losses(met, ambient_c, body_area_m2):
    """Compute evaporative/respiratory loss and mechanical power in W.

    The MET factor is 58.15 W/m². Ambient temperature and activity determine
    the E and W inputs to Eq. (2).
    """
    if not all(math.isfinite(value) for value in (met, ambient_c, body_area_m2)):
        raise ValueError("Heat-loss inputs must be finite.")
    if met < 0 or body_area_m2 <= 0:
        raise ValueError("MET must be nonnegative and body area positive.")
    resting_metabolic_w = 58.15 * body_area_m2
    activity_factor = min(1.5, 1.0 + 0.15 * (met - 1.0))
    temperature_factor = 1.0 + 0.02 * max(0.0, ambient_c - 20.0)
    evaporation_w = 0.1 * resting_metabolic_w * activity_factor * temperature_factor
    mechanical_work_w = 0.1 * max(met - 1.0, 0.0) * 58.15 * body_area_m2
    return evaporation_w, mechanical_work_w


def heating_requirement(ambient_c, met, body_area_m2, clothing_resistance,
                        air_resistance, skin_setpoint_c, evaporation_w, mechanical_work_w):
    """Eq. (2): compute supplemental whole-body heating demand for one instant.

    Args:
        ambient_c, skin_setpoint_c: Ambient and reference skin temperatures, degrees C.
        met: Activity level in MET; 1 MET = 58.15 W/m² in the heat-balance model.
        body_area_m2: Whole-body surface area in m².
        clothing_resistance, air_resistance: Area-normalized resistances in m² K/W.
        evaporation_w: Explicit total evaporative/respiratory heat loss in W.
        mechanical_work_w: External mechanical power in W.

    Returns:
        Required heating power in W using the supplied evaporative and mechanical powers.
    """
    values = (ambient_c, met, body_area_m2, clothing_resistance, air_resistance,
              skin_setpoint_c, evaporation_w, mechanical_work_w)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Heat-balance inputs must be finite.")
    if body_area_m2 <= 0 or met < 0:
        raise ValueError("Body area must be positive and MET nonnegative.")
    if clothing_resistance < 0 or air_resistance <= 0:
        raise ValueError("Clothing resistance must be nonnegative and air resistance positive.")
    if evaporation_w < 0 or mechanical_work_w < 0:
        raise ValueError("Evaporative loss and external work must be nonnegative powers.")

    # Eq. (2): area converts the environmental heat flux (W/m²) into total W.
    environmental_loss_w = body_area_m2 * (skin_setpoint_c - ambient_c) / (clothing_resistance + air_resistance)
    metabolic_w = 58.15 * body_area_m2 * met
    # Apply the nonnegative cap to the final supplemental demand.
    return max(0.0, environmental_loss_w - (metabolic_w - evaporation_w - mechanical_work_w))


def regional_fulfillment(required_power_w, regional_areas_m2, weights,
                         efficiencies, rated_powers_w, mask, pwm):
    """Eq. (3): localized thermal-comfort score as modeled heating-demand fulfillment.

    Normalize areas over all twelve on-body nodes, cap each regional
    supply/demand ratio at one and combine using normalized scoring weights.
    Missing pads supply zero heat. Zero whole-body demand gives score one.
    Positive allocation proxies are accepted and normalized to conserve watts.
    Equipment heating is accounted for in the electrical-load balance.
    """
    areas = np.asarray(regional_areas_m2, dtype=float)
    omega = np.asarray(weights, dtype=float)
    if (areas.shape != (12,) or not np.all(np.isfinite(areas)) or
            np.any(areas <= 0) or not math.isfinite(float(areas.sum()))):
        raise ValueError("Twelve finite positive regional areas are required.")
    if (omega.shape != (12,) or not np.all(np.isfinite(omega)) or
            np.any(omega < 0) or not math.isclose(
                float(omega.sum()), 1.0, rel_tol=0, abs_tol=WEIGHT_SUM_ATOL)):
        raise ValueError("Twelve nonnegative scoring weights must sum to one.")
    eta, rated, available, duty = (
        np.asarray(value, dtype=float)
        for value in (efficiencies, rated_powers_w, mask, pwm))
    if any(value.shape != (12,) or not np.all(np.isfinite(value))
           for value in (eta, rated, available, duty)):
        raise ValueError("Each regional input must contain exactly 12 finite entries.")
    if not math.isfinite(required_power_w) or required_power_w < 0:
        raise ValueError("Demand must be nonnegative and finite.")
    if np.any((eta < 0) | (eta > 1)) or np.any((duty < 0) | (duty > 1)):
        raise ValueError("Efficiencies and PWM duty cycles must lie in [0, 1].")
    if np.any(rated < 0) or np.any((available != 0) & (available != 1)):
        raise ValueError("Pad ratings must be nonnegative and garment masks binary.")
    electrical = rated * available * duty
    effective = eta * electrical
    local_demand = required_power_w * (areas / areas.sum())
    fulfillment = np.ones(12)
    if required_power_w > 0:
        fulfillment = np.minimum(
            1.0, effective / local_demand)
    return {
        'installed': available == 1,
        'electrical_w': electrical,
        'effective_heat_w': effective,
        'regional_demand_w': local_demand,
        'regional_fulfillment': fulfillment,
        'regional_unmet_heat_w': np.maximum(
            local_demand - effective, 0.0),
        'score': 1.0 if required_power_w == 0 else float(omega @ fulfillment),
    }


def equipment_heater_action(ambient_c, installed):
    """Switch on installed equipment heating at air temperature <= -20 degrees C."""
    if not math.isfinite(ambient_c):
        raise ValueError("Equipment air temperature must be finite.")
    if installed not in (0, 1):
        raise ValueError("Equipment installation must be binary.")
    return int(bool(installed) and ambient_c <= -20.0)


def demand_series(config, garment, air_c, met):
    """Evaluate the declared heat balance on aligned interval-start inputs."""
    air, activity = np.asarray(air_c, dtype=float), np.asarray(met, dtype=float)
    if (air.ndim != 1 or air.size == 0 or air.shape != activity.shape or
            not np.all(np.isfinite(air)) or not np.all(np.isfinite(activity))):
        raise ValueError('Temperature and MET must be aligned finite nonempty vectors.')
    person = config.person
    area, skin = person.body_area_m2, person.skin_setpoint_c
    result = np.empty(air.size)
    for i, (temperature, value) in enumerate(zip(air, activity)):
        evaporation, work = heat_losses(value, temperature, area)
        result[i] = heating_requirement(
            temperature, value, area, garment.clothing_resistance,
            config.scenario.air_resistance, skin, evaporation, work)
    return result
