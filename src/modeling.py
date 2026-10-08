'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Implement the paper's thermal-demand, regional-service and equipment models.
Physical parameters are explicit inputs. Equation numbers follow
docs/sections/2_methods.tex: Eq. (2), Eq. (3), Eq. (4).
'''
import math
import numpy as np


# This tolerance only accommodates floating-point sums of normalized weights.
# It is not a thermal-service criterion or an experimental uncertainty bound.
WEIGHT_SUM_ATOL = 1e-12


def weighted_skin_temperature(skin_temperatures_c, weights):
    """Compute the reference skin temperature used in Eq. (2).

    Args:
        skin_temperatures_c: Twelve regional reference temperatures in degrees C.
        weights: Twelve nonnegative, normalized paper weights (not area fractions).

    Returns:
        Weighted reference temperature in degrees C; not a measured skin temperature.
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
    """Return the declared heuristic evaporative/respiratory loss and work in W.

    The calculation uses a 58.15 W/m² MET factor and supplies E and W to Eq. (2).
    It is an explicit engineering prescription, not a physiological model or
    human validation.
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
        met: Activity level in MET; 1 MET = 58.15 W/m² in the accepted manuscript.
        body_area_m2: Whole-body surface area in m².
        clothing_resistance, air_resistance: Area-normalized resistances in m² K/W.
        evaporation_w: Explicit total evaporative/respiratory heat loss in W.
        mechanical_work_w: Explicit external mechanical work in W.

    Returns:
        Nonnegative heating demand in W. E/W prescriptions are caller decisions,
        not assumed fractions of metabolism and not silently set to zero.
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
    # Do not clip M-E-W: the paper clips only the final supplemental demand.
    return max(0.0, environmental_loss_w - (metabolic_w - evaporation_w - mechanical_work_w))


# Service-definition identifier (R3). Whole-body total-heat acceptance; the
# legacy per-region area-share definition is removed (author, 2026-10-04).
SERVICE_DEFINITION_ID = 'whole_body_total_heat_v1'


def whole_body_fulfillment(required_power_w, efficiencies, rated_powers_w,
                           mask, pwm):
    """Eq. (3): whole-body modeled heating-demand fulfillment (new definition).

    Effective supplied heat P_supp(t) = sum_i eta_i * P_i * M_i(g) * u_i(t) over
    the installed set I_g = {i : M_i(g) = 1}; x13 is excluded. The score is
    U_B = min(1, P_supp / P_req), with U_B = 1 when P_req = 0.

    Args:
        required_power_w: Nonnegative whole-body heating demand in W (P_req).
        efficiencies: Twelve effective heat-transfer fractions in [0, 1] (eta_i).
        rated_powers_w: Twelve nonnegative full-duty electrical ratings in W (P_i).
        mask: Twelve binary garment availability flags (M_i).
        pwm: Twelve commanded duty cycles in [0, 1] (u_i).

    Returns:
        Dict with 'installed' (bool[12]), 'electrical_w' (12),
        'effective_heat_w' (12, zero where uninstalled), 'total_effective_heat_w'
        (float P_supp), and 'score' (U_B; exactly 1.0 when P_req == 0).
        This score does not validate regional skin temperature, comfort, or safety.
    """
    eta, rated, available, duty = (
        np.asarray(value, dtype=float)
        for value in (efficiencies, rated_powers_w, mask, pwm)
    )
    if any(value.shape != (12,) or not np.all(np.isfinite(value))
           for value in (eta, rated, available, duty)):
        raise ValueError("Each regional input must contain exactly 12 finite entries.")
    if not math.isfinite(required_power_w) or required_power_w < 0:
        raise ValueError("Demand must be nonnegative and finite.")
    if np.any((eta < 0) | (eta > 1)) or np.any((duty < 0) | (duty > 1)):
        raise ValueError("Efficiencies and PWM duty cycles must lie in [0, 1].")
    if np.any(rated < 0) or np.any((available != 0) & (available != 1)):
        raise ValueError("Pad ratings must be nonnegative and garment masks binary.")

    installed = available == 1
    electrical_w = rated * available * duty
    effective_heat_w = eta * electrical_w
    total_effective_heat_w = float(effective_heat_w.sum())
    if required_power_w == 0:
        score = 1.0
    else:
        score = min(1.0, total_effective_heat_w / required_power_w)
    return {
        "installed": installed,
        "electrical_w": electrical_w,
        "effective_heat_w": effective_heat_w,
        "total_effective_heat_w": total_effective_heat_w,
        "score": score,
    }


def allocation_reference(required_power_w, regional_areas_m2, weights,
                         efficiencies, rated_powers_w, mask, pwm):
    """Per-node allocation preference for lexicographic control (not a service floor).

    beta_i(g) = M_i * BSA_i / sum_{j in I_g} M_j * BSA_j is the installed-node
    allocation share; the allocated task is D_i = beta_i * P_req. These shares
    are preferences only — they never impose q_i <= D_i and never decide S.
    Uninstalled nodes carry fulfillment NaN ("not applicable").

    Args are as in whole_body_fulfillment plus:
        regional_areas_m2: Twelve fixed positive regional areas (BSA_i proxies).
        weights: Twelve nonnegative weights summing to one.

    Returns:
        Dict with 'installed' (bool[12]), 'beta' (12, zero where uninstalled),
        'allocated_task_w' (12, D_i), 'effective_heat_w' (12, q_i), and
        'allocation_fulfillment' (12): min(1, q_i/D_i) where D_i > 0, 1.0 where
        installed and P_req == 0, NaN where uninstalled.
    """
    areas, omega, eta, rated, available, duty = (
        np.asarray(value, dtype=float)
        for value in (regional_areas_m2, weights, efficiencies,
                      rated_powers_w, mask, pwm)
    )
    if any(value.shape != (12,) or not np.all(np.isfinite(value))
           for value in (areas, omega, eta, rated, available, duty)):
        raise ValueError("Each regional input must contain exactly 12 finite entries.")
    if not math.isfinite(required_power_w) or required_power_w < 0 or np.any(areas <= 0):
        raise ValueError("Demand must be nonnegative and regional areas positive.")
    if np.any(omega < 0) or not math.isclose(float(omega.sum()), 1.0, rel_tol=0, abs_tol=WEIGHT_SUM_ATOL):
        raise ValueError("Regional weights must be nonnegative and sum to one.")
    if np.any((eta < 0) | (eta > 1)) or np.any((duty < 0) | (duty > 1)):
        raise ValueError("Efficiencies and PWM duty cycles must lie in [0, 1].")
    if np.any(rated < 0) or np.any((available != 0) & (available != 1)):
        raise ValueError("Pad ratings must be nonnegative and garment masks binary.")

    installed = available == 1
    weighted = available * areas
    denom = float(weighted.sum())
    beta = weighted / denom if denom > 0 else np.zeros(12)
    allocated_task_w = beta * required_power_w
    effective_heat_w = eta * rated * available * duty

    allocation_fulfillment = np.full(12, np.nan)
    pos = installed & (allocated_task_w > 0)
    allocation_fulfillment[pos] = np.minimum(
        1.0, effective_heat_w[pos] / allocated_task_w[pos])
    if required_power_w == 0:
        allocation_fulfillment[installed] = 1.0
    return {
        "installed": installed,
        "beta": beta,
        "allocated_task_w": allocated_task_w,
        "effective_heat_w": effective_heat_w,
        "allocation_fulfillment": allocation_fulfillment,
    }


def battery_temperature_step(battery_c, ambient_c, heater_power_w,
                             heat_capacity_j_per_k, thermal_resistance_k_per_w, dt_s):
    """Eq. (4): exact single-node thermal update under constant interval inputs.

    Args:
        battery_c, ambient_c: Initial battery and imposed air temperatures, degrees C.
        heater_power_w: Actual equipment-heater electrical power in W; coupling is one.
        heat_capacity_j_per_k: Positive pack heat capacity in J/K.
        thermal_resistance_k_per_w: Positive pack-to-air resistance in K/W.
        dt_s: Nonnegative interval duration in seconds, not the electrical model's hours.

    Returns:
        End-of-interval latent battery temperature in degrees C. The imposed air
        temperature is not changed. No self-heating or capacity derating is added.
    """
    values = (battery_c, ambient_c, heater_power_w, heat_capacity_j_per_k,
              thermal_resistance_k_per_w, dt_s)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Equipment thermal inputs must be finite.")
    if heat_capacity_j_per_k <= 0 or thermal_resistance_k_per_w <= 0:
        raise ValueError("Pack heat capacity and thermal resistance must be positive.")
    if heater_power_w < 0 or dt_s < 0:
        raise ValueError("Heater power and time interval must be nonnegative.")

    # Eq. (4): T_next = T_eq + (T_initial - T_eq) exp(-dt / (R*C)).
    # expm1 avoids cancellation for short timesteps; dt=0 returns the initial state.
    equilibrium_c = ambient_c + heater_power_w * thermal_resistance_k_per_w
    approach_fraction = -math.expm1(-dt_s / (thermal_resistance_k_per_w * heat_capacity_j_per_k))
    return battery_c + (equilibrium_c - battery_c) * approach_fraction
