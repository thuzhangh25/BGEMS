'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Test three-objective dominance, deterministic selection and unit scaling.
'''

from pathlib import Path
import tempfile
import unittest
import numpy as np
from src import configuration
from src.weather import read_scenario_weather
from test_run import synthetic_config


def candidate(key, score, mass, cost, uncovered=0.0):
    return {
        'configuration': key,
        'objectives': (-score, mass, cost),
        'score': score,
        'mass_kg': mass,
        'cost_usd': cost,
        'uncovered_demand_ratio': uncovered,
    }


class ConfigurationTests(unittest.TestCase):
    def test_three_objective_front_retains_equal_points_and_real_tradeoffs(self):
        a = candidate((1, 0, 0), 0.8, 1.0, 10.0)
        equal = candidate((1, 0, 1), 0.8, 1.0, 10.0)
        comfort = candidate((2, 0, 0), 1.0, 2.0, 20.0)
        dominated = candidate((3, 0, 0), 0.7, 2.0, 11.0)
        front = configuration.pareto_front([dominated, equal, comfort, a])
        self.assertEqual(
            {r['configuration'] for r in front},
            {a['configuration'], equal['configuration'], comfort['configuration']},
        )
        self.assertEqual(
            configuration.select_configuration(front, 'lightweight')['configuration'],
            a['configuration'],
        )
        self.assertEqual(
            configuration.select_configuration(front, 'comfort')['configuration'],
            comfort['configuration'],
        )

    def test_balanced_selection_filters_service_and_uncovered_demand_before_cost(self):
        best = candidate((3, 0, 0), 1.0, 3.0, 30.0)
        covered = candidate((2, 0, 0), 0.96, 2.0, 20.0, 0.2)
        uncovered = candidate((1, 0, 0), 0.99, 1.0, 10.0, 0.3)
        low_service = candidate((1, 0, 1), 0.94, 0.5, 5.0)
        self.assertEqual(
            configuration.select_configuration(
                [best, covered, uncovered, low_service], 'balanced'
            ),
            covered,
        )
        self.assertIsNone(configuration.select_configuration([], 'balanced'))

    def test_pv_scaling_and_battery_limits_follow_selected_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            config = synthetic_config(Path(directory))
            weather = read_scenario_weather(config.scenario, 0.5)
            weather['reference_pv_w'] = np.full(2, 50.0)
            prepared = configuration.prepare_mission(config, weather)[0]
            none = configuration.build_candidate(config, prepared, weather, 1, 0, 0)
            solar = configuration.build_candidate(config, prepared, weather, 2, 1, 0)
            np.testing.assert_array_equal(none['actual']['pv_w'], [0.0, 0.0])
            np.testing.assert_array_equal(solar['actual']['pv_w'], [10.0, 10.0])
            self.assertEqual(
                solar['hardware']['capacity_wh'], 2 * none['hardware']['capacity_wh']
            )
            self.assertEqual(
                solar['hardware']['max_discharge_w'],
                2 * none['hardware']['max_discharge_w'],
            )
            expected = (
                config.endurance_multiplier
                * (solar['full_activation_energy_wh'] - 10.0)
                - 120.0 * 0.75
            )
            self.assertAlmostEqual(
                solar['diagnostics']['full_activation_energy_residual_wh'], expected
            )

    def test_catalogue_search_matches_individually_built_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            config = synthetic_config(Path(directory))
            config.max_battery_units = 2
            weather = read_scenario_weather(config.scenario, 0.5)
            prepared = configuration.prepare_mission(config, weather)
            result = configuration.search_configuration(config, weather)
            self.assertEqual(
                {r['configuration'] for r in result['records']},
                {(y, h, g.id) for y in (1, 2) for h in (0, 1) for g in config.garments},
            )
            for record in result['records']:
                y, h, g = record['configuration']
                direct = configuration.build_candidate(
                    config, prepared[g], weather, y, h, g
                )
                np.testing.assert_allclose(
                    record['objectives'], direct['objectives'], rtol=0, atol=1e-10
                )
                self.assertEqual(
                    record['design_admissible'], direct['design_admissible']
                )
                for group in ('diagnostics', 'design_constraints'):
                    for key in direct[group]:
                        self.assertAlmostEqual(record[group][key], direct[group][key])
                self.assertAlmostEqual(
                    record['reference_energy_budget_wh'],
                    direct['reference_energy_budget_wh'],
                )

    def test_cold_air_never_installs_equipment_without_a_backpack(self):
        with tempfile.TemporaryDirectory() as directory:
            config = synthetic_config(Path(directory), -30.0)
            weather = read_scenario_weather(config.scenario, 0.5)
            prepared = configuration.prepare_mission(config, weather)[0]
            absent = configuration.build_candidate(config, prepared, weather, 1, 0, 0)
            config.scenario.equipment_heater_installed = True
            config.device.equipment_available = True
            installed = configuration.build_candidate(
                config, prepared, weather, 1, 0, 0
            )
            self.assertAlmostEqual(
                installed['full_activation_energy_wh']
                - absent['full_activation_energy_wh'],
                config.device.equipment_power_w,
            )
            self.assertAlmostEqual(
                installed['reference_load_energy_wh']
                - absent['reference_load_energy_wh'],
                config.device.equipment_power_w,
            )
            self.assertAlmostEqual(
                installed['mass_kg'] - absent['mass_kg'],
                config.device.equipment_mass_kg,
            )

    def test_zero_demand_passes_reference_budget_despite_positive_all_on_residual(self):
        with tempfile.TemporaryDirectory() as directory:
            config = synthetic_config(Path(directory))
            config.device.base_power_w = 5.0
            weather = read_scenario_weather(config.scenario, 0.5)
            prepared = configuration.prepare_mission(config, weather)[0]
            prepared['demand_w'] = np.zeros(2)
            result = configuration.build_candidate(config, prepared, weather, 1, 0, 0)
            self.assertEqual(result['reference_load_energy_wh'], 5.0)
            self.assertEqual(result['reference_energy_budget_wh'], 5.0)
            self.assertGreater(result['diagnostics']['full_activation_energy_residual_wh'], 0)
            self.assertTrue(result['design_admissible'])

    def test_reference_power_preserves_uncovered_and_zero_efficiency_demands(self):
        with tempfile.TemporaryDirectory() as directory:
            config = synthetic_config(Path(directory))
            config.person.area_allocation_proxies = [1.0] * 12
            config.person.weights = [1 / 12] * 12
            config.device.rated_powers_w = [10.0] * 12
            config.device.efficiencies = [0.5, 0.0] + [0.5] * 10
            config.device.base_power_w = 5.0
            config.garments[0].mask = [0] + [1] * 11
            weather = read_scenario_weather(config.scenario, 0.5)
            prepared = configuration.prepare_mission(config, weather)[0]
            prepared['demand_w'] = np.full(2, 12.0)
            result = configuration.build_candidate(config, prepared, weather, 1, 0, 0)
            # Each of ten usable regions needs 1 W of heat, hence 2 W electrical.
            np.testing.assert_allclose(result['reference_load_w'], [25.0, 25.0])
            self.assertAlmostEqual(result['score'], 10 / 12)
            self.assertEqual(result['reference_load_energy_wh'], 25.0)

    def test_later_pv_surplus_does_not_cancel_an_earlier_battery_deficit(self):
        with tempfile.TemporaryDirectory() as directory:
            config = synthetic_config(Path(directory))
            config.device.base_power_w = 5.0
            config.device.battery_unit_capacity_wh = 3.0
            config.endurance_multiplier = 1.1
            weather = read_scenario_weather(config.scenario, 0.5)
            prepared = configuration.prepare_mission(config, weather)[0]
            prepared['demand_w'] = np.zeros(2)
            # Selected PV is [0, 10] W: equal total PV/load Wh, but 2.5 Wh deficit.
            weather['reference_pv_w'] = np.array([0.0, 50.0])
            delayed = configuration.build_candidate(config, prepared, weather, 1, 1, 0)
            self.assertAlmostEqual(delayed['reference_energy_budget_wh'], 2.75)
            self.assertAlmostEqual(delayed['design_constraints']['reference_energy_residual_wh'], 0.5)
            self.assertFalse(delayed['design_admissible'])
            # The same total PV, now coincident with both 5 W load intervals.
            weather['reference_pv_w'] = np.array([25.0, 25.0])
            matched = configuration.build_candidate(config, prepared, weather, 1, 1, 0)
            self.assertEqual(matched['reference_energy_budget_wh'], 0.0)
            self.assertTrue(matched['design_admissible'])

    def test_reference_power_limits_can_reject_an_energy_sufficient_design(self):
        with tempfile.TemporaryDirectory() as directory:
            config = synthetic_config(Path(directory))
            config.device.base_power_w = 5.0
            weather = read_scenario_weather(config.scenario, 0.5)
            prepared = configuration.prepare_mission(config, weather)[0]
            prepared['demand_w'] = np.zeros(2)
            config.device.max_discharge_w = 4.0
            discharge = configuration.build_candidate(config, prepared, weather, 1, 0, 0)
            self.assertLess(discharge['design_constraints']['reference_energy_residual_wh'], 0)
            self.assertEqual(discharge['design_constraints']['discharge_power_residual_w'], 1.0)
            self.assertFalse(discharge['design_admissible'])
            config.device.max_discharge_w = 5.0
            config.device.bus_limit_w = 4.0
            bus = configuration.build_candidate(config, prepared, weather, 1, 0, 0)
            self.assertEqual(bus['design_constraints']['bus_power_residual_w'], 1.0)
            self.assertFalse(bus['design_admissible'])
            config.device.bus_limit_w = 5.0
            boundary = configuration.build_candidate(config, prepared, weather, 1, 0, 0)
            self.assertTrue(boundary['design_admissible'])


if __name__ == '__main__':
    unittest.main()
