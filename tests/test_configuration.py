'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Verify configuration mathematics and acceptance/selection boundaries.
Synthetic evaluators test search semantics only, not real operating feasibility.
'''
import unittest
from types import SimpleNamespace

from src import configuration
from src.inputs import REGION_ORDER


def candidate(y, h, g):
    return {'objectives': (y, h), 'capability': 0.9 if g else 0.2,
            'diagnostics': {'energy_wh': 100}, 'structural_residuals': {}}


def accepted(y, h, g, data):
    return {'completed': True, 'residuals': {'service': 0, 'reserve': -1},
            'failure_kind': None}


class ConfigurationTests(unittest.TestCase):
    def test_mathematics_and_energy_units(self):
        # Eq. (6)/(7) diagnostic under the whole-body service definition:
        # full activation u = ones(12); step scores min(1, P_supp/P_req) with
        # the zero-demand convention, averaged over uniform steps.
        score = configuration.thermal_capability([0, 24], [0.5]*12, [2]*12, [1]*12)
        self.assertAlmostEqual(score, 0.75)
        # A masked-out pad contributes nothing: one installed 2 W pad against
        # a 12 W demand scores 1/6, whatever the unused pads could deliver.
        partial = configuration.thermal_capability([12], [1.]*12, [2.]*12,
                                                   [1] + [0]*11)
        self.assertAlmostEqual(partial, 2 / 12)
        # Zero demand scores 1 even for a no-pad garment (declared convention;
        # the equipment heater is excluded and this is not feasibility evidence).
        self.assertEqual(configuration.thermal_capability([0.], [0.]*12, [0.]*12, [0.]*12), 1.0)
        with self.assertRaises(ValueError):
            configuration.thermal_capability([], [0.5]*12, [2]*12, [1]*12)
        self.assertEqual(configuration.configuration_objectives(
            2, 2, 1, 1, 2, 0.1, 10, 20, 3, 4), (6.2, 44))
        screens = configuration.diagnostic_screens(
            2, 1, 10, 0.8, 0.2, [4, 0], [10, 10], 12, [0, 24], 0.5)
        self.assertAlmostEqual(screens['energy_wh'], -4)
        self.assertEqual(screens['body_power_w'], 12)

    def test_scenario_requirement_never_installs_or_removes_equipment(self):
        # Explicit one-step solver check; Device defaults are covered by integration tests.
        person = SimpleNamespace(regional_areas_m2=(1 / 12,) * 12,
                                 weights=(1 / 12,) * 12,
                                 region_order=REGION_ORDER)
        device = SimpleNamespace(
            cell_capacity_wh=10.0,
            cell_mass_kg=0.1,
            cell_cost=3.0,
            pv_rating_w=20.0,
            pv_mass_kg=0.5,
            pv_cost=4.0,
            base_mass_kg=1.0,
            base_cost=10.0,
            base_power_w=1.0,
            rated_powers_w=(0.0,) * 12,
            efficiencies=(1.0,) * 12,
            equipment_power_w=5.0,
            equipment_available=True,
            voltage_v=5.0,
            heat_capacity_j_per_k=1000.0,
            thermal_resistance_k_per_w=2.0,
            max_discharge_w=10.0,
            max_charge_w=5.0,
            bus_limit_w=10.0,
        )
        scenario = SimpleNamespace(
            initial_state={'soc': 0.8, 'battery_c': 10.0,
                           'equipment_on': 0, 'air_history_c': []},
            equipment_heater_required=False,
            pv_reference_rating_w=100.0,
        )
        garment = SimpleNamespace(
            id=0, mask=(0,) * 12, mass_kg=2.0, cost=20.0)
        config = SimpleNamespace(
            person=person, device=device, scenario=scenario, garments=(garment,),
            max_battery_cells=2, max_pv_units=2,
            settings={'soc_min': 0.2, 'dt_hours': 1.0},
        )
        scenario_data = {
            'demand_w': [0.0], 'air_c': [10.0],
        }
        weather_data = {'reference_pv_w': [0.0]}
        optional = configuration.build_candidate(
            config, scenario_data, weather_data, 1, 0, 0)
        scenario.equipment_heater_required = True
        required = configuration.build_candidate(
            config, scenario_data, weather_data, 1, 0, 0)
        self.assertEqual(optional['hardware']['equipment_available'], 1)
        self.assertEqual(optional['hardware']['equipment_power_w'], 5.0)
        self.assertEqual(optional['diagnostics'], required['diagnostics'])
        self.assertAlmostEqual(optional['diagnostics']['energy_wh'], 0.0)
        self.assertEqual(required['structural_residuals']['equipment_unavailable'], 0.0)
        doubled = configuration.build_candidate(
            config, scenario_data, weather_data, 2, 0, 0)
        self.assertEqual(doubled['hardware']['capacity_wh'], 20.0)
        self.assertEqual(doubled['hardware']['max_discharge_w'], 20.0)
        self.assertEqual(doubled['hardware']['max_charge_w'], 10.0)
        self.assertEqual(doubled['hardware']['heat_capacity_j_per_k'], 2000.0)
        self.assertEqual(doubled['hardware']['bus_limit_w'], 10.0)
        weather_data['reference_pv_w'] = [50.0]
        solar = configuration.build_candidate(
            config, scenario_data, weather_data, 1, 2, 0)
        self.assertEqual(solar['actual']['pv_w'].tolist(), [20.0])

    def test_diagnostic_failure_does_not_exclude_and_ties_are_deterministic(self):
        result = configuration.search_configuration(2, 1, [1, 0], candidate, accepted,
                                                     ('mass', 'cost'))
        self.assertEqual(len(result['feasible']), 8)
        self.assertEqual({r['configuration'] for r in result['pareto']}, {(1, 0, 0), (1, 0, 1)})
        self.assertEqual(result['selected']['configuration'], (1, 0, 0))

    def test_failures_and_incomplete_runs_never_enter_pareto(self):
        def operation(y, h, g, data):
            run = accepted(y, h, g, data)
            if g == 0:
                run['completed'] = False
            elif g == 1:
                run['residuals']['reserve'] = 0.01
            elif g == 2:
                run['residuals']['reserve'] = float('nan')
            else:
                run['failure_kind'] = 'physical_infeasibility'
            return run
        result = configuration.search_configuration(1, 0, range(4), candidate, operation,
                                                     ('mass', 'cost'))
        self.assertIsNone(result['selected'])
        self.assertEqual(result['pareto'], [])
        self.assertEqual([r['status'] for r in result['records']],
                         ['policy_failure', 'policy_failure', 'numerical_failure', 'physical_infeasibility'])

    def test_mass_and_cost_priorities_choose_opposite_pareto_members(self):
        def tradeoff(y, h, g):
            data = candidate(y, h, g)
            data['objectives'] = (y, 3-y)
            return data
        first = configuration.search_configuration(2, 0, [0], tradeoff, accepted,
                                                    ('cost', 'mass'))
        second = configuration.search_configuration(2, 0, [0], tradeoff, accepted,
                                                     ('mass', 'cost'))
        self.assertEqual(len(first['pareto']), 2)
        self.assertEqual(first['selected']['configuration'], (2, 0, 0))
        self.assertEqual(second['selected']['configuration'], (1, 0, 0))

    def test_missing_evaluator_and_structural_exclusion(self):
        with self.assertRaises(TypeError):
            configuration.search_configuration(1, 0, [0], candidate, None,
                                               ('mass', 'cost'))
        def incompatible(y, h, g):
            data = candidate(y, h, g)
            data['structural_residuals'] = {'equipment_unavailable': 1}
            return data
        def forbidden(*args):
            self.fail('Structurally excluded candidate must not run.')
        result = configuration.search_configuration(1, 0, [0], incompatible, forbidden,
                                                     ('mass', 'cost'))
        self.assertIsNone(result['selected'])


if __name__ == '__main__':
    unittest.main()
