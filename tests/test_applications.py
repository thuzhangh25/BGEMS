'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Verify cashflow recovery and cycle-closed building/electricity boundaries.
'''

from copy import deepcopy
import unittest

from scripts.evaluate_applications import cashflow, recharge_energy_kwh, building_state
from test_operation import forcing, hardware, settings, replay


def record(setpoint=20.):
    hw = hardware()
    hw['base_power_w'] = 1.
    actual = forcing([0., 0.], air=[setpoint, setpoint])
    return {'run': replay(actual, hw, controller='rbc'), 'actual': actual,
            'initial_soc': .8, 'hardware': hw, 'settings': settings(),
            'evaluation_tolerances': {'balance_tolerance_wh': 1e-6, 'time_tolerance_hours': 1e-9},
            'configuration': [1, 0, 0], 'experiment': {'setpoint_c': setpoint}}


def assumptions():
    return {'charging_efficiency': .9, 'building': {
        'outdoor_c': -20., 'cop_valid_temperature_c': [-30., 10.],
        'cop_rated': 3., 'cop_alpha_per_k2': .0004, 'cop_rated_temperature_c': 7.,
        'wearable_heat_credit_fraction': 0., 'day_hours': 24., 'occupants': 30,
        'metabolic_gain_w_per_person': 134., 'heat_loss_kw_per_k': 2.,
        'nonmission_setpoint_c': 25., 'baseline_setpoint_c': 25.}}


class ApplicationTests(unittest.TestCase):
    def test_cashflow_includes_capex_and_discounted_year_end_flows(self):
        result = cashflow(100., [60., 60.], .1)
        self.assertAlmostEqual(result['npv_usd'], -100 + 60 / 1.1 + 60 / 1.1 ** 2)
        self.assertAlmostEqual(result['net_payback_years'], 1 + 40 / 60)
        self.assertAlmostEqual(result['discounted_payback_years'], 1 + (100 - 60 / 1.1) / (60 / 1.1 ** 2))

    def test_no_payback_is_not_extrapolated_beyond_lifetime(self):
        result = cashflow(100., [20., -10., 20.], .05)
        self.assertIsNone(result['net_payback_years'])
        self.assertIsNone(result['discounted_payback_years'])
        self.assertLess(result['npv_usd'], 0)

    def test_zero_discount_and_exact_last_year_recovery(self):
        result = cashflow(100., [20., 80.], 0.)
        self.assertEqual(result['npv_usd'], 0.)
        self.assertEqual(result['net_payback_years'], 2.)
        self.assertEqual(result['discounted_payback_years'], 2.)

    def test_grid_recharge_counts_electronics_and_pv_once(self):
        rec = record()
        rec['actual']['pv_w'][:] = .25
        rec['run'] = replay(rec['actual'], rec['hardware'], controller='rbc')
        energy, metrics = recharge_energy_kwh(rec, .9)
        self.assertEqual(metrics['body_heating_energy_wh'], 0.)
        self.assertAlmostEqual(energy, 1.5 / 900)

    def test_incomplete_or_corrupt_mission_cannot_supply_cashflow_energy(self):
        rec = record()
        rec['run']['states'][-1]['soc'] += .1
        with self.assertRaises(ValueError):
            recharge_energy_kwh(rec, .9)

    def test_building_savings_use_task_hours_not_whole_day_heating(self):
        result = building_state(record(), assumptions())
        cop = 3 * (1 - .0004 * 27 ** 2)
        self.assertAlmostEqual(result['setpoint_only_saving_kwh'], 2 * 5 * 2 / cop)
        self.assertAlmostEqual(result['saving_kwh'], 20 / cop - 30 * 2 / 900)
        self.assertAlmostEqual(result['reference_kwh'], (2 * 45 - 30 * .134) * 24 / cop)

    def test_baseline_wearable_has_cost_without_setpoint_savings(self):
        result = building_state(record(25.), assumptions())
        self.assertAlmostEqual(result['setpoint_only_saving_kwh'], 0.)
        self.assertAlmostEqual(result['saving_kwh'], -30 * 2 / 900)

    def test_invalid_cop_domain_cannot_produce_a_saving(self):
        result = building_state(record(), assumptions(), outdoor_c=-50.)
        self.assertEqual(result['status'], 'outside_cop_model_domain')
        self.assertNotIn('saving_kwh', result)


if __name__ == '__main__':
    unittest.main()
