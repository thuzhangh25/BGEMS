'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: LP and electrical-boundary regression tests using synthetic fixtures.
'''

import unittest
from types import SimpleNamespace
import numpy as np
from src import operation
from src.inputs import REGION_GROUPS, REGION_ORDER


def hardware():
    return {
        'rated_powers_w': np.ones(12),
        'mask': np.ones(12),
        'efficiencies': np.ones(12),
        'weights': np.full(12, 1 / 12),
        'regional_areas_m2': np.ones(12),
        'capacity_wh': 100.0,
        'voltage_v': 10.0,
        'equipment_power_w': 2.0,
        'equipment_available': False,
        'base_power_w': 0.0,
        'max_discharge_w': 20.0,
        'max_charge_w': 10.0,
        'bus_limit_w': 20.0,
    }


def settings():
    return {
        'dt_hours': 1.0,
        'horizon_steps': 2,
        'soc_min': 0.05,
        'soc_safe': 0.15,
        'lambda_deg': 0.0,
        'lambda_soc': 0.0,
        'solver_tolerance': 1e-7,
        'constraint_tolerance': 1e-8,
    }


def parameters():
    return {
        'rbc': {'cold_threshold': 0.3, 'comfort_threshold': 0.8},
        'pid': {'kp': 10.0, 'ki': 1.0, 'kd': 0.0, 'setpoint': 0.9},
        'tacs': {
            'base_threshold': 0.7,
            'soc_factor': 0.2,
            'after_chill_multiplier': 1.3,
            'deadband': 0.05,
        },
        'rhc': {
            'pwl_segments': 8,
            'priority_lock_tolerance': 1e-6,
            'solve_time_limit_s': 3.0,
        },
    }


def forcing(demand, pv=None, air=None):
    count = len(demand)
    return {
        'demand_w': np.asarray(demand, dtype=float),
        'pv_w': np.zeros(count) if pv is None else np.asarray(pv, dtype=float),
        'air_c': np.zeros(count) if air is None else np.asarray(air, dtype=float),
        'met': np.ones(count),
        'planned_met': np.ones(count),
    }


def replay(actual, hw=None, cfg=None, controller='rhc', soc=0.8):
    hw = hardware() if hw is None else hw
    cfg = settings() if cfg is None else cfg

    def forecast(k, n):
        return {
            key: np.full(n, actual[key][k]) for key in ('demand_w', 'air_c', 'pv_w')
        }

    return operation.simulate(actual, hw, cfg, soc, controller, parameters(), forecast)


class OperationTests(unittest.TestCase):
    def test_electrical_load_does_not_use_heat_transfer_efficiency(self):
        hw = hardware()
        hw['efficiencies'][:] = 0.2
        result = operation.electrical_step(0.8, np.ones(12), 0, 2.0, hw, 1.0)
        self.assertEqual(result['load_w'], 12.0)
        self.assertAlmostEqual(result['soc'], 0.7)
        self.assertEqual(result['discharged_ah'], 1.0)

    def test_common_actuator_limits_charge_and_preserves_equipment_rule(self):
        hw = hardware()
        hw.update(equipment_available=True, base_power_w=1.0)
        result = operation.execute_action(
            1.0, np.zeros(12), -20.0, 20.0, hw, settings(), remaining_h=1.0
        )
        self.assertEqual(result['equipment_on'], 1)
        self.assertEqual(result['pv_used_w'], 3.0)
        self.assertEqual(result['curtailed_pv_w'], 17.0)
        self.assertEqual(result['soc'], 1.0)
        hw['equipment_available'] = False
        no_pack = operation.execute_action(
            1.0, np.zeros(12), -30.0, 20.0, hw, settings(), remaining_h=1.0
        )
        self.assertEqual(no_pack['equipment_on'], 0)
        self.assertEqual(no_pack['load_w'], 1.0)

    def test_energy_limit_scales_body_without_clipping_negative_soc(self):
        hw = hardware()
        hw.update(capacity_wh=10.0, base_power_w=1.0)
        step = operation.execute_action(0.25, np.ones(12), 0.0, 0.0, hw, settings(), remaining_h=1.0)
        self.assertAlmostEqual(sum(step['pwm']), 1.0)
        self.assertAlmostEqual(step['soc'], 0.05)
        failed = operation.execute_action(0.1, np.zeros(12), 0.0, 0.0, hw, settings(), remaining_h=1.0)
        self.assertFalse(failed['success'])

    def test_priority_allocates_scarce_bus_to_core_before_distal(self):
        hw = hardware()
        hw['bus_limit_w'] = 2.0
        run = replay(forcing([12.0]), hw)
        self.assertTrue(run['completed'], run['failure'])
        pwm = np.asarray(run['actions'][0]['pwm'])
        core = [REGION_ORDER.index(name) for name in REGION_GROUPS[0]]
        distal = [REGION_ORDER.index(name) for name in REGION_GROUPS[1]]
        self.assertAlmostEqual(pwm[core].sum(), 2.0, places=4)
        self.assertLess(pwm[distal].sum(), 1e-4)
        self.assertAlmostEqual(run['states'][-1]['soc'], 0.78, places=6)

    def test_zero_weight_groups_and_zero_demand_have_finite_actions(self):
        hw = hardware()
        hw['weights'] = np.array([1.0] + [0.0] * 11)
        run = replay(forcing([12.0, 0.0]), hw)
        self.assertTrue(run['completed'], run['failure'])
        self.assertAlmostEqual(
            run['actions'][0]['regional_fulfillment'][0], 1.0, places=5
        )
        self.assertEqual(run['actions'][1]['load_w'], 0.0)
        self.assertEqual(run['actions'][1]['score'], 1.0)

    def test_four_policies_have_distinct_realized_decisions(self):
        actual = forcing([24.0] * 4)
        runs = {c: replay(actual, controller=c) for c in operation.CONTROLLERS}
        self.assertEqual(
            [a['load_w'] for a in runs['rbc']['actions']], [12.0, 6.0, 12.0, 6.0]
        )
        self.assertEqual(runs['pid']['actions'][0]['load_w'], 0.0)
        self.assertAlmostEqual(runs['pid']['actions'][1]['load_w'], 9.9)
        self.assertTrue(
            all(abs(a['score'] - 0.5) < 1e-5 for a in runs['rhc']['actions'])
        )
        policies = {
            tuple(round(a['load_w'], 5) for a in run['actions'])
            for run in runs.values()
        }
        self.assertEqual(len(policies), 4)

    def test_insufficient_initial_reserve_rejects_before_any_controller(self):
        hw = hardware()
        hw.update(capacity_wh=10.0, base_power_w=1.0)
        for controller in operation.CONTROLLERS:
            run = replay(forcing([0.0] * 3), hw, controller=controller, soc=0.2)
            self.assertFalse(run['completed'])
            self.assertEqual(run['failure']['kind'], 'reserve_domain_rejection')
            self.assertEqual(run['failure']['time_h'], 0.0)
            self.assertEqual(run['actions'], [])
            self.assertEqual(run['states'], [{'time_h': 0.0, 'soc': 0.2}])

    def test_future_pv_does_not_rescue_initial_reserve_domain(self):
        hw = hardware()
        hw.update(capacity_wh=10.0, base_power_w=1.0)
        run = replay(forcing([0.0, 0.0], pv=[100.0, 100.0]), hw, soc=0.2)
        self.assertEqual(run['failure']['kind'], 'reserve_domain_rejection')
        self.assertEqual(run['optimizer_calls'], 0)

    def test_short_forecast_keeps_mandatory_energy_beyond_horizon(self):
        hw = hardware()
        hw.update(capacity_wh=10.0, base_power_w=1.0)
        decision = operation.solve_rhc(
            0.45, forcing([12.0]), hw, settings(), parameters()['rhc'], remaining_h=3.0
        )
        self.assertTrue(decision['success'], decision)
        self.assertAlmostEqual(sum(decision['command_pwm']), 1.0, places=5)
        self.assertAlmostEqual(decision['planned_soc'][-1], 0.25, places=5)

    def test_reserve_clipping_is_separate_from_physical_clipping(self):
        hw = hardware()
        hw.update(capacity_wh=10.0, base_power_w=1.0)
        step = operation.execute_action(
            0.45, np.ones(12), 0.0, 0.0, hw, settings(), remaining_h=3.0
        )
        self.assertTrue(step['success'])
        self.assertAlmostEqual(sum(step['pwm']), 1.0)
        self.assertAlmostEqual(step['physical_curtailed_body_w'], 9.0)
        self.assertAlmostEqual(step['reserve_curtailed_body_w'], 2.0)
        self.assertAlmostEqual(step['reserve_margin_after_wh'], 0.0)

    def test_threshold_changes_preserve_full_mandatory_mission_for_all_policies(self):
        hw = hardware()
        hw.update(capacity_wh=10.0, base_power_w=1.0, equipment_available=True)
        cfg = settings()
        cfg['horizon_steps'] = 1
        for controller in operation.CONTROLLERS:
            run = replay(forcing([20.0] * 3, air=[0, -20, -30]), hw, cfg, controller, 1.0)
            self.assertTrue(run['completed'], run['failure'])
            self.assertEqual([a['equipment_on'] for a in run['actions']], [0, 1, 1])
            for k, state in enumerate(run['states']):
                self.assertGreaterEqual((state['soc'] - .05) * 10, 3 * (3 - k) - 1e-6)

    def test_reserve_accounts_for_timestep_and_uses_current_pv(self):
        hw = hardware()
        hw.update(capacity_wh=10.0, base_power_w=1.0)
        cfg = settings()
        cfg['dt_hours'] = .5
        step = operation.execute_action(
            .25, np.ones(12), 0., 2., hw, cfg, remaining_h=2.
        )
        self.assertTrue(step['success'])
        self.assertAlmostEqual(step['load_w'], 3.0)
        self.assertAlmostEqual(step['reserve_required_after_wh'], 1.5)
        self.assertAlmostEqual(step['reserve_margin_after_wh'], 0.)

    def test_mandatory_power_domain_rejected_even_when_current_air_is_warm(self):
        hw = hardware()
        hw.update(base_power_w=1.0, equipment_available=True, max_discharge_w=2.0)
        run = replay(forcing([0.0]), hw, soc=1.0)
        self.assertEqual(run['failure']['kind'], 'reserve_domain_rejection')
        self.assertEqual(run['optimizer_calls'], 0)

    def test_causal_forecast_never_reads_future_realized_disturbance(self):
        config = SimpleNamespace(
            person=SimpleNamespace(body_area_m2=1.0, skin_setpoint_c=30.0),
            scenario=SimpleNamespace(air_resistance=0.1),
        )
        garment = SimpleNamespace(clothing_resistance=0.1)
        actual = forcing([12.0, 999.0], [5.0, 999.0], [-10.0, -70.0])
        actual['met'] = np.array([1.0, 9.0])
        actual['planned_met'] = np.array([1.0, 2.0])
        forecast = operation.make_forecast(config, garment, actual)(0, 2)
        np.testing.assert_array_equal(forecast['air_c'], [-10.0, -10.0])
        np.testing.assert_array_equal(forecast['pv_w'], [5.0, 5.0])
        self.assertEqual(forecast['demand_w'][0], 12.0)
        self.assertNotEqual(forecast['demand_w'][1], 999.0)
        actual['met'][1] = 3.0
        actual['air_c'][1] = 20.0
        np.testing.assert_array_equal(
            operation.make_forecast(config, garment, actual)(0, 2)['demand_w'],
            forecast['demand_w'],
        )

    def test_soc_bands_use_interval_discharge_ah(self):
        for soc, weight in ((0.1999, 1.0), (0.2, 0.5), (0.4999, 0.5), (0.5, 0.2)):
            self.assertAlmostEqual(operation.discharge_penalty(soc, 2.0), 2 * weight)


if __name__ == '__main__':
    unittest.main()
