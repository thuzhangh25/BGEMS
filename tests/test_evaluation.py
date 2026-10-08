'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Verify independent trajectory auditing and scheduled-run accounting boundaries.
Synthetic traces exercise evidence semantics only, not controller performance.
The service audit is the whole-body total-heat definition: a regional
allocation shortfall is a diagnostic, never an S failure, and the no-waste
oversupply acceptance shares operation.py's declared _RESIDUAL_TOLERANCE.
'''
import math
import unittest

import numpy as np
import scipy.optimize  # Preload extension dependencies before temporary flat modules.

from src import evaluation, modeling, operation
from src.operation import _RESIDUAL_TOLERANCE


def hardware():
    return {
        'rated_powers_w': [4.] + [0.]*11,
        'mask': [1] + [0]*11,
        'efficiencies': [1.]*12,
        'weights': [1.] + [0.]*11,
        'regional_areas_m2': [1.]*12,
        'capacity_wh': 10.,
        'voltage_v': 5.,
        'equipment_power_w': 2.,
        'equipment_available': 0,
        'base_power_w': 0.,
        'max_discharge_w': 10.,
        'max_charge_w': 10.,
        'bus_limit_w': 10.,
        'heat_capacity_j_per_k': 1000.,
        'thermal_resistance_k_per_w': 2.,
    }


def settings():
    return {
        'dt_hours': 1.,
        'horizon_steps': 2,
        'soc_min': .2,
        'soc_safe': .3,
        'terminal_soc': .2,
        'lambda_deg': 0.,
        'lambda_soc': 0.,
        'on_c': -5.,
        'off_c': 0.,
        'filter_window': 2,
        'solver_tolerance': 1e-7,
        'constraint_tolerance': 1e-9,
    }


def initial(soc=.9):
    return {'soc': soc, 'battery_c': 10., 'equipment_on': 0, 'air_history_c': []}


def forcing(demand, pv=None, air=None):
    count = len(demand)
    return {
        'demand_w': np.asarray(demand, dtype=float),
        'pv_w': np.zeros(count) if pv is None else np.asarray(pv, dtype=float),
        'air_c': np.full(count, 5.) if air is None else np.asarray(air, dtype=float),
    }


def trajectory(actual, pwms, pv_used, state, hw, cfg, failure_kind=None):
    """Build an operation.py-shaped prefix with shared physical functions."""
    soc = state['soc']
    battery = state['battery_c']
    previous = state['equipment_on']
    history = list(state['air_history_c'])
    states = [{'time_hours': 0., 'soc': soc, 'battery_c': battery}]
    steps = []
    for index, (pwm, used) in enumerate(zip(pwms, pv_used)):
        history.append(float(actual['air_c'][index]))
        on, filtered = operation.equipment_switch(
            history, previous, hw['equipment_available'], cfg['on_c'], cfg['off_c'],
            cfg['filter_window'])
        balance = operation.electrical_step(soc, pwm, on, used, hw, cfg['dt_hours'])
        whole = modeling.whole_body_fulfillment(
            float(actual['demand_w'][index]), hw['efficiencies'],
            hw['rated_powers_w'], hw['mask'], pwm)
        alloc = modeling.allocation_reference(
            float(actual['demand_w'][index]), hw['regional_areas_m2'], hw['weights'],
            hw['efficiencies'], hw['rated_powers_w'], hw['mask'], pwm)
        next_battery = modeling.battery_temperature_step(
            battery, actual['air_c'][index], on * hw['equipment_power_w'],
            hw['heat_capacity_j_per_k'], hw['thermal_resistance_k_per_w'],
            cfg['dt_hours'] * 3600)
        steps.append({
            'time_hours': index * cfg['dt_hours'],
            'pwm': np.asarray(pwm, dtype=float),
            'equipment_on': on,
            'filtered_air_c': filtered,
            'air_c': float(actual['air_c'][index]),
            'demand_w': float(actual['demand_w'][index]),
            'pv_available_w': float(actual['pv_w'][index]),
            'pv_used_w': float(used),
            'pv_curtailed_w': float(actual['pv_w'][index] - used),
            **balance,
            'thermal_score': whole['score'],
            'allocation_fulfillment': alloc['allocation_fulfillment'],
            'installed': whole['installed'],
            'total_effective_heat_w': whole['total_effective_heat_w'],
            'unmet_heat_w': max(float(actual['demand_w'][index])
                                - whole['total_effective_heat_w'], 0.),
            'allocation_unmet_heat_w': np.where(
                alloc['installed'],
                np.maximum(alloc['allocated_task_w'] - alloc['effective_heat_w'], 0.),
                np.nan),
            'residuals': {'dishonest_reported_residual': -999.},
        })
        soc, battery, previous = balance['soc'], next_battery, on
        states.append({'time_hours': (index + 1) * cfg['dt_hours'],
                       'soc': soc, 'battery_c': battery})
    return {
        'completed': True,
        'failure_kind': failure_kind,
        'residuals': {'dishonest_reported_residual': -999.},
        'steps': steps,
        'states': states,
        'scheduled_steps': len(actual['demand_w']),
    }


def audit(run, actual, floors, state, hw, cfg):
    return evaluation.evaluate_trajectory(
        run, actual, floors, state, hw, cfg,
        balance_tolerance_wh=0., time_tolerance_hours=0.)


class EvaluationTests(unittest.TestCase):
    def test_dishonest_success_fields_do_not_hide_service_failure(self):
        hw, cfg, state = hardware(), settings(), initial()
        actual = forcing([12.])
        # Whole-body alpha = .5: the floor needs P_supp >= 6 W and the pad is
        # off; the falsified step-row score and fulfillment must not be
        # trusted — the auditor recomputes from the executed PWM.
        floors = np.array([.5])
        run = trajectory(actual, [np.zeros(12)], [0.], state, hw, cfg)
        run['steps'][0]['thermal_score'] = 1.
        run['steps'][0]['allocation_fulfillment'] = np.ones(12)
        run['steps'][0]['total_effective_heat_w'] = 6.
        run['steps'][0]['unmet_heat_w'] = 0.

        verdict = audit(run, actual, floors, state, hw, cfg)

        self.assertTrue(verdict['completed'])
        self.assertFalse(verdict['components']['S'])
        self.assertEqual(verdict['status'], 'failed')
        self.assertEqual(verdict['failure_kind'], 'policy_failure')
        self.assertGreater(verdict['residuals']['service'], 0)

    def test_allocation_shortfall_passes_S_with_positive_deficit_diagnostic(self):
        # One installed node undersupplies its beta_i allocation share
        # (q_i < D_i) while the whole-body total still clears the floor:
        # the audit MUST pass S and report the shortfall only as the
        # allocation diagnostics (regional shortfall is diagnostic, not an
        # S failure).
        hw, cfg, state = hardware(), settings(), initial()
        hw.update(rated_powers_w=[2., 2.] + [0.] * 10, mask=[1, 1] + [0.] * 10,
                  weights=[.5, .5] + [0.] * 10)
        actual = forcing([12.])
        floors = np.array([.2])  # whole-body floor: P_supp >= 2.4 W
        # q = (2, .5) W: P_supp = 2.5 >= 2.4, but beta_i = .5 gives
        # D_i = 6 W each, so r = (1/3, 1/12) < 1 for BOTH installed nodes.
        pwm = np.array([1., .25] + [0.] * 10)
        run = trajectory(actual, [pwm], [0.], state, hw, cfg)

        verdict = audit(run, actual, floors, state, hw, cfg)

        self.assertEqual(verdict['status'], 'success', verdict)
        self.assertTrue(verdict['components']['S'])
        self.assertTrue(all(verdict['components'][key] for key in 'CESR'))
        deficit = verdict['metrics']['allocation_deficit_duration_hours']
        self.assertGreater(deficit[0], 0)
        self.assertGreater(deficit[1], 0)
        self.assertIsNone(deficit[2])  # uninstalled nodes export as null
        self.assertGreater(verdict['metrics']['allocation_unmet_heat_wh'][0], 0)
        # The whole-body diagnostics stay structurally zero on S success.
        self.assertEqual(verdict['metrics']['whole_body_unmet_heat_wh'], 0.)
        self.assertEqual(verdict['metrics']['service_floor_shortfall_duration_hours'], 0.)
        self.assertAlmostEqual(verdict['metrics']['mean_thermal_score_observed'],
                               min(1., 2.5 / 12.), places=12)

    def test_oversupply_acceptance_mirrors_the_shared_planner_margin(self):
        # The no-waste cap acceptance must use operation.py's declared
        # _RESIDUAL_TOLERANCE with the same rounding as the planner's cap
        # bound (simulate's exact mirror), never an independent constant.
        self.assertEqual(_RESIDUAL_TOLERANCE, {'service_oversupply': 1e-9})
        hw, cfg, state = hardware(), settings(), initial()
        hw.update(rated_powers_w=[2., 2.] + [0.] * 10, mask=[1, 1] + [0.] * 10)
        actual = forcing([2.])
        floors = np.array([0.])
        # P_supp = 2 + 1e-9: exactly one planner margin of oversupply. The
        # residual is accepted at the mirrored rounded cap bound, so S passes.
        # The exported residual is the excess over that declared bound, so it
        # reads <= 0 for an accepted candidate (zero-tolerance search rule).
        pwm_boundary = np.array([1., 5e-10] + [0.] * 10)
        verdict = audit(trajectory(actual, [pwm_boundary], [0.], state, hw, cfg),
                        actual, floors, state, hw, cfg)
        self.assertEqual(verdict['status'], 'success', verdict)
        self.assertTrue(verdict['components']['S'])
        self.assertLessEqual(verdict['residuals']['service_oversupply'],
                             _RESIDUAL_TOLERANCE['service_oversupply'] * 1e-6)
        # Clearly beyond the declared margin the cap fails the audit.
        pwm_over = np.array([1., 1.] + [0.] * 10)  # P_supp = 4 = demand + 2
        verdict_over = audit(trajectory(actual, [pwm_over], [0.], state, hw, cfg),
                             actual, floors, state, hw, cfg)
        self.assertFalse(verdict_over['components']['S'])
        self.assertGreater(verdict_over['residuals']['service_oversupply'],
                           _RESIDUAL_TOLERANCE['service_oversupply'])
        # Zero demand enforces the exact cap P_supp = 0 (no margin): any
        # positive supplied heat fails S.
        actual_zero = forcing([0.])
        verdict_zero = audit(trajectory(actual_zero, [pwm_over], [0.], state, hw, cfg),
                             actual_zero, np.array([0.]), state, hw, cfg)
        self.assertFalse(verdict_zero['components']['S'])
        self.assertGreater(verdict_zero['residuals']['service_oversupply'], 0)

    def test_intermediate_soc_violation_is_not_erased_by_terminal_recovery(self):
        hw, cfg, state = hardware(), settings(), initial(.5)
        actual = forcing([0., 0.], pv=[0., 5.])
        floors = np.zeros(2)
        run = trajectory(actual, [[1.] + [0.]*11, np.zeros(12)], [0., 5.],
                         state, hw, cfg)

        verdict = audit(run, actual, floors, state, hw, cfg)

        self.assertTrue(verdict['completed'])
        self.assertFalse(verdict['components']['E'])
        self.assertTrue(verdict['components']['R'])
        self.assertEqual(verdict['status'], 'failed')
        self.assertAlmostEqual(verdict['metrics']['minimum_soc'], .1)
        self.assertAlmostEqual(verdict['metrics']['terminal_soc'], .6)
        self.assertGreater(verdict['residuals']['soc_lower'], 0)

    def test_last_step_reserve_failure_keeps_complete_policy_verdict(self):
        hw, cfg, state = hardware(), settings(), initial(.3)
        cfg['terminal_soc'] = .4
        actual = forcing([0.])
        floors = np.zeros(1)
        run = trajectory(actual, [np.zeros(12)], [0.], state, hw, cfg, 'policy_failure')

        verdict = audit(run, actual, floors, state, hw, cfg)

        self.assertTrue(verdict['completed'])
        self.assertTrue(verdict['components']['C'])
        self.assertFalse(verdict['components']['R'])
        self.assertEqual(verdict['status'], 'failed')
        self.assertEqual(verdict['failure_kind'], 'policy_failure')

    def test_balance_tolerance_never_relaxes_terminal_reserve(self):
        hw, cfg, state = hardware(), settings(), initial(.4)
        cfg['terminal_soc'] = .4
        actual = forcing([0.])
        floors = np.zeros(1)
        run = trajectory(actual, [np.zeros(12)], [0.], state, hw, cfg)
        run['states'][-1]['soc'] = .399

        verdict = evaluation.evaluate_trajectory(
            run, actual, floors, state, hw, cfg,
            balance_tolerance_wh=.02, time_tolerance_hours=0.)

        self.assertTrue(verdict['completed'])
        self.assertLessEqual(verdict['residuals']['electrical_balance_wh'], 0)
        self.assertGreater(verdict['residuals']['terminal_soc'], 0)
        self.assertFalse(verdict['components']['R'])
        self.assertEqual(verdict['status'], 'failed')

    def test_off_grid_full_trace_has_unknown_completion(self):
        hw, cfg, state = hardware(), settings(), initial()
        actual = forcing([0.])
        floors = np.zeros(1)
        run = trajectory(actual, [np.zeros(12)], [0.], state, hw, cfg)
        run['steps'][0]['time_hours'] = .01

        verdict = audit(run, actual, floors, state, hw, cfg)

        self.assertFalse(verdict['completed'])
        self.assertIsNone(verdict['components']['C'])
        self.assertEqual(verdict['status'], 'unresolved')
        self.assertEqual(verdict['failure_kind'], 'numerical_failure')

    def test_partial_policy_failure_and_numerical_outcome_are_distinct(self):
        hw, cfg, state = hardware(), settings(), initial()
        actual = forcing([0., 0.])
        floors = np.zeros(2)
        actions = [np.zeros(12)]
        policy = audit(trajectory(actual, actions, [0.], state, hw, cfg, 'policy_failure'),
                       actual, floors, state, hw, cfg)
        numerical = audit(trajectory(actual, actions, [0.], state, hw, cfg, 'numerical_failure'),
                          actual, floors, state, hw, cfg)

        self.assertEqual(policy['components'], {'C': False, 'E': None, 'S': None, 'R': None})
        self.assertEqual(policy['status'], 'failed')
        self.assertEqual(policy['failure_kind'], 'policy_failure')
        self.assertTrue(all(math.isfinite(value) for value in policy['residuals'].values()))
        self.assertEqual(numerical['components'], {'C': False, 'E': None, 'S': None, 'R': None})
        self.assertEqual(numerical['status'], 'unresolved')
        self.assertEqual(numerical['failure_kind'], 'numerical_failure')
        self.assertEqual(numerical['metrics']['terminal_soc'], None)
        self.assertAlmostEqual(numerical['metrics']['observed_end_soc'], .9)
        self.assertEqual(numerical['metrics']['observed_duration_hours'], 1.)

    def test_missing_runs_remain_in_scheduled_denominator(self):
        summary = evaluation.summarize_runs(
            [{'status': 'success'}, {'status': 'failed'}], scheduled_runs=3)

        self.assertEqual(summary['scheduled'], 3)
        self.assertEqual(summary['available'], 2)
        self.assertEqual(summary['resolved'], 2)
        self.assertEqual(summary['success'], 1)
        self.assertEqual(summary['failed'], 1)
        self.assertEqual(summary['unresolved'], 0)
        self.assertEqual(summary['missing'], 1)
        self.assertEqual(summary['unresolved_or_missing'], 1)
        self.assertEqual(summary['candidate_feasible_fraction'], 1/3)
        self.assertEqual(summary['candidate_denominator'], 3)


if __name__ == '__main__':
    unittest.main()
