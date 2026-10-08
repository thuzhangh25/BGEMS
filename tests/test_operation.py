'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Test real RHC solves, causal replay, electrical boundaries and configuration integration.
All values are synthetic mathematical fixtures, not approved scenario parameters.
'''
import unittest
from types import SimpleNamespace

import numpy as np
import scipy.optimize  # Load extension dependencies outside the temporary module map.

from src import configuration, operation


def hardware():
    return {'rated_powers_w': [2.] + [0.]*11, 'mask': [1] + [0]*11,
            'efficiencies': [1.]*12, 'weights': [.89] + [.01]*11,
            'regional_areas_m2': [1.]*12, 'capacity_wh': 10., 'voltage_v': 5.,
            'equipment_power_w': 0., 'equipment_available': 0, 'base_power_w': 0.,
            'max_discharge_w': 10., 'max_charge_w': 5., 'bus_limit_w': 10.,
            'heat_capacity_j_per_k': 1000., 'thermal_resistance_k_per_w': 2.}


def settings():
    return {'dt_hours': .1, 'horizon_steps': 2, 'soc_min': .1, 'soc_safe': .2,
            'terminal_soc': .1, 'lambda_deg': 1., 'lambda_soc': 1.,
            'on_c': -5., 'off_c': 0., 'filter_window': 2,
            'solver_tolerance': 1e-7, 'constraint_tolerance': 1e-9}


def initial(soc=.8):
    return {'soc': soc, 'battery_c': 10., 'equipment_on': 0, 'air_history_c': []}


def forcing(demand, pv=None, air=None):
    return {'demand_w': np.array(demand), 'pv_w': np.zeros(len(demand)) if pv is None else np.array(pv),
            'air_c': np.full(len(demand), 5.) if air is None else np.array(air)}


def persistence(t, observed, count):
    return {key: np.full(count, observed[key]) for key in ('demand_w', 'pv_w', 'air_c')}


def supplied_heat(h, pwm):
    """Independent whole-body P_supp = sum_i eta_i P_i M_i u_i."""
    return float(np.dot(np.asarray(h['efficiencies']) * np.asarray(h['rated_powers_w'])
                        * np.asarray(h['mask']), pwm))


class OperationTests(unittest.TestCase):
    def test_scenario_builder_and_declared_forecast_information(self):
        # Explicit two-step solver check; heuristic E/W is computed by modeling.py.
        scenario = SimpleNamespace(
            air_resistance=0.1,
            get_activity=lambda hour: 1.0 if hour < 0.5 else 2.0,
            get_service_requirement=lambda hour: 0.25,
        )
        person = SimpleNamespace(body_area_m2=1.0, skin_setpoint_c=30.0)
        config = SimpleNamespace(
            scenario=scenario,
            person=person,
            settings={'dt_hours': 0.5},
            forecast_mode='persistence',
        )
        garment = SimpleNamespace(clothing_resistance=0.1)
        built = operation.build_scenario(
            config,
            garment,
            {'times': (object(), object()), 'air_c': [-10.0, 10.0]},
        )
        self.assertEqual(built['met'].tolist(), [1.0, 2.0])
        # Whole-body service definition: service_floor is a (steps,) scalar
        # alpha array, never a (steps, 12) per-region matrix.
        self.assertEqual(built['service_floor'].shape, (2,))
        self.assertTrue(np.all(built['service_floor'] == 0.25))
        self.assertAlmostEqual(built['evaporation_w'][0], 5.815)
        self.assertEqual(built['mechanical_work_w'][0], 0.0)
        self.assertAlmostEqual(built['demand_w'][0], 147.665)

        actual = {
            'demand_w': built['demand_w'],
            'pv_w': np.array([0.0, 5.0]),
            'air_c': built['air_c'],
        }
        observed = {key: float(values[0]) for key, values in actual.items()}
        forecast = operation.make_forecast(
            config, garment, built, actual)(0, observed, 2)
        np.testing.assert_array_equal(forecast['air_c'], [-10.0, -10.0])
        np.testing.assert_array_equal(forecast['pv_w'], [0.0, 0.0])
        self.assertEqual(forecast['demand_w'][0], observed['demand_w'])
        self.assertAlmostEqual(forecast['demand_w'][1], 96.20225)

        config.forecast_mode = 'perfect-preview'
        preview = operation.make_forecast(
            config, garment, built, actual)(0, observed, 2)
        for key in ('demand_w', 'pv_w', 'air_c'):
            np.testing.assert_array_equal(preview[key], actual[key])

        # A uniform hand-expanded 12-vector folds to its scalar alpha; a
        # non-uniform 12-vector is rejected (never averaged or reduced).
        scenario.get_service_requirement = lambda hour: [0.25] * 12
        folded = operation.build_scenario(
            config, garment,
            {'times': (object(), object()), 'air_c': [-10.0, 10.0]})
        self.assertEqual(folded['service_floor'].shape, (2,))
        self.assertTrue(np.all(folded['service_floor'] == 0.25))
        scenario.get_service_requirement = lambda hour: [0.25] + [0.3] * 11
        with self.assertRaises(ValueError):
            operation.build_scenario(
                config, garment,
                {'times': (object(), object()), 'air_c': [-10.0, 10.0]})

    def test_interval_electrical_charge_and_soc_not_clipped(self):
        h = hardware()
        h['efficiencies'] = [.1]*12
        value = operation.electrical_step(.15, [1]+[0]*11, 0, 0, h, 1.)
        self.assertAlmostEqual(value['soc'], -.05)
        self.assertEqual(value['load_w'], 2)
        self.assertAlmostEqual(value['discharged_ah'], .4)
        charge = operation.electrical_step(.5, [0]*12, 0, 1, h, 1.)
        self.assertEqual(charge['discharged_ah'], 0)
        self.assertAlmostEqual(charge['soc'], .6)

    def test_piecewise_soc_weights_are_not_smoothed(self):
        self.assertEqual(operation.discharge_penalty(.199999, 2), 2)
        self.assertEqual(operation.discharge_penalty(.2, 2), 1)
        self.assertEqual(operation.discharge_penalty(.499999, 2), 1)
        self.assertEqual(operation.discharge_penalty(.5, 2), .4)

    def test_hysteresis_filter_and_equality(self):
        self.assertEqual(operation.equipment_switch([-10, 0], 0, 1, -5, 0, 2), (0, -5))
        self.assertEqual(operation.equipment_switch([-10, 0], 1, 1, -5, 0, 2), (1, -5))
        self.assertEqual(operation.equipment_switch([-10], 0, 1, -5, 0, 3), (1, -10))
        self.assertEqual(operation.equipment_switch([-10], 1, 0, -5, 0, 3), (0, -10))

    def test_real_optimizer_solves_distinct_horizon_actions(self):
        result = operation.solve_rhc(.8, 0, [], forcing([12, 24]), np.zeros(2),
                                     True, hardware(), settings())
        self.assertTrue(result['success'], result)
        np.testing.assert_allclose(result['pwm'][:, 0], [1, 1], atol=2e-4)
        self.assertGreater(result['objective'], 0)
        # Only the nonempty installed group (node 0 = left_chest -> core)
        # contributes a priority target; empty groups are dropped silently.
        self.assertEqual(len(result['priority_targets']), 1)
        self.assertEqual(result['priority_group_labels'], ['core'])
        self.assertLessEqual(result['max_constraint_residual'], 0)

    def test_terminal_reserve_is_applied_only_at_scenario_end(self):
        h, s = hardware(), settings()
        h['capacity_wh'] = 1.
        s.update(dt_hours=1., terminal_soc=.5, lambda_deg=0., lambda_soc=0.)
        short = operation.solve_rhc(1., 0, [], forcing([12]), np.zeros(1), False, h, s)
        final = operation.solve_rhc(1., 0, [], forcing([12]), np.zeros(1), True, h, s)
        self.assertTrue(short['success'], short)
        self.assertTrue(final['success'], final)
        self.assertAlmostEqual(short['pwm'][0, 0]*2, .9, places=4)
        self.assertAlmostEqual(final['pwm'][0, 0]*2, .5, places=4)

    def test_equipment_enters_shared_power_before_solve(self):
        h = hardware()
        h.update(equipment_available=1, equipment_power_w=3., bus_limit_w=2.)
        result = operation.solve_rhc(.8, 0, [], forcing([0], air=[-10]), np.zeros(1),
                                     True, h, settings())
        self.assertFalse(result['success'])
        self.assertEqual(result['failure_kind'], 'policy_failure')

    def test_pv_curtailment_and_full_grid(self):
        actual = forcing([0, 0], pv=[10, 10])
        h = hardware()
        h['mask'] = [0]*12  # No controllable loads: every generated watt must be curtailed.
        result = operation.simulate(actual, persistence, 'persistence', np.zeros(2),
                                    initial(1.), h, settings())
        self.assertTrue(result['completed'], result)
        self.assertIsNone(result['failure_kind'])
        self.assertEqual(len(result['states']), 3)
        self.assertEqual(result['states'][-1]['time_hours'], .2)
        for step in result['steps']:
            self.assertEqual(step['pv_used_w'], 0)
            self.assertEqual(step['pv_curtailed_w'], 10)
            self.assertEqual(step['soc'], 1)

    def test_forecast_observation_does_not_contain_future_or_battery_temperature(self):
        actual = forcing([0, 12], air=[5, 20])
        observations = []
        def observe(t, observation, count):
            observations.append(observation)
            return persistence(t, observation, count)
        result = operation.simulate(actual, observe, 'persistence', np.zeros(2),
                                    initial(), hardware(), settings())
        self.assertTrue(result['completed'], result)
        self.assertNotIn('battery_c', observations[0])
        self.assertEqual(observations[0]['air_history_c'], ())
        self.assertEqual(observations[0]['air_c'], 5)
        self.assertEqual(observations[1]['air_history_c'], (5,))
        self.assertEqual(result['steps'][1]['air_c'], 20)
        def invalid(t, observation, count):
            return forcing([999]*count)
        with self.assertRaises(ValueError):
            operation.simulate(actual, invalid, 'incorrect', np.zeros(2), initial(), hardware(), settings())

    def test_infeasible_service_and_numeric_failure_stop_without_padding(self):
        # Whole-body floor alpha = .5 at the second step needs P_supp >= 6 W;
        # the single installed 2 W pad cannot meet it, whatever the regional
        # layout — a whole-body shortfall, not a per-region one.
        floors = np.zeros(2)
        floors[1] = .5
        s = settings()
        s['horizon_steps'] = 1
        result = operation.simulate(forcing([0, 12]), persistence, 'persistence', floors,
                                    initial(), hardware(), s)
        self.assertFalse(result['completed'])
        self.assertEqual(result['failure_kind'], 'policy_failure')
        self.assertEqual(len(result['steps']), 1)
        self.assertEqual(len(result['states']), 2)
        s['horizon_steps'] = 2
        numeric = operation.simulate(forcing([12, 24]), persistence, 'persistence', np.zeros(2),
                                     initial(), hardware(), s)
        self.assertTrue(numeric['completed'])
        self.assertIsNone(numeric['failure_kind'])

    def test_service_floor_boundary_plans_pass_raw_acceptance(self):
        # Exactly binding whole-body floor plans must pass the raw realized
        # acceptance (<= 0, no hidden tolerance): the declared 1e-9 planning
        # margin absorbs the last-ulp flip between the LP arithmetic path and
        # the independent recomputation. The demand = 5 case pins P_supp at
        # fl(alpha * demand + margin) where the installed capability is 4 W.
        h = hardware()
        h.update(rated_powers_w=[2., 2.] + [0.] * 10, mask=[1, 1] + [0.] * 10,
                 weights=[.85, .09] + [.006] * 10)
        s = settings()
        s.update(lambda_deg=50.)
        alpha = .8
        for demand in (.4, 2.0, 3.2, 4.0, 5.0, 1.0, 2.5):
            result = operation.solve_rhc(.8, 0, [], forcing([demand]), np.array([alpha]),
                                         True, h, s)
            self.assertTrue(result['success'], (demand, result))
            pwm = result['pwm'][0]
            supplied = supplied_heat(h, pwm)
            # Raw deficit-side acceptance: P_supp >= alpha * demand exactly.
            self.assertGreaterEqual(supplied, alpha * demand, (demand, pwm))
            # No-waste cap: P_supp <= demand + declared margin.
            self.assertLessEqual(
                supplied, demand + operation._RESIDUAL_TOLERANCE['service_oversupply'],
                (demand, pwm))
        # Above the installed whole-body capability the floor is infeasible:
        # a declared policy failure, never a fabricated plan.
        result = operation.solve_rhc(.8, 0, [], forcing([6.]), np.array([alpha]),
                                     True, h, s)
        self.assertFalse(result['success'])
        self.assertEqual(result['failure_kind'], 'policy_failure')

    def test_alpha_one_coincident_boundary_stays_feasible_and_passes_acceptance(self):
        # alpha = 1: the floor row and the no-waste cap row coincide at
        # P_supp = P_req. The declared margin on BOTH rows keeps the LP
        # feasible (the spec forbids stacking a margin on an exact upper
        # bound), the pinned plan passes the raw realized deficit acceptance,
        # and the mirrored shared _RESIDUAL_TOLERANCE lets the oversupply
        # residual (at most one margin width) through simulate's audit.
        h = hardware()
        h.update(rated_powers_w=[2., 2.] + [0.] * 10, mask=[1, 1] + [0.] * 10)
        s = settings()
        s.update(lambda_deg=0., lambda_soc=0.)
        demand = 4.  # exactly the installed capability sum eta_i P_i M_i
        result = operation.solve_rhc(.8, 0, [], forcing([demand]), np.array([1.]),
                                     True, h, s)
        self.assertTrue(result['success'], result)
        pwm = result['pwm'][0]
        supplied = supplied_heat(h, pwm)
        self.assertGreaterEqual(supplied, demand, pwm)
        self.assertLessEqual(supplied,
                             demand + operation._RESIDUAL_TOLERANCE['service_oversupply'], pwm)
        self.assertAlmostEqual(result['pwm'][0].max() * 2., 2., places=4)
        # The same boundary replayed through simulate: completed with the
        # shared-margin oversupply residual and no service deficit.
        run = operation.simulate(forcing([demand]), persistence, 'persistence',
                                 np.array([1.]), initial(.8), h, s)
        self.assertTrue(run['completed'], run)
        self.assertIsNone(run['failure_kind'])
        step = run['steps'][0]
        self.assertEqual(step['thermal_score'], 1.)
        self.assertGreaterEqual(step['total_effective_heat_w'], demand)
        self.assertLessEqual(run['residuals']['service'], 0.)
        self.assertLessEqual(run['residuals']['service_oversupply'],
                             operation._RESIDUAL_TOLERANCE['service_oversupply'])
        # Zero demand under alpha = 1: the exact cap row P_supp <= 0 forces
        # every installed pad to zero duty and the score is exactly 1.
        zero = operation.solve_rhc(.8, 0, [], forcing([0.]), np.array([1.]), True, h, s)
        self.assertTrue(zero['success'], zero)
        np.testing.assert_array_equal(zero['pwm'][0], np.zeros(12))

    def test_reported_residual_and_objective_match_exact_equations(self):
        # The returned max_constraint_residual must be a computed bound audit,
        # never a placeholder. The reported objective equals the LP objective
        # under the exact arithmetic: the whole-body minimum of Eq. (3), the
        # zero-demand whole-body convention, the extrapolated Eq. (15)/(16)
        # weights actually handed to the LP, and Eq. (16) as the declared
        # 8-tangent piecewise-linear form. The realized-weights diagnostic
        # uses the executed branch weights and the exact quadratic instead.
        for demand in ([12, 24], [0, 12]):
            h, s = hardware(), settings()
            result = operation.solve_rhc(.8, 0, [], forcing(demand),
                                         np.zeros(2), True, h, s)
            self.assertTrue(result['success'], result)
            self.assertLessEqual(result['max_constraint_residual'], 0.)
            np.testing.assert_array_equal(result['deg_weights'], [.2, .2])
            rated = np.asarray(h['rated_powers_w']) * np.asarray(h['mask'])
            plan = np.column_stack((result['pwm'], result['pv_used_w']))
            power = plan[:, :12] @ rated + h['base_power_w'] - plan[:, 12]
            states = np.r_[.8, .8 - np.cumsum(power * s['dt_hours']
                                              / h['capacity_wh'])]
            np.testing.assert_allclose(result['planned_start_soc'], states[:-1],
                                       atol=1e-9)
            # U_B = min(1, P_supp / P_req), exactly 1 when P_req = 0.
            scores = []
            for k, required in enumerate(demand):
                if required > 0:
                    scores.append(min(1., supplied_heat(h, plan[k, :12]) / required))
                else:
                    scores.append(1.)
            scores = np.asarray(scores)
            ah = np.maximum(power, 0) * s['dt_hours'] / h['voltage_v']
            shortfall = np.maximum(0, s['soc_safe'] - states[:-1])
            tangents = np.array([s['soc_safe'] * j / 8 for j in range(1, 9)])
            pwl = np.maximum(0, np.max(2 * tangents[None, :] * shortfall[:, None]
                                       - tangents[None, :] ** 2, axis=1))
            reference = float(np.sum(
                scores - s['lambda_deg'] * result['deg_weights'] * ah
                - s['lambda_soc'] * pwl))
            self.assertAlmostEqual(result['objective'], reference, places=9)
            branch = np.where(states[:-1] < .2, 1.,
                              np.where(states[:-1] < .5, .5, .2))
            realized = float(np.sum(
                scores - s['lambda_deg'] * branch * ah
                - s['lambda_soc'] * shortfall ** 2))
            self.assertAlmostEqual(result['objective_realized_weights'],
                                   realized, places=9)
            self.assertEqual(len(result['priority_stages']), 1)

    def test_soc_soft_penalty_uses_prior_net_power_and_fixed_load(self):
        h, s = hardware(), settings()
        h['base_power_w'] = 3.
        s.update(soc_safe=.8, lambda_soc=100., lambda_deg=0.)
        tangents = tuple(.8 * j / 8 for j in range(1, 9))
        def pwl(y):
            # Eq. (16) as the declared 8-tangent piecewise-linear form.
            return max(0.0, max(2.0 * b * y - b * b for b in tangents))
        end_socs = []
        for pv in ([0., 0.], [5., 0.]):
            actual = forcing([12., 12.], pv=pv)
            result = operation.solve_rhc(.75, 0, [], actual, np.zeros(2),
                                         True, h, s)
            self.assertTrue(result['success'], result)
            load_first = (np.asarray(result['pwm'][0]) @
                          np.asarray(h['rated_powers_w'])) + h['base_power_w']
            next_soc = .75 - .1 / h['capacity_wh'] * (load_first - result['pv_used_w'][0])
            end_socs.append(next_soc)
            scores = sum(min(1., supplied_heat(h, pwm) / 12.) for pwm in result['pwm'])
            expected = (scores - s['lambda_soc'] * (
                pwl(max(.8 - .75, 0.)) + pwl(max(.8 - next_soc, 0.))))
            self.assertAlmostEqual(result['objective'], expected, places=7)
        self.assertGreater(end_socs[1], end_socs[0])

    def test_strict_groups_protect_core_then_distal_under_shared_bus(self):
        from src.evaluation import evaluate_trajectory
        h, s = hardware(), settings()
        h.update(rated_powers_w=[2.]*12, mask=[int(i in (2, 4, 6)) for i in range(12)],
                 weights=[.1 if i in (2, 4) else .8 if i == 6 else 0.
                          for i in range(12)], bus_limit_w=1., capacity_wh=60.)
        s.update(priority_mode='lexicographic', horizon_steps=1)
        # Whole-body alpha = .05: floor P_supp >= .6 W sits inside the 1 W
        # shared bus, so the priority stages decide the split of the scarce
        # bus between the installed groups (node 2 core, node 6 distal, node
        # 4 other; equal areas give beta = 1/3 and D_i = 4 W each).
        alpha = .05
        actual = forcing([12])
        result = operation.simulate(actual, persistence, 'persistence', np.array([alpha]),
                                    initial(), h, s)
        self.assertTrue(result['completed'], result)
        self.assertEqual(result['solves'][0]['priority_group_labels'],
                         ['core', 'distal', 'other'])
        grouped = operation.group_service(h, 12., result['steps'][0]['pwm'])
        self.assertEqual(len(grouped), 3)
        # Core allocation preference is maximized first: q_2 = 2 * .5 = 1 W
        # against D_2 = 4 W gives r = .25; the bus leaves distal/other at 0.
        self.assertAlmostEqual(grouped[0], .25, places=5)
        self.assertAlmostEqual(grouped[1], 0., places=5)
        self.assertAlmostEqual(grouped[2], 0., places=5)
        tolerances = {'balance_tolerance_wh': 1e-8, 'time_tolerance_hours': 1e-9}
        clean = evaluate_trajectory(result, actual, np.array([alpha]), initial(), h, s, **tolerances)
        self.assertEqual(clean['status'], 'success', clean)
        # Moving bus share from the core pad to the distal pad keeps the
        # whole-body total (equal ratings) but sacrifices the locked core
        # allocation preference: S must fail through priority conformance.
        altered = result['steps'][0]['pwm'].copy()
        altered[2] -= .1
        altered[6] += .1
        result['steps'][0]['pwm'] = altered
        degraded = evaluate_trajectory(result, actual, np.array([alpha]), initial(), h, s, **tolerances)
        self.assertEqual(degraded['status'], 'failed', degraded)
        self.assertFalse(degraded['components']['S'])
        result['solves'][0].pop('forecast')
        unresolved = evaluate_trajectory(result, actual, np.array([alpha]), initial(), h, s,
                                         **tolerances)
        self.assertEqual(unresolved['status'], 'unresolved', unresolved)

    def test_one_step_myopic_uses_real_terminal_only_at_mission_end(self):
        h, s = hardware(), settings()
        h['capacity_wh'] = 1.
        s.update(dt_hours=.1, terminal_soc=.5, lambda_deg=0., lambda_soc=0.)
        actual = forcing([12, 12])
        rhc = operation.simulate(actual, persistence, 'persistence',
                                 np.zeros(2), initial(1.), h, s,
                                 strategy='myopic_qp')
        self.assertEqual(len(rhc['solves'][0]['forecast']['demand_w']), 1)
        self.assertTrue(rhc['completed'], rhc)
        self.assertGreaterEqual(rhc['states'][-1]['soc'], .5)

    def test_configuration_calls_real_operating_evaluator(self):
        actual = forcing([12])
        # Whole-body alpha = .1 needs P_supp >= 1.2 W: feasible only with two
        # battery cells (y = 2), so the search must reject y = 1 on the real
        # operating evaluator.
        floors = np.array([.1])
        s = settings()
        # A dominant degradation cost pins the plan at the declared service
        # floor (P_supp = fl(alpha*demand + margin)) instead of the SOC
        # boundary, so the exactly-binding service plan exercises the raw
        # realized acceptance rather than an exact SOC row.
        s.update(lambda_deg=50.)
        def candidate(y, h, g):
            return {'objectives': (y, 1), 'diagnostics': {'energy_wh': 100},
                    'structural_residuals': {}}
        def run(y, pv_count, g, candidate_data):
            h = hardware()
            h['capacity_wh'] = .1*y
            return operation.simulate(actual, persistence, 'persistence', floors, initial(), h, s)
        result = configuration.search_configuration(2, 0, [0], candidate, run, ('mass', 'cost'))
        self.assertIsNotNone(result['selected'], result)
        self.assertEqual(result['selected']['configuration'], (2, 0, 0))
        self.assertTrue(result['selected']['operation']['completed'])


    def test_pwl_penalty_engages_when_shortfall_positive(self):
        # With SOC below soc_safe the tangent epigraph p_k must carry the
        # declared 8-tangent piecewise-linear penalty of Eq. (16): strictly
        # below the exact quadratic off the declared breakpoints.
        h, s = hardware(), settings()
        result = operation.solve_rhc(.17, 0, [], forcing([0.]), np.zeros(1),
                                     False, h, s)
        self.assertTrue(result['success'], result)
        self.assertEqual(result['solver'], 'HiGHS-LP')
        self.assertEqual(result['evaluations'], 1)
        np.testing.assert_array_equal(result['deg_weights'], [.2])
        shortfall = .2 - .17
        tangents = tuple(.2 * j / 8 for j in range(1, 9))
        pwl = max(0.0, max(2.0 * b * shortfall - b * b for b in tangents))
        self.assertGreater(pwl, 0.)
        self.assertLess(pwl, shortfall ** 2)
        score = 1.  # U_B = 1 exactly when P_req = 0 (zero-demand convention).
        self.assertAlmostEqual(result['objective'], score - s['lambda_soc'] * pwl,
                               places=9)
        self.assertAlmostEqual(result['objective_realized_weights'],
                               score - s['lambda_soc'] * shortfall ** 2, places=9)

    def test_deg_weights_validation_and_default(self):
        h, s = hardware(), settings()
        result = operation.solve_rhc(.8, 0, [], forcing([12]), np.zeros(1),
                                     False, h, s)
        self.assertTrue(result['success'], result)
        np.testing.assert_array_equal(result['deg_weights'], [.2])
        for bad in ([.3], [.2, .2], [float('nan')]):
            with self.assertRaises(ValueError):
                operation.solve_rhc(.8, 0, [], forcing([12]), np.zeros(1),
                                    False, h, s, deg_weights=np.array(bad))

    def test_one_step_weight_extrapolation_from_previous_plan(self):
        # The second control step's weights are the Eq. (15)/(16) branches of
        # the first plan's start-of-interval SOC shifted one interval; a deep
        # first-interval discharge moves the second solve onto the 1.0 branch.
        h, s = hardware(), settings()
        h['capacity_wh'] = 1.
        floors = np.zeros(2)
        floors[0] = .1  # Whole-body floor P_supp >= 1.2 W forces a discharge
        # below the 0.2 branch without violating soc_min = .1 (SOC .25 -> .13).
        result = operation.simulate(forcing([12, 12]), persistence, 'persistence',
                                    floors, initial(.25), h, s)
        self.assertTrue(result['completed'], result)
        first, second = result['solves'][0], result['solves'][1]
        np.testing.assert_array_equal(first['deg_weights'], [.2, .2])
        projected = np.asarray(first['planned_start_soc'][1:2])
        expected = np.where(projected < .2, 1., np.where(projected < .5, .5, .2))
        np.testing.assert_array_equal(second['deg_weights'], expected)
        self.assertEqual(second['deg_weights'][0], 1.0)
        self.assertLess(first['planned_start_soc'][1], .2)


if __name__ == '__main__':
    unittest.main()
