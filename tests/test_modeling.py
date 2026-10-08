'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Check paper equations with synthetic hand-calculable inputs.
These examples are not physical calibration or approved scenario parameters.
'''
import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("modeling", Path(__file__).parents[1] / "src/modeling.py")
modeling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(modeling)


class ModelingTests(unittest.TestCase):
    def test_heat_balance_uses_total_watts_and_signed_endogenous_heat(self):
        # Environmental loss = 2*(30-0)/0.5 = 120 W; metabolism = 116.3 W.
        result = modeling.heating_requirement(0, 1, 2, 0.4, 0.1, 30, 20, 10)
        self.assertAlmostEqual(result, 33.7)
        # E+W exceeds M: retaining the signed term must INCREASE demand.
        result = modeling.heating_requirement(0, 1, 2, 0.4, 0.1, 30, 150, 10)
        self.assertAlmostEqual(result, 163.7)
        self.assertEqual(modeling.heating_requirement(30, 2, 2, 0.4, 0.1, 30, 0, 0), 0)

    def test_heat_losses_preserve_declared_heuristic_with_5815_factor(self):
        evaporation, work = modeling.heat_losses(3, 25, 2)
        self.assertAlmostEqual(evaporation, 0.1 * (58.15 * 2) * 1.3 * 1.1)
        self.assertAlmostEqual(work, 0.1 * 2 * 58.15 * 2)

        capped, high_work = modeling.heat_losses(6, 20, 2)
        self.assertAlmostEqual(capped, 0.1 * (58.15 * 2) * 1.5)
        self.assertAlmostEqual(high_work, 0.1 * 5 * 58.15 * 2)
        _, resting_work = modeling.heat_losses(0.5, -20, 2)
        self.assertEqual(resting_work, 0)

    def test_service_definition_identifier_is_declared_once(self):
        self.assertEqual(modeling.SERVICE_DEFINITION_ID, 'whole_body_total_heat_v1')
        # The legacy per-region area-share function was deleted (author, 2026-10-04);
        # production code must not be able to reach it again.
        self.assertFalse(hasattr(modeling, 'regional_fulfillment'))

    def test_whole_body_score_sums_installed_effective_heat_only(self):
        # Whole-body Eq. (3): P_supp = sum_i eta_i P_i M_i u_i over I_g; x13
        # (here node 0, masked out) never enters P_supp.
        areas = np.array([2.0] + [1.0] * 11)  # areas are NOT an argument any more
        mask = np.array([0] + [1] * 11)
        result = modeling.whole_body_fulfillment(130, np.full(12, 0.5),
                                                 np.full(12, 10), mask, np.ones(12))
        self.assertEqual(result['installed'].tolist(), [False] + [True] * 11)
        self.assertAlmostEqual(result['electrical_w'].sum(), 110)
        self.assertEqual(result['electrical_w'][0], 0)
        self.assertEqual(result['effective_heat_w'][0], 0)
        self.assertAlmostEqual(result['total_effective_heat_w'], 55)
        self.assertAlmostEqual(result['score'], 55 / 130)

    def test_zero_demand_and_oversupply_conventions_are_whole_body(self):
        eta, rated, mask = np.ones(12), np.full(12, 100.), np.ones(12)
        # P_req = 0: score exactly 1.0 and P_supp = 0 under zero duty; no
        # consumer may divide by zero.
        zero = modeling.whole_body_fulfillment(0, eta, rated, mask, np.zeros(12))
        self.assertEqual(zero['score'], 1.0)
        self.assertEqual(zero['total_effective_heat_w'], 0.)
        # Score saturates at 1 under oversupply (min, not a ratio).
        supplied = modeling.whole_body_fulfillment(12, eta, rated, mask, np.ones(12))
        self.assertEqual(supplied['score'], 1.0)
        self.assertEqual(supplied['total_effective_heat_w'], 1200.)

    def test_allocation_reference_normalizes_beta_over_the_installed_subset(self):
        # beta_i = M_i BSA_i / sum_{j in I_g} M_j BSA_j: removing a node
        # renormalizes the remaining shares instead of leaving them unmet.
        areas = np.array([2.0] + [1.0] * 11)
        weights = np.array([0.5] + [0.5 / 11] * 11)
        mask = np.array([0] + [1] * 11)
        result = modeling.allocation_reference(130, areas, weights, np.full(12, 0.5),
                                               np.full(12, 10), mask, np.ones(12))
        beta = result['beta']
        self.assertEqual(beta[0], 0.)
        self.assertAlmostEqual(float(beta.sum()), 1.)
        np.testing.assert_allclose(beta[1:], np.full(11, 1 / 11))
        np.testing.assert_allclose(result['allocated_task_w'][1:], np.full(11, 130 / 11))
        self.assertEqual(result['allocated_task_w'][0], 0.)
        self.assertEqual(result['effective_heat_w'][0], 0)
        # Uninstalled nodes are explicitly "not applicable": NaN, never zero
        # and never read as satisfied.
        self.assertTrue(np.isnan(result['allocation_fulfillment'][0]))
        np.testing.assert_allclose(result['allocation_fulfillment'][1:],
                                   np.full(11, min(1., 5. / (130. / 11.))))
        self.assertEqual(result['installed'].tolist(), mask.astype(bool).tolist())

    def test_allocation_reference_zero_demand_and_partial_supply(self):
        areas = np.array([2.0] + [1.0] * 11)
        weights = np.array([0.5] + [0.5 / 11] * 11)
        mask = np.array([0] + [1] * 11)
        # P_req = 0: every installed node has allocation fulfillment exactly 1.
        zero = modeling.allocation_reference(0, areas, weights, np.full(12, 0.5),
                                             np.full(12, 10), mask, np.zeros(12))
        np.testing.assert_array_equal(zero['allocation_fulfillment'][1:], np.ones(11))
        self.assertTrue(np.isnan(zero['allocation_fulfillment'][0]))
        # Positive demand with one node supplied and the rest idle: a node can
        # undersupply its allocation share (a diagnostic, never a service
        # failure) while another node compensates.
        pwm = np.zeros(12)
        pwm[1] = 1.
        partial = modeling.allocation_reference(12, np.ones(12), np.full(12, 1 / 12),
                                                np.ones(12), np.ones(12), np.ones(12), pwm)
        self.assertAlmostEqual(partial['beta'][0], 1 / 12)
        self.assertAlmostEqual(partial['allocated_task_w'][0], 1.)
        self.assertAlmostEqual(partial['allocation_fulfillment'][1], 1.)
        self.assertAlmostEqual(partial['allocation_fulfillment'][0], 0.)
        self.assertTrue(np.all(np.isfinite(partial['allocation_fulfillment'])))

    def test_reference_temperature_uses_score_weights(self):
        value = modeling.weighted_skin_temperature([30] + [34] * 11, [0.5] + [0.5 / 11] * 11)
        self.assertAlmostEqual(value, 32)

    def test_equipment_step_matches_analytic_decay_and_substeps(self):
        # R=2 K/W, C=1000 J/K: a 2000-second interval is one time constant.
        self.assertAlmostEqual(modeling.battery_temperature_step(10, 0, 0, 1000, 2, 2000), 10 / math.e)
        heated = modeling.battery_temperature_step(-10, -10, 5, 1000, 2, 2000)
        self.assertAlmostEqual(heated, -10 / math.e)
        half = modeling.battery_temperature_step(-10, -10, 5, 1000, 2, 1000)
        self.assertAlmostEqual(modeling.battery_temperature_step(half, -10, 5, 1000, 2, 1000), heated)
        self.assertEqual(modeling.battery_temperature_step(-10, -10, 5, 1000, 2, 0), -10)
        self.assertAlmostEqual(modeling.battery_temperature_step(-10, -10, 5, 1000, 2, 1e9), 0)

    def test_invalid_domains_are_not_silently_clipped_or_renormalized(self):
        with self.assertRaises(ValueError):
            modeling.weighted_skin_temperature([34] * 12, [0.1] * 12)
        with self.assertRaises(ValueError):
            modeling.battery_temperature_step(0, 0, 1, 0, 2, 1)
        with self.assertRaises(ValueError):
            modeling.heating_requirement(0, 1, 2, 0.4, 0.1, 30, float("nan"), 0)
        with self.assertRaises(ValueError):
            modeling.heat_losses(-1, 0, 1)
        with self.assertRaises(ValueError):
            modeling.heat_losses(1, 0, 0)
        # whole_body_fulfillment: PWM outside [0, 1], negative demand and
        # wrong-length inputs are rejected, never clipped or renormalized.
        with self.assertRaises(ValueError):
            modeling.whole_body_fulfillment(12, [1.] * 12, [10.] * 12, [1] * 12, [1.1] * 12)
        with self.assertRaises(ValueError):
            modeling.whole_body_fulfillment(-1, [1.] * 12, [10.] * 12, [1] * 12, [0.] * 12)
        with self.assertRaises(ValueError):
            modeling.whole_body_fulfillment(12, [1.] * 13, [10.] * 13, [1] * 13, [0.] * 13)
        # allocation_reference additionally validates areas and score weights.
        with self.assertRaises(ValueError):
            modeling.allocation_reference(12, [1.] * 12, [1.] * 12, [1.] * 12,
                                          [10.] * 12, [1] * 12, [0.] * 12)
        with self.assertRaises(ValueError):
            modeling.allocation_reference(12, [0.] * 12, [1. / 12] * 12, [1.] * 12,
                                          [10.] * 12, [1] * 12, [0.] * 12)
        with self.assertRaises(ValueError):
            modeling.allocation_reference(12, [1.] * 12, [1. / 12] * 12, [1.] * 12,
                                          [10.] * 12, [1] * 12, [1.1] * 12)


if __name__ == "__main__":
    unittest.main()
