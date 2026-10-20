'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Hand-calculable equation tests using synthetic inputs.
'''
import importlib.util
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

    def test_total_demand_scales_with_area_without_double_conversion(self):
        results = []
        for area in (1.0, 2.0):
            evaporation, work = modeling.heat_losses(1.0, 0.0, area)
            results.append(modeling.heating_requirement(
                0.0, 1.0, area, 0.4, 0.1, 30.0, evaporation, work))
        self.assertAlmostEqual(results[0], 7.665)
        self.assertAlmostEqual(results[1], 15.33)

    def test_heat_losses_preserve_declared_heuristic_with_5815_factor(self):
        evaporation, work = modeling.heat_losses(3, 25, 2)
        self.assertAlmostEqual(evaporation, 0.1 * (58.15 * 2) * 1.3 * 1.1)
        self.assertAlmostEqual(work, 0.1 * 2 * 58.15 * 2)

        capped, high_work = modeling.heat_losses(6, 20, 2)
        self.assertAlmostEqual(capped, 0.1 * (58.15 * 2) * 1.5)
        self.assertAlmostEqual(high_work, 0.1 * 5 * 58.15 * 2)
        _, resting_work = modeling.heat_losses(0.5, -20, 2)
        self.assertEqual(resting_work, 0)

    def test_regional_demand_is_conserved_without_mask_renormalization(self):
        areas = np.array([2.0] + [1.0] * 11)
        weights = np.array([0.5] + [0.5 / 11] * 11)
        mask = np.array([0] + [1] * 11)
        result = modeling.regional_fulfillment(
            130, areas, weights, np.ones(12), np.full(12, 10.), mask, np.ones(12))
        np.testing.assert_allclose(result['regional_demand_w'], [20.] + [10.] * 11)
        self.assertAlmostEqual(result['regional_demand_w'].sum(), 130.)
        np.testing.assert_allclose(result['regional_fulfillment'], [0.] + [1.] * 11)
        self.assertAlmostEqual(result['score'], 0.5)
        self.assertEqual(result['regional_unmet_heat_w'][0], 20.)

    def test_regional_surplus_cannot_cover_other_nodes(self):
        pwm = np.zeros(12)
        pwm[0] = 1.
        result = modeling.regional_fulfillment(
            12., np.ones(12), np.full(12, 1 / 12), np.ones(12),
            np.full(12, 12.), np.ones(12), pwm)
        self.assertEqual(result['effective_heat_w'].sum(), 12.)
        self.assertAlmostEqual(result['score'], 1 / 12)
        self.assertAlmostEqual(result['regional_unmet_heat_w'].sum(), 11.)
        zero = modeling.regional_fulfillment(
            0., np.ones(12), np.full(12, 1 / 12), np.ones(12),
            np.full(12, 12.), np.zeros(12), np.zeros(12))
        np.testing.assert_array_equal(zero['regional_fulfillment'], np.ones(12))
        self.assertAlmostEqual(zero['score'], 1.)
        self.assertEqual(zero['electrical_w'].sum(), 0.)

    def test_regional_model_rejects_invalid_mask_weights_and_dimensions(self):
        base = [12., np.ones(12), np.full(12, 1 / 12), np.ones(12),
                np.ones(12), np.ones(12), np.ones(12)]
        for index, invalid in ((1, np.zeros(12)), (2, np.full(12, .1)),
                               (3, np.full(13, 1.)), (5, np.full(12, .5)),
                               (6, np.full(12, 1.1))):
            args = list(base)
            args[index] = invalid
            with self.subTest(index=index), self.assertRaises(ValueError):
                modeling.regional_fulfillment(*args)

    def test_equipment_rule_separates_installation_and_threshold(self):
        for temperature, expected in ((-20.001, 1), (-20., 1), (-19.999, 0)):
            with self.subTest(temperature=temperature):
                self.assertEqual(modeling.equipment_heater_action(temperature, True), expected)
                self.assertEqual(modeling.equipment_heater_action(temperature, False), 0)
        for invalid in (float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                modeling.equipment_heater_action(invalid, True)
        with self.assertRaises(ValueError):
            modeling.equipment_heater_action(-25., .5)

    def test_reference_temperature_uses_score_weights(self):
        value = modeling.weighted_skin_temperature([30] + [34] * 11, [0.5] + [0.5 / 11] * 11)
        self.assertAlmostEqual(value, 32)


    def test_invalid_domains_are_not_silently_clipped_or_renormalized(self):
        with self.assertRaises(ValueError):
            modeling.weighted_skin_temperature([34] * 12, [0.1] * 12)
        with self.assertRaises(ValueError):
            modeling.heating_requirement(0, 1, 2, 0.4, 0.1, 30, float("nan"), 0)
        with self.assertRaises(ValueError):
            modeling.heat_losses(-1, 0, 1)
        with self.assertRaises(ValueError):
            modeling.heat_losses(1, 0, 0)


if __name__ == "__main__":
    unittest.main()
