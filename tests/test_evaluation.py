'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Check independent accounting, corruption detection and failed prefixes.
'''

from copy import deepcopy
import unittest
import numpy as np
from src.evaluation import evaluate_trajectory, summarize_runs
from test_operation import forcing, hardware, settings, replay


def audit(run, actual, hw=None, cfg=None, soc=0.8):
    return evaluate_trajectory(
        run,
        actual,
        soc,
        hardware() if hw is None else hw,
        settings() if cfg is None else cfg,
        balance_tolerance_wh=1e-6,
        time_tolerance_hours=1e-9,
    )


class EvaluationTests(unittest.TestCase):
    def test_full_trace_metrics_are_independently_reconstructed(self):
        actual = forcing([12.0, 12.0])
        run = replay(actual, controller='rbc')
        result = audit(run, actual)
        self.assertTrue(result['valid'], result['issues'])
        metrics = result['metrics']
        self.assertAlmostEqual(metrics['load_energy_wh'], 12.0)
        self.assertAlmostEqual(metrics['mean_service_full_mission'], 0.5)
        self.assertAlmostEqual(metrics['unmet_regional_heat_wh'], 12.0)
        self.assertAlmostEqual(metrics['terminal_soc'], 0.68)
        self.assertAlmostEqual(metrics['discharge_penalty_ah'], 0.24)

    def test_soc_score_and_device_corruption_are_not_trusted(self):
        actual = forcing([12.0])
        run = replay(actual, controller='rbc')
        for field in ('soc', 'score', 'equipment_on', 'pv_used_w'):
            changed = deepcopy(run)
            if field == 'soc':
                changed['states'][-1]['soc'] += 0.1
            elif field == 'score':
                changed['actions'][0]['score'] = 0.2
            elif field == 'equipment_on':
                changed['actions'][0]['equipment_on'] = 1
            else:
                changed['actions'][0]['pv_used_w'] = 10.0
            with self.subTest(field=field):
                self.assertFalse(audit(changed, actual)['valid'])

    def test_partial_metrics_cannot_masquerade_as_full_mission_metrics(self):
        hw = hardware()
        hw.update(base_power_w=1.0)
        actual = forcing([0.0, 0.0, 0.0])
        run = replay(actual, hw, controller='rbc', soc=0.2)
        # A saved solver failure after one executed interval retains only that prefix.
        run['actions'] = run['actions'][:1]
        run['states'] = run['states'][:2]
        run['completed'] = False
        run['failure'] = {'kind': 'numerical_failure', 'step': 1, 'time_h': 1.0,
                          'reason': 'Recorded numerical solver failure.'}
        result = audit(run, actual, hw, soc=0.2)
        self.assertTrue(result['valid'], result['issues'])
        self.assertFalse(result['mission_completed'])
        self.assertIsNone(result['metrics']['terminal_soc'])
        self.assertIsNone(result['metrics']['mean_service_full_mission'])
        self.assertEqual(result['metrics']['mean_service_completed_prefix'], 1.0)
        self.assertAlmostEqual(result['metrics']['completion_fraction'], 1 / 3)
        totals = summarize_runs([result], 4)
        self.assertEqual(totals['failed_runs'], 1)
        self.assertEqual(totals['not_run'], 3)
        self.assertEqual(totals['completion_rate'], 0.0)

    def test_last_interval_and_charge_energy_are_included(self):
        hw = hardware()
        hw['base_power_w'] = 1.0
        actual = forcing([0.0, 0.0], [0.0, 5.0])
        run = replay(actual, hw, controller='rbc')
        result = audit(run, actual, hw)
        self.assertTrue(result['valid'], result['issues'])
        self.assertEqual(result['metrics']['load_energy_wh'], 2.0)
        self.assertEqual(result['metrics']['battery_discharge_wh'], 1.0)
        self.assertEqual(result['metrics']['battery_charge_wh'], 4.0)
        self.assertAlmostEqual(result['metrics']['terminal_soc'], 0.83)
        changed = deepcopy(run)
        changed['states'].pop()
        self.assertFalse(audit(changed, actual, hw)['valid'])

    def test_missing_actions_cannot_be_labeled_complete(self):
        actual = forcing([0.0])
        run = replay(actual, controller='rbc')
        run['actions'].clear()
        run['states'] = run['states'][:1]
        self.assertFalse(audit(run, actual)['valid'])

    def test_reserve_record_and_boundary_tampering_are_detected(self):
        hw = hardware()
        hw.update(capacity_wh=10.0, base_power_w=1.0)
        actual = forcing([12.0, 12.0, 12.0])
        run = replay(actual, hw, controller='rbc', soc=0.45)
        self.assertTrue(audit(run, actual, hw, soc=0.45)['valid'])
        for field in ('reserve_required_after_wh', 'reserve_margin_after_wh',
                      'reserve_curtailed_body_w', 'physical_curtailed_body_w'):
            changed = deepcopy(run)
            changed['actions'][0][field] += 1.0
            self.assertFalse(audit(changed, actual, hw, soc=0.45)['valid'])

    def test_reserve_domain_rejection_is_valid_but_not_a_completed_mission(self):
        hw = hardware()
        hw.update(capacity_wh=10.0, base_power_w=1.0)
        actual = forcing([0.0] * 3)
        run = replay(actual, hw, controller='rbc', soc=.2)
        result = audit(run, actual, hw, soc=.2)
        self.assertTrue(result['valid'], result['issues'])
        self.assertFalse(result['mission_completed'])
        self.assertFalse(result['metrics']['reserve_domain_admissible'])
        self.assertIsNone(result['metrics']['mean_service_completed_prefix'])


if __name__ == '__main__':
    unittest.main()
