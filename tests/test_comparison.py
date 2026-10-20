'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Test cumulative perturbations and genuinely shared controller samples.
'''

import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scripts.compare import sample_forcing, run_monte_carlo
from test_run import synthetic_config


class ComparisonTests(unittest.TestCase):
    def test_cold_events_are_cumulative_and_clipped_not_independent_dips(self):
        p = {
            'temperature_bounds_c': [-25.0, 60.0],
            'temp_sigma_c': 0.0,
            'met_sigma': 0.0,
            'met_bounds': [0.1, 12.0],
            'pv_alpha': 2.0,
            'pv_beta': 3.0,
            'event_probability_per_step': 1.0,
            'event_drop_c': -10.0,
        }
        weather = {'air_c': np.zeros(4), 'reference_pv_w': np.full(4, 20.0)}
        result = sample_forcing(weather, np.ones(4), p, [1, 2, 3, 4])
        np.testing.assert_array_equal(result['air_c'], [-10.0, -20.0, -25.0, -25.0])
        self.assertEqual(result['event_steps'], [0, 1, 2, 3])
        repeated = sample_forcing(weather, np.ones(4), p, [1, 2, 3, 4])
        np.testing.assert_array_equal(
            result['reference_pv_w'], repeated['reference_pv_w']
        )
        self.assertTrue(
            np.all((result['reference_pv_w'] > 0) & (result['reference_pv_w'] < 20.0))
        )
        np.testing.assert_array_equal(weather['air_c'], np.zeros(4))

    def test_mc_hardware_and_realization_are_shared_without_mutating_unperturbed_config(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = synthetic_config(root)
            config.scenario.scenario_type = 2
            original_step = config.settings['dt_hours']
            result = run_monte_carlo(
                config, root / 'mc', 1, configuration=(1, 0, 0), progress=None
            )
            self.assertEqual(result['scheduled_controller_runs'], 4)
            self.assertEqual(config.settings['dt_hours'], original_step)
            records = [
                json.loads((root / 'mc/sample_0000' / c / 'run.json').read_text())
                for c in ('rbc', 'pid', 'tacs', 'rhc')
            ]
            for record in records:
                self.assertEqual(record['configuration'], [1, 0, 0])
                self.assertEqual(record['actual'], records[0]['actual'])
                self.assertEqual(
                    record['experiment']['seeds'], records[0]['experiment']['seeds']
                )
                self.assertTrue(
                    record['evaluation']['valid'], record['evaluation']['issues']
                )
            self.assertEqual(records[0]['run']['scheduled_steps'], 5)


if __name__ == '__main__':
    unittest.main()
