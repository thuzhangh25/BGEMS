'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Verify offline corruption detection and incomplete-cohort denominators.
'''

import json
from pathlib import Path
import tempfile
import unittest
from scripts.report_evidence import report
from src.run import execute
from test_run import synthetic_config


class ReportTests(unittest.TestCase):
    def test_missing_controller_remains_in_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = synthetic_config(root)
            execute(config, 'fixed', root / 'runs', configuration=(1, 0, 0))
            (root / 'runs/pid/run.json').unlink()
            evidence = report(root / 'runs', root / 'report')
            self.assertEqual(evidence['outcomes']['scheduled_runs'], 4)
            self.assertEqual(evidence['outcomes']['not_run'], 1)
            self.assertEqual(evidence['outcomes']['completion_rate'], 0.75)

    def test_edited_action_cannot_reuse_a_cached_positive_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = synthetic_config(root)
            execute(config, 'fixed', root / 'runs', configuration=(1, 0, 0))
            path = root / 'runs/rbc/run.json'
            record = json.loads(path.read_text())
            record['run']['states'][-1]['soc'] = 0.99
            self.assertTrue(record['evaluation']['valid'])
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'Invalid trajectory'):
                report(root / 'runs', root / 'report')

    def test_modified_realized_weather_fails_identity_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = synthetic_config(root)
            execute(config, 'fixed', root / 'runs', configuration=(1, 0, 0))
            path = root / 'runs/rbc/run.json'
            record = json.loads(path.read_text())
            record['actual']['air_c'][0] += 1.0
            path.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, 'Forcing identity mismatch'):
                report(root / 'runs', root / 'report')

    def test_individually_valid_different_hardware_is_not_a_paired_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = synthetic_config(root)
            execute(config, 'fixed', root / 'runs', configuration=(1, 0, 0))
            execute(config, 'fixed', root / 'other',
                    controllers=('pid',), configuration=(2, 0, 0))
            (root / 'runs/pid/run.json').write_bytes(
                (root / 'other/pid/run.json').read_bytes())
            with self.assertRaisesRegex(ValueError, 'Unpaired'):
                report(root / 'runs', root / 'report')


if __name__ == '__main__':
    unittest.main()
