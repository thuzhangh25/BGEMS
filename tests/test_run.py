'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Offline execution integration tests with synthetic one-hour inputs.
'''

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest

from src import inputs, run


def synthetic_config(root, ambient_c=30.0):
    config = inputs.load_system_config(
        'subject.json', 'scenario_profiles.json', 'components.json', 1
    )
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    temperature = Path(root) / 'temperature.csv'
    pv = Path(root) / 'pv.csv'
    temperature.write_text(
        'timestamp,sample_duration_hours,air_temperature_c\n'
        f'{start.isoformat()},1.0,{ambient_c}\n'
    )
    pv.write_text(
        'timestamp,sample_duration_hours,reference_rating_w,pv_power_w\n'
        f'{(start+timedelta(hours=1)).isoformat()},1.0,100.0,0.0\n'
    )
    config.scenario.start = start
    config.scenario.set_phases([('Synthetic', 1.0, 1.0)])
    config.scenario.temperature_files = str(temperature)
    config.scenario.pv_files = str(pv)
    config.scenario.initial_state = {'soc': 0.8}
    config.settings.update(dt_hours=0.5, horizon_steps=2)
    config.max_battery_units = 2
    config.max_pv_units = 1
    return config


class RunTests(unittest.TestCase):
    def test_fixed_hardware_shared_across_real_controller_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = synthetic_config(root)
            result = run.execute(
                config, 'fixed', root / 'runs', configuration=(1, 0, 0)
            )
            self.assertEqual(result['outcomes']['completed_runs'], 4)
            records = [
                json.loads((root / 'runs' / c / 'run.json').read_text())
                for c in run.CONTROLLERS
            ]
            self.assertEqual(len({r['forcing_sha256'] for r in records}), 1)
            for record in records:
                self.assertTrue(record['evaluation']['valid'])
                self.assertAlmostEqual(
                    record['evaluation']['metrics']['load_energy_wh'], 5.0
                )
                self.assertAlmostEqual(
                    record['evaluation']['metrics']['terminal_soc'], 0.8 - 5 / 60
                )
            with self.assertRaises(FileExistsError):
                run.execute(config, 'fixed', root / 'runs', configuration=(1, 0, 0))
            self.assertEqual(
                json.loads((root / 'runs/rbc/run.json').read_text())['evaluation'],
                records[0]['evaluation'],
            )

    def test_empty_design_domain_keeps_scheduled_controller_denominator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = synthetic_config(root, -100.0)
            result = run.execute(config, 'search', root / 'search')
            self.assertIsNone(result['selected'])
            self.assertEqual(result['candidate_count'], 16)
            self.assertEqual(result['outcomes']['scheduled_runs'], 4)
            self.assertEqual(result['outcomes']['not_run'], 4)
            self.assertEqual(result['outcomes']['completed_runs'], 0)

    def test_forcing_cannot_trigger_hardware_redesign(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = synthetic_config(root)
            with self.assertRaises(ValueError):
                run.execute(config, 'search', root / 'invalid', forcing={})
            self.assertFalse((root / 'invalid').exists())


if __name__ == '__main__':
    unittest.main()
