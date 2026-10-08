'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Exercise the JSON CLI, explicit execution gates, independent evidence,
and required user-facing outputs. Synthetic runtime values are solver checks,
not accepted scenario parameters or formal results.
'''
import csv
from contextlib import redirect_stderr
from datetime import datetime, timedelta, timezone
import json
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

from src import inputs, run
from src.modeling import SERVICE_DEFINITION_ID


ROOT = Path(__file__).parents[1]
SYSTEMCONF = ROOT / 'data' / 'systemconf'


def synthetic_config(root, *, service_floor=0.0, active_pad=False,
                     heat_capacity=1000.0, dst=False, ambient_c=10.0,
                     run_date='20260101'):
    config = inputs.load_system_config(
        SYSTEMCONF / 'subject.json', SYSTEMCONF / 'scenario_profiles.json',
        SYSTEMCONF / 'components.json', 1, run_date)
    if dst:
        start = datetime(2026, 3, 8, 1, 30, tzinfo=ZoneInfo('America/New_York'))
    else:
        start = datetime.strptime(run_date, '%Y%m%d').replace(tzinfo=timezone.utc)
    source_start = start.astimezone(timezone.utc)
    temperature = root / 'temperature.csv'
    temperature.write_text(
        'timestamp,sample_duration_hours,air_temperature_c\n'
        f'{source_start.isoformat()},1.0,{ambient_c}\n', encoding='utf-8')
    pv = root / 'pv.csv'
    pv.write_text(
        'timestamp,sample_duration_hours,reference_rating_w,pv_power_w\n'
        f'{(source_start + timedelta(hours=1)).isoformat()},1.0,100.0,0.0\n',
        encoding='utf-8')

    config.scenario.start = start
    config.scenario.set_phases((
        ('First phase', 0.5, 1.0, service_floor),
        ('Second phase', 0.5, 1.2, service_floor),
    ))
    config.scenario.initial_state = {
        'soc': 0.8, 'battery_c': 10.0, 'equipment_on': 0, 'air_history_c': []}
    config.scenario.terminal_soc = 0.1
    config.scenario.temperature_files = (temperature,)
    config.scenario.pv_files = (pv,)
    config.scenario.temperature_sample_hours = 1.0
    config.scenario.pv_sample_hours = 1.0
    config.scenario.equipment_heater_required = False
    config.settings.update({
        'dt_hours': 0.5,
        'horizon_steps': 2,
        'soc_min': 0.1,
        'soc_safe': 0.2,
        'lambda_deg': 1.0,
        'lambda_soc': 1.0,
        'equipment_on_c': -5.0,
        'equipment_off_c': 0.0,
        'filter_window_h': 1.0,
        'solver_tolerance': 1e-7,
        'constraint_tolerance': 1e-9,
    })
    config.forecast_mode = 'persistence'
    config.device.voltage_v = 9.0
    config.device.heat_capacity_j_per_k = heat_capacity
    config.device.thermal_resistance_k_per_w = 2.0
    config.device.max_discharge_w = 500.0
    config.device.max_charge_w = 500.0
    config.device.bus_limit_w = 500.0
    config.device.equipment_available = False
    config.garments = [config.garments[0]]
    config.garments[0].mask = [1] + [0] * 11 if active_pad else [0] * 12
    config.max_battery_cells = 1
    config.max_pv_units = 1
    config.fixed_configuration = (1, 0, config.garments[0].id)
    return config


class RunTests(unittest.TestCase):
    def test_fixed_and_search_emit_evidence_and_required_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = synthetic_config(root)
            fixed = run.execute(config, 'fixed', root / 'fixed')
            self.assertEqual(fixed['status'], 'success')
            evidence = json.loads((root / 'fixed/runs/y1_h0_g0.json').read_text())
            self.assertTrue(all(evidence['evaluation']['components'][key] is True
                                for key in ('C', 'E', 'S', 'R')))
            # The whole-body service definition is stamped on every evidence
            # artifact so pre-refactor outputs can never be silently mixed in.
            self.assertEqual(evidence['service_definition'], SERVICE_DEFINITION_ID)
            parameters = json.loads((root / 'fixed/parameters.json').read_text())
            self.assertEqual(parameters['service_definition'], SERVICE_DEFINITION_ID)
            summary = json.loads((root / 'fixed/summary.json').read_text())
            self.assertEqual(summary['service_definition'], SERVICE_DEFINITION_ID)
            self.assertTrue((root / 'fixed/output_operation.pdf').is_file())
            text = (root / 'fixed/output_configuration.txt').read_text()
            self.assertIn('garment_type:', text)
            self.assertIn('battery_units: 1', text)
            parameters = json.loads((root / 'fixed/parameters.json').read_text())
            self.assertEqual(parameters['config']['person']['height_cm'], 175.0)
            self.assertEqual(set(parameters['config_sources']),
                             {'subject', 'scenario_profiles', 'components'})

            unrelated = root / 'fixed/keep.txt'
            unrelated.write_text('keep', encoding='utf-8')
            stale = root / 'fixed/runs/y99_h99_g99.json'
            stale.write_text('{}', encoding='utf-8')
            repeated = run.execute(config, 'fixed', root / 'fixed')
            self.assertEqual(repeated['status'], 'success')
            self.assertEqual(unrelated.read_text(encoding='utf-8'), 'keep')
            self.assertFalse(stale.exists())
            progress = []
            search = run.execute(
                config, 'search', root / 'search', progress=progress.append)
            self.assertEqual(search['selected'], (1, 0, 0))
            with (root / 'search/candidates.csv').open() as stream:
                candidates = list(csv.DictReader(stream))
            self.assertEqual({row['pv_units'] for row in candidates}, {'0', '1'})
            self.assertEqual({row['service_definition'] for row in candidates},
                             {SERVICE_DEFINITION_ID})
            self.assertEqual(progress[0], '[search] Preparing 2 candidates')
            self.assertIn('[search] Candidate 1/2: y1_h0_g0', progress)
            self.assertEqual(progress[-1], '[search] Completed 2 candidates: success')

    def test_policy_failure_remains_a_recorded_partial_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = synthetic_config(root, service_floor=0.5, ambient_c=-20.0)
            summary = run.execute(config, 'fixed', root / 'failed')
            self.assertEqual(summary['status'], 'failed')
            evidence = json.loads((root / 'failed/runs/y1_h0_g0.json').read_text())
            self.assertFalse(evidence['evaluation']['components']['C'])
            self.assertIsNone(evidence['evaluation']['components']['R'])
            self.assertEqual(evidence['trajectory']['steps'], [])

    def test_exported_elapsed_grid_crosses_dst_without_a_phantom_hour(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = synthetic_config(root, dst=True)
            run.execute(config, 'fixed', root / 'dst')
            with (root / 'dst/runs/y1_h0_g0_states.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            times = [datetime.fromisoformat(row['timestamp']) for row in rows]
            self.assertEqual([(value.hour, value.minute) for value in times],
                             [(1, 30), (3, 0), (3, 30)])

    def test_unresolved_runtime_parameters_fail_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = synthetic_config(root, heat_capacity=None)
            with self.assertRaisesRegex(ValueError, 'heat_capacity_j_per_k'):
                run.execute(config, 'fixed', root / 'invalid')
            self.assertFalse((root / 'invalid').exists())

    def test_module_cli_help_and_api_failure_precedes_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            help_result = subprocess.run(
                [sys.executable, '-m', 'src.run', '--help'], cwd=ROOT,
                capture_output=True, text=True, timeout=30)
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            for option in ('--subject', '--scenario_type', '--date', '--output_dir'):
                self.assertIn(option, help_result.stdout)

            error_output = io.StringIO()
            with patch.object(
                    run, 'fetch_scenario_weather',
                    side_effect=run.requests.HTTPError(
                        'Open-Meteo HTTP 400: requested date unavailable')) as fetch, \
                    redirect_stderr(error_output):
                result = run.main([
                    '--scenario_type', '3',
                    '--date', '20260101',
                    '--output_dir', str(root / 'api-failure'),
                ])
            self.assertEqual(result, 2)
            self.assertIn('Open-Meteo HTTP 400', error_output.getvalue())
            self.assertEqual(
                fetch.call_args.args[0].start.date().isoformat(), '2026-01-01')
            self.assertFalse((root / 'api-failure').exists())


if __name__ == '__main__':
    unittest.main()
