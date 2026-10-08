'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Exercise independent design-date comparison, date-ledger denominators,
and the explicit four-argument comparison CLI on offline synthetic weather.
The aggregator must reject legacy (pre-whole-body-definition) folders instead
of silently mixing service definitions.
'''
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from scripts.compare import compare, frozen_reference
from scripts.aggregate_sweep import aggregate, RANGES
from src import run
from src.modeling import SERVICE_DEFINITION_ID
from test_run import synthetic_config


class ComparisonTests(unittest.TestCase):
    def test_design_date_is_frozen_and_all_four_arms_replay_real_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design_weather = root / 'design_weather'
            design_weather.mkdir()
            design = synthetic_config(design_weather)
            design_output = root / 's1_20260101'
            baseline = run.execute(design, 'search', design_output)
            self.assertEqual(baseline['status'], 'success')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({
                'scenario_type': 1, 'subject_id': design.person.subject_id,
                'design_days': [{'date': '20260101', 'directory': 's1_20260101'}]
            }), encoding='utf-8')
            evaluation_weather = root / 'evaluation_weather'
            evaluation_weather.mkdir()
            current = synthetic_config(evaluation_weather, run_date='20260102')
            frozen, _ = frozen_reference(manifest, current, '20260102')
            self.assertEqual(frozen, (1, 0, 0))
            with self.assertRaisesRegex(ValueError, 'distinct'):
                frozen_reference(manifest, current, '20260101')
            paired = compare(current, '20260102', manifest, root / 'paired')
            self.assertEqual(current.fixed_configuration, (1, 0, 0))
            self.assertEqual(set(paired['arms']), {
                'adaptive_rhc', 'frozen_rhc', 'adaptive_myopic', 'frozen_myopic'})
            self.assertEqual(paired['frozen_configuration'], (1, 0, 0))
            for arm in paired['arms'].values():
                self.assertEqual(arm['date_status'], 'success', arm)
                self.assertTrue(all(arm['evaluation']['components'][key] for key in 'CESR'))
            with (root / 'paired/comparison.csv').open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            rhc = json.loads((root / 'paired/adaptive_search/runs/y1_h0_g0.json').read_text())
            myopic = json.loads((root / 'paired/adaptive_myopic.json').read_text())
            self.assertEqual(rhc['trajectory']['states'], myopic['trajectory']['states'])
            self.assertEqual(rhc['trajectory']['steps'][0]['pwm'],
                             myopic['trajectory']['steps'][0]['pwm'])
            self.assertEqual(len(rhc['trajectory']['solves'][0]['forecast']['air_c']), 2)
            self.assertEqual(len(myopic['trajectory']['solves'][0]['forecast']['air_c']), 1)
            ledger = aggregate(root)
            self.assertEqual(len(ledger), sum(map(len, RANGES.values())))
            selected = next(row for row in ledger if row['scenario_type'] == 1 and
                            row['date'] == '20260101')
            self.assertEqual(selected['status'], 'success', selected)
            self.assertEqual((selected['selected_y'], selected['selected_h'],
                              selected['selected_g']), (1, 0, 0))
            self.assertEqual(sum(row['status'] == 'not_run' for row in ledger),
                             len(ledger)-1)
            # A pre-refactor folder (no service_definition in parameters.json)
            # is rejected as an error row, never aggregated with whole-body
            # evidence — old-model outputs are frozen, not mixed.
            legacy = root / 's1_20251201'
            shutil.copytree(design_output, legacy)
            legacy_parameters = json.loads((legacy / 'parameters.json').read_text())
            legacy_parameters.pop('service_definition', None)
            (legacy / 'parameters.json').write_text(json.dumps(legacy_parameters),
                                                    encoding='utf-8')
            legacy_row = next(row for row in aggregate(root) if row['date'] == '20251201')
            self.assertEqual(legacy_row['status'], 'error', legacy_row)
            self.assertIn('service definition', legacy_row['reason'])
            current.comparison = {'enabled': True, 'design_manifest': str(manifest)}
            with patch.object(run, 'load_system_config', return_value=current), \
                 patch.object(run, 'fetch_scenario_weather'), \
                 redirect_stdout(io.StringIO()):
                code = run.main(['--scenario_type', '1', '--date', '20260102',
                                 '--output_dir', str(root / 'cli_paired')])
            self.assertEqual(code, 0)
            self.assertTrue((root / 'cli_paired/comparison.json').is_file())


if __name__ == '__main__':
    unittest.main()
