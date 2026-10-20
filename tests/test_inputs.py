'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Check the fixed three-file JSON contract and its conversion into the
five simple parameter classes. No values in this file are formal results.
'''
from datetime import date
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).parents[1]
SYSTEMCONF = ROOT / 'data' / 'systemconf'
from src import inputs


def load(scenario_type=2, run_date='20260101', **paths):
    return inputs.load_system_config(
        paths.get('subject', 'subject.json'),
        paths.get('scenarios', 'scenario_profiles.json'),
        paths.get('components', 'components.json'),
        scenario_type,
        run_date,
    )


class InputTests(unittest.TestCase):
    def test_run_date_timezone_and_weather_names_are_explicit(self):
        config = load(run_date=date(2026, 2, 3))
        scenario = config.scenario

        self.assertEqual(scenario.start.isoformat(), '2026-02-03T06:00:00+08:00')
        self.assertEqual(scenario.timezone, 'Asia/Shanghai')

    def test_decimal_phase_boundaries_are_half_open(self):
        scenario = load().scenario
        scenario.set_phases([
            ('First', .1, 1.),
            ('Second', .2, 2.),
            ('Third', .3, 3.),
        ])
        self.assertEqual(scenario.get_duration_hours(), .6)
        self.assertEqual(scenario.get_activity(.3 - 1e-10), 2.)
        self.assertEqual(scenario.get_activity(.3), 3.)
        self.assertEqual(scenario.get_activity(.3 + 1e-10), 3.)
        self.assertEqual(scenario.get_activity(.6 - 1e-10), 3.)
        with self.assertRaises(ValueError):
            scenario.get_activity(.6)
        with self.assertRaises(ValueError):
            scenario.get_activity(-1e-10)

    def test_body_area_tracks_anthropometry_and_conserves_regional_area(self):
        person = inputs.Person('synthetic',175,70,30,'male',[1/12]*12,[34.]*12,[.1]*12)
        expected = 0.202 * 70 ** .425 * 1.75 ** .725
        self.assertAlmostEqual(person.body_area_m2, expected)
        self.assertAlmostEqual(sum(person.regional_areas_m2), expected)
        person.height_cm, person.weight_kg = 180, 80
        self.assertAlmostEqual(person.body_area_m2, .202 * 80**.425 * 1.8**.725)
        self.assertAlmostEqual(sum(person.regional_areas_m2), person.body_area_m2)

    def test_fractional_mask_and_uninstalled_backpack_are_rejected(self):
        from dataclasses import replace
        config = load()
        with self.assertRaises(ValueError):
            replace(config.garments[0], mask=[.5]*12)
        with self.assertRaises(ValueError):
            replace(config.scenario, backpack_included=False, equipment_heater_installed=True)

    def test_subject_contract_rejects_missing_and_unknown_keys(self):
        with (SYSTEMCONF / 'subject.json').open(encoding='utf-8') as stream:
            subject = json.load(stream)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'subject.json'
            missing = dict(subject)
            missing.pop('sex')
            path.write_text(json.dumps(missing), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'missing'):
                load(subject=path)

            unknown = dict(subject)
            unknown['legacy_weight'] = 70
            path.write_text(json.dumps(unknown), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'unknown'):
                load(subject=path)


if __name__ == '__main__':
    unittest.main()
