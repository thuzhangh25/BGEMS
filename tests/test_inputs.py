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
    def test_bare_filenames_load_json_into_exactly_five_parameter_classes(self):
        config = load()

        self.assertIsInstance(config, inputs.SystemConfig)
        self.assertIsInstance(config.person, inputs.Person)
        self.assertIsInstance(config.device, inputs.Device)
        self.assertIsInstance(config.scenario, inputs.ScenarioProfile)
        self.assertTrue(all(isinstance(item, inputs.Garment)
                            for item in config.garments))
        public_classes = {name for name, value in vars(inputs).items()
                          if isinstance(value, type) and value.__module__ == inputs.__name__}
        self.assertEqual(public_classes,
                         {'Person', 'Garment', 'Device', 'ScenarioProfile', 'SystemConfig'})

    def test_configuration_file_order_controls_every_regional_array(self):
        config = load()
        person = config.person
        device = config.device

        self.assertEqual(inputs.REGION_ORDER[4:8], (
            'left_upper_arm', 'right_upper_arm',
            'left_lower_arm', 'right_lower_arm'))
        self.assertEqual(person.region_order, inputs.REGION_ORDER)
        self.assertEqual(person.weights[4:8], [0.032, 0.032, 0.017, 0.017])
        self.assertEqual(person.skin_temperatures_c[4:8],
                         [34.39, 34.39, 33.99, 33.99])
        self.assertEqual(person.area_allocation_proxies[4:8],
                         [0.075, 0.075, 0.045, 0.045])
        self.assertEqual(device.rated_powers_w,
                         [9, 9, 18, 18, 15, 15, 9, 9, 18, 18, 15, 15])
        self.assertEqual(device.efficiencies[4:8], [0.75, 0.75, 0.68, 0.68])

    def test_component_values_and_integrated_garment_semantics_are_preserved(self):
        config = load()
        device = config.device

        self.assertEqual((device.cell_capacity_wh, device.cell_mass_kg, device.cell_cost),
                         (60, 0.25, 10))
        self.assertEqual((device.pv_rating_w, device.pv_mass_kg, device.pv_cost),
                         (20, 0.2, 20))
        self.assertEqual((device.base_mass_kg, device.base_cost, device.base_power_w),
                         (0.5, 150, 5))
        self.assertEqual(device.equipment_power_w, 18)
        self.assertEqual([garment.id for garment in config.garments], [0, 1, 2, 3])
        self.assertEqual([garment.clothing_resistance for garment in config.garments],
                         [0.39215, 0.1147, 0.31, 0.0527])
        self.assertEqual([garment.mass_kg for garment in config.garments],
                         [1.5, 1.0, 0.7, 0.5])
        self.assertEqual([garment.cost for garment in config.garments],
                         [800, 600, 350, 250])
        self.assertTrue(all(garment.mass_and_cost_include_heating_pads
                            for garment in config.garments))
        self.assertEqual(config.garments[0].mask, [1] * 12)
        self.assertEqual(config.garments[2].mask, [1] * 8 + [0] * 4)

    def test_adopted_execution_parameters_are_loaded_without_implicit_defaults(self):
        config = load()
        self.assertEqual(config.device.voltage_v, 10.8)
        self.assertEqual(config.device.heat_capacity_j_per_k, 275.0)
        self.assertEqual(config.device.thermal_resistance_k_per_w, 3.0)
        self.assertEqual(config.device.max_discharge_w, 60.0)
        self.assertEqual(config.device.max_charge_w, 30.0)
        self.assertEqual(config.device.bus_limit_w, 200.0)
        self.assertEqual(config.scenario.terminal_soc, 0.15)
        self.assertEqual(config.scenario.initial_state['battery_c'], 20.0)
        self.assertEqual([phase[3] for phase in config.scenario.get_phases()],
                         [.80, .70, .50, .40, .75, .65])
        self.assertEqual(config.settings, {
            'dt_hours': 0.2,
            'horizon_steps': 60,
            'soc_min': 0.05,
            'soc_safe': 0.15,
            'lambda_deg': 0.1,
            'lambda_soc': 100.0,
            'equipment_on_c': 0.0,
            'equipment_off_c': 5.0,
            'filter_window_h': 0.2,
            'solver_tolerance': 1e-7,
            'constraint_tolerance': 1e-9,
        })
        self.assertEqual(config.forecast_mode, 'persistence')

    def test_run_date_timezone_weather_names_and_phase_boundaries_are_explicit(self):
        config = load(run_date=date(2026, 2, 3))
        scenario = config.scenario

        self.assertEqual(scenario.start.isoformat(), '2026-02-03T06:00:00+08:00')
        self.assertEqual(scenario.timezone, 'Asia/Shanghai')
        self.assertTrue(str(scenario.temperature_files).endswith(
            'data/weather/temperature/Muztagh_Ata_2026-02-03.csv'))
        self.assertTrue(str(scenario.pv_files).endswith(
            'data/weather/pv/Muztagh_Ata_2026-02-03.csv'))
        self.assertEqual(scenario.get_duration_hours(), 12)
        self.assertEqual(scenario.get_phase_label(0), 'Initialization')
        self.assertEqual(scenario.get_phase_label(1.999), 'Initialization')
        self.assertEqual(scenario.get_phase_label(2), 'Approach hike')
        self.assertEqual(scenario.get_activity(7), 6)
        self.assertEqual(scenario.get_service_requirement(7), 0.40)
        with self.assertRaises(ValueError):
            scenario.get_activity(12)

    def test_qinling_scenario_and_backpack_constraints_match_configuration(self):
        config = load(scenario_type=3)

        self.assertEqual(config.scenario.name, 'Qinling Outdoor Scientific Expedition')
        self.assertEqual(config.scenario.category, 'long-term heavy heating-demand')
        self.assertEqual(config.scenario.get_phase_label(1.0), 'Low-intensity field activity')
        self.assertEqual(config.scenario.get_phase_label(10.0), 'Equipment transport')
        self.assertEqual(config.scenario.get_activity(10.0), 4.0)
        self.assertEqual(config.scenario.get_duration_hours(), 14.0)
        self.assertEqual(
            Path(config.scenario.temperature_files).name,
            'Qinling_Station_2026-01-01.csv')
        self.assertEqual(
            Path(config.scenario.pv_files).name,
            'Qinling_Station_2026-01-01.csv')
        self.assertTrue(config.scenario.backpack_included)
        self.assertTrue(config.scenario.equipment_heater_required)
        self.assertTrue(config.device.equipment_available)
        self.assertEqual(config.max_pv_units, 8)

    def test_absolute_paths_and_source_file_provenance_are_preserved(self):
        subject = (SYSTEMCONF / 'subject.json').resolve()
        scenarios = (SYSTEMCONF / 'scenario_profiles.json').resolve()
        components = (SYSTEMCONF / 'components.json').resolve()
        config = load(subject=subject, scenarios=scenarios, components=components)

        self.assertEqual(config.source_files, {
            'subject': str(subject),
            'scenario_profiles': str(scenarios),
            'components': str(components),
        })

    def test_body_and_skin_properties_follow_current_values_without_aliasing(self):
        first = load()
        second = load()
        person = first.person
        expected = 0.202 * 70 ** 0.425 * 1.75 ** 0.725

        self.assertAlmostEqual(person.body_area_m2, expected)
        self.assertAlmostEqual(sum(person.regional_areas_m2), expected)
        self.assertAlmostEqual(person.skin_setpoint_c, 34.62621)
        person.height_cm = 180
        person.weight_kg = 80
        person.weights[0] = 0
        first.garments[0].mask[0] = 0
        self.assertAlmostEqual(person.body_area_m2,
                               0.202 * 80 ** 0.425 * 1.80 ** 0.725)
        self.assertEqual(second.person.weights[0], 0.105)
        self.assertEqual(second.garments[0].mask[0], 1)

    def test_service_floor_scalar_uniform_vector_and_non_uniform_rejection(self):
        # Whole-body service definition: each phase carries one scalar alpha.
        scenario = load().scenario
        scenario.set_phases((('Scalar', 1.0, 1.0, 0.5),))
        requirement = scenario.get_service_requirement(0.5)
        self.assertIsInstance(requirement, float)
        self.assertEqual(requirement, 0.5)
        # A hand-expanded uniform 12-vector folds to its scalar (never stored
        # as a per-region array); None (no requirement) is still accepted.
        scenario.set_phases((('Uniform', 1.0, 1.0, [0.4] * 12),
                             ('Free', 1.0, 1.0, None)))
        self.assertEqual(scenario.get_service_requirement(0.0), 0.4)
        self.assertIsNone(scenario.get_service_requirement(1.0))
        self.assertEqual([phase[3] for phase in scenario.get_phases()], [0.4, None])
        # A non-uniform 12-vector is rejected: never averaged, never
        # min/max-reduced — the whole-body definition has no per-region floors.
        for bad in ([0.4, 0.5] + [0.4] * 10, [0.4] * 11 + [0.5]):
            with self.assertRaisesRegex(ValueError, 'alpha'):
                scenario.set_phases((('Mixed', 1.0, 1.0, bad),))
        # Stored phase floors stay scalar floats after the fold.
        scenario.set_phases((('Folded', 1.0, 1.0, (0.3,) * 12),))
        self.assertIsInstance(scenario.get_phases()[0][3], float)
        self.assertEqual(scenario.get_service_requirement(0.9), 0.3)

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
