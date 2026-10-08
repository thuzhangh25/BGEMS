'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Check source time semantics, dated CSV output, PV conversion and errors offline.
Synthetic API responses isolate the tests from network access and live weather data.
Units: Temperature in degrees C, irradiance in W/m², rated capacity and output in W.
'''
from datetime import datetime, timezone
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
from zoneinfo import ZoneInfo

from src import weather



class WeatherReaderTests(unittest.TestCase):
    def test_default_source_semantics_and_dst_elapsed_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            temperature = root / 'temperature.csv'
            temperature.write_text(
                'timestamp,sample_duration_hours,air_temperature_c\n'
                '2026-03-08T06:30:00+00:00,1.0,10.0\n',
                encoding='utf-8',
            )
            pv = root / 'pv.csv'
            pv.write_text(
                'timestamp,sample_duration_hours,reference_rating_w,pv_power_w\n'
                '2026-03-08T07:30:00+00:00,1.0,100.0,50.0\n',
                encoding='utf-8',
            )
            scenario = SimpleNamespace(
                start=datetime(2026, 3, 8, 1, 30, tzinfo=ZoneInfo('America/New_York')),
                temperature_files=(temperature,),
                pv_files=(pv,),
                temperature_sample_hours=1.0,
                pv_sample_hours=1.0,
                pv_reference_rating_w=100.0,
                get_phases=lambda: (('Synthetic', 1.0, 1.0, 0.0),),
            )
            data = weather.read_scenario_weather(scenario, 0.5)
            self.assertEqual(
                [(stamp.hour, stamp.minute) for stamp in data['times']],
                [(1, 30), (3, 0)],
            )
            elapsed = [
                (stamp.astimezone(timezone.utc) -
                 data['times'][0].astimezone(timezone.utc)).total_seconds()
                for stamp in data['times']
            ]
            self.assertEqual(elapsed, [0, 1800])
            self.assertEqual(data['air_c'].tolist(), [10.0, 10.0])
            self.assertEqual(data['reference_pv_w'].tolist(), [50.0, 50.0])
            self.assertEqual(
                data['semantics'],
                {'temperature': 'sample_hold', 'pv': 'interval_end_mean'},
            )

    def test_metadata_override_gap_coverage_and_reference_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            temperature = root / 'temperature.csv'
            pv = root / 'pv.csv'
            temperature.write_text(
                'timestamp,sample_duration_hours,semantics,air_temperature_c\n'
                '2026-01-01T00:00:00+00:00,1.0,interval_start_mean,0.0\n'
                '2026-01-01T02:00:00+00:00,1.0,interval_start_mean,1.0\n',
                encoding='utf-8',
            )
            pv.write_text(
                'timestamp,sample_duration_hours,semantics,reference_rating_w,pv_power_w\n'
                '2026-01-01T00:00:00+00:00,1.0,interval_start_mean,100.0,0.0\n'
                '2026-01-01T01:00:00+00:00,1.0,interval_start_mean,100.0,0.0\n',
                encoding='utf-8',
            )
            scenario = SimpleNamespace(
                start=datetime(2026, 1, 1, tzinfo=timezone.utc),
                temperature_files=(temperature,),
                pv_files=(pv,),
                temperature_sample_hours=1.0,
                pv_sample_hours=1.0,
                pv_reference_rating_w=100.0,
                get_phases=lambda: (('Synthetic', 2.0, 1.0, 0.0),),
            )
            with self.assertRaisesRegex(ValueError, 'gap inconsistent'):
                weather.read_scenario_weather(scenario, 0.5)

            temperature.write_text(
                'timestamp,sample_duration_hours,semantics,air_temperature_c\n'
                '2026-01-01T00:00:00+00:00,1.0,interval_start_mean,0.0\n'
                '2026-01-01T01:00:00+00:00,1.0,interval_start_mean,1.0\n',
                encoding='utf-8',
            )
            scenario.pv_reference_rating_w = 200.0
            with self.assertRaisesRegex(ValueError, 'reference_rating_w disagrees'):
                weather.read_scenario_weather(scenario, 0.5)

            scenario.pv_reference_rating_w = 100.0
            scenario.temperature_files = (temperature,)
            scenario.get_phases = lambda: (('Synthetic', 3.0, 1.0, 0.0),)
            with self.assertRaisesRegex(ValueError, 'does not cover'):
                weather.read_scenario_weather(scenario, 0.5)

class WeatherWriterTests(unittest.TestCase):
    def setUp(self):
        self.processor = weather.WeatherDataProcessor()
        self.payload = {
            "timezone": "UTC",
            "hourly": {
                "time": ["2025-12-01T00:00", "2025-12-01T01:00"],
                "global_tilted_irradiance": [0, 1000],
                "shortwave_radiation": [0, 800],
                "direct_radiation": [0, 600],
                "diffuse_radiation": [0, 200],
            },
            "hourly_units": {
                "global_tilted_irradiance": "W/m²",
                "shortwave_radiation": "W/m²",
                "direct_radiation": "W/m²",
                "diffuse_radiation": "W/m²",
            },
        }

    def test_both_csv_paths_apply_losses_once_and_preserve_preceding_hour_mean(self):
        # At reference irradiance, a 100 Wp panel with 95% efficiency gives 95 W.
        response = SimpleNamespace(ok=True, json=lambda: copy.deepcopy(self.payload))
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(weather.requests, "get", return_value=response), \
                patch.object(self.processor, "_get_coordinates", return_value=(45.75, 126.63)), \
                patch.object(weather.time, "sleep"):
            for name in ("single", "multiple"):
                csv_dir = Path(directory) / name
                arguments = dict(start_date="2025-12-01", end_date="2025-12-01",
                                 csv_dir=str(csv_dir), img_dir=str(csv_dir), kwp=0.1)
                if name == "single":
                    self.processor.scrape_and_save_pv("Harbin", **arguments)
                else:
                    self.processor.scrape_multiple_locations_pv(["Harbin"], **arguments)
                data = pd.read_csv(csv_dir / "Harbin_2025-12-01.csv")
                self.assertEqual(data["pv_power_w"].tolist(), [0.0, 95.0])
                self.assertEqual(data["reference_rating_w"].tolist(), [100.0, 100.0])
                self.assertEqual(data["sample_duration_hours"].tolist(), [1.0, 1.0])
                self.assertEqual(data["semantics"].tolist(),
                                 ["interval_end_mean", "interval_end_mean"])
                self.assertEqual(data["timestamp"].tolist(),
                                 ["2025-12-01T00:00:00+00:00",
                                  "2025-12-01T01:00:00+00:00"])

    def test_temperature_is_an_instant_sample_hold_and_multiday_output_is_split(self):
        payload = {
            "timezone": "UTC", "elevation": 100,
            "hourly": {
                "time": ["2025-12-01T23:00", "2025-12-02T00:00"],
                "temperature_2m": [-10.0, -11.0],
            },
            "hourly_units": {"temperature_2m": "°C"},
        }
        response = SimpleNamespace(json=lambda: payload, raise_for_status=lambda: None)
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(weather.requests, "get", return_value=response), \
                patch.object(self.processor, "_get_coordinates", return_value=(45.75, 126.63)), \
                patch.object(weather.time, "sleep"):
            self.processor.scrape_and_save(
                "Harbin", "2025-12-01", "2025-12-02",
                csv_dir=directory, img_dir=directory)
            first = pd.read_csv(Path(directory) / "Harbin_2025-12-01.csv")
            second = pd.read_csv(Path(directory) / "Harbin_2025-12-02.csv")
            self.assertEqual(first["air_temperature_c"].tolist(), [-10.0])
            self.assertEqual(second["air_temperature_c"].tolist(), [-11.0])
            self.assertEqual(first["semantics"].tolist(), ["sample_hold"])
            self.assertEqual(first["timestamp"].tolist(),
                             ["2025-12-01T23:00:00+00:00"])

    def test_fetch_scenario_weather_uses_requested_date_and_configured_paths(self):
        temperature_payload = {
            "timezone": "UTC",
            "elevation": 100,
            "hourly": {
                "time": ["2025-12-01T00:00", "2025-12-01T01:00"],
                "temperature_2m": [-10.0, -11.0],
            },
            "hourly_units": {"temperature_2m": "°C"},
        }
        temperature_response = SimpleNamespace(
            ok=True, status_code=200, text="", json=lambda: temperature_payload)
        pv_response = SimpleNamespace(
            ok=True, status_code=200, text="", json=lambda: copy.deepcopy(self.payload))
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(weather.WeatherDataProcessor, "_get_coordinates",
                             return_value=(38.27, 75.10)), \
                patch.object(weather.requests, "get",
                             side_effect=(temperature_response, pv_response)) as request:
            root = Path(directory)
            scenario = SimpleNamespace(
                start=datetime(2025, 12, 1, tzinfo=timezone.utc),
                timezone="UTC",
                weather_location="Muztagh Ata",
                temperature_files=root / "temperature/Muztagh_Ata_2025-12-01.csv",
                pv_files=root / "pv/Muztagh_Ata_2025-12-01.csv",
                pv_reference_rating_w=100.0,
            )
            progress = []
            temperature_path, pv_path = weather.fetch_scenario_weather(
                scenario, progress=progress.append)

            self.assertTrue(temperature_path.is_file())
            self.assertTrue(pv_path.is_file())
            self.assertEqual(request.call_count, 2)
            for call in request.call_args_list:
                self.assertEqual(call.args[0], self.processor.base_url)
                self.assertEqual(call.kwargs["params"]["start_date"], "2025-12-01")
                self.assertEqual(call.kwargs["params"]["end_date"], "2025-12-01")
                self.assertEqual(call.kwargs["params"]["timezone"], "UTC")
                self.assertEqual(call.kwargs["timeout"], weather.REQUEST_TIMEOUT)
            self.assertEqual(progress, [
                "[weather 1/3] Geocoding Muztagh Ata",
                "[weather 2/3] Fetching temperature for 2025-12-01",
                "[weather 3/3] Fetching PV for 2025-12-01",
                "[weather] Saved Muztagh_Ata_2025-12-01.csv and "
                "Muztagh_Ata_2025-12-01.csv",
            ])
            self.assertEqual(
                pd.read_csv(temperature_path)["semantics"].unique().tolist(),
                ["sample_hold"])
            self.assertEqual(
                pd.read_csv(pv_path)["reference_rating_w"].unique().tolist(),
                [100.0])

    def test_missing_irradiance_is_not_zero_generation(self):
        self.payload["hourly"]["global_tilted_irradiance"][1] = None
        response = SimpleNamespace(ok=True, json=lambda: self.payload)
        with patch.object(weather.requests, "get", return_value=response):
            with self.assertRaisesRegex(ValueError, "missing or non-finite"):
                self.processor._fetch_pv_from_api(45.75, 126.63, "2025-12-01", "2025-12-01")

    def test_http_error_keeps_server_reason(self):
        response = SimpleNamespace(ok=False, status_code=400,
                                   text='{"reason":"invalid hourly variable"}')
        with patch.object(weather.requests, "get", return_value=response):
            with self.assertRaisesRegex(weather.requests.HTTPError, "invalid hourly variable"):
                self.processor._fetch_pv_from_api(45.75, 126.63, "2025-12-01", "2025-12-01")


if __name__ == "__main__":
    unittest.main()
