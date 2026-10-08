'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Read causal scenario CSVs and fetch/write timezone-aware weather source files.
Temperature is a sample hold; irradiance/PV is a preceding-interval mean by default.
Units: Temperature in degrees C, irradiance in W/m², capacity in kWp, PV output in W.
'''
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import math
import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import requests

SLEEP_TIME = 1  # seconds delay between API calls to avoid rate limits
REQUEST_TIMEOUT = (10, 60)  # connect and read timeouts in seconds

_ROUNDOFF = 1e-12
_SUPPORTED_SEMANTICS = ('interval_start_mean', 'sample_hold', 'interval_end_mean')


def _save_temperature_plot(series, title, path, title_size=None):
    """Retain the original plot appearance except for font and grid."""
    font_manager.findfont(
        font_manager.FontProperties(family='Times New Roman', weight='bold'),
        fallback_to_default=False)
    with plt.rc_context({
        'font.family': 'serif', 'font.serif': ['Times New Roman'],
        'font.weight': 'bold', 'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
    }):
        fig, ax = plt.subplots(figsize=(14, 7))
        try:
            for dates, values, label, color in series:
                ax.plot(dates, values, marker='o', color=color, label=label)
            ax.set_title(title, fontsize=title_size)
            ax.set_xlabel('Date')
            ax.set_ylabel('Temperature (°C)')
            ax.grid(False)
            ax.legend()
            fig.tight_layout()
            fig.savefig(path, dpi=200)
        finally:
            plt.close(fig)


def _scenario_times(scenario, dt_hours):
    """Return timezone-aware interval starts on the scenario's uniform elapsed grid."""
    if not math.isfinite(dt_hours) or dt_hours <= 0:
        raise ValueError('Operation dt_hours must be finite and positive.')
    start = scenario.start
    if not isinstance(start, datetime) or start.tzinfo is None or start.utcoffset() is None:
        raise ValueError('Scenario start must be a timezone-aware datetime.')
    dt_decimal = Decimal(str(dt_hours))
    phase_steps = []
    for index, phase in enumerate(scenario.get_phases()):
        duration = phase[1]
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f'Scenario phase {index} duration must be finite and positive.')
        steps = Decimal(str(duration)) / dt_decimal
        if steps != steps.to_integral_value():
            raise ValueError(
                f'Scenario phase {index} duration must be an exact multiple of dt_hours.')
        phase_steps.append(int(steps))
    if not phase_steps:
        raise ValueError('Scenario must contain at least one phase.')
    step_count = sum(phase_steps)
    start_utc = start.astimezone(timezone.utc)
    return tuple(
        (start_utc + timedelta(seconds=index * dt_hours * 3600)).astimezone(start.tzinfo)
        for index in range(step_count)
    )


def _paths(values, name):
    """Normalize one explicit path or path sequence without an implicit base directory."""
    if isinstance(values, (str, os.PathLike)):
        values = (values,)
    else:
        try:
            values = tuple(values)
        except TypeError as error:
            raise ValueError(f'{name} must be a path or a nonempty sequence of paths.') from error
    if not values:
        raise ValueError(f'{name} must contain at least one path.')
    paths = tuple(Path(value).expanduser().resolve() for value in values)
    missing = next((path for path in paths if not path.is_file()), None)
    if missing is not None:
        raise ValueError(f'{name} is not a file: {missing}')
    return paths


def _timestamp(value, name):
    """Parse one source timestamp and reject timezone-naive source records."""
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} is not a valid timestamp.') from error
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError(f'{name} must include a UTC offset.')
    return stamp.to_pydatetime()


def _read_csv_series(paths, times, dt_hours, sample_hours, value_column, default_semantics,
                     name, *, nonnegative, reference_rating_w=None):
    """Read and conservatively resample one complete piecewise-constant source."""
    if not math.isfinite(sample_hours) or sample_hours <= 0:
        raise ValueError(f'{name} sample duration must be finite and positive.')
    source_paths = _paths(paths, name + ' files')
    frames = []
    semantics = []
    for path in source_paths:
        try:
            frame = pd.read_csv(path)
        except Exception as error:
            raise ValueError(f'Cannot read {name} CSV: {path}') from error
        if frame.empty or 'timestamp' not in frame or value_column not in frame:
            raise ValueError(
                f"{name} CSV must contain nonempty 'timestamp' and {value_column!r} columns.")
        if 'semantics' in frame:
            labels = frame['semantics'].dropna().astype(str).unique()
            if len(labels) != 1 or len(frame['semantics'].dropna()) != len(frame):
                raise ValueError(name + ' CSV must declare one source-time semantics value.')
            semantics.append(labels[0])
        else:
            semantics.append(default_semantics)
        frames.append(frame)
    if len(set(semantics)) != 1 or semantics[0] not in _SUPPORTED_SEMANTICS:
        raise ValueError(name + ' CSV files have inconsistent or unsupported source-time semantics.')
    semantics = semantics[0]
    frame = pd.concat(frames, ignore_index=True)
    stamps = tuple(
        _timestamp(value, f'{name} timestamp row {row + 2}')
        for row, value in enumerate(frame['timestamp'])
    )
    instants = np.asarray([stamp.timestamp() for stamp in stamps], dtype=float)
    if len(set(instants)) != len(instants):
        raise ValueError(name + ' CSV contains duplicate or overlapping timestamps.')
    if np.any(np.diff(instants) <= 0):
        raise ValueError(name + ' CSV timestamps must be strictly increasing across declared files.')
    sample_seconds = sample_hours * 3600
    spacing = np.diff(instants)
    if np.any(spacing < sample_seconds - 1e-6):
        raise ValueError(name + ' CSV intervals overlap.')
    if np.any(spacing > sample_seconds + 1e-6):
        raise ValueError(name + ' CSV has a gap inconsistent with its sample duration.')
    try:
        values = pd.to_numeric(frame[value_column], errors='raise').to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(name + ' values must be numeric.') from error
    if not np.all(np.isfinite(values)) or (nonnegative and np.any(values < 0)):
        qualifier = 'finite and nonnegative' if nonnegative else 'finite'
        raise ValueError(f'{name} values must be {qualifier}.')
    if 'sample_duration_hours' in frame:
        recorded = pd.to_numeric(
            frame['sample_duration_hours'], errors='coerce').to_numpy(dtype=float)
        if (not np.all(np.isfinite(recorded)) or
                not np.allclose(recorded, sample_hours, rtol=0, atol=_ROUNDOFF)):
            raise ValueError(name + ' CSV sample_duration_hours disagrees with the scenario.')
    if reference_rating_w is not None:
        if not math.isfinite(reference_rating_w) or reference_rating_w <= 0:
            raise ValueError('Scenario PV reference rating must be finite and positive.')
        if 'reference_rating_w' in frame:
            recorded = pd.to_numeric(
                frame['reference_rating_w'], errors='coerce').to_numpy(dtype=float)
            if (not np.all(np.isfinite(recorded)) or
                    not np.allclose(recorded, reference_rating_w, rtol=0, atol=_ROUNDOFF)):
                raise ValueError('PV CSV reference_rating_w disagrees with the scenario.')

    starts = instants - sample_seconds if semantics == 'interval_end_mean' else instants
    scenario_start = times[0].timestamp()
    scenario_end = times[-1].timestamp() + dt_hours * 3600
    source_end = starts[-1] + sample_seconds
    if scenario_start < starts[0] or scenario_end > source_end:
        raise ValueError(name + ' CSV does not cover the complete scenario interval.')
    target_seconds = dt_hours * 3600
    result = np.empty(len(times), dtype=float)
    for target, target_start in enumerate(stamp.timestamp() for stamp in times):
        target_end = target_start + target_seconds
        cursor = target_start
        total = 0.0
        while cursor < target_end - 1e-9:
            index = int(np.searchsorted(starts, cursor, side='right') - 1)
            if index < 0 or cursor >= starts[index] + sample_seconds - 1e-9:
                raise ValueError(name + ' CSV has an uncovered scenario interval.')
            segment_end = min(target_end, starts[index] + sample_seconds)
            total += values[index] * (segment_end - cursor)
            cursor = segment_end
        result[target] = total / target_seconds
    return result, source_paths, semantics


def read_scenario_weather(scenario, dt_hours):
    """Load temperature and reference-PV trajectories for one declared scenario.

    Temperature defaults to causal sample-hold semantics. PV defaults to a
    preceding-interval mean. Synthetic CSVs may override either default through a
    uniform ``semantics`` column. No interpolation, extrapolation or gap filling is
    performed.
    """
    times = _scenario_times(scenario, dt_hours)
    air_c, temperature_paths, temperature_semantics = _read_csv_series(
        scenario.temperature_files, times, dt_hours, scenario.temperature_sample_hours,
        'air_temperature_c', 'sample_hold', 'Temperature', nonnegative=False)
    reference_pv_w, pv_paths, pv_semantics = _read_csv_series(
        scenario.pv_files, times, dt_hours, scenario.pv_sample_hours, 'pv_power_w',
        'interval_end_mean', 'PV', nonnegative=True,
        reference_rating_w=scenario.pv_reference_rating_w)
    return {
        'times': times,
        'air_c': air_c,
        'reference_pv_w': reference_pv_w,
        'source_files': tuple(dict.fromkeys((*temperature_paths, *pv_paths))),
        'sample_hours': {
            'temperature': float(scenario.temperature_sample_hours),
            'pv': float(scenario.pv_sample_hours),
        },
        'semantics': {'temperature': temperature_semantics, 'pv': pv_semantics},
    }


class WeatherDataProcessor:
    """
    A class for fetching, processing, and visualizing weather data using Open-Meteo API.
    
    Features:
    - Automatically geocodes place names to coordinates
    - Uses the Open-Meteo temperature as archived (no elevation correction)
    - Plots daily max/min/avg temperature curves for single location
    - Plots daily temperature for multiple locations in one chart (default: mean, configurable)
    - Saves results (CSV + PNG) in separate folders
    """

    def __init__(self):
        self.base_url = "https://archive-api.open-meteo.com/v1/era5"
        self.forecast_url = "https://api.open-meteo.com/v1/forecast"

        # PV system parameters
        self.pv_tilt = 30.0  # Panel tilt angle (degrees)
        self.pv_azimuth = 180.0  # Panel orientation (degrees, 180=south-facing)
        self.pv_kwp = 0.1  # Installed capacity (kW peak)
        self.pv_efficiency = 0.95  # System efficiency (MPPT, wiring, etc.)

    def _get_coordinates(self, location_name):
        """Fetch latitude and longitude using OpenStreetMap's Nominatim API."""
        geocoding_url = "https://nominatim.openstreetmap.org/search"
        params = {"q": location_name, "format": "json", "limit": 1}
        headers = {"User-Agent": "WeatherScraper"}

        response = requests.get(
            geocoding_url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                lat = float(data[0]["lat"])
                lon = float(data[0]["lon"])
                return lat, lon
            else:
                raise ValueError(f"Cannot find coordinates for location '{location_name}'.")
        raise requests.HTTPError(
            f"Geocoding HTTP {response.status_code}: {response.text}",
            response=response,
        )

    @staticmethod
    def _hourly_timestamps(data):
        """Return strict timezone-aware hourly timestamps from one API response."""
        if not isinstance(data.get("timezone"), str) or not data["timezone"]:
            raise ValueError("Weather response is missing its IANA timezone.")
        try:
            timestamps = pd.DatetimeIndex(pd.to_datetime(data["hourly"]["time"], errors="raise"))
            timestamps = (timestamps.tz_localize(
                data["timezone"], ambiguous="raise", nonexistent="raise")
                if timestamps.tz is None else timestamps.tz_convert(data["timezone"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Weather response contains invalid timestamps.") from error
        if timestamps.empty or timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
            raise ValueError("Weather timestamps must be nonempty, unique and increasing.")
        if len(timestamps) > 1 and not np.all(np.diff(timestamps.asi8) == 3_600_000_000_000):
            raise ValueError("Weather response must use an uninterrupted hourly grid.")
        return timestamps

    def _temperature_frame(self, data, location):
        """Preserve instantaneous temperature samples with an explicit forward hold."""
        if data.get("hourly_units", {}).get("temperature_2m") != "°C":
            raise ValueError("Open-Meteo temperature_2m must be reported in °C.")
        timestamps = self._hourly_timestamps(data)
        try:
            temperature = np.asarray(data["hourly"]["temperature_2m"], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Temperature response is incomplete.") from error
        if temperature.shape != (len(timestamps),) or not np.all(np.isfinite(temperature)):
            raise ValueError("Temperature response contains missing or non-finite values.")
        source = pd.DataFrame({"temperature_2m": temperature})
        # Use the archived Open-Meteo temperature directly; no elevation correction.
        air = source["temperature_2m"].to_numpy(dtype=float)
        if not np.all(np.isfinite(air)):
            raise ValueError("Air temperature contains non-finite values.")
        return pd.DataFrame({
            "timestamp": timestamps,
            "sample_duration_hours": np.ones(len(timestamps)),
            "semantics": ["sample_hold"] * len(timestamps),
            "air_temperature_c": air,
        })

    def _pv_frame(self, data):
        """Preserve preceding-hour irradiance/PV means and their reference rating."""
        expected_units = {
            "pv_power_output": "W",
            "global_tilted_irradiance": "W/m²",
            "shortwave_radiation": "W/m²",
            "direct_radiation": "W/m²",
            "diffuse_radiation": "W/m²",
        }
        units = data.get("hourly_units", {})
        if any(units.get(name) != unit for name, unit in expected_units.items()):
            raise ValueError("Open-Meteo PV response has unexpected or missing units.")
        timestamps = self._hourly_timestamps(data)
        source_names = (
            "pv_power_output", "global_tilted_irradiance", "shortwave_radiation",
            "direct_radiation", "diffuse_radiation")
        try:
            arrays = {name: np.asarray(data["hourly"][name], dtype=float)
                      for name in source_names}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("PV response is incomplete.") from error
        if any(values.shape != (len(timestamps),) or not np.all(np.isfinite(values)) or
               np.any(values < 0) for values in arrays.values()):
            raise ValueError("PV response contains missing, non-finite or negative values.")
        rating = data.get("_pv_reference_rating_w")
        if not isinstance(rating, (int, float)) or not np.isfinite(rating) or rating <= 0:
            raise ValueError("PV response is missing its positive reference rating.")
        return pd.DataFrame({
            "timestamp": timestamps,
            "sample_duration_hours": np.ones(len(timestamps)),
            "semantics": ["interval_end_mean"] * len(timestamps),
            "reference_rating_w": np.full(len(timestamps), float(rating)),
            "pv_power_w": arrays["pv_power_output"],
            "global_tilted_irradiance_w_m2": arrays["global_tilted_irradiance"],
            "shortwave_radiation_w_m2": arrays["shortwave_radiation"],
            "direct_radiation_w_m2": arrays["direct_radiation"],
            "diffuse_radiation_w_m2": arrays["diffuse_radiation"],
        })

    @staticmethod
    def _write_daily_csv(frame, location, csv_dir):
        """Split timestamped rows by local calendar date without changing timestamps."""
        os.makedirs(csv_dir, exist_ok=True)
        paths = []
        dates = frame["timestamp"].dt.date
        for date_value in dates.drop_duplicates():
            output = frame.loc[dates == date_value].copy()
            output["timestamp"] = output["timestamp"].map(lambda value: value.isoformat())
            path = os.path.join(csv_dir, f"{location}_{date_value.isoformat()}.csv")
            output.to_csv(path, index=False)
            paths.append(path)
        return paths

    def _cached_temperature_frame(self, location, start_date, end_date, csv_dir):
        """Read existing dated CSVs; fetch only contiguous runs of missing dates."""
        dates = pd.date_range(start_date, end_date, freq='D')
        if dates.empty:
            raise ValueError("start_date must not be after end_date")
        frames = {}
        missing = []
        for day in dates:
            key = day.date()
            path = Path(csv_dir) / f"{location}_{key.isoformat()}.csv"
            if path.exists():
                frame = pd.read_csv(path)
                # Preserve local calendar dates, including UTC-offset changes.
                frame["timestamp"] = frame["timestamp"].map(
                    lambda value: _timestamp(value, str(path)))
                frames[key] = frame
            else:
                missing.append(day)
        if missing:
            lat, lon = self._get_coordinates(location)
            groups = []
            for day in missing:
                if not groups or day != groups[-1][-1] + pd.Timedelta(days=1):
                    groups.append([])
                groups[-1].append(day)
            for group in groups:
                response = requests.get(self.base_url, params={
                    "latitude": lat, "longitude": lon,
                    "start_date": group[0].date().isoformat(),
                    "end_date": group[-1].date().isoformat(),
                    "hourly": "temperature_2m", "timezone": "auto",
                }, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                frame = self._temperature_frame(response.json(), location)
                local_dates = frame["timestamp"].dt.date
                requested = {day.date() for day in group}
                frame = frame.loc[local_dates.isin(requested)]
                if set(frame["timestamp"].dt.date) != requested:
                    raise ValueError("Weather response is missing requested dates.")
                for path in self._write_daily_csv(frame, location, csv_dir):
                    print(f"Data saved to: {path}")
                for key, daily in frame.groupby(frame["timestamp"].dt.date):
                    frames[key] = daily
                time.sleep(SLEEP_TIME)
        return pd.concat([frames[day.date()] for day in dates], ignore_index=True)

    def scrape_and_save(
        self, location, start_date, end_date,
        csv_dir="data/weather/temperature",
        img_dir="docs/figures/weather/temperature"
    ):
        """Read cached temperature samples, fetching only missing dated CSVs.

        ``start_date`` and ``end_date`` are inclusive YYYY-MM-DD API dates. Each
        ``location_YYYY-MM-DD.csv`` retains timezone-aware timestamps, degrees C,
        one-hour sample duration and explicit ``sample_hold`` semantics.
        """
        try:
            frame = self._cached_temperature_frame(
                location, start_date, end_date, csv_dir)
            daily = frame.assign(date=frame["timestamp"].map(lambda value: value.date())).groupby(
                "date")["air_temperature_c"].agg(["min", "max", "mean"]).reset_index()
            os.makedirs(img_dir, exist_ok=True)
            img_path = os.path.join(img_dir, f"{location}_temperature.png")
            _save_temperature_plot([
                (daily["date"], daily["max"], "Daily Max (°C)", "red"),
                (daily["date"], daily["min"], "Daily Min (°C)", "blue"),
                (daily["date"], daily["mean"], "Daily Avg (°C)", "gold"),
            ], f"Temperature of {location} ({start_date} - {end_date})", img_path, title_size=12)
            print(f"✅ Chart saved to: {img_path}")
        except Exception as e:
            print(f"❌ Error: {e}")

    def scrape_multiple_locations(
        self, locations, start_date, end_date,
        csv_dir="data/weather/temperature",
        img_dir="docs/figures/weather/temperature",
        temp_type="mean"
    ):
        """Read cached dated temperatures for each location, fetching missing days.

        ``temp_type`` selects only the daily comparison plot; every CSV retains the
        full hourly temperature samples in degrees C.
        """
        try:
            if temp_type not in ["mean", "min", "max"]:
                raise ValueError("temp_type must be 'mean', 'min', or 'max'")
            all_daily_stats = {}
            for location in locations:
                frame = self._cached_temperature_frame(
                    location, start_date, end_date, csv_dir)
                daily = frame.assign(date=frame["timestamp"].map(lambda value: value.date())).groupby(
                    "date")["air_temperature_c"].agg(["min", "max", "mean"]).reset_index()
                all_daily_stats[location] = daily

            if all_daily_stats:
                os.makedirs(img_dir, exist_ok=True)
                img_path = os.path.join(img_dir, f"multiple_locations_{temp_type}_temperature.png")
                display = {"mean": "Average", "min": "Minimum", "max": "Maximum"}
                colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
                series = [
                    (daily["date"], daily[temp_type], location,
                     colors[index % len(colors)])
                    for index, (location, daily) in enumerate(all_daily_stats.items())
                ]
                _save_temperature_plot(
                    series, f"{display[temp_type]} Temperature ({start_date} - {end_date})",
                    img_path)
                print(f"✅ Multi-location chart saved to: {img_path}")
            else:
                print("❌ No valid data to plot.")
        except Exception as e:
            print(f"❌ Error: {e}")

    # =============================================================================
    # PV (Photovoltaic) Data Fetching Methods
    # =============================================================================

    def _fetch_pv_from_api(
        self,
        lat,
        lon,
        start_date,
        end_date,
        tilt=None,
        azimuth=None,
        kwp=None,
        api_url=None,
        timezone_name="auto"
    ):
        """Fetch irradiance means and estimate PV output for one reference array.

        ``tilt`` and compass ``azimuth`` are degrees; ``kwp`` is the positive rated
        capacity in kWp. Open-Meteo irradiance is W/m² averaged over the preceding
        hour. The returned electrical W already includes ``pv_efficiency`` once.
        """
        tilt = self.pv_tilt if tilt is None else tilt
        azimuth = self.pv_azimuth if azimuth is None else azimuth
        kwp = self.pv_kwp if kwp is None else kwp
        if (not all(not isinstance(value, bool) and isinstance(value, (int, float)) and
                    np.isfinite(value) for value in (tilt, azimuth, kwp, self.pv_efficiency)) or
                kwp <= 0 or not 0 <= self.pv_efficiency <= 1):
            raise ValueError("PV tilt, azimuth, positive rating and efficiency must be finite.")
        api_azimuth = (azimuth % 360) - 180
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "global_tilted_irradiance,shortwave_radiation,direct_radiation,diffuse_radiation",
            "timezone": timezone_name,
            "tilt": tilt,
            "azimuth": api_azimuth
        }
        response = requests.get(
            api_url or self.base_url, params=params, timeout=REQUEST_TIMEOUT)
        if not response.ok:
            raise requests.HTTPError(
                f"Open-Meteo HTTP {response.status_code}: {response.text}",
                response=response
            )
        data = response.json()
        try:
            irradiance = np.asarray(data["hourly"]["global_tilted_irradiance"], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("PV irradiance response is incomplete.") from error
        if not np.all(np.isfinite(irradiance)):
            raise ValueError("PV irradiance contains missing or non-finite values.")
        # P(W) = G(W/m²) / 1000 * P_rated(W) * efficiency; kwp makes 1000 cancel.
        data["hourly"]["pv_power_output"] = (irradiance * kwp * self.pv_efficiency).tolist()
        data.setdefault("hourly_units", {})["pv_power_output"] = "W"
        data["_pv_reference_rating_w"] = float(kwp) * 1000.0
        return data

    def scrape_and_save_pv(
        self,
        location,
        start_date,
        end_date,
        csv_dir="data/weather/pv",
        img_dir="docs/figures/weather/pv",
        tilt=None,
        azimuth=None,
        kwp=None
    ):
        """Write dated preceding-hour PV means and their explicit W reference rating.

        Files retain source timestamps, a one-hour duration and ``interval_end_mean``
        semantics. Downstream panel scaling must not apply system efficiency again.
        """
        try:
            print(f"Fetching coordinates for '{location}' ...")
            lat, lon = self._get_coordinates(location)
            data = self._fetch_pv_from_api(lat, lon, start_date, end_date, tilt, azimuth, kwp)
            frame = self._pv_frame(data)
            for path in self._write_daily_csv(frame, location, csv_dir):
                print(f"✅ PV data saved to: {path}")
            daily = frame.assign(date=frame["timestamp"].dt.date).groupby(
                "date")["pv_power_w"].max().reset_index()
            os.makedirs(img_dir, exist_ok=True)
            plt.figure(figsize=(14, 7))
            plt.plot(daily["date"], daily["pv_power_w"], marker='o',
                     color="orange", label="Daily Max PV Power (W)")
            plt.fill_between(daily["date"], 0, daily["pv_power_w"], alpha=0.3, color="orange")
            plt.title(f"PV Power Output - {location} ({start_date} - {end_date})", fontsize=12)
            plt.xlabel("Date")
            plt.ylabel("PV Power (Watts)")
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.legend()
            plt.tight_layout()
            img_path = os.path.join(img_dir, f"{location}_pv.png")
            plt.savefig(img_path, dpi=200)
            plt.close()
            print(f"✅ PV chart saved to: {img_path}")
            time.sleep(SLEEP_TIME)
        except Exception as e:
            print(f"❌ Error: {e}")

    def scrape_multiple_locations_pv(
        self,
        locations,
        start_date,
        end_date,
        csv_dir="data/weather/pv",
        img_dir="docs/figures/weather/pv",
        tilt=None,
        azimuth=None,
        kwp=None
    ):
        """Write the same dated PV/reference-rating contract for every location."""
        try:
            all_daily_data = {}
            for location in locations:
                print(f"Fetching PV data for '{location}' ...")
                lat, lon = self._get_coordinates(location)
                data = self._fetch_pv_from_api(lat, lon, start_date, end_date,
                                               tilt, azimuth, kwp)
                frame = self._pv_frame(data)
                daily = frame.assign(date=frame["timestamp"].dt.date).groupby(
                    "date")["pv_power_w"].max().reset_index()
                all_daily_data[location] = daily
                for path in self._write_daily_csv(frame, location, csv_dir):
                    print(f"✅ PV data saved to: {path}")
                time.sleep(SLEEP_TIME)

            if all_daily_data:
                os.makedirs(img_dir, exist_ok=True)
                img_path = os.path.join(img_dir, "multiple_locations_pv_comparison.png")
                plt.figure(figsize=(14, 7))
                colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
                for index, (location, daily) in enumerate(all_daily_data.items()):
                    plt.plot(daily["date"], daily["pv_power_w"], marker='o',
                             color=colors[index % len(colors)], label=location)
                plt.title(f"PV Power Comparison ({start_date} - {end_date})", fontsize=12)
                plt.xlabel("Date")
                plt.ylabel("Daily Max PV Power (Watts)")
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.legend()
                plt.tight_layout()
                plt.savefig(img_path, dpi=200)
                plt.close()
                print(f"✅ Multi-location PV chart saved to: {img_path}")
            else:
                print("❌ No valid data to plot.")
        except Exception as e:
            print(f"❌ Error: {e}")


def _single_scenario_path(value, label):
    """Return the one dated source file used by the command-line scenarios."""
    if isinstance(value, (str, Path)):
        return Path(value).expanduser().resolve()
    paths = [Path(item).expanduser().resolve() for item in value]
    if len(paths) != 1:
        raise ValueError(f"Scenario {label} must identify exactly one dated file.")
    return paths[0]


def _write_scenario_frame(frame, path, requested_date):
    """Write one API frame only when every row belongs to the requested local date."""
    dates = frame["timestamp"].dt.date.drop_duplicates().tolist()
    if dates != [requested_date]:
        raise ValueError(
            f"Weather API returned dates {dates}, expected only {requested_date}.")
    output = frame.copy()
    output["timestamp"] = output["timestamp"].map(lambda value: value.isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)


def fetch_scenario_weather(scenario, progress=None):
    """Fetch and overwrite the CLI scenario's temperature and PV files.

    Recent and future requests use the forecast endpoint; older requests use the
    ERA5 archive endpoint. API and geocoding failures propagate to the CLI.
    """
    if progress is not None and not callable(progress):
        raise TypeError("Weather progress reporter must be callable.")
    report = progress or (lambda message: None)
    requested_date = scenario.start.date()
    date_text = requested_date.isoformat()
    processor = WeatherDataProcessor()
    today = datetime.now(scenario.start.tzinfo).date()
    api_url = (processor.forecast_url
               if requested_date >= today - timedelta(days=5)
               else processor.base_url)
    report(f"[weather 1/3] Geocoding {scenario.weather_location}")
    latitude, longitude = processor._get_coordinates(scenario.weather_location)

    report(f"[weather 2/3] Fetching temperature for {date_text}")
    temperature_response = requests.get(api_url, params={
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date_text,
        "end_date": date_text,
        "hourly": "temperature_2m",
        "timezone": scenario.timezone,
    }, timeout=REQUEST_TIMEOUT)
    if not temperature_response.ok:
        raise requests.HTTPError(
            f"Open-Meteo HTTP {temperature_response.status_code}: "
            f"{temperature_response.text}",
            response=temperature_response,
        )
    temperature = processor._temperature_frame(
        temperature_response.json(), scenario.weather_location)
    report(f"[weather 3/3] Fetching PV for {date_text}")
    pv = processor._pv_frame(processor._fetch_pv_from_api(
        latitude,
        longitude,
        date_text,
        date_text,
        kwp=scenario.pv_reference_rating_w / 1000.0,
        api_url=api_url,
        timezone_name=scenario.timezone,
    ))

    temperature_path = _single_scenario_path(
        scenario.temperature_files, "temperature_files")
    pv_path = _single_scenario_path(scenario.pv_files, "pv_files")
    _write_scenario_frame(temperature, temperature_path, requested_date)
    _write_scenario_frame(pv, pv_path, requested_date)
    report(f"[weather] Saved {temperature_path.name} and {pv_path.name}")
    return temperature_path, pv_path


if __name__ == "__main__":
    processor = WeatherDataProcessor()

    # =============================================================================
    # Temperature Data Examples
    # =============================================================================

    # Single location examples
    print("=== Processing Single Locations (Temperature) ===")
    location = "Harbin"
    start_date = "2025-12-01"
    end_date = "2026-02-28"
    processor.scrape_and_save(
        location,
        start_date,
        end_date
    )

    location = "Muztagh Ata"
    start_date = "2026-06-01"
    end_date = "2026-08-31"
    processor.scrape_and_save(
        location,
        start_date,
        end_date
    )

    # Multiple locations example
    print("\n=== Processing Multiple Locations (Minimum Temperature) ===")
    locations = [
        "Great Wall Station",
        "Zhongshan Station",
        "Taishan Station",
        "Kunlun Station",
        "Qinling Station"
    ]
    start_date = "2025-11-01"
    end_date = "2026-03-31"
    processor.scrape_multiple_locations(
        locations,
        start_date,
        end_date,
        temp_type="min"
    )

    # =============================================================================
    # PV Data Examples (uncomment to use)
    # =============================================================================

    # print("\n=== Processing Single Location (PV Data) ===")
    # processor.scrape_and_save_pv(
    #     location="Harbin",
    #     start_date="2025-12-01",
    #     end_date="2025-12-31",
    #     tilt=45.0,  # Higher tilt for high-latitude regions
    #     kwp=0.1  # 100W panel
    # )
