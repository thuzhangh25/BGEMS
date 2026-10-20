'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Convert the three fixed system-configuration JSON files into the five
parameter objects shared by the BGEMS pipeline. Weather loading remains in
weather.py.
'''
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Sequence
from zoneinfo import ZoneInfo

from .modeling import weighted_skin_temperature


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEMCONF_DIR = PROJECT_ROOT / 'data' / 'systemconf'
WEATHER_DIR = PROJECT_ROOT / 'data' / 'weather'
REGION_ORDER = (
    'left_chest', 'right_chest', 'back', 'pelvis',
    'left_upper_arm', 'right_upper_arm', 'left_lower_arm', 'right_lower_arm',
    'left_thigh', 'right_thigh', 'left_lower_leg', 'right_lower_leg',
)
REGION_GROUPS = (
    ('back', 'pelvis', 'left_chest', 'right_chest'),
    ('left_lower_arm', 'right_lower_arm', 'left_lower_leg', 'right_lower_leg'),
    ('left_upper_arm', 'right_upper_arm', 'left_thigh', 'right_thigh'),
)


def _number(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite.')
    value = float(value)
    if positive and value <= 0:
        raise ValueError(f'{name} must be positive.')
    if nonnegative and value < 0:
        raise ValueError(f'{name} must be nonnegative.')
    return value


def _optional_number(value, name, **bounds):
    return None if value is None else _number(value, name, **bounds)


def _integer(value, name, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f'{name} must be an integer.')
    value = int(value)
    if nonnegative and value < 0:
        raise ValueError(f'{name} must be nonnegative.')
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{name} must be a nonempty string.')
    return value


def _boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError(f'{name} must be boolean.')
    return value


def _array(value, name):
    if not isinstance(value, list):
        raise ValueError(f'{name} must be a JSON array.')
    return value


def _record(value, keys, name):
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be a JSON object.')
    expected, actual = set(keys), set(value)
    if actual != expected:
        details = []
        if expected - actual:
            details.append(f'missing {sorted(expected - actual)}')
        if actual - expected:
            details.append(f'unknown {sorted(actual - expected)}')
        raise ValueError(f'{name} has invalid keys: {"; ".join(details)}.')
    return value


def _regional(values, name):
    values = _array(values, name)
    if len(values) != 12:
        raise ValueError(f'{name} must contain twelve values.')
    return [_number(value, name) for value in values]


def _json(path, name):
    path = Path(path)
    resolved = path if path.is_absolute() else (
        SYSTEMCONF_DIR / path if path.parent == Path('.') else PROJECT_ROOT / path)
    try:
        value = json.loads(resolved.read_text(encoding='utf-8'))
    except FileNotFoundError as error:
        raise ValueError(f'{name} file does not exist: {resolved}') from error
    except json.JSONDecodeError as error:
        raise ValueError(f'{name} is not valid JSON: {resolved}: {error.msg}') from error
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be a JSON object.')
    return value, resolved.resolve()


def _date(value):
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.strptime(value, '%Y%m%d').date()
        except ValueError as error:
            raise ValueError('run_date must use YYYYMMDD.') from error
    raise ValueError('run_date must be a date or YYYYMMDD string.')


@dataclass
class Person:
    subject_id: str
    height_cm: float
    weight_kg: float
    age_years: int
    sex: str
    weights: list[float]
    skin_temperatures_c: list[float]
    area_allocation_proxies: list[float]
    region_order: tuple[str, ...] = REGION_ORDER

    def __post_init__(self):
        self.subject_id = _text(self.subject_id, 'Person subject_id')
        self.height_cm = _number(self.height_cm, 'Person height_cm', positive=True)
        self.weight_kg = _number(self.weight_kg, 'Person weight_kg', positive=True)
        self.age_years = _integer(self.age_years, 'Person age_years', nonnegative=True)
        self.sex = _text(self.sex, 'Person sex')
        self.region_order = tuple(self.region_order)
        if self.region_order != REGION_ORDER:
            raise ValueError('Person region_order must follow the adopted configuration order.')
        self.weights = _regional(list(self.weights), 'Person weights')
        self.skin_temperatures_c = _regional(
            list(self.skin_temperatures_c), 'Person skin temperatures')
        self.area_allocation_proxies = _regional(
            list(self.area_allocation_proxies), 'Person area allocation proxies')
        if not math.isclose(sum(self.weights), 1.0, abs_tol=1e-12):
            raise ValueError('Person weights must sum to one.')
        if any(value <= 0 for value in self.area_allocation_proxies):
            raise ValueError('Person area allocation proxies must be positive.')

    @property
    def body_area_m2(self):
        height_m = _number(self.height_cm, 'Person height_cm', positive=True) / 100
        weight_kg = _number(self.weight_kg, 'Person weight_kg', positive=True)
        return 0.202 * weight_kg ** 0.425 * height_m ** 0.725

    @property
    def regional_areas_m2(self):
        total = sum(self.area_allocation_proxies)
        return [self.body_area_m2 * value / total
                for value in self.area_allocation_proxies]

    @property
    def skin_setpoint_c(self):
        return weighted_skin_temperature(self.skin_temperatures_c, self.weights)


@dataclass
class Garment:
    id: int
    name: str
    clothing_resistance: float
    heating_pad_locations: tuple[str, ...]
    mask: list[int]
    mass_kg: float
    cost: float
    mass_and_cost_include_heating_pads: bool

    def __post_init__(self):
        self.id = _integer(self.id, 'Garment id', nonnegative=True)
        self.name = _text(self.name, 'Garment name')
        self.clothing_resistance = _number(
            self.clothing_resistance, 'Garment clothing_resistance', nonnegative=True)
        self.mass_kg = _number(self.mass_kg, 'Garment mass_kg', nonnegative=True)
        self.cost = _number(self.cost, 'Garment cost', nonnegative=True)
        self.mass_and_cost_include_heating_pads = _boolean(
            self.mass_and_cost_include_heating_pads,
            'Garment mass_and_cost_include_heating_pads')
        self.heating_pad_locations = tuple(self.heating_pad_locations)
        if (len(set(self.heating_pad_locations)) != len(self.heating_pad_locations) or
                any(item not in REGION_ORDER for item in self.heating_pad_locations)):
            raise ValueError('Garment heating_pad_locations must be unique configured regions.')
        self.mask = _regional(list(self.mask), 'Garment mask')
        if any(value not in (0, 1) for value in self.mask):
            raise ValueError('Garment mask must be binary.')
        self.mask = [int(value) for value in self.mask]


@dataclass
class Device:
    battery_unit_capacity_wh: float
    battery_unit_mass_kg: float
    battery_unit_cost: float
    pv_rating_w: float
    pv_mass_kg: float
    pv_cost: float
    base_mass_kg: float
    base_cost: float
    base_power_w: float
    rated_powers_w: list[float]
    efficiencies: list[float]
    equipment_power_w: float
    equipment_mass_kg: float
    equipment_cost: float
    equipment_available: bool
    voltage_v: float
    # Electrical limits of the standardized battery module.
    max_discharge_w: float
    max_charge_w: float
    bus_limit_w: float

    def __post_init__(self):
        positive = ('battery_unit_capacity_wh', 'battery_unit_mass_kg', 'pv_rating_w', 'pv_mass_kg',
                    'base_mass_kg', 'base_power_w', 'equipment_power_w',
                    'equipment_mass_kg')
        nonnegative = ('battery_unit_cost', 'pv_cost', 'base_cost', 'equipment_cost')
        nullable = ('voltage_v', 'max_discharge_w', 'max_charge_w', 'bus_limit_w')
        for name in positive:
            setattr(self, name, _number(getattr(self, name), f'Device {name}', positive=True))
        for name in nonnegative:
            setattr(self, name, _number(getattr(self, name), f'Device {name}', nonnegative=True))
        for name in nullable:
            setattr(self, name, _number(
                getattr(self, name), f'Device {name}', positive=True))
        self.rated_powers_w = _regional(list(self.rated_powers_w), 'Device rated_powers_w')
        self.efficiencies = _regional(list(self.efficiencies), 'Device efficiencies')
        if any(value <= 0 for value in self.rated_powers_w):
            raise ValueError('Device rated_powers_w must be positive.')
        if any(not 0 <= value <= 1 for value in self.efficiencies):
            raise ValueError('Device efficiencies must lie in [0, 1].')
        self.equipment_available = _boolean(
            self.equipment_available, 'Device equipment_available')


@dataclass
class ScenarioProfile:
    scenario_type: int
    name: str
    category: str
    weather_location: str
    timezone: str
    start: datetime
    phases: list[tuple]
    air_resistance: float
    initial_state: dict
    temperature_files: str | Path | Sequence[str | Path]
    pv_files: str | Path | Sequence[str | Path]
    temperature_sample_hours: float
    pv_sample_hours: float
    pv_reference_rating_w: float
    backpack_included: bool
    equipment_heater_installed: bool

    def __post_init__(self):
        self.scenario_type = _integer(self.scenario_type, 'Scenario scenario_type', nonnegative=True)
        for name in ('name', 'category', 'weather_location', 'timezone'):
            setattr(self, name, _text(getattr(self, name), f'Scenario {name}'))
        if (not isinstance(self.start, datetime) or self.start.tzinfo is None or
                self.start.utcoffset() is None):
            raise ValueError('Scenario start must be a timezone-aware datetime.')
        self.air_resistance = _number(self.air_resistance, 'Scenario air_resistance', positive=True)
        self.temperature_sample_hours = _number(
            self.temperature_sample_hours, 'Scenario temperature_sample_hours', positive=True)
        self.pv_sample_hours = _number(
            self.pv_sample_hours, 'Scenario pv_sample_hours', positive=True)
        self.pv_reference_rating_w = _number(
            self.pv_reference_rating_w, 'Scenario pv_reference_rating_w', positive=True)
        self.backpack_included = _boolean(self.backpack_included, 'Scenario backpack_included')
        self.equipment_heater_installed = _boolean(
            self.equipment_heater_installed, 'Scenario equipment_heater_installed')
        if self.equipment_heater_installed and not self.backpack_included:
            raise ValueError('Equipment heating requires the scenario backpack.')
        self.initial_state = dict(self.initial_state)
        if not isinstance(self.temperature_files, (str, Path)):
            self.temperature_files = list(self.temperature_files)
        if not isinstance(self.pv_files, (str, Path)):
            self.pv_files = list(self.pv_files)
        self.set_phases(self.phases)

    def set_phases(self, phases):
        prepared = []
        for index, phase in enumerate(phases):
            if not isinstance(phase, (list, tuple)) or len(phase) != 3:
                raise ValueError(
                    f'Scenario phase {index} must be (label, duration_hours, met).')
            label, duration, met = phase
            label = _text(label, f'Scenario phase {index} label')
            duration = _number(duration, f'Scenario phase {index} duration', positive=True)
            met = _number(met, f'Scenario phase {index} MET', nonnegative=True)
            prepared.append((label, duration, met))
        self.phases = prepared

    def get_duration_hours(self):
        return float(sum((Decimal(str(phase[1])) for phase in self.phases), Decimal(0)))

    def _phase_at(self, elapsed_hours):
        elapsed = Decimal(str(_number(
            elapsed_hours, 'Scenario elapsed_hours', nonnegative=True)))
        # Decimal arithmetic preserves configured activity boundaries (e.g. 2.8 h).
        end = Decimal(0)
        for phase in self.phases:
            end += Decimal(str(phase[1]))
            if elapsed < end:
                return phase
        raise ValueError('Scenario elapsed_hours lies outside the half-open scenario interval.')

    def get_phase_label(self, elapsed_hours):
        return self._phase_at(elapsed_hours)[0]

    def get_activity(self, elapsed_hours):
        return self._phase_at(elapsed_hours)[2]

    def get_phases(self):
        return tuple(self.phases)


@dataclass
class SystemConfig:
    scenario: ScenarioProfile
    person: Person
    device: Device
    garments: list[Garment]
    settings: dict
    evaluation: dict
    max_battery_units: int
    max_pv_units: int
    preference: str
    endurance_multiplier: float
    controllers: dict
    monte_carlo: dict
    source_files: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.set_scenario(self.scenario)
        self.set_person(self.person)
        self.set_device(self.device)
        self.garments = list(self.garments)
        if not self.garments or any(not isinstance(value, Garment) for value in self.garments):
            raise ValueError('System garments must contain Garment objects.')
        if len({garment.id for garment in self.garments}) != len(self.garments):
            raise ValueError('System garment IDs must be distinct.')
        self.settings, self.evaluation = dict(self.settings), dict(self.evaluation)
        self.max_battery_units = _integer(
            self.max_battery_units, 'System max_battery_units', nonnegative=True)
        self.max_pv_units = _integer(self.max_pv_units, 'System max_pv_units', nonnegative=True)
        if self.preference not in ('lightweight', 'comfort', 'balanced'):
            raise ValueError('Unknown scenario selection preference.')
        self.endurance_multiplier = _number(
            self.endurance_multiplier, 'Endurance multiplier', positive=True)
        if self.endurance_multiplier < 1:
            raise ValueError('Endurance multiplier cannot shorten the design task.')
        self.controllers = dict(self.controllers)
        self.monte_carlo = dict(self.monte_carlo)
        self.source_files = dict(self.source_files)

    def get_person(self):
        return self.person

    def set_person(self, person):
        if not isinstance(person, Person):
            raise TypeError('person must be a Person.')
        self.person = person

    def get_device(self):
        return self.device

    def set_device(self, device):
        if not isinstance(device, Device):
            raise TypeError('device must be a Device.')
        self.device = device

    def get_scenario(self):
        return self.scenario

    def set_scenario(self, scenario):
        if not isinstance(scenario, ScenarioProfile):
            raise TypeError('scenario must be a ScenarioProfile.')
        self.scenario = scenario


def _component_objects(data, equipment_available):
    _record(data, {'system', 'battery_unit', 'photovoltaic_unit', 'regional_model',
                   'garments', 'heating_pads', 'pad_types', 'equipment_backpack',
                   'controllers', 'monte_carlo'},
            'components')
    system = _record(data['system'], {
        'base_electronics_mass_kg', 'base_electronics_cost_usd',
        'base_electronics_power_w', 'air_resistance_m2k_per_w',
        'max_battery_units', 'max_pv_units'}, 'components.system')
    battery = _record(data['battery_unit'], {
        'capacity_wh', 'mass_kg', 'cost_usd', 'voltage_v',
        'max_discharge_w', 'max_charge_w'},
        'components.battery_unit')
    pv = _record(data['photovoltaic_unit'],
                 {'rated_power_w', 'mass_kg', 'cost_usd'},
                 'components.photovoltaic_unit')

    pad_types = {}
    pad_type_keys = {'id', 'dimensions_cm', 'rated_power_w', 'resistance_ohm',
                     'voltage_v', 'cost_usd'}
    for index, item in enumerate(_array(data['pad_types'], 'components.pad_types')):
        item = _record(item, pad_type_keys, f'components.pad_types[{index}]')
        pad_id = _text(item['id'], f'components.pad_types[{index}].id')
        dimensions = _array(item['dimensions_cm'], f'components.pad_types[{index}].dimensions_cm')
        if pad_id in pad_types or len(dimensions) != 2:
            raise ValueError('Pad type IDs must be unique and dimensions_cm must have two values.')
        for value in dimensions:
            _number(value, 'Pad dimension', positive=True)
        for key in ('rated_power_w', 'resistance_ohm', 'voltage_v'):
            _number(item[key], f'components.pad_types[{index}].{key}', positive=True)
        _optional_number(item['cost_usd'], f'components.pad_types[{index}].cost_usd',
                         nonnegative=True)
        pad_types[pad_id] = item

    pads = {}
    pad_ids = set()
    pad_keys = {'id', 'location', 'zone', 'pad_type', 'rated_power_w', 'efficiency'}
    for index, item in enumerate(_array(data['heating_pads'], 'components.heating_pads')):
        item = _record(item, pad_keys, f'components.heating_pads[{index}]')
        pad_id = _integer(item['id'], f'components.heating_pads[{index}].id', nonnegative=True)
        location = _text(item['location'], f'components.heating_pads[{index}].location')
        pad_type = _text(item['pad_type'], f'components.heating_pads[{index}].pad_type')
        _text(item['zone'], f'components.heating_pads[{index}].zone')
        power = _number(item['rated_power_w'], 'Heating-pad rated_power_w', positive=True)
        efficiency = _number(item['efficiency'], 'Heating-pad efficiency')
        if (pad_id in pad_ids or location in pads or pad_type not in pad_types or
                not math.isclose(power, float(pad_types[pad_type]['rated_power_w'])) or
                not 0 <= efficiency <= 1):
            raise ValueError('Heating-pad IDs, locations, type ratings and efficiencies are inconsistent.')
        pad_ids.add(pad_id)
        pads[location] = item
    if set(pads) != set(REGION_ORDER):
        raise ValueError('components.heating_pads must define every region exactly once.')

    garment_keys = {'id', 'name', 'clothing_resistance_m2k_per_w',
                    'heating_pad_locations', 'mass_kg', 'cost_usd',
                    'mass_and_cost_include_heating_pads'}
    garments = []
    for index, item in enumerate(_array(data['garments'], 'components.garments')):
        item = _record(item, garment_keys, f'components.garments[{index}]')
        locations = tuple(_text(value, 'Garment heating-pad location') for value in
                          _array(item['heating_pad_locations'], 'Garment heating_pad_locations'))
        garments.append(Garment(
            item['id'], item['name'], item['clothing_resistance_m2k_per_w'], locations,
            [int(location in locations) for location in REGION_ORDER], item['mass_kg'],
            item['cost_usd'], item['mass_and_cost_include_heating_pads']))

    equipment = _record(data['equipment_backpack'], {
        'id', 'location', 'pad_type', 'rated_power_w', 'mass_kg', 'cost_usd',
        'bus_limit_w'}, 'components.equipment_backpack')
    equipment_type = _text(equipment['pad_type'], 'components.equipment_backpack.pad_type')
    _integer(equipment['id'], 'components.equipment_backpack.id', nonnegative=True)
    _text(equipment['location'], 'components.equipment_backpack.location')
    if (equipment_type not in pad_types or not math.isclose(
            float(equipment['rated_power_w']),
            float(pad_types[equipment_type]['rated_power_w']))):
        raise ValueError('Equipment heater type and rating are inconsistent.')

    device = Device(
        battery['capacity_wh'], battery['mass_kg'], battery['cost_usd'],
        pv['rated_power_w'], pv['mass_kg'], pv['cost_usd'],
        system['base_electronics_mass_kg'], system['base_electronics_cost_usd'],
        system['base_electronics_power_w'],
        [pads[name]['rated_power_w'] for name in REGION_ORDER],
        [pads[name]['efficiency'] for name in REGION_ORDER],
        equipment['rated_power_w'], equipment['mass_kg'], equipment['cost_usd'],
        equipment_available, battery['voltage_v'], battery['max_discharge_w'],
        battery['max_charge_w'], equipment['bus_limit_w'])
    return device, garments, system


def _scenario_parts(scenario, run_date, air_resistance):
    """Parse explicit mission timing independently of the prediction horizon."""
    location = _text(scenario['weather_location'], 'scenario.weather_location')
    zone = ZoneInfo(_text(scenario['timezone'], 'scenario.timezone'))
    local_time = datetime.strptime(scenario['local_start_time'], '%H:%M').time()
    initial = _record(scenario['initial_state'], {'soc'}, 'scenario.initial_state')
    initial_soc = _number(initial['soc'], 'Initial SOC', nonnegative=True)
    if initial_soc > 1:
        raise ValueError('Initial SOC must lie in [0, 1].')
    weather = _record(scenario['weather'], {
        'temperature_sample_hours', 'pv_sample_hours', 'pv_reference_rating_w',
        'temperature_filename_template', 'pv_filename_template'}, 'scenario.weather')
    tokens = {'weather_location': location.replace(' ', '_'),
              'date': run_date.strftime('%Y-%m-%d')}
    files = []
    for directory in ('temperature', 'pv'):
        filename = weather[f'{directory}_filename_template'].format(**tokens)
        if Path(filename).is_absolute() or Path(filename).name != filename:
            raise ValueError('Weather templates must produce a filename.')
        files.append(str(WEATHER_DIR / directory / filename))
    backpack = _record(scenario['backpack'],
                       {'included', 'equipment_heater_installed', 'max_pv_units'},
                       'scenario.backpack')
    included = _boolean(backpack['included'], 'Backpack included')
    installed = _boolean(backpack['equipment_heater_installed'], 'Equipment installed')
    max_pv = _integer(backpack['max_pv_units'], 'PV bound', nonnegative=True)
    if not included and (installed or max_pv):
        raise ValueError('Backpack equipment cannot be installed without the backpack.')
    search = _record(scenario['search'],
                     {'max_battery_units', 'max_pv_units', 'preference',
                      'endurance_multiplier'}, 'scenario.search')
    if search['max_pv_units'] != max_pv:
        raise ValueError('Search and backpack PV bounds must agree.')
    operation_keys = {'step_h', 'horizon_h', 'soc_min', 'soc_safe', 'lambda_deg',
                      'lambda_soc', 'solver_tolerance', 'constraint_tolerance',
                      'forecast_mode'}
    op = _record(scenario['operation'], operation_keys, 'scenario.operation')
    dt = _number(op['step_h'], 'Time step', positive=True)
    horizon = _number(op['horizon_h'], 'Prediction horizon', positive=True)
    ratio = horizon / dt
    if not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-10):
        raise ValueError('Prediction horizon must be a multiple of step_h.')
    settings = {'dt_hours': dt, 'horizon_steps': int(round(ratio))}
    for key in operation_keys - {'step_h', 'horizon_h', 'forecast_mode'}:
        settings[key] = _number(op[key], key, nonnegative=True)
    if not 0 <= settings['soc_min'] <= settings['soc_safe'] <= 1:
        raise ValueError('SOC thresholds must satisfy 0 <= min <= safe <= 1.')
    if initial_soc < settings['soc_min']:
        raise ValueError('Initial SOC lies below the hard floor.')
    if op['forecast_mode'] != 'persistence':
        raise ValueError('The approved forecast is causal persistence.')
    settings['forecast_mode'] = op['forecast_mode']
    phases, end = [], Decimal(0)
    for phase in _array(scenario['phases'], 'Scenario phases'):
        _record(phase, {'label', 'start_h', 'end_h', 'met'}, 'Phase')
        begin, stop = Decimal(str(phase['start_h'])), Decimal(str(phase['end_h']))
        if begin != end or stop <= begin:
            raise ValueError('Phases must be contiguous, positive and start at zero.')
        phases.append((phase['label'], float(stop - begin), phase['met']))
        end = stop
    if not phases:
        raise ValueError('A mission requires at least one phase.')
    scenario_obj = ScenarioProfile(
        scenario['scenario_type'], scenario['display_name'], scenario['category'],
        location, scenario['timezone'], datetime.combine(run_date, local_time, tzinfo=zone),
        phases, air_resistance, {'soc': initial_soc}, files[0], files[1],
        weather['temperature_sample_hours'], weather['pv_sample_hours'],
        weather['pv_reference_rating_w'], included, installed)
    evaluation = _record(scenario['evaluation'],
                         {'balance_tolerance_wh', 'time_tolerance_hours'},
                         'scenario.evaluation')
    evaluation = {k: _number(v, k, positive=True) for k, v in evaluation.items()}
    return scenario_obj, settings, evaluation, search


def load_system_config(subject_path, scenario_profiles_path, components_path,
                       scenario_type, run_date=None):
    """Load the three-file contract; default to the explicitly specified date."""
    subject, subject_file = _json(subject_path, 'subject')
    profiles, profiles_file = _json(scenario_profiles_path, 'scenario_profiles')
    components, components_file = _json(components_path, 'components')
    _record(subject, {'subject_id', 'height_cm', 'weight_kg', 'age_years', 'sex'},
            'subject')
    regional = _record(components.get('regional_model'),
                       {'region_order', 'weights', 'reference_skin_temperatures_c',
                        'area_allocation_proxies'}, 'components.regional_model')
    requested = _integer(scenario_type, 'scenario_type', nonnegative=True)
    keys = {'scenario_type', 'display_name', 'category', 'weather_location',
            'timezone', 'local_start_time', 'mission_date', 'initial_state',
            'weather', 'backpack', 'search', 'operation', 'evaluation', 'phases'}
    matches = []
    for name, profile in profiles.items():
        _record(profile, keys, f'scenario_profiles.{name}')
        if profile['scenario_type'] == requested:
            matches.append(profile)
    if len(matches) != 1:
        raise ValueError('Scenario type must identify exactly one scenario.')
    profile = matches[0]
    run_date = _date(run_date if run_date is not None else
                     profile['mission_date'].replace('-', ''))
    installed = profile['backpack']['equipment_heater_installed']
    device, garments, system = _component_objects(components, installed)
    person = Person(subject['subject_id'], subject['height_cm'], subject['weight_kg'],
                    subject['age_years'], subject['sex'], regional['weights'],
                    regional['reference_skin_temperatures_c'],
                    regional['area_allocation_proxies'], tuple(regional['region_order']))
    scenario, settings, evaluation, search = _scenario_parts(
        profile, run_date, system['air_resistance_m2k_per_w'])
    max_battery = _integer(search['max_battery_units'], 'Battery bound', nonnegative=True)
    max_pv = _integer(search['max_pv_units'], 'PV bound', nonnegative=True)
    if max_battery < 1 or max_battery > system['max_battery_units'] or max_pv > system['max_pv_units']:
        raise ValueError('Scenario search bounds exceed the component catalogue.')
    return SystemConfig(
        scenario, person, device, garments, settings, evaluation, max_battery, max_pv,
        search['preference'], search['endurance_multiplier'],
        components['controllers'], components['monte_carlo'],
        {'subject': str(subject_file), 'scenario_profiles': str(profiles_file),
         'components': str(components_file)})
