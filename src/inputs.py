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
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
        self.mask = [int(value) for value in _regional(list(self.mask), 'Garment mask')]
        if any(value not in (0, 1) for value in self.mask):
            raise ValueError('Garment mask must be binary.')


@dataclass
class Device:
    cell_capacity_wh: float
    cell_mass_kg: float
    cell_cost: float
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
    voltage_v: float | None
    heat_capacity_j_per_k: float | None
    thermal_resistance_k_per_w: float | None
    max_discharge_w: float | None
    max_charge_w: float | None
    bus_limit_w: float | None

    def __post_init__(self):
        positive = ('cell_capacity_wh', 'cell_mass_kg', 'pv_rating_w', 'pv_mass_kg',
                    'base_mass_kg', 'base_power_w', 'equipment_power_w',
                    'equipment_mass_kg')
        nonnegative = ('cell_cost', 'pv_cost', 'base_cost', 'equipment_cost')
        nullable = ('voltage_v', 'heat_capacity_j_per_k', 'thermal_resistance_k_per_w',
                    'max_discharge_w', 'max_charge_w', 'bus_limit_w')
        for name in positive:
            setattr(self, name, _number(getattr(self, name), f'Device {name}', positive=True))
        for name in nonnegative:
            setattr(self, name, _number(getattr(self, name), f'Device {name}', nonnegative=True))
        for name in nullable:
            setattr(self, name, _optional_number(
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
    terminal_soc: float | None
    temperature_files: str | Path | Sequence[str | Path]
    pv_files: str | Path | Sequence[str | Path]
    temperature_sample_hours: float
    pv_sample_hours: float
    pv_reference_rating_w: float
    backpack_included: bool
    equipment_heater_required: bool

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
        self.equipment_heater_required = _boolean(
            self.equipment_heater_required, 'Scenario equipment_heater_required')
        self.terminal_soc = _optional_number(
            self.terminal_soc, 'Scenario terminal_soc', nonnegative=True)
        if self.terminal_soc is not None and self.terminal_soc > 1:
            raise ValueError('Scenario terminal_soc must lie in [0, 1].')
        self.initial_state = dict(self.initial_state)
        if not isinstance(self.temperature_files, (str, Path)):
            self.temperature_files = list(self.temperature_files)
        if not isinstance(self.pv_files, (str, Path)):
            self.pv_files = list(self.pv_files)
        self.set_phases(self.phases)

    def set_phases(self, phases):
        prepared = []
        for index, phase in enumerate(phases):
            if not isinstance(phase, (list, tuple)) or len(phase) != 4:
                raise ValueError(
                    f'Scenario phase {index} must be (label, duration_hours, met, service_floor).')
            label, duration, met, floor = phase
            label = _text(label, f'Scenario phase {index} label')
            duration = _number(duration, f'Scenario phase {index} duration', positive=True)
            met = _number(met, f'Scenario phase {index} MET', nonnegative=True)
            if floor is not None:
                if isinstance(floor, Real) and not isinstance(floor, bool):
                    floor = _number(floor, f'Scenario phase {index} service floor')
                elif isinstance(floor, (list, tuple)):
                    values = _regional(list(floor), f'Scenario phase {index} service floor')
                    if any(value != values[0] for value in values):
                        raise ValueError(
                            f'Scenario phase {index} service floor must be a scalar alpha: '
                            'the whole-body service definition requires a single alpha per '
                            'phase, and a twelve-entry array is accepted only when all '
                            'twelve entries are exactly equal; it is never averaged or '
                            'min/max-reduced.')
                    floor = float(values[0])
                else:
                    raise ValueError(
                        f'Scenario phase {index} service floor must be a JSON number '
                        '(scalar alpha) or a twelve-entry JSON array of equal values.')
                if not 0 <= floor <= 1:
                    raise ValueError('Scenario service floors must lie in [0, 1].')
            prepared.append((label, duration, met, floor))
        self.phases = prepared

    def get_duration_hours(self):
        return sum(phase[1] for phase in self.phases)

    def _phase_at(self, elapsed_hours):
        elapsed = _number(elapsed_hours, 'Scenario elapsed_hours', nonnegative=True)
        end = 0.0
        for phase in self.phases:
            end += phase[1]
            if elapsed < end:
                return phase
        raise ValueError('Scenario elapsed_hours lies outside the half-open scenario interval.')

    def get_phase_label(self, elapsed_hours):
        return self._phase_at(elapsed_hours)[0]

    def get_activity(self, elapsed_hours):
        return self._phase_at(elapsed_hours)[2]

    def get_service_requirement(self, elapsed_hours):
        return self._phase_at(elapsed_hours)[3]

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
    max_battery_cells: int
    max_pv_units: int
    priority: tuple[str, ...]
    fixed_configuration: tuple[int, int, int]
    forecast_mode: str | None
    source_files: dict[str, str] = field(default_factory=dict)
    comparison: dict = field(default_factory=lambda: {'enabled': False, 'design_manifest': None})

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
        self.max_battery_cells = _integer(
            self.max_battery_cells, 'System max_battery_cells', nonnegative=True)
        self.max_pv_units = _integer(self.max_pv_units, 'System max_pv_units', nonnegative=True)
        self.priority = tuple(self.priority)
        if len(self.priority) != 2 or set(self.priority) != {'mass', 'cost'}:
            raise ValueError('System priority must order mass and cost exactly once.')
        self.fixed_configuration = tuple(self.fixed_configuration)
        if len(self.fixed_configuration) != 3 or any(
                isinstance(value, bool) or not isinstance(value, Integral) or value < 0
                for value in self.fixed_configuration):
            raise ValueError('System fixed_configuration must contain three nonnegative integers.')
        if self.forecast_mode is not None:
            self.forecast_mode = _text(self.forecast_mode, 'System forecast_mode')
        self.source_files = dict(self.source_files)
        self.comparison = _record(
            dict(self.comparison), {'enabled', 'design_manifest'}, 'System comparison')
        self.comparison['enabled'] = _boolean(
            self.comparison['enabled'], 'System comparison.enabled')
        if self.comparison['enabled']:
            self.comparison['design_manifest'] = _text(
                self.comparison['design_manifest'], 'System comparison.design_manifest')
        elif self.comparison['design_manifest'] is not None:
            raise ValueError('Disabled comparison cannot declare a design manifest.')

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
                   'garments', 'heating_pads', 'pad_types', 'equipment_backpack'},
            'components')
    system = _record(data['system'], {
        'base_electronics_mass_kg', 'base_electronics_cost_usd',
        'base_electronics_power_w', 'air_resistance_m2k_per_w',
        'max_battery_units', 'max_pv_units'}, 'components.system')
    battery = _record(data['battery_unit'], {
        'capacity_wh', 'mass_kg', 'cost_usd', 'voltage_v', 'heat_capacity_j_per_k',
        'thermal_resistance_k_per_w', 'max_discharge_w', 'max_charge_w'},
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
        equipment_available, battery['voltage_v'], battery['heat_capacity_j_per_k'],
        battery['thermal_resistance_k_per_w'], battery['max_discharge_w'],
        battery['max_charge_w'], equipment['bus_limit_w'])
    return device, garments, system


def _scenario_parts(scenario, run_date, air_resistance):
    scenario = _record(scenario, {
        'scenario_type', 'display_name', 'category', 'weather_location', 'timezone',
        'local_start_time', 'initial_state', 'terminal_soc', 'weather', 'backpack',
        'search', 'operation', 'evaluation', 'comparison', 'phases'}, 'selected scenario')
    location = _text(scenario['weather_location'], 'scenario.weather_location')
    timezone_name = _text(scenario['timezone'], 'scenario.timezone')
    try:
        zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as error:
        raise ValueError(f'Unknown IANA timezone: {timezone_name}.') from error
    try:
        local_time = datetime.strptime(
            _text(scenario['local_start_time'], 'scenario.local_start_time'), '%H:%M').time()
    except ValueError as error:
        raise ValueError('scenario.local_start_time must use HH:MM.') from error

    initial = _record(scenario['initial_state'],
                      {'soc', 'battery_c', 'equipment_on', 'air_history_c'},
                      'scenario.initial_state')
    initial_state = {
        'soc': _number(initial['soc'], 'scenario.initial_state.soc', nonnegative=True),
        'battery_c': _optional_number(initial['battery_c'], 'scenario.initial_state.battery_c'),
        'equipment_on': _integer(initial['equipment_on'],
                                 'scenario.initial_state.equipment_on', nonnegative=True),
        'air_history_c': [_number(value, 'scenario.initial_state.air_history_c item')
                          for value in _array(initial['air_history_c'],
                                              'scenario.initial_state.air_history_c')],
    }
    if initial_state['soc'] > 1 or initial_state['equipment_on'] not in (0, 1):
        raise ValueError('Initial SOC must lie in [0, 1] and equipment_on must be 0 or 1.')

    weather = _record(scenario['weather'], {
        'temperature_sample_hours', 'pv_sample_hours', 'pv_reference_rating_w',
        'temperature_filename_template', 'pv_filename_template'}, 'scenario.weather')
    tokens = {'weather_location': location.replace(' ', '_'),
              'date': run_date.strftime('%Y-%m-%d')}
    filenames = []
    templates = (
        ('temperature', 'temperature_filename_template'),
        ('pv', 'pv_filename_template'),
    )
    for directory, key in templates:
        try:
            filename = _text(weather[key], f'scenario.weather.{key}').format(**tokens)
        except (KeyError, ValueError) as error:
            raise ValueError(f'scenario.weather.{key} has an invalid template.') from error
        if Path(filename).is_absolute() or Path(filename).name != filename:
            raise ValueError(f'scenario.weather.{key} must produce a filename.')
        filenames.append(str(WEATHER_DIR / directory / filename))

    backpack = _record(scenario['backpack'],
                       {'included', 'equipment_heater_installed', 'max_pv_units'},
                       'scenario.backpack')
    included = _boolean(backpack['included'], 'scenario.backpack.included')
    heater = _boolean(backpack['equipment_heater_installed'],
                      'scenario.backpack.equipment_heater_installed')
    backpack_pv = _integer(backpack['max_pv_units'],
                           'scenario.backpack.max_pv_units', nonnegative=True)
    if not included and (heater or backpack_pv):
        raise ValueError('A scenario without a backpack cannot install its heater or PV units.')

    search = _record(scenario['search'],
                     {'max_battery_units', 'max_pv_units', 'priority', 'fixed_configuration'},
                     'scenario.search')
    max_battery = _integer(search['max_battery_units'],
                           'scenario.search.max_battery_units', nonnegative=True)
    max_pv = _integer(search['max_pv_units'], 'scenario.search.max_pv_units', nonnegative=True)
    if max_pv != backpack_pv:
        raise ValueError('Scenario search and backpack max_pv_units must agree.')
    priority = tuple(_text(value, 'scenario.search.priority item')
                     for value in _array(search['priority'], 'scenario.search.priority'))
    fixed = tuple(_integer(value, 'scenario.search.fixed_configuration item', nonnegative=True)
                  for value in _array(search['fixed_configuration'],
                                      'scenario.search.fixed_configuration'))

    comparison = _record(scenario['comparison'],
                         {'enabled', 'design_manifest'}, 'scenario.comparison')
    _boolean(comparison['enabled'], 'scenario.comparison.enabled')
    if comparison['enabled']:
        _text(comparison['design_manifest'], 'scenario.comparison.design_manifest')
    elif comparison['design_manifest'] is not None:
        raise ValueError('Disabled comparison cannot declare a design manifest.')
    operation_keys = {'step_h', 'horizon_h', 'soc_min', 'soc_safe', 'lambda_deg',
                      'lambda_soc', 'equipment_on_c', 'equipment_off_c',
                      'filter_window_h', 'solver_tolerance',
                      'constraint_tolerance', 'forecast_mode'}
    operation = _record(scenario['operation'], operation_keys, 'scenario.operation')
    step_h = _number(operation['step_h'], 'scenario.operation.step_h', positive=True)
    horizon = _number(operation['horizon_h'], 'scenario.operation.horizon_h', positive=True)
    horizon_steps = horizon / step_h
    if not float(horizon_steps).is_integer():
        raise ValueError('scenario.operation.horizon_h must be an exact multiple of step_h.')
    settings = {'dt_hours': step_h, 'horizon_steps': int(horizon_steps)}
    for key in operation_keys - {'step_h', 'horizon_h', 'forecast_mode'}:
        settings[key] = _optional_number(
            operation[key], f'scenario.operation.{key}', nonnegative=True)
    forecast = operation['forecast_mode']
    if forecast is not None:
        forecast = _text(forecast, 'scenario.operation.forecast_mode')

    phases, next_start = [], 0.0
    phase_keys = {'label', 'start_h', 'end_h', 'met', 'service_floor'}
    for index, phase in enumerate(_array(scenario['phases'], 'scenario.phases')):
        phase = _record(phase, phase_keys, f'scenario.phases[{index}]')
        start_h = _number(phase['start_h'], 'Scenario phase start_h', nonnegative=True)
        end_h = _number(phase['end_h'], 'Scenario phase end_h', positive=True)
        if not math.isclose(start_h, next_start, abs_tol=1e-12) or end_h <= start_h:
            raise ValueError('Scenario phases must be contiguous, positive and start at zero.')
        duration = float(Decimal(str(end_h)) - Decimal(str(start_h)))
        phases.append((phase['label'], duration, phase['met'], phase['service_floor']))
        next_start = end_h
    if not math.isclose(next_start, horizon, abs_tol=1e-12):
        raise ValueError('Scenario phases must end at operation.horizon_h.')

    evaluation = _record(scenario['evaluation'],
                         {'balance_tolerance_wh', 'time_tolerance_hours'},
                         'scenario.evaluation')
    evaluation = {key: _number(value, f'scenario.evaluation.{key}', positive=True)
                  for key, value in evaluation.items()}
    scenario = ScenarioProfile(
        scenario['scenario_type'], scenario['display_name'], scenario['category'], location,
        timezone_name, datetime.combine(run_date, local_time, tzinfo=zone), phases,
        air_resistance, initial_state, scenario['terminal_soc'], filenames[0], filenames[1],
        weather['temperature_sample_hours'], weather['pv_sample_hours'],
        weather['pv_reference_rating_w'], included, heater)
    return (scenario, settings, evaluation, max_battery, max_pv, priority, fixed,
            forecast, heater, comparison)


def load_system_config(subject_path, scenario_profiles_path, components_path,
                       scenario_type, run_date):
    """Load the fixed three-file contract and return one SystemConfig."""
    subject, subject_file = _json(subject_path, 'subject')
    profiles, profiles_file = _json(scenario_profiles_path, 'scenario_profiles')
    components, components_file = _json(components_path, 'components')
    subject = _record(subject,
                      {'subject_id', 'height_cm', 'weight_kg', 'age_years', 'sex'},
                      'subject')
    regional = _record(components.get('regional_model'),
                       {'region_order', 'weights', 'reference_skin_temperatures_c',
                        'area_allocation_proxies'}, 'components.regional_model')
    order = tuple(_text(value, 'components.regional_model.region_order item')
                  for value in _array(regional['region_order'],
                                      'components.regional_model.region_order'))
    if order != REGION_ORDER:
        raise ValueError('components.regional_model.region_order must match the adopted order.')

    scenario_keys = {'scenario_type', 'display_name', 'category', 'weather_location',
                     'timezone', 'local_start_time', 'initial_state', 'terminal_soc',
                     'weather', 'backpack', 'search', 'operation', 'evaluation',
                     'comparison', 'phases'}
    requested = _integer(scenario_type, 'scenario_type', nonnegative=True)
    matches = []
    for key, scenario in profiles.items():
        scenario = _record(scenario, scenario_keys, f'scenario_profiles.{key}')
        declared = _integer(scenario['scenario_type'],
                            f'scenario_profiles.{key}.scenario_type', nonnegative=True)
        if declared == requested:
            matches.append(scenario)
    if len(matches) != 1:
        raise ValueError(f'scenario_type {requested} must identify exactly one scenario.')
    scenario = matches[0]
    equipment_available = _boolean(
        _record(scenario['backpack'],
                {'included', 'equipment_heater_installed', 'max_pv_units'},
                'selected scenario.backpack')['equipment_heater_installed'],
        'selected scenario.backpack.equipment_heater_installed')

    device, garments, system = _component_objects(components, equipment_available)
    person = Person(
        subject['subject_id'], subject['height_cm'], subject['weight_kg'],
        subject['age_years'], subject['sex'], regional['weights'],
        regional['reference_skin_temperatures_c'], regional['area_allocation_proxies'], order)
    (scenario, settings, evaluation, max_battery, max_pv, priority, fixed,
     forecast, _, comparison) = _scenario_parts(
        scenario, _date(run_date), system['air_resistance_m2k_per_w'])
    component_max_battery = _integer(
        system['max_battery_units'], 'components.system.max_battery_units', nonnegative=True)
    component_max_pv = _integer(
        system['max_pv_units'], 'components.system.max_pv_units', nonnegative=True)
    if max_battery > component_max_battery or max_pv > component_max_pv:
        raise ValueError('Scenario search bounds exceed component catalogue bounds.')

    return SystemConfig(
        scenario, person, device, garments, settings, evaluation, max_battery, max_pv,
        priority, fixed, forecast,
        {'subject': str(subject_file), 'scenario_profiles': str(profiles_file),
         'components': str(components_file)}, comparison)
