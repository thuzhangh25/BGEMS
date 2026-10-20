'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Reproduce seasonal context figures from supplied daily temperature
summaries. Uses the weather plot style and never requests network data.
'''
import argparse
from pathlib import Path

import pandas as pd

from src.weather import _save_temperature_plot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-file', type=Path,
                        default=Path('data/weather/seasonal_temperature.csv'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    frame = pd.read_csv(args.data_file, parse_dates=['date'])
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, stem in [('Harbin', 'weather_harbin'), ('Muztagh Ata', 'weather_muztagh')]:
        daily = frame.loc[frame.location == name].sort_values('date')
        start, end = daily.date.min().date(), daily.date.max().date()
        _save_temperature_plot([
            (daily.date.dt.date, daily['max'], 'Daily Max (°C)', 'red'),
            (daily.date.dt.date, daily['min'], 'Daily Min (°C)', 'blue'),
            (daily.date.dt.date, daily['mean'], 'Daily Avg (°C)', 'gold'),
        ], f'Temperature of {name} ({start} - {end})',
            args.output_dir / f'{stem}.png', title_size=12)
    stations = ['Great Wall', 'Zhongshan', 'Taishan', 'Kunlun', 'Qinling']
    series = []
    for name, color in zip(stations, ['red', 'blue', 'green', 'orange', 'purple']):
        daily = frame.loc[frame.location == name + ' Station'].sort_values('date')
        series.append((daily.date.dt.date, daily['min'], name + ' Station', color))
    _save_temperature_plot(series, 'Minimum Temperature (2025-11-01 - 2026-03-31)',
                           args.output_dir / 'weather_antarctic.png')
    print(f'Seasonal figures written to {args.output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
