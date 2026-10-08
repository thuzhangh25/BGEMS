'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Render the five Results figures from self-contained derived evidence.
No frozen archives, scientific model imports, weather API or paper tree required.
Usage: python3 -m scripts.plot_results --data-dir <evidence> --output-dir <new-figures>
'''
import argparse
import json
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

# Pin the noninteractive renderer: macOSX measures tight-layout text differently.
matplotlib.use('Agg')

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

SCENARIOS = {1: 'Harbin', 2: 'Muztagh Ata', 3: 'Qinling'}
COLORS = ['#3498DB', '#F39C12', '#2ECC71', '#E74C3C']
GARMENTS = ['Down jumpsuit', 'Fleece jumpsuit', 'Down top', 'Fleece top']
RC = {
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'font.weight': 'bold', 'axes.labelweight': 'bold', 'axes.titleweight': 'bold',
    'text.color': 'black', 'axes.labelcolor': 'black', 'axes.edgecolor': 'black',
    'xtick.color': 'black', 'ytick.color': 'black', 'figure.facecolor': 'white',
    'axes.facecolor': 'white', 'savefig.facecolor': 'white',
}
SCEN = [(1, 'Harbin'), (2, 'Muztagh Ata'), (3, 'Qinling')]
COLOR = '#3498DB'
PV_COLOR = '#F39C12'
RANGE_ALPHA = 0.18
HATCHES = ['', '//', 'xx', '\\\\']
GARMENT_LABELS = {
    'a_down_jumpsuit': 'Down\njumpsuit',
    'b_fleece_jumpsuit': 'Fleece\njumpsuit',
    'c_down_top': 'Down\ntop',
    'd_fleece_top': 'Fleece\ntop',
}

def _style(ax, xlabel, ylabel, title, *, configuration=False, right_axis=False, title_pad=None):
    """Preserve the original evidence and configuration panel typography."""
    ax.set_xlabel(xlabel, fontsize=13 if configuration else 14, fontweight='bold', color='black')
    ax.set_ylabel(ylabel, fontsize=13 if configuration else 14, fontweight='bold', color='black')
    title_options = {'pad': title_pad if title_pad is not None else 6} if configuration else {}
    ax.set_title(title, fontsize=14 if configuration else 16, fontweight='bold',
                 color='black', **title_options)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(right_axis)
    ax.spines['left'].set_visible(not right_axis)
    ax.spines['bottom'].set_visible(not right_axis)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(axis='both', direction='in', width=1.5, length=5,
                   labelsize=10 if configuration else 11, colors='black')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_color('black')


def _plot_forcing(ax, records, name, temperature_limits, pv_limits):
    dates = [datetime.strptime(record['date'], '%Y%m%d') for record in records]
    temp_range = ax.fill_between(
        dates, [record['temperature_min_c'] for record in records],
        [record['temperature_max_c'] for record in records],
        color=COLOR, alpha=RANGE_ALPHA, linewidth=0, zorder=1)
    pv_ax, pv_range, pv_mean = None, None, None
    if records[0]['pv_unit_rating_w'] is not None:
        pv_ax = ax.twinx()
        # Keep the temperature mean in front of the orange range and mean.
        pv_ax.set_zorder(ax.get_zorder() - 1)
        ax.patch.set_visible(False)
        pv_range = pv_ax.fill_between(
            dates, [record['pv_min_w'] for record in records],
            [record['pv_max_w'] for record in records],
            color=PV_COLOR, alpha=RANGE_ALPHA, linewidth=0, zorder=1)
        pv_mean, = pv_ax.plot(
            dates, [record['pv_mean_w'] for record in records],
            color=PV_COLOR, linestyle='--', linewidth=1.5, zorder=2)
        rating = records[0]['pv_unit_rating_w']
        _style(pv_ax, '', f'PV power (W per {rating:g}-W unit)', '', right_axis=True, configuration=True)
        pv_ax.set_ylim(pv_limits)
    temp_mean, = ax.plot(
        dates, [record['temperature_mean_c'] for record in records],
        color=COLOR, linestyle='-', linewidth=1.5, zorder=3)
    ax.set_ylim(temperature_limits)
    ax.set_xlim(dates[0] - timedelta(days=1), dates[-1] + timedelta(days=1))
    locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    years = str(dates[0].year)
    if dates[-1].year != dates[0].year:
        years += f'–{dates[-1].year}'
    _style(ax, f'Mission date ({years})', 'Temperature (°C)',
           f'{name} — forcing (n={len(records)})', title_pad=43, configuration=True)
    handles = [temp_mean, temp_range]
    labels = ['T mean', 'T min–max']
    if pv_ax is not None:
        handles.extend([pv_mean, pv_range])
        labels.extend(['PV mean', 'PV min–max'])
    ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 1.015),
              ncol=2, frameon=False, fontsize=10, handlelength=1.8,
              columnspacing=1.0, borderaxespad=0)
    return pv_ax


def build_figure(rows, garments, forcing):
    """Return (figure, 3-by-4 primary axes, two PV axes), without saving."""
    font_manager.findfont(
        font_manager.FontProperties(family='Times New Roman', weight='bold'),
        fallback_to_default=False)
    minimum = min(record['temperature_min_c'] for record in forcing)
    maximum = max(record['temperature_max_c'] for record in forcing)
    margin = max((maximum - minimum) * 0.06, 1.0)
    temperature_limits = (minimum - margin, maximum + margin)
    maximum_pv = max(record['pv_max_w'] for record in forcing
                     if record['pv_max_w'] is not None)
    pv_limits = (0, maximum_pv + max(maximum_pv * 0.08, 0.5))
    rc = {
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'axes.edgecolor': 'black',
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            3, 4, figsize=(16.5, 11.5),
            gridspec_kw={'width_ratios': [1.3, 1.15, 0.85, 1.15]})
        pv_axes = []
        for row_i, (scen, name) in enumerate(SCEN):
            records = [record for record in forcing if record['scenario_type'] == scen]
            pv_ax = _plot_forcing(axes[row_i][0], records, name,
                                  temperature_limits, pv_limits)
            if pv_ax is not None:
                pv_axes.append(pv_ax)
            succ = [r for r in rows
                    if int(r['scenario_type']) == scen and r['status'] == 'success']
            n = len(succ)
            batt = Counter(int(r['selected_y']) for r in succ)
            pv = Counter(int(r['selected_h']) for r in succ)
            gar = Counter(int(r['selected_g']) for r in succ)

            # Battery units (bars, zero baseline)
            ax = axes[row_i][1]
            keys = sorted(batt)
            ax.bar([str(k) for k in keys], [batt[k] for k in keys],
                   color=COLOR, edgecolor='black', linewidth=1.0)
            for i, k in enumerate(keys):
                ax.text(i, batt[k], str(batt[k]), ha='center', va='bottom',
                        fontsize=9, fontweight='bold')
            ax.set_ylim(0, max(batt.values()) * 1.18)
            _style(ax, 'Battery units', 'Dates',
                   f'{name}\nbattery (n={n})', configuration=True)
            if len(keys) > 10:
                ax.tick_params(axis='x', labelsize=7.5, labelrotation=90)

            # PV units (bars, zero baseline)
            ax = axes[row_i][2]
            keys = sorted(pv)
            ax.bar([str(k) for k in keys], [pv[k] for k in keys],
                   color=COLOR, edgecolor='black', linewidth=1.0)
            for i, k in enumerate(keys):
                ax.text(i, pv[k], str(pv[k]), ha='center', va='bottom',
                        fontsize=9, fontweight='bold')
            ax.set_ylim(0, max(pv.values()) * 1.18)
            _style(ax, 'PV units', 'Dates',
                   f'{name}\nPV (n={n})', configuration=True)

            # Garments (bars, zero baseline, redundant hatch per garment)
            ax = axes[row_i][3]
            gids = list(range(len(garments)))
            counts = [gar.get(g, 0) for g in gids]
            labels = [GARMENT_LABELS.get(garments[g], garments[g]) for g in gids]
            bars = ax.bar(labels, counts, color=COLOR,
                          edgecolor='black', linewidth=1.0)
            for bar, g in zip(bars, gids):
                bar.set_hatch(HATCHES[g])
                if counts[g] == 0:
                    bar.set_facecolor('#dddddd')
                    ax.text(bar.get_x() + bar.get_width() / 2, 0.3, '0',
                            ha='center', va='bottom', fontsize=9,
                            fontweight='bold', color='#555555')
            for i, g in enumerate(gids):
                if counts[g] > 0:
                    ax.text(i, counts[g], str(counts[g]), ha='center',
                            va='bottom', fontsize=9, fontweight='bold')
            ymax = max(counts) if max(counts) > 0 else 1
            ax.set_ylim(0, ymax * 1.18)
            _style(ax, 'Garment', 'Dates',
                   f'{name}\ngarment (n={n})', configuration=True)

        fig.tight_layout(h_pad=2.0, w_pad=1.6)
    return fig, axes, pv_axes


def legend(ax, **kwargs):
    ax.legend(frameon=False, prop={'size': 11, 'weight': 'bold'}, **kwargs)


def configuration_figure(evidence, candidates):
    keys = ['s1_20251201', 's1_20260210', evidence['max_front_date']]
    fig, mosaic = plt.subplot_mosaic([['a', 'b'], ['c', 'd']], figsize=(13, 9),
                                    constrained_layout=True)
    axes = [mosaic[k] for k in ('a', 'b', 'c')]
    for panel, (ax, key) in enumerate(zip(axes, keys)):
        rows = candidates[key]
        for g, (label, color) in enumerate(zip(GARMENTS, COLORS)):
            subset = [r for r in rows if r['tuple'][2] == g]
            for accepted in [False, True]:
                group = [r for r in subset if (r['outcome'] == 'success') == accepted]
                ax.scatter([r['mass'] for r in group], [r['cost'] for r in group],
                           marker='o' if accepted else 'x', color=color,
                           s=35 if accepted else 22, alpha=.8 if accepted else .23,
                           label=label if accepted else None)
        front = evidence['fronts'][key]
        ax.scatter([r['mass'] for r in front], [r['cost'] for r in front],
                   s=115, facecolors='none', edgecolors='black', linewidth=1.6, label='Pareto')
        selected = min(front, key=lambda r: (r['mass'], r['cost'], r['tuple']))
        ax.scatter(selected['mass'], selected['cost'], marker='*', s=250,
                   color='black', zorder=5, label='Selected')
        if panel < 2:
            ax.scatter(2.5, 970, marker='s', s=120, facecolors='none',
                       edgecolors='black', linewidth=1.2, label='Specified fixed')
            ax.set_xlim(1, 10); ax.set_ylim(380, 1310)
        title = f"({chr(97 + panel)}) {SCENARIOS[int(key[1])]} {key[3:]}"
        _style(ax, 'Mass subtotal (kg)', 'Cost subtotal (USD)', title)
        ax.text(.98, .98, f"Accepted: {sum(r['outcome'] == 'success' for r in rows)}/{len(rows)}\nPareto: {len(front)}",
                transform=ax.transAxes, ha='right', va='top', fontsize=12)
    legend(axes[0], loc='lower right', ncol=1)
    ax = mosaic['d']
    for comparison in evidence['seasonal_comparison']:
        scenario = comparison['scenario']
        reference = comparison['references']['mass_first']
        if reference is None:
            continue
        color = COLORS[scenario - 1]
        x, y = comparison['selected_mean_mass_kg'], comparison['selected_mean_cost_usd']
        ax.scatter(x, y, marker='D', s=85, color=color,
                   label=f'{SCENARIOS[scenario]}: daily mean')
        ax.plot([x, reference['mass_kg']], [y, reference['cost_usd']],
                '--', color=color, linewidth=1.5)
        ax.scatter(reference['mass_kg'], reference['cost_usd'], marker='o',
                   s=110, facecolors='none', edgecolors=color, linewidth=2,
                   label='Season fixed: mass-first' if scenario == 1 else None)
        cost_reference = comparison['references']['cost_first']
        if cost_reference['tuple'] != reference['tuple']:
            ax.scatter(cost_reference['mass_kg'], cost_reference['cost_usd'],
                       marker='s', s=100, facecolors='none', edgecolors=color,
                       linewidth=2, label='Season fixed: cost-first')
    ax.text(.03, .96, 'Full-season coverage: Harbin 90/90; Muztagh 46/46\n'
            'Qinling: no full-season feasible configuration',
            transform=ax.transAxes, va='top', fontsize=10)
    _style(ax, 'Mass subtotal (kg)', 'Cost subtotal (USD)',
           '(d) Seasonal fixed references and daily means')
    ax.set_xlim(2, 6); ax.set_ylim(760, 1200)
    legend(ax, loc='lower right')
    return fig


def tradeoff_figure(evidence):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for s, name in SCENARIOS.items():
        rows = [r for r in evidence['selected'] if r['scenario'] == s]
        for ax, metric in zip(axes, ['terminal_soc', 'mean_thermal_score_observed']):
            ax.scatter([r['mass_kg'] for r in rows], [r[metric] for r in rows],
                       color=COLORS[s - 1], marker=['o', '^', 's'][s - 1],
                       s=45, alpha=.55, edgecolors='black', linewidth=.35,
                       label=f'{name} (n={len(rows)})')
    axes[0].axhline(.15, color='black', linestyle='--', linewidth=1.2, label='Terminal floor')
    _style(axes[0], 'Catalogue mass subtotal (kg)', 'Terminal SOC', '(a) Resource–reserve distribution')
    _style(axes[1], 'Catalogue mass subtotal (kg)', 'Mean modeled service', '(b) Resource–service distribution')
    legend(axes[0], loc='upper right'); legend(axes[1], loc='lower left')
    return fig


def operation_figure(evidence, priority):
    tr = evidence['operation']; steps = tr['steps']; solves = tr['solves']
    times = np.array([s['time_hours'] for s in steps]); dt = tr['dt_hours']
    edges = np.r_[times, times[-1] + dt]
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    def intervals(ax, values, **kwargs):
        ax.stairs(values, edges, baseline=None, **kwargs)
    demand = np.array([s['demand_w'] for s in steps])
    floor = tr['floor']
    intervals(axes[0, 0], demand, color='black', label='Full demand')
    intervals(axes[0, 0], demand * floor, color=COLORS[1], linestyle='--', label='Service floor')
    intervals(axes[0, 0], [s['total_effective_heat_w'] for s in steps], color=COLORS[0], label='Effective supply')
    _style(axes[0, 0], 'Elapsed mission time (h)', 'Effective heat (W)', '(a) Demand and delivered heat')
    for j, label in enumerate(solves[0]['priority_group_labels']):
        intervals(axes[0, 1], [s['priority_realized'][j] for s in solves], color=COLORS[j], label=label.title())
        axes[0, 1].scatter(times, [s['priority_targets'][j] for s in solves], s=12,
                           facecolors='none', edgecolors=COLORS[j], linewidth=.7)
    _style(axes[0, 1], 'Elapsed mission time (h)', 'Group allocation fulfillment', '(b) Targets (circles) and execution (lines)')
    axes[1, 0].plot([s['time_hours'] for s in tr['states']], [s['soc'] for s in tr['states']],
                    color=COLORS[0], linewidth=2, label='SOC')
    axes[1, 0].axhline(.15, color='black', linestyle='--', label='Safe/terminal level')
    axes[1, 0].axhline(.05, color=COLORS[3], linestyle=':', label='Hard lower bound')
    _style(axes[1, 0], 'Elapsed mission time (h)', 'SOC', '(c) Stored energy and reserve')
    intervals(axes[1, 1], [s['load_w'] for s in steps], color=COLORS[0], label='Electrical load')
    intervals(axes[1, 1], [s['pv_used_w'] for s in steps], color=COLORS[1], linestyle='--', label='PV used')
    _style(axes[1, 1], 'Elapsed mission time (h)', 'Electrical power (W)', '(d) Supply-side energy flow')
    intervals(axes[2, 0], [s['deg_weights'][0] for s in solves], color=COLORS[1],
              label='Weight supplied to first action')
    _style(axes[2, 0], 'Elapsed mission time (h)', 'Discharge weight', '(e) Extrapolated degradation proxy')
    axes[2, 0].set_yticks([.2, .5, 1.0])
    for s, name in SCENARIOS.items():
        rows = [r for r in priority if r['scenario'] == s]
        axes[2, 1].scatter([r['minimum_planned_soc'] for r in rows],
                           [r['forecast_soc_penalty'] for r in rows],
                           s=16, alpha=.4, color=COLORS[s - 1], label=name)
    axes[2, 1].axvline(.15, color='black', linestyle='--', linewidth=1)
    _style(axes[2, 1], 'Minimum planned SOC in forecast', 'Summed weighted SOC penalty',
           '(f) Reserve preference activation: all dates')
    for ax in axes.flat:
        legend(ax, loc='best')
    for ax in list(axes.flat)[:5]:
        ax.set_xlim(0, edges[-1])
    return fig


def stress_figure(evidence):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    levels = [['dT+0.0', 'dT-2.0', 'dT-5.0', 'dT-10.0'],
              ['scale1.00', 'scale0.75', 'scale0.50', 'uniform0.50']]
    labels = [['0', '−2', '−5', '−10'], ['Baseline', '0.75η', '0.50η', 'Uniform 0.5']]
    for column in range(2):
        ax = axes[0, column]
        for row in evidence['stress']:
            if row['level'] not in levels[column]:
                continue
            x, y = levels[column].index(row['level']), 3 - row['scenario']
            ok = row['status'] == 'success'
            ax.scatter(x, y, marker='o' if ok else 'x', s=125,
                       color=COLORS[0] if ok else COLORS[3], linewidth=2)
            text = 'Pass' if ok else f"{row['certificate'][0].upper()}: {row['solve_time_h']:g} h"
            ax.annotate(text, (x, y), xytext=(0, 12), textcoords='offset points',
                        ha='center', fontsize=11)
        ax.set_xticks(range(4), labels[column]); ax.set_yticks([2, 1, 0], SCENARIOS.values())
        ax.set_ylim(-.4, 2.6); ax.set_xlim(-.4, 3.4)
        _style(ax, 'Temperature shift (°C)' if column == 0 else 'Efficiency condition',
               '', '(a) Ambient stress' if column == 0 else '(b) Transfer-factor stress')
    baseline = next(r for r in evidence['stress'] if r['scenario'] == 1 and r['level'] == 'dT+0.0')
    cold = next(r for r in evidence['stress'] if r['scenario'] == 1 and r['level'] == 'dT-2.0')
    axes[1, 0].bar([0, 1], [baseline['current_required_w'], cold['current_required_w']],
                    color=[COLORS[0], COLORS[3]], edgecolor='black', width=.55)
    axes[1, 0].axhline(baseline['max_heat_w'], color='black', linestyle='--', label='Installed maximum')
    axes[1, 0].set_xticks([0, 1], ['Baseline', '−2°C'])
    _style(axes[1, 0], 'Harbin: first interval', 'Required effective heat (W)', '(c) Thermal-capacity certificate')
    axes[1, 0].set_ylim(0, 110); legend(axes[1, 0], loc='upper left')
    energy = [r for r in evidence['stress'] if r['certificate'] == 'energy']
    for s in [2, 3]:
        rows = [r for r in energy if r['scenario'] == s]
        axes[1, 1].scatter([r['thermal_margin_w'] for r in rows],
                           [r['energy_margin_wh'] for r in rows], s=80,
                           marker='^' if s == 2 else 's', color=COLORS[s - 1],
                           edgecolors='black', label=SCENARIOS[s])
    axes[1, 1].axhline(0, color='black', linestyle='--')
    _style(axes[1, 1], 'Minimum forecast thermal margin (W)',
           'Optimistic energy margin (Wh)', '(d) Energy certificates: 10 failures')
    legend(axes[1, 1], loc='lower left')
    return fig


def save_figure(fig, directory, name):
    for suffix in ('png', 'pdf'):
        fig.savefig(directory / f'{name}.{suffix}', dpi=300,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='A new directory; original paper images are never overwritten.')
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('--output-dir must not already exist')
    font_manager.findfont(font_manager.FontProperties(family='Times New Roman', weight='bold'),
                          fallback_to_default=False)
    evidence = json.loads((args.data_dir / 'evidence.json').read_text())
    if evidence['schema_version'] != 1:
        raise ValueError('Unsupported evidence schema version')
    args.output_dir.mkdir(parents=True)
    config = evidence['configuration']
    # Configuration historically finishes its private rc_context before saving.
    fig, _, _ = build_figure(config['ledger'], config['garments'], config['forcing'])
    save_figure(fig, args.output_dir, 'configuration_outcomes')
    with plt.rc_context(RC):
        figures = {
            'design_pareto_comparison': configuration_figure(evidence, evidence['candidate_points']),
            'resource_service_reserve': tradeoff_figure(evidence),
            'allocation_operation_evidence': operation_figure(evidence, evidence['priority_records']),
            'stress_mechanism_evidence': stress_figure(evidence),
        }
        for name, fig in figures.items():
            save_figure(fig, args.output_dir, name)
    print(f'Wrote five figure pairs to {args.output_dir}')


if __name__ == '__main__':
    main()
