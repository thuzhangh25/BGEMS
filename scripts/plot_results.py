'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Plot recorded finite-task controller trajectories without re-simulation.
Input is the self-contained report_evidence export; incomplete runs stop at their
last observed state, never at an invented terminal value.
'''

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

COLORS = {'rbc': '#3498DB', 'pid': '#F39C12', 'tacs': '#2ECC71', 'rhc': '#E74C3C'}
RC = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'axes.edgecolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
}


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(direction='in', width=1.5, length=5, labelsize=11)


def plot_trajectories(evidence, output_dir):
    if evidence.get('schema') != 'bgems-evidence-v1':
        raise ValueError('Expected a current finite-task evidence export.')
    groups = {}
    for record in evidence['records']:
        # One figure per common realization, including MC samples and stress arms.
        case = str(Path(record['source']).parent.parent)
        groups.setdefault(case, []).append(record)
    if not groups:
        raise ValueError('No observed trajectories to plot.')
    font_manager.findfont(
        font_manager.FontProperties(family='Times New Roman', weight='bold'),
        fallback_to_default=False,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    artifacts = []
    with plt.rc_context(RC):
        for index, (case, records) in enumerate(sorted(groups.items())):
            if len({r['forcing_sha256'] for r in records}) != 1:
                raise ValueError('Controller panel mixes different realized inputs.')
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            for record in records:
                run = record['run']
                color = COLORS[record['controller']]
                label = record['controller'].upper()
                actions = run['actions']
                states = run['states']
                if actions:
                    times = [a['time_h'] for a in actions] + [states[-1]['time_h']]
                    values = [a['score'] for a in actions] + [actions[-1]['score']]
                    axes[0].step(
                        times,
                        values,
                        where='post',
                        label=label,
                        color=color,
                        linewidth=1.5,
                    )
                else:
                    axes[0].plot(
                        [],
                        [],
                        label=f'{label} (no completed interval)',
                        color=color,
                        linewidth=1.5,
                    )
                axes[1].plot(
                    [s['time_h'] for s in states],
                    [s['soc'] for s in states],
                    label=label,
                    color=color,
                    linewidth=1.5,
                )
                if not run['completed']:
                    axes[1].plot(
                        states[-1]['time_h'],
                        states[-1]['soc'],
                        marker='x',
                        color=color,
                        markersize=7,
                    )
            first = records[0]
            _style(
                axes[0],
                'Elapsed mission time (h)',
                'Localized thermal-comfort score',
                first['scenario_name'],
            )
            _style(axes[1], 'Elapsed mission time (h)', 'Battery SOC', first['date'])
            axes[1].axhline(
                first['settings']['soc_min'],
                color='black',
                linestyle='--',
                linewidth=1,
                label='Hard SOC floor',
            )
            axes[1].axhline(
                first['settings']['soc_safe'],
                color='black',
                linestyle=':',
                linewidth=1,
                label='Soft SOC preference',
            )
            for ax in axes:
                ax.set_xlim(0, first['evaluation']['metrics']['scheduled_duration_h'])
                ax.set_ylim(-0.02, 1.04)
                ax.legend(frameon=False, fontsize=10, loc='best')
            fig.tight_layout()
            stem = f'controller_trajectories_{index+1:02d}'
            for suffix in ('png', 'pdf'):
                path = output / f'{stem}.{suffix}'
                fig.savefig(path, dpi=200, bbox_inches='tight')
                artifacts.append(str(path))
            plt.close(fig)
    (output / 'figure_sources.json').write_text(
        json.dumps(
            {
                'cases': list(sorted(groups)),
                'source_files_sha256': evidence['source_files_sha256'],
                'optimizer_calls': 0,
            },
            indent=2,
        )
        + '\n'
    )
    return artifacts


def plot_study(evidence, output_dir):
    """Render the specified finite study from one self-contained numeric export."""
    if evidence.get('schema') != 'bgems-study-results-v1':
        raise ValueError('Expected the specified paper evidence schema.')
    font_manager.findfont('Times New Roman', fallback_to_default=False)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    artifacts = []
    controllers = tuple(COLORS)
    styles = dict(zip(controllers, ('--', ':', '-.', '-')))
    trajectories = evidence['trajectories']
    names = {1: 'Harbin', 2: 'Muztagh Ata', 3: 'Qinling indoor'}

    def records(prefix):
        return sorted((r for path, r in trajectories.items() if path.startswith(prefix)),
                      key=lambda r: controllers.index(r['run']['controller']))

    fleece = records('experiments/fixed_fleece/indoor/setpoint_15/')
    if tuple(record['run']['controller'] for record in fleece) != controllers:
        raise ValueError('Qinling fixed-fleece panel requires one record per controller.')
    reference = fleece[0]
    for record in fleece:
        for key in ('actual', 'forcing_sha256', 'hardware', 'settings',
                    'configuration', 'initial_soc'):
            if record[key] != reference[key]:
                raise ValueError(f'Qinling fixed-fleece panel mixes different {key}.')
        run = record['run']
        if (not run['completed'] or run['scheduled_steps'] != 140
                or run['dt_hours'] != .1 or len(run['actions']) != 140
                or len(run['states']) != 141):
            raise ValueError('Qinling fixed-fleece panel requires complete 14-h trajectories.')
        for samples, count in ((run['actions'], 140), (run['states'], 141)):
            if not np.allclose([sample['time_h'] for sample in samples],
                               np.arange(count) * .1, rtol=0, atol=1e-9):
                raise ValueError('Qinling fixed-fleece panel has inconsistent time boundaries.')
    if (reference['configuration'] != [8, 0, 1]
            or reference['hardware']['capacity_wh'] != 480
            or reference['hardware']['base_power_w'] != 5
            or reference['initial_soc'] != 1
            or reference['settings']['dt_hours'] != .1
            or reference['settings']['soc_min'] != .05
            or any(reference['actual']['pv_w'])
            or any(state['soc'] != reference['initial_soc']
                   for state in (record['run']['states'][0] for record in fleece))):
        raise ValueError('Qinling fixed-fleece panel requires the specified shared 480-Wh setup.')

    def axes_style(ax, x, y, title):
        _style(ax, x, y, title)
        ax.title.set_fontsize(13)
        ax.xaxis.label.set_fontsize(12)
        ax.yaxis.label.set_fontsize(12)

    def save(fig, stem, legend=False):
        fig.tight_layout(rect=(0, .12 if legend else 0, 1, 1))
        for suffix in ('png', 'pdf'):
            path = output / f'{stem}.{suffix}'
            fig.savefig(path, dpi=300, bbox_inches='tight')
            artifacts.append(str(path))
        plt.close(fig)

    def interval(ax, record, values, **kwargs):
        actions = record['run']['actions']
        if actions:
            time = [a['time_h'] for a in actions] + [record['run']['states'][-1]['time_h']]
            ax.step(time, list(values) + [values[-1]], where='post', **kwargs)

    def controller_line(ax, record, key):
        controller = record['run']['controller']
        interval(ax, record, [a[key] for a in record['run']['actions']],
                 color=COLORS[controller], linestyle=styles[controller], linewidth=1.6,
                 label=controller.upper())

    def legend(fig, ax, ncol=4):
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', frameon=False, ncol=ncol, fontsize=11)

    with plt.rc_context({**RC, 'pdf.fonttype': 42, 'font.size': 11}):
        fig, axes = plt.subplots(3, 2, figsize=(8.8, 9))
        for index, unperturbed in enumerate(evidence['unperturbed']):
            summary = unperturbed['summary']
            scenario = summary['scenario_type']
            trace = records(f'results/s{scenario}_')[0]
            ax = axes[index, 0]
            interval(ax, trace, trace['actual']['air_c'], color=COLORS['rbc'], linewidth=1.6)
            axes_style(ax, 'Elapsed time (h)', 'Air temperature (°C)', f'{chr(97+2*index)}) {names[scenario]}')
            if trace['configuration'][1]:
                pvax = ax.twinx()
                interval(pvax, trace, trace['actual']['pv_w'], color=COLORS['pid'], linestyle='--')
                pvax.set_ylabel('Available PV (W)', color=COLORS['pid'], fontsize=11)
                pvax.tick_params(labelsize=11, colors=COLORS['pid'])
                pvax.spines['top'].set_visible(False)
                pvax.grid(False)
            front = [r for r in unperturbed['candidates']
                     if r['configuration'] in summary['pareto_configurations']]
            ax = axes[index, 1]
            scatter = ax.scatter([r['mass_kg'] for r in front], [r['cost_usd'] for r in front],
                                 c=[r['score'] for r in front], vmin=.75, vmax=1,
                                 cmap='Blues', s=65, edgecolors='black', linewidths=.7,
                                 label='Pareto configuration')
            chosen = summary['selected']
            ax.scatter(chosen['mass_kg'], chosen['cost_usd'], marker='*', s=130,
                       color=COLORS['rhc'], edgecolors='black', linewidths=.5, label='Selected configuration')
            masses, costs = [r['mass_kg'] for r in front], [r['cost_usd'] for r in front]
            dx, dy = max(.025, np.ptp(masses)*.8), max(10., np.ptp(costs)*.8)
            ax.set_xlim(min(masses)-dx, max(masses)+dx)
            ax.set_ylim(min(costs)-dy, max(costs)+dy)
            ax.ticklabel_format(useOffset=False, style='plain')
            for point in front:
                ax.annotate(f'{tuple(point["configuration"])}\nU$_{{\\mathrm{{cfg}}}}$={point["score"]:.4f}',
                            (point['mass_kg'], point['cost_usd']), xytext=(0, 12),
                            textcoords='offset points', ha='center', fontsize=11)
            axes_style(ax, 'Mass (kg)', 'Cost (USD)', f'{chr(98+2*index)}) Selected {tuple(chosen["configuration"])}')
            bar = fig.colorbar(scatter, ax=ax, fraction=.045, pad=.03)
            bar.set_label(r'Capability $U_{\mathrm{cfg}}$', fontsize=11)
            bar.ax.tick_params(labelsize=11)
        axes[0, 1].legend(frameon=False, fontsize=11)
        save(fig, 'configuration')

        fig, axes = plt.subplots(3, 2, figsize=(8.8, 8.2))
        for index, scenario in enumerate((1, 2, 3)):
            group = records(f'results/s{scenario}_')
            for record in group:
                controller_line(axes[index, 0], record, 'score')
                controller = record['run']['controller']
                states = record['run']['states']
                axes[index, 1].plot([s['time_h'] for s in states], [s['soc'] for s in states],
                    color=COLORS[controller], linestyle=styles[controller], linewidth=1.6)
            for column, ylabel in enumerate(('Localized thermal-\ncomfort score', 'SOC')):
                axes_style(axes[index, column], 'Elapsed time (h)', ylabel,
                           f'{chr(97+2*index+column)}) {names[scenario]}')
                axes[index, column].set_ylim(-.02, 1.05)
            axes[index, 1].axhline(.05, color='black', linestyle='--', linewidth=.8)
        legend(fig, axes[0, 0])
        save(fig, 'controllers', True)

        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
        for record in fleece:
            controller_line(axes[0], record, 'score')
            controller = record['run']['controller']
            states = record['run']['states']
            axes[1].plot([s['time_h'] for s in states], [s['soc'] for s in states],
                         color=COLORS[controller], linestyle=styles[controller], linewidth=1.6)
        for index, ylabel in enumerate(('Localized thermal-\ncomfort score', 'SOC')):
            axes_style(axes[index], 'Elapsed time (h)', ylabel,
                       f'{chr(97+index)}) Qinling fixed fleece jumpsuit')
            axes[index].set_xlim(0, 14)
            axes[index].set_xticks(np.arange(0, 15, 2))
            axes[index].set_ylim(-.02, 1.05)
        axes[1].axhline(.05, color='black', linestyle='--', linewidth=.8)
        legend(fig, axes[0])
        save(fig, 'qinling_fleece_controllers', True)

        stressed = records('experiments/stress/s2/temperature_-5/')
        unperturbed = records('results/s2_')
        fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.8))
        x = np.arange(4)
        axes[0, 0].bar(x-.18, [r['evaluation']['metrics']['mean_service_full_mission'] for r in unperturbed],
                       width=.36, color=[COLORS[c] for c in controllers], alpha=.35, label='Unperturbed')
        axes[0, 0].bar(x+.18, [r['evaluation']['metrics']['mean_service_full_mission'] for r in stressed],
                       width=.36, color=[COLORS[c] for c in controllers], hatch='//', label='−5°C')
        axes[0, 0].axhline(.8, color='black', linestyle='--', linewidth=1, label='Mean reference 0.80')
        axes[0, 0].set_xticks(x, [c.upper() for c in controllers])
        axes[0, 0].set_ylim(0, 1.04)
        axes[0, 0].legend(frameon=False, fontsize=11, loc='lower left')
        for record in stressed:
            controller = record['run']['controller']
            controller_line(axes[0, 1], record, 'score')
            states = record['run']['states']
            time = np.array([s['time_h'] for s in states])
            axes[1, 0].plot(time, [s['soc'] for s in states], color=COLORS[controller],
                            linestyle=styles[controller], linewidth=1.6)
            hardware = record['hardware']
            area = np.asarray(hardware['regional_areas_m2'])
            deficit = [np.maximum(a['demand_w']*area/area.sum() -
                       np.asarray(hardware['rated_powers_w'])*hardware['efficiencies']*a['pwm'], 0).sum()
                       for a in record['run']['actions']]
            axes[1, 1].plot(time, np.r_[0., np.cumsum(deficit)*record['settings']['dt_hours']],
                            color=COLORS[controller], linestyle=styles[controller], linewidth=1.6)
        hardware = stressed[0]['hardware']
        settings = stressed[0]['settings']
        mandatory = hardware['base_power_w'] + hardware['equipment_available']*hardware['equipment_power_w']
        reserve = settings['soc_min'] + mandatory*(time[-1]-time)/hardware['capacity_wh']
        axes[1, 0].plot(time, reserve, color='black', linewidth=1.3, label='Mandatory reserve')
        axes[1, 0].axhline(settings['soc_min'], color='black', linestyle='--', linewidth=.8, label='Hard floor')
        axes[1, 0].axhline(settings['soc_safe'], color='black', linestyle=':', linewidth=.8, label='Soft preference')
        axes[1, 0].legend(frameon=False, fontsize=11)
        axes[0, 1].set_ylim(-.02, 1.05)
        for ax, xlabel, ylabel, title in (
            (axes[0, 0], 'Controller', 'Mean localized\nthermal-comfort score', 'a) Unperturbed versus −5°C'),
            (axes[0, 1], 'Elapsed time (h)', 'Localized thermal-\ncomfort score', 'b) −5°C heating-demand fulfillment'),
            (axes[1, 0], 'Elapsed time (h)', 'SOC', 'c) Common reserve boundary'),
            (axes[1, 1], 'Elapsed time (h)', 'Unmet regional heat (Wh)', 'd) Cumulative unmet heat')):
            axes_style(ax, xlabel, ylabel, title)
        legend(fig, axes[0, 1])
        save(fig, 'temperature_perturbation', True)

        fig, axes = plt.subplots(2, 3, figsize=(8.8, 6.5))
        order = ['baseline', 'temperature_-2', 'temperature_-5', 'temperature_-10',
                 'eta_0.75', 'eta_0.5', 'pv_0.75', 'pv_0.5', 'pv_0']
        short = ['Base', '−2°C', '−5°C', '−10°C', 'η ×0.75', 'η ×0.5', 'PV ×0.75', 'PV ×0.5', 'PV ×0']
        for index, scenario in enumerate((1, 2, 3)):
            rows = [r for r in evidence['rows'] if r['batch'] == 'stress' and r['scenario_type'] == scenario]
            arms = [arm for arm in order if any(r['case'].endswith('/'+arm) for r in rows)]
            for controller in controllers:
                values = [next(r['mean_service_full_mission'] for r in rows
                          if r['controller'] == controller and r['case'].endswith('/'+arm)) for arm in arms]
                axes[0, index].plot(range(len(arms)), values, marker='o', markersize=3,
                                    color=COLORS[controller], linestyle=styles[controller], label=controller.upper())
            axes[0, index].set_xticks(range(len(arms)), [short[order.index(a)] for a in arms], rotation=55)
            axes[0, index].set_ylim(0, 1.05)
            axes_style(axes[0, index], 'Perturbation condition', 'Mean localized\nthermal-comfort score', f'{chr(97+index)}) {names[scenario]}')
        for index, (key, ylabel) in enumerate((
                ('mean_service_full_mission', 'Mean localized\nthermal-comfort score'),
                ('load_energy_wh', 'Electrical load energy (Wh)'), ('terminal_soc', 'Terminal SOC'))):
            samples = [[r[key] for r in evidence['rows'] if r['batch']=='formal_mc' and
                        r['controller']==c and r['completed']] for c in controllers]
            boxes = axes[1, index].boxplot(samples, tick_labels=[c.upper() for c in controllers],
                                           patch_artist=True, showfliers=False, whis=(2.5, 97.5))
            for patch, controller in zip(boxes['boxes'], controllers):
                patch.set_facecolor(COLORS[controller])
                patch.set_alpha(.55)
            axes_style(axes[1, index], 'Controller', ylabel, f'{chr(100+index)}) 1000 paired samples')
        legend(fig, axes[0, 0])
        save(fig, 'robustness', True)

        record = next(r for r in stressed if r['run']['controller'] == 'rhc')
        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
        weights = np.asarray(record['hardware']['weights'])
        for indices, name, color in zip(([0,1,2,3], [6,7,10,11], [4,5,8,9]),
                                         ('Core', 'Distal', 'Other'), list(COLORS.values())[:3]):
            values = [float(np.dot(weights[indices], np.asarray(a['regional_fulfillment'])[indices]) /
                            weights[indices].sum()) for a in record['run']['actions']]
            interval(axes[0], record, values, color=color, label=name, linewidth=1.6)
        interval(axes[1], record, record['actual']['demand_w'], color='black', linestyle='--', label='Required heating power')
        supplied = [sum(np.asarray(record['hardware']['rated_powers_w'])*
                         record['hardware']['efficiencies']*a['pwm']) for a in record['run']['actions']]
        interval(axes[1], record, supplied, color=COLORS['rhc'], label='Effective heating power')
        axes_style(axes[0], 'Elapsed time (h)', 'Weighted group fulfillment', 'a) RHC regional priorities')
        axes_style(axes[1], 'Elapsed time (h)', 'Heating power (W)', 'b) −5°C demand and supply')
        axes[0].set_ylim(-.02, 1.05)
        for ax in axes:
            ax.legend(frameon=False, fontsize=11)
        save(fig, 'regional_allocation')

        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
        factors = (('price_factor', 'Rental price'), ('utilization_factor', 'Utilization'),
                   ('maintenance_factor', 'Maintenance'), ('lifetime_years', 'Lifetime'))
        for ax, case, name in zip(axes, evidence['applications']['economics'], ('Harbin', 'Muztagh Ata')):
            for index, (factor, _) in enumerate(factors):
                rows = sorted((r for r in case['joint_sensitivity'] if all(
                    r[k] == case['baseline'][k] for k, _ in factors if k != factor)),
                    key=lambda r: r[factor])
                low, high = rows[0]['npv_usd'], rows[-1]['npv_usd']
                ax.hlines(index, min(low, high), max(low, high), color='gray', linewidth=2)
                ax.plot(low, index, 's', color=COLORS['rbc'], markersize=5,
                        label='Lower input' if index == 0 else None)
                ax.plot(high, index, 'o', color=COLORS['pid'], markersize=5,
                        label='Higher input' if index == 0 else None)
            ax.axvline(0, color='black', linewidth=.8)
            ax.axvline(case['baseline']['npv_usd'], color=COLORS['rhc'], linestyle=':', label='Baseline')
            ax.set_yticks(range(len(factors)), labels=[label for _, label in factors])
            ax.set_ylim(-.5, 3.5)
            axes_style(ax, 'NPV (USD)', '', name)
        legend(fig, axes[0], 3)
        save(fig, 'economics', True)

        fig, axes = plt.subplots(2, 3, figsize=(8.8, 7.8), sharey='col')
        states = (25, 20, 18, 15)
        fleece_rows = {
            controller: [next(r for r in evidence['fixed_fleece']['building_states']
                              if r['controller'] == controller and r['setpoint_c'] == state)
                         for state in states]
            for controller in controllers
        }
        down_rows = [next(r for r in evidence['applications']['building_states']
                          if r['arm'] == 'baseline' and r['setpoint_c'] == state)
                     for state in states]
        # Bind the configuration comparison to RHC, including its accounting export.
        down_metrics = [trajectories[f'experiments/indoor/setpoint_{state}/rhc/run.json']
                        ['evaluation']['metrics'] for state in states]
        for row, metrics in zip(down_rows, down_metrics):
            for key in ('mean_service_full_mission', 'body_heating_energy_wh', 'load_energy_wh'):
                if not np.isclose(row['mission_metrics'][key], metrics[key], rtol=0, atol=1e-8):
                    raise ValueError('Selected-down accounting must match the RHC trajectory.')
        configuration_series = (
            ('Selected down top — RHC', down_rows, down_metrics, 'black', ':', 'x', 5),
            ('Fixed fleece jumpsuit — RHC', fleece_rows['rhc'],
             [r['mission_metrics'] for r in fleece_rows['rhc']], COLORS['rhc'], '-', 'o', 7),
        )
        for label, rows, metrics, color, style, marker, size in configuration_series:
            series = ([m['mean_service_full_mission'] for m in metrics],
                      [m['body_heating_energy_wh'] for m in metrics],
                      [r['saving_percent'] for r in rows])
            for ax, values in zip(axes[0], series):
                ax.plot(states, values, color=color, linestyle=style, marker=marker,
                        markersize=size, markerfacecolor='none', label=label)
        # Nested open markers expose coincident results without shifting their coordinates.
        markers = dict(zip(controllers, ('s', 'v', 'o', 'D')))
        sizes = dict(zip(controllers, (8, 6, 4, 5)))
        for controller in controllers:
            rows = fleece_rows[controller]
            series = ([r['mission_metrics']['mean_service_full_mission'] for r in rows],
                      [r['mission_metrics']['body_heating_energy_wh'] for r in rows],
                      [r['saving_percent'] for r in rows])
            for ax, values in zip(axes[1], series):
                ax.plot(states, values, color=COLORS[controller], linestyle=styles[controller],
                        marker=markers[controller], markersize=sizes[controller],
                        markerfacecolor='none', label=controller.upper())
        titles = ('Heating-demand fulfillment', 'Active heating', '24-h accounting')
        ylabels = ('Mean localized\nthermal-comfort score',
                   'Body-heating energy (Wh)', 'Net building saving (%)')
        for row_index, row_axes in enumerate(axes):
            for column, (ax, ylabel, title) in enumerate(zip(row_axes, ylabels, titles)):
                panel = chr(ord('a') + row_index * 3 + column)
                axes_style(ax, 'Indoor state (°C)', ylabel, f'{panel}) {title}')
                ax.set_xticks(states)
                ax.invert_xaxis()
            row_axes[0].set_ylim(0, 1.05)
            row_axes[1].set_ylim(-15, 420)
            handles, labels = row_axes[0].get_legend_handles_labels()
            row_axes[1].legend(handles, labels, loc='lower center',
                               bbox_to_anchor=(.5, 1.16), ncol=2 if row_index == 0 else 4,
                               frameon=False, fontsize=11)
        for row_axes, heading in zip(axes, (
                'Configuration comparison under RHC',
                'Controller comparison on the fixed fleece jumpsuit')):
            row_axes[1].text(.5, 1.47, heading, transform=row_axes[1].transAxes,
                            ha='center', va='bottom', fontsize=13, fontweight='bold')
        save(fig, 'indoor')

        fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.4))
        sizes = [r['garments'] for r in evidence['catalogue_scaling']]
        for key, label, color in (('candidate_count','Candidates',COLORS['rbc']),
                                  ('admissible_count','Admissible',COLORS['pid']),
                                  ('pareto_count','Pareto',COLORS['rhc'])):
            axes[0].plot(sizes, [r[key] for r in evidence['catalogue_scaling']], marker='o', color=color, label=label)
        axes[0].set_yscale('log')
        axes[0].set_yticks([2, 10, 100, 1000], labels=['2', '10', '100', '1000'])
        axes[0].legend(frameon=False, fontsize=11)
        axes[1].plot(sizes, [1000*r['elapsed_seconds'] for r in evidence['catalogue_scaling']],
                      marker='o', color=COLORS['rbc'])
        axes_style(axes[0], 'Garment entries', 'Count (log scale)', 'a) Finite search space')
        axes_style(axes[1], 'Garment entries', 'Observed search time (ms)', 'b) Catalogue scaling')
        for ax in axes:
            ax.set_xticks(sizes)
        save(fig, 'catalogue_scaling')
    (output / 'figure_sources.json').write_text(json.dumps({
        'schema': evidence['schema'], 'source_files_sha256': evidence['source_files_sha256'],
        'artifacts': [Path(path).name for path in artifacts],
        'optimizer_calls': 0, 'style': RC}, indent=2) + '\n')
    return artifacts


def plot_equipment(directory, output_dir):
    """Plot recorded equipment switching and the associated battery response."""
    records = {c: json.loads((directory / c / 'run.json').read_text()) for c in COLORS}
    if len({r['forcing_sha256'] for r in records.values()}) != 1:
        raise ValueError('Equipment comparison requires identical forcing.')
    output_dir.mkdir(parents=True, exist_ok=False)
    with plt.rc_context({**RC, 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        styles = {'rbc': '-', 'pid': '--', 'tacs': ':', 'rhc': '-.'}
        first = records['rhc']['run']
        times = [a['time_h'] for a in first['actions']] + [first['states'][-1]['time_h']]
        air = [a['air_c'] for a in first['actions']]
        axes[0].step(times, air + air[-1:], where='post', color=COLORS['rbc'], linewidth=1.8)
        axes[0].axhline(-20, color='black', linestyle='--', linewidth=1, label='Switching threshold')
        axes[0].legend(frameon=False, fontsize=9)
        for controller, record in records.items():
            run = record['run']
            power = [a['equipment_on'] * record['hardware']['equipment_power_w'] for a in run['actions']]
            axes[1].step(times, power + power[-1:], where='post',
                         color=COLORS[controller], linestyle=styles[controller], linewidth=1.8)
            axes[2].plot([s['time_h'] for s in run['states']], [s['soc'] for s in run['states']],
                         color=COLORS[controller], linestyle=styles[controller],
                         linewidth=1.8, label=controller.upper())
        for ax, ylabel, title in zip(axes, ('Air temperature (°C)', 'Equipment power (W)', 'SOC'),
                                     ('a) Temperature input', 'b) Heater activation', 'c) Stored energy')):
            _style(ax, 'Elapsed time (h)', ylabel, title)
            ax.set_xlim(0, 2)
            ax.set_xticks([0, 0.5, 1, 1.5, 2])
        axes[1].set_ylim(-1, 21)
        axes[2].legend(frameon=False, fontsize=9)
        fig.tight_layout()
        artifacts = []
        for extension in ('pdf', 'png'):
            target = output_dir / f'equipment_activation.{extension}'
            fig.savefig(target, dpi=300, bbox_inches='tight')
            artifacts.append(str(target))
        plt.close(fig)
    return artifacts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--data-dir', type=Path)
    inputs.add_argument('--equipment-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.equipment_dir:
        artifacts = plot_equipment(args.equipment_dir, args.output_dir)
    else:
        evidence = json.loads((args.data_dir / 'evidence.json').read_text())
        renderer = plot_study if evidence.get('schema') == 'bgems-study-results-v1' else plot_trajectories
        artifacts = renderer(evidence, args.output_dir)
    for path in artifacts:
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
