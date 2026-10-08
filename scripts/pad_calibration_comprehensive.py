'''
Author: Hong ZHANG
Affiliation: Tsinghua University
Copyright (c) 2026 by Tsinghua University, All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");

Purpose: Fit the four-condition benchtop heating-pad temperature records and
report descriptive apparent responses, not heat-delivery efficiencies.
Data, per-condition powers and the declared normalization are documented in
data/experiments/README.md.

T0_fit is a free fitted sensor-temperature intercept, distinct from the
independently measured ambient temperature. delta_T_fit is the predicted
plateau rise above that intercept, not above ambient:
    G_app = P_in / delta_T_fit
    I_ins = G_ref / G_app, with G_ref = H_REF * PAD_AREA = 0.231 W/K.
H_REF = 10 W/(m^2 K) is a declared normalization, not an identified heat-loss
coefficient. G_app is an apparent response ratio, not a measurement of total
conductance. I_ins > 1 only means G_app < G_ref; it does not identify physical
losses or body-side efficiency. The descriptive index is not clamped.
'''
import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


# --- Recorded metadata and declared normalization (see data/experiments/README.md) ---

# Per-condition measured supply powers in W (TPS3010 constant-power mode).
CONDITION_POWER_W = {
    'air': 4.0,           # 4.0 W / 4.3 V
    'fleece_edge': 5.0,   # 5.0 W / 4.8 V
    'fleece_center': 5.8, # 5.8 W / 5.2 V
    'down': 3.2,          # 3.2 W / 3.9 V
}

# Single-face area of the prototype pad, 22 cm x 10.5 cm (README equipment list).
PAD_AREA_M2 = 0.0231

# Independently recorded room temperature; never imposed on the fitted intercept.
MEASURED_AMBIENT_C = 25.5

# Declared normalization coefficient, not measured or inferred from these records.
H_REF_W_PER_M2K = 10.0

# Alternative normalizations, not a confidence interval (I_ins scales linearly).
H_REF_SENSITIVITY = (5.0, 10.0, 15.0)

G_REF_W_PER_K = H_REF_W_PER_M2K * PAD_AREA_M2  # = 0.231 W/K


CONDITIONS = {
    'air': {
        'name': 'Air Exposure',
        'short_label': 'Air',
        'csv': '0521_pad_air.csv',
        'description': 'iButton directly exposed to air',
        'color': '#E74C3C',
        'marker': 'o',
    },
    'fleece_edge': {
        'name': 'Fleece - Edge',
        'short_label': 'Fleece Edge',
        'csv': '0521_pad_fleece_edge.csv',
        'description': 'Covered with fleece, heating pad near edge (like clothing hem)',
        'color': '#F39C12',
        'marker': 's',
    },
    'fleece_center': {
        'name': 'Fleece - Center',
        'short_label': 'Fleece Center',
        'csv': '0521_pad_fleece_center.csv',
        'description': 'Covered with fleece, heating pad in center (like clothing back)',
        'color': '#3498DB',
        'marker': '^',
    },
    'down': {
        'name': 'Down - Center',
        'short_label': 'Down Center',
        'csv': '0522_pad_down_center.csv',
        'description': 'Covered with down jacket, heating pad in center (like clothing back)',
        'color': '#9B59B6',
        'marker': 'D',
    },
}


def first_order_response(t, T0_fit, delta_T_fit, tau):
    """Empirical sensor response with a free intercept and rise above it."""
    return T0_fit + delta_T_fit * (1 - np.exp(-t / tau))


def load_csv_data(csv_path):
    """Load a benchtop CSV and return (dataframe, dropped_row_count).

    Unparseable time rows are dropped but counted and reported; a silent
    mismatch between dropped rows and the parsed index is made explicit.
    """
    df = pd.read_csv(csv_path)
    time_col, temp_col = df.columns[0], df.columns[1]

    datetimes = []
    good_index = []
    dropped = 0
    for idx, ts in enumerate(df[time_col].astype(str).values):
        try:
            if ':' in ts and '/' not in ts:
                # Format: "17:40:01" - anchor to a dummy date.
                dt = datetime.strptime(ts, '%H:%M:%S').replace(year=2026, month=1, day=1)
            elif '/' in ts:
                # Format: "5/21/2026 21:48[:ss]" (minute or second precision).
                try:
                    dt = datetime.strptime(ts, '%m/%d/%Y %H:%M:%S')
                except ValueError:
                    dt = datetime.strptime(ts, '%m/%d/%Y %H:%M')
            else:
                raise ValueError(f"Unknown time format: {ts}")
        except ValueError:
            dropped += 1
            continue
        datetimes.append(dt)
        good_index.append(idx)

    parsed = df.iloc[good_index].copy()
    parsed['Time'] = datetimes
    parsed['Temperature'] = parsed[temp_col].astype(float)
    parsed['Time_elapsed'] = (parsed['Time'] - parsed['Time'].iloc[0]).dt.total_seconds()
    return parsed, dropped


def fit_model(df):
    """Fit one record without imposing the independently measured ambient.

    Standard errors are conditional curve_fit covariance estimates, not
    variability across independent experiments.
    """
    t = df['Time_elapsed'].values
    T = df['Temperature'].values

    p0 = [T.min(), T.max() - T.min(), 300]  # tau guess 5 minutes
    bounds = ([0, 0, 10], [60, 60, 10000])
    popt, pcov = curve_fit(first_order_response, t, T, p0=p0, bounds=bounds, maxfev=10000)
    T0_fit, delta_T_fit, tau = popt

    T_pred = first_order_response(t, *popt)
    ss_res = np.sum((T - T_pred) ** 2)
    ss_tot = np.sum((T - np.mean(T)) ** 2)
    perr = np.sqrt(np.diag(pcov))

    return {
        'T0_fit': T0_fit,
        'delta_T_fit': delta_T_fit,
        'tau': tau,
        'T_inf_fit': T0_fit + delta_T_fit,
        'r_squared': 1 - (ss_res / ss_tot),
        'rmse': float(np.sqrt(np.mean((T - T_pred) ** 2))),
        'T0_fit_stderr': perr[0],
        'delta_T_fit_stderr': perr[1],
        'tau_stderr': perr[2],
        'predictions': T_pred,
    }


def apparent_response_indices(results, power_w):
    """Return (G_app, I_ins, {h_ref: I_ins_h}) as descriptive response ratios.

    G_app = P_in / delta_T_fit has units W/K but is not an identified total
    conductance. I_ins = G_ref / G_app is a declared normalization, not a
    heat-delivery efficiency; values above one carry no heat-loss inference.
    """
    g_app = power_w / results['delta_T_fit']
    i_ins = G_REF_W_PER_K / g_app
    sensitivity = {h: (h * PAD_AREA_M2) / g_app for h in H_REF_SENSITIVITY}
    return g_app, i_ins, sensitivity


def add_bar_labels(ax, bars, values, errors=None, fmt='{:.2f}', offset=0.0, suffix=''):
    """Add value labels to bar charts (publication style)."""
    if errors is None:
        errors = [0.0] * len(values)
    for bar, value, err in zip(bars, values, errors):
        ax.annotate(
            fmt.format(value) + suffix,
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + err),
            xytext=(0, 8),
            textcoords='offset points',
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
            clip_on=False,
            zorder=6,
            bbox=dict(boxstyle='round,pad=0.18', facecolor='white',
                      edgecolor='#666666', linewidth=0.6),
        )


def save_comparison_plot(all_results, output_path, bar_width=0.5):
    """Plot observed/fitted sensor responses and descriptive fit quantities."""
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
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))

        # Panel 1: transient curves.
        ax1 = axes[0, 0]
        for data in all_results.values():
            df, results, cond = data['df'], data['results'], data['cond']
            t = df['Time_elapsed'].values
            ax1.scatter(t, df['Temperature'].values, alpha=0.35, s=15,
                        color=cond['color'], marker=cond['marker'])
            t_smooth = np.linspace(0, t.max(), 300)
            ax1.plot(t_smooth,
                     first_order_response(t_smooth, results['T0_fit'],
                                          results['delta_T_fit'], results['tau']),
                     '-', linewidth=2.5, color=cond['color'],
                     label=f"{cond['short_label']} (tau={results['tau']:.0f} s)")
        ax1.axhline(MEASURED_AMBIENT_C, color='black', linestyle='--', linewidth=1.2,
                    label=f'Measured ambient ({MEASURED_AMBIENT_C:.1f} °C)')
        _style_axes(ax1, 'Time (seconds)', 'Sensor temperature (°C)',
                    'Observed and Fitted Sensor Responses')
        ax1.legend(fontsize=12, loc='lower right', bbox_to_anchor=(1.0, 0.06),
                   frameon=False, prop={'weight': 'bold'})

        keys = list(all_results.keys())
        x_pos = np.arange(len(keys))
        colors = [all_results[k]['cond']['color'] for k in keys]
        labels_short = [all_results[k]['cond']['short_label'] for k in keys]

        def style_bar(ax, title, ylabel):
            _style_axes(ax, 'Condition', ylabel, title)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels_short, rotation=15, ha='right', fontsize=12)
            ax.tick_params(axis='x', which='both', length=0, pad=6)

        # Panel 2: tau.
        ax2 = axes[0, 1]
        taus = [all_results[k]['results']['tau'] for k in keys]
        taus_err = [all_results[k]['results']['tau_stderr'] for k in keys]
        bars = ax2.bar(x_pos, taus, yerr=taus_err, capsize=5, width=bar_width,
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        style_bar(ax2, 'Time Constant Comparison', 'τ (tau) [seconds]')
        add_bar_labels(ax2, bars, taus, errors=taus_err, fmt='{:.0f}', suffix='s')
        ax2.set_ylim(0, max(v + e for v, e in zip(taus, taus_err)) * 1.25)

        # Panel 3: descriptive I_ins (not an efficiency; no clamping).
        ax3 = axes[1, 0]
        indices = [all_results[k]['i_ins'] for k in keys]
        bars = ax3.bar(x_pos, indices, width=bar_width, color=colors, alpha=0.8,
                       edgecolor='black', linewidth=1.5)
        style_bar(ax3, 'Descriptive Insulation Index', 'I_ins = G_ref / G_app')
        ax3.set_ylim(0, max(indices) * 1.25)
        add_bar_labels(ax3, bars, indices, fmt='{:.2f}')

        # Panel 4: fitted plateau rise above T0_fit, not above measured ambient.
        ax4 = axes[1, 1]
        dts = [all_results[k]['results']['delta_T_fit'] for k in keys]
        dts_err = [all_results[k]['results']['delta_T_fit_stderr'] for k in keys]
        bars = ax4.bar(x_pos, dts, yerr=dts_err, capsize=5, width=bar_width,
                       color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        style_bar(ax4, 'Fitted Rise Above Initial Sensor Temperature', 'ΔT_fit (K)')
        add_bar_labels(ax4, bars, dts, errors=dts_err, fmt='{:.1f}', suffix=' K')
        ax4.set_ylim(0, max(v + e for v, e in zip(dts, dts_err)) * 1.25)

        fig.text(0.5, 0.015,
                 'One record per condition; error bars: conditional fit SE, not replicate variability.\n'
                 'I_ins uses declared h_ref = 10 W/(m² K); it is not a heat-delivery efficiency.',
                 ha='center', fontsize=11, fontweight='bold')
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)


def _style_axes(ax, xlabel, ylabel, title):
    """Apply the shared academic style to one axes."""
    ax.set_xlabel(xlabel, fontsize=14, fontweight='bold', color='black')
    ax.set_ylabel(ylabel, fontsize=14, fontweight='bold', color='black')
    ax.set_title(title, fontsize=16, fontweight='bold', color='black')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.tick_params(axis='both', direction='in', width=1.5, length=5, labelsize=12, colors='black')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        label.set_color('black')




def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path,
                        default=Path(__file__).parents[1] / 'data' / 'experiments')
    parser.add_argument('--output-dir', type=Path, default=Path('output/experiments'))
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("HEATING PAD SENSOR RESPONSES - COMPREHENSIVE 4-CONDITION ANALYSIS")
    print(f"Measured ambient: {MEASURED_AMBIENT_C:.1f} °C (independent of fitted T0_fit)")
    print(f"Declared normalization: G_ref = h_ref*A = "
          f"{H_REF_W_PER_M2K} * {PAD_AREA_M2} = {G_REF_W_PER_K:.3f} W/K")
    print("=" * 80)

    all_results = {}
    for key, cond in CONDITIONS.items():
        csv_path = args.data_dir / cond['csv']
        print(f"\n{'─' * 60}\nCondition: {cond['name']}  (P_in = {CONDITION_POWER_W[key]} W)\n{'─' * 60}")
        try:
            df, dropped = load_csv_data(csv_path)
            print(f"  - Data points: {len(df)}  (dropped unparseable rows: {dropped})")
            print(f"  - Duration: {df['Time_elapsed'].max() / 60:.1f} minutes")

            results = fit_model(df)
            g_app, i_ins, sensitivity = apparent_response_indices(results, CONDITION_POWER_W[key])

            print(f"  T0_fit={results['T0_fit']:.3f}±{results['T0_fit_stderr']:.3f} °C  "
                  f"delta_T_fit={results['delta_T_fit']:.3f}±{results['delta_T_fit_stderr']:.3f} K  "
                  f"tau={results['tau']:.1f}±{results['tau_stderr']:.1f}s  "
                  f"R2={results['r_squared']:.4f}  RMSE={results['rmse']:.3f} °C")
            print(f"  T_inf_fit={results['T_inf_fit']:.3f} °C  "
                  f"plateau above measured ambient={results['T_inf_fit'] - MEASURED_AMBIENT_C:.3f} K")
            print(f"  G_app={g_app:.3f} W/K   I_ins={i_ins:.3f} "
                  "(descriptive ratios, not measured conductance or efficiency)")

            all_results[key] = {'df': df, 'results': results, 'g_app_w_per_k': g_app,
                                'i_ins': i_ins, 'i_ins_sensitivity': sensitivity, 'cond': cond}
        except Exception as error:  # noqa: BLE001 - report and continue other conditions
            print(f"ERROR analyzing {key}: {error}")
            import traceback
            traceback.print_exc()

    if not all_results:
        raise SystemExit("No condition analyzed successfully; aborting output.")

    keys = list(all_results.keys())

    # Plot (single publication style).
    plot_path = args.output_dir / 'pad_calibration_4_conditions.png'
    save_comparison_plot(all_results, plot_path)
    print(f"\nPlot saved to: {plot_path}")

    # JSON results.
    results_json = {}
    for key in keys:
        r = all_results[key]['results']
        results_json[key] = {
            'name': all_results[key]['cond']['name'],
            'power_w': CONDITION_POWER_W[key],
            'measured_ambient_c': MEASURED_AMBIENT_C,
            'T0_fit': round(r['T0_fit'], 3), 'T0_fit_stderr': round(r['T0_fit_stderr'], 3),
            'delta_T_fit': round(r['delta_T_fit'], 3),
            'delta_T_fit_stderr': round(r['delta_T_fit_stderr'], 3),
            'tau': round(r['tau'], 1), 'tau_stderr': round(r['tau_stderr'], 1),
            'T_inf_fit': round(r['T_inf_fit'], 3),
            'plateau_above_ambient_k': round(r['T_inf_fit'] - MEASURED_AMBIENT_C, 3),
            'r_squared': round(r['r_squared'], 4), 'rmse': round(r['rmse'], 3),
            'g_app_w_per_k': round(all_results[key]['g_app_w_per_k'], 4),
            'i_ins': round(all_results[key]['i_ins'], 4),
            'i_ins_sensitivity': {str(h): round(v, 4)
                                  for h, v in all_results[key]['i_ins_sensitivity'].items()},
            'h_ref_w_per_m2k': H_REF_W_PER_M2K, 'pad_area_m2': PAD_AREA_M2,
            'g_ref_w_per_k': round(G_REF_W_PER_K, 4),
        }
    json_path = args.output_dir / 'pad_calibration_4_conditions.json'
    json_path.write_text(json.dumps(results_json, indent=2) + "\n", encoding='utf-8')
    print(f"Results saved to: {json_path}")


    print(f"\n{'Condition':<16} {'T0_fit':<8} {'R2':<7} {'RMSE':<7} "
          f"{'tau(s)':<8} {'G_app':<7} {'I_ins':<7}")
    print("─" * 72)
    for key in keys:
        r = all_results[key]['results']
        print(f"{all_results[key]['cond']['name']:<16} {r['T0_fit']:<8.3f} "
              f"{r['r_squared']:<7.4f} {r['rmse']:<7.3f} {r['tau']:<8.0f} "
              f"{all_results[key]['g_app_w_per_k']:<7.3f} {all_results[key]['i_ins']:<7.3f}")
    print("T0_fit: °C; RMSE: °C; G_app: W/K; I_ins: dimensionless. "
          "Fit standard errors are conditional, not replicate variability.")
    print("\nANALYSIS COMPLETE")
    return all_results


if __name__ == '__main__':
    main()
