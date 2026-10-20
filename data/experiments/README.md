# Benchtop Experiment Documentation

## Purpose

These benchtop experiments record local sensor temperature-rise curves for a prototype heating pad under four covering and placement conditions. The analysis describes the first-order temperature response and differences between the recorded operating points.

## Principle

An iButton (DS1922L) records the local temperature at the heating-pad contact in a sandwich assembly. The fitted parameters are the initial sensor temperature $T_{0,\mathrm{fit}}$, the asymptotic rise above that initial temperature $\Delta T_{\mathrm{fit}}$, and the time constant $\tau$. Ambient temperature was recorded independently as 25.5°C.

The recorded supply power for each condition is used to calculate the descriptive apparent response quantity $G_{\mathrm{app}}=P_{in}/\Delta T_{\mathrm{fit}}$. A declared normalization reference then defines the dimensionless index $I_{\mathrm{ins}}=G_{ref}/G_{\mathrm{app}}$. The interpretation of these descriptive quantities is specified below.

### Equipment

- Power supply: WANPTEK Programmable DC Power Supply, TPS3010.
- Ambient temperature/humidity instrument: Miaoxin Temperature & Humidity Sensor, TH22R-EX.
- Heating-pad dimensions: 22 cm × 10.5 cm; **single-face area: 0.0231 m²**, consistent with `PAD_AREA_M2` in `scripts/pad_temperature_response.py`.
- Temperature logger: iButton, DS1922L.

## Experimental Procedure — SOP for the iButton Benchtop Test

1. Assemble the sandwich intended to approximate a wearing configuration. The physical stack, from bottom to top, is:
   - Tabletop at room temperature, also acting as the bottom heat sink.
   - Fabric layer representing clothing thermal resistance $R_{clothing}$: a layer of down-garment material or fleece garment fabric.
   - Heating layer: the prototype heating pad (22 cm × 10.5 cm), placed at the specified location on the target garment, either the center or edge.
   - Sensor layer: attach the iButton firmly to the center of the heating pad using medical tape to record local contact temperature.
   - Covering fabric layer: place the same garment material above the iButton, fully covering both the heating pad and sensor.
   - Book used as a compression load: place a book on top to maintain contact between layers and remove air gaps. Compression affects contact thermal resistance and should be kept consistent across conditions.

2. Collect transient data:
   - Set the iButton sampling interval according to the four-condition design below.
   - Leave the assembly at room temperature until the iButton reading stabilizes.
   - Energize the heating pad and continue recording until the reading reaches a relatively flat temperature plateau. Nominal durations are listed below.
   - Switch off the power. Post-switch-off cooling samples are retained in the original records but excluded from the heating-response analysis, as detailed under “Collected Data.”
   - Export time and temperature from the iButton to CSV.

3. Process the records and extract response quantities:
   - Fit the existing heating segment with the empirical first-order response:

     $$T(t)=T_{0,\mathrm{fit}}+\Delta T_{\mathrm{fit}}(1-e^{-t/\tau}).$$

     Fit $T_{0,\mathrm{fit}}$ as a free parameter. The fitted plateau is $T_{\infty,\mathrm{fit}}=T_{0,\mathrm{fit}}+\Delta T_{\mathrm{fit}}$. Report $R^2$, RMSE (°C), and conditional fit standard errors for each of the four records.
   - Report both temperature differences with their references: $\Delta T_{\mathrm{fit}}$ is the fitted rise above initial sensor temperature; $T_{\infty,\mathrm{fit}}-25.5^\circ\mathrm{C}$ is the plateau above recorded ambient temperature.
   - Calculate $G_{\mathrm{app}}=P_{in}/\Delta T_{\mathrm{fit}}$ and $I_{\mathrm{ins}}=G_{ref}/G_{\mathrm{app}}$. The declared references are $h_{ref}=10$ W/(m²·K) and $A=0.0231$ m², giving $G_{ref}=h_{ref}A=0.231$ W/K. Values above one indicate $G_{\mathrm{app}}<G_{ref}$.

## Four-Condition Experimental Design

Location: Information Building, Tsinghua campus, Xili University Town, Nanshan District, Shenzhen, Guangdong, China. Recorded ambient conditions: **25.5°C and 77.5% RH**.

First attach the heating layer and sensor layer to one another as specified in the SOP.

| Condition | Covering and placement | Sampling interval | Recorded nominal duration | Supply power / voltage |
|---|---|---|---|---|
| 1 — Air | Pad/sensor assembly exposed to air | 10 s | 40 min | 4.0 W / 4.3 V |
| 2 — Fleece edge | Fleece covering; pad at the lower hem edge of the fleece garment | 60 s | 46 min heating segment; two additional cooling rows in the original record | 5.0 W / 4.8 V |
| 3 — Fleece center | Fleece covering; pad at the center of the garment back | 3 s | 50 min | 5.8 W / 5.2 V |
| 4 — Down center | Down-garment covering; pad at the center of the garment back | 3 s | 70 min | 3.2 W / 3.9 V |

Condition 2 has a temperature quantization step of 0.5°C, compared with 0.1°C for the other three conditions. This reflects different export-resolution settings and warrants conservative statements about fit precision. The approximate durations above describe the experiments; CSV timestamps give the recorded windows.

## Collected Data

- `0521_pad_air.csv`: Condition 1.
- `0521_pad_fleece_edge.csv`: Condition 2.
- `0521_pad_fleece_center.csv`: Condition 3.
- `0522_pad_down_center.csv`: Condition 4.

## Data Processing and Figure Generation

- `scripts/pad_temperature_response.py` produces first-order fit parameters for all four conditions, JSON containing the descriptive $G_{\mathrm{app}}/I_{\mathrm{ins}}$ quantities, and a four-panel PNG showing observations, fitted curves, and response quantities.
- Default outputs are `output/experiments/pad_temperature_response.{json,png}`: numerical fit results and a four-panel figure.
- JSON fit fields are `T0_fit`, `T0_fit_stderr`, `delta_T_fit`, `delta_T_fit_stderr`, `T_inf_fit`, `tau`, `tau_stderr`, `r_squared`, and `rmse`. The experimental environment is recorded separately as `measured_ambient_c=25.5`; `plateau_above_ambient_k` is the fitted plateau above that ambient temperature. Derived fields are `g_app_w_per_k`, `i_ins`, and `i_ins_sensitivity`. Each condition's power, pad area, and normalization constants are also recorded.
- Figure error bars show conditional fit standard errors from the `curve_fit` covariance. Each condition contributes one experimental record.
- Run from the repository root:

  ```bash
  python3 -m scripts.pad_temperature_response
  ```

  Use `--data-dir` and `--output-dir` to specify input and output directories explicitly.

## Analysis Settings and Interpretation

The `curve_fit` initial guess is `[T.min(), T.max()-T.min(), 300]`, with parameter order `[T0_fit, delta_T_fit, tau]`, bounds `[0, 0, 10]` to `[60, 60, 10000]`, and `maxfev=10000`. Ambient temperature is environmental metadata; the initial sensor temperature is fitted. The time constant describes the local measured response. Conditional fit standard errors represent model-fit uncertainty for one record per condition; replication, sensor attachment, temperature quantization and supply-power uncertainty are separate sources of uncertainty.

The condition-specific operating powers and voltages in the table describe constant-power supply operation. Algebraic $V^2/P$ values are approximately 4.62/4.61/4.66/4.75 Ω. The available electrical measurements are operating power and voltage; catalogue heater specifications are listed separately in the model configuration.

The descriptive $G_{\mathrm{app}}$ uses the fitted rise above initial sensor temperature. The reference $G_{ref}=0.231$ W/K uses $h_{ref}=10$ W/(m²·K) and the pad area. The script reports sensitivity at $h_{ref}=5/10/15$; the indices scale linearly with this reference, with values above one corresponding to $G_{\mathrm{app}}<G_{ref}$.

**Scope.** The four CSVs and this description document local temperature responses at the specified powers and placements. The fits provide response curves and time constants; independent heat-flow measurements are needed to calibrate body-side heat delivery. Regional model factors $\eta_i$ are adopted inputs in the configuration. Sensor attachment, temperature quantization and supply-power uncertainty should be considered when comparing the four operating points.
