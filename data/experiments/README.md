# Benchtop Experiment Documentation

## Purpose

These benchtop experiments record local sensor temperature-rise curves for a prototype heating pad under four covering and placement conditions. The analysis describes the first-order temperature response and differences between the recorded operating points. The records contain no independent heat-flow measurements: they cannot calibrate the fraction of heat transferred to the body or sensor, or identify the regional effective heat-delivery coefficients $\eta_{\mathrm{model}}$ used in configuration and operation.

## Principle

An iButton (DS1922L) records the local temperature at the heating-pad contact in a sandwich assembly. The fitted parameters are the free initial temperature $T_{0,\mathrm{fit}}$, the asymptotic temperature rise relative to that initial temperature $\Delta T_{\mathrm{fit}}$, and the time constant $\tau$. The fitted intercept $T_{0,\mathrm{fit}}$ is not the independently recorded ambient temperature of 25.5°C.

The recorded supply power for each condition is used to calculate the descriptive apparent response quantity $G_{\mathrm{app}}=P_{in}/\Delta T_{\mathrm{fit}}$. A declared normalization reference then defines the dimensionless index $I_{\mathrm{ins}}=G_{ref}/G_{\mathrm{app}}$. The W/K unit does not imply that total heat-loss conductance has been identified. Likewise, $I_{\mathrm{ins}}$ is not a measured pad-to-sensor or body-side heat-delivery efficiency.

### Equipment

- Power supply: WANPTEK Programmable DC Power Supply, TPS3010.
- Ambient temperature/humidity instrument: Miaoxin Temperature & Humidity Sensor, TH22R-EX.
- Heating-pad dimensions: 22 cm × 10.5 cm; **single-face area: 0.0231 m²**, consistent with `PAD_AREA_M2` in `scripts/pad_calibration_comprehensive.py`.
- Temperature logger: iButton, DS1922L.

## Experimental Procedure — SOP for the iButton Benchtop Test

1. Assemble the sandwich intended to approximate a wearing configuration. The physical stack, from bottom to top, is:
   - Tabletop at room temperature, also acting as the bottom heat sink.
   - Fabric layer representing clothing thermal resistance $R_{clothing}$: a layer of down-garment material or fleece garment fabric.
   - Heating layer: the prototype heating pad (22 cm × 10.5 cm), placed at the specified location on the target garment, either the center or edge.
   - Sensor layer: attach the iButton firmly to the center of the heating pad using medical tape. It records local contact temperature, not heat flux.
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

     Fit $T_{0,\mathrm{fit}}$ freely rather than fixing it to the measured ambient temperature. The fitted asymptotic plateau is $T_{\infty,\mathrm{fit}}=T_{0,\mathrm{fit}}+\Delta T_{\mathrm{fit}}$. Report $R^2$, RMSE (°C), and conditional fit standard errors; do not interpret the number of samples as the number of independent experimental replicates.
   - Distinguish the two temperature differences: $\Delta T_{\mathrm{fit}}$ is the plateau rise above the fitted initial temperature, whereas $T_{\infty,\mathrm{fit}}-25.5^\circ\mathrm{C}$ is the fitted plateau above the recorded ambient temperature. Both are descriptive quantities derived from fitted values; they are not interchangeable as temperature differences driving heat loss to the environment.
   - Calculate $G_{\mathrm{app}}=P_{in}/\Delta T_{\mathrm{fit}}$ and $I_{\mathrm{ins}}=G_{ref}/G_{\mathrm{app}}$. The declared references are $h_{ref}=10$ W/(m²·K) and $A=0.0231$ m², giving $G_{ref}=h_{ref}A=0.231$ W/K. This normalization is not a physical heat-transfer coefficient identified from the four curves. $I_{\mathrm{ins}}>1$ means only that $G_{\mathrm{app}}<G_{ref}$; it does not establish that actual losses are lower than a bare-pad reference.

## Four-Condition Experimental Design

Location: Information Building, Tsinghua campus, Xili University Town, Nanshan District, Shenzhen, Guangdong, China. Recorded ambient conditions: **25.5°C and 77.5% RH**.

First attach the heating layer and sensor layer to one another as specified in the SOP.

| Condition | Covering and placement | Sampling interval | Recorded nominal duration | Supply power / voltage |
|---|---|---|---|---|
| 1 — Air | Pad/sensor assembly exposed to air | 10 s | 40 min | 4.0 W / 4.3 V |
| 2 — Fleece edge | Fleece covering; pad at the lower hem edge of the fleece garment | 60 s | 46 min heating segment; two additional cooling rows in the original record | 5.0 W / 4.8 V |
| 3 — Fleece center | Fleece covering; pad at the center of the garment back | 3 s | 50 min | 5.8 W / 5.2 V |
| 4 — Down center | Down-garment covering; pad at the center of the garment back | 3 s | 70 min | 3.2 W / 3.9 V |

Condition 2 has a temperature quantization step of 0.5°C, compared with 0.1°C for the other three conditions. This reflects different export-resolution settings and warrants conservative statements about fit precision. The nominal durations above reproduce the experimental description; the exact retained record windows are documented below.

## Collected Data

- `0521_pad_air.csv`: Condition 1.
- `0521_pad_fleece_edge.csv`: Condition 2.
- `0521_pad_fleece_center.csv`: Condition 3.
- `0522_pad_down_center.csv`: Condition 4.

## Data Processing and Figure Generation

- `scripts/pad_calibration_comprehensive.py` produces first-order fit parameters for all four conditions, JSON containing the descriptive $G_{\mathrm{app}}/I_{\mathrm{ins}}$ quantities, and a four-panel PNG showing observations, fitted curves, and response quantities.
- Default outputs are `output/experiments/pad_calibration_4_conditions.{json,png}`: numerical fit results and a four-panel figure. The response quantities do not constitute heat-delivery-efficiency calibration.
- JSON fit fields are `T0_fit`, `T0_fit_stderr`, `delta_T_fit`, `delta_T_fit_stderr`, `T_inf_fit`, `tau`, `tau_stderr`, `r_squared`, and `rmse`. The experimental environment is recorded separately as `measured_ambient_c=25.5`; `plateau_above_ambient_k` is the fitted plateau above that ambient temperature. Derived fields are `g_app_w_per_k`, `i_ins`, and `i_ins_sensitivity`. Each condition's power, pad area, and normalization constants are also recorded.
- Figure error bars are conditional standard errors from the `curve_fit` covariance, not between-experiment variability or total measurement uncertainty. There is only one record per condition; measured ambient remains distinct from fitted initial temperature.
- Run from the repository root:

  ```bash
  python3 -m scripts.pad_calibration_comprehensive
  ```

  Use `--data-dir` and `--output-dir` to specify input and output directories explicitly.

## Notes: Operating Points, Fitting, and Interpretation Limits

1. **Preserve the actual operating points.** Recorded supply power/voltage pairs are 4.0 W/4.3 V, 5.0 W/4.8 V, 5.8 W/5.2 V, and 3.2 W/3.9 V. The original experimental description states that the TPS3010 operated in constant-power mode. The analysis uses these condition-specific powers, rather than replacing all values with 4 W or treating different covering conditions as a power sweep under otherwise identical conditions. The temperature curves do not validate linearity over the full power range, material tolerance limits, or wearing safety.

2. **Fitting settings and interpretation.** The `curve_fit` initial guess is `[T.min(), T.max()-T.min(), 300]`, with parameter order `[T0_fit, delta_T_fit, tau]`, lower bounds `[0, 0, 10]`, upper bounds `[60, 60, 10000]`, and `maxfev=10000`. The independently measured 25.5°C is environmental metadata only, not a fitting constraint. The time constant $\tau$ is a diagnostic of the local record and is not an input to the system model. Fit standard errors are conditional on the model and record; they do not cover uncertainties in temperature quantization, sensor attachment, supply-power readings, or experimental replication.

3. **The apparent response is not measured total conductance.** The reference for $\Delta T_{\mathrm{fit}}=T_{\infty,\mathrm{fit}}-T_{0,\mathrm{fit}}$ is the fitted initial temperature, not ambient temperature. Accordingly, $G_{\mathrm{app}}=P_{in}/\Delta T_{\mathrm{fit}}$ reports only the ratio of power to fitted temperature rise for that record. Even using the plateau-to-ambient difference would not make these local temperature observations sufficient to independently identify separate heat-flow paths, net heat received by the sensor, total heat loss, or the body-side heat fraction.

4. **$h_{ref}$ is a declared normalization, not an experimentally identified value.** The values $h_{ref}=10$ W/(m²·K), $A=0.0231$ m², and $G_{ref}=0.231$ W/K are retained as the definition of the descriptive index $I_{\mathrm{ins}}$. No independent heat-flow or boundary measurements establish that this is the actual heat-transfer coefficient of the sandwich, bare pad, or covered conditions. Natural-convection and radiation estimates are not presented as measured evidence precisely matching the bench setup. The script also reports indices for $h_{ref}=5/10/15$ W/(m²·K). These show sensitivity to the normalization choice and vary linearly with $h_{ref}$; they are not confidence intervals. $I_{\mathrm{ins}}$ is not clipped to [0,1]. Values above 1 do not mean efficiency above 100% and do not establish the magnitude of actual heat losses.

5. **Prototype identity and extrapolation limits.** Applying $R=V^2/P$ to the four voltage/power pairs gives operating-point resistance estimates of approximately 4.62/4.61/4.66/4.75 Ω. These are algebraic conversions of the readings, not independent resistance calibrations. No nominal rated voltage or resistance is recorded for this prototype, and it is not the Type B device in the system catalogue. The records do not validate high-power temperature predictions or establish thermal or safety transferability to Type B devices or the human body.

6. **Response indices and model coefficients are distinct.** $I_{\mathrm{ins}}$ does not map directly to $\eta_{\mathrm{model}}$ for the 12 body regions. Clipping, scaling, or analogies between covering conditions cannot convert a temperature-response index into heat-delivery efficiency. System values of $\eta_{\mathrm{model}}$ are independently declared modeling assumptions. Numerical sensitivity analysis can show how those assumptions affect system outcomes, but cannot establish that the bench experiment measured the efficiency. The supplied data consist of the four CSV files listed above and this experimental description.
