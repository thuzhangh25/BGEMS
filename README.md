# BGEMS

BGEMS (Body Grid End-to-End Management System) is a scenario-driven configuration and control framework for wearable thermal management, linking regional heating demand, finite hardware configuration and predictive operation. Receding horizon control (RHC) coordinates regional heating with state-of-charge (SOC)-weighted discharge costs and remaining-load reserves. Comparisons with rule-based control (RBC), proportional–integral–derivative (PID) control and thermal adaptive control strategy (TACS) evaluate three scenarios, paired perturbations, catalogue scaling, rental returns and building-electricity savings.

**Reproduce numerical results and data-derived figures from the supplied inputs and simulation records.**

Version: **v1.1.1-resubmit**. Code: https://github.com/thuzhangh25/BGEMS/tree/v1.1.1-resubmit . Code and research-data archive: https://doi.org/10.5281/zenodo.23607419 . Extract this version's code and data archives into the same directory to reproduce the study results.

This presentation-only update moves the Fig. 9(a) legend above the bars and refines its layout, and changes the Fig. 12 axis wording to net building-electricity savings. Numerical inputs, simulation records and results are unchanged. The previous GitHub tag and Zenodo version remain available for the earlier figure presentation.

## Install

Run from the package root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Python 3.13 and the pinned dependencies were used for verification. `requirements-lock.txt` records the complete environment. Figures use Times New Roman, white backgrounds, grid-free axes and a consistent blue/orange/green/red palette. Obtain Times New Roman from a licensed source for figure generation.

The recorded seasonal-context PNG layout uses Matplotlib's MacOSX backend on macOS; the result figures use Agg.

## Files and responsibilities

| Location | Responsibility |
|---|---|
| `src/inputs.py` | Parameter loading and validation |
| `src/weather.py` | Cached temperature/reference-PV loading and optional acquisition |
| `src/modeling.py` | Regional demand, fulfillment and equipment-heater activation |
| `src/configuration.py` | Design screens, Pareto set and deterministic selection |
| `src/operation.py` | Four controllers and common electrical constraints |
| `src/evaluation.py` | Independent trajectory calculations and checks |
| `src/run.py` | Mission planning, search/fixed execution and source identities |
| `scripts/compare.py` | Paired Monte Carlo simulations on fixed hardware |
| `scripts/stress_sensitivity.py` | Fixed-hardware single-factor perturbations |
| `scripts/reproduce_study.py` | Fixed-configuration, catalogue and indoor comparisons |
| `scripts/evaluate_applications.py` | Cashflow and building-energy accounting |
| `scripts/report_evidence.py` | Independent metrics and self-contained plotting data |
| `scripts/plot_results.py` | Data-derived PNG/PDF figures |
| `scripts/pad_temperature_response.py` | Four-condition benchtop response analysis |
| `data/{systemconf,weather,experiments}/` | Model parameters, cached inputs and benchtop records |
| `output/results/` | Three mission comparisons under unperturbed conditions |
| `output/experiments/` | Monte Carlo, single-factor, fixed-configuration, catalogue and indoor records |
| `output/evaluation/` | Application calculations, design checks and plotting data |
| `output/paperfigures/` | Generated figures |

Use a new destination for each execution, report and figure set. The workflows read the supplied records and required cached weather inputs offline.

## Mission conditions

| Scenario | Date | Local interval | Equipment heater installed | Candidates |
|---|---|---|---|---:|
| Harbin skiing | 2026-01-01 | 09:00–12:00 | No | 160 |
| Muztagh Ata mountaineering | 2025-07-15 | 06:00–18:00 | Yes | 960 |
| Qinling indoor work | 2025-07-01 | 08:00–22:00 | No | 160 |

Times use Asia/Shanghai. Unperturbed conditions use the specified time-varying mission profiles and model parameters. Qinling uses a prescribed indoor profile; Muztagh Ata uses the supplied weather temperature series directly. Temperature uses forward sample hold and PV uses preceding-hour means. PV inputs describe a 100-W reference array, scaled once to the installed rating. `data/weather/weather_sources.json` gives sources, units, transformations and checksums.

## 1. Regenerate figures

The code package includes self-contained plotting data for nine figure groups:

```bash
python -m scripts.plot_results --data-dir output/evaluation/study --output-dir reproduced/figures
python -m scripts.pad_temperature_response --output-dir reproduced/pad_response
```

The nine groups are configuration, controller trajectories, Qinling fixed-fleece-jumpsuit controller trajectories, temperature perturbation, robustness, regional allocation, economics, indoor heating and catalogue scaling. Seasonal weather context is generated separately from the supplied daily summaries using `scripts.plot_weather_context`.

## 2. Independently evaluate supplied trajectories

Extract the code and full-data archives into the same parent directory; both contain a `BGEMS/` root. The code package contains 12 trajectories and the supplement adds 4166 disjoint records: 4162 from the main study and four equipment-heater activation runs.

```bash
python -m scripts.report_evidence --study --data-dir output --output-dir reproduced/evaluation/study
python -m scripts.plot_results --data-dir reproduced/evaluation/study --output-dir reproduced/audited_figures
```

The expected layout is `results/`, `experiments/{pilot_mc,formal_mc,stress,static,indoor,fixed_fleece,catalogue_scaling}/`, and `evaluation/{applications/applications.json,fixed_fleece_applications.json,design_checks.json}`. The report recalculates every trajectory, checks paired hardware/input identities and requires all specified controller cohorts.

The main study cohort contains 4174 valid executions and 4173 electrical completions: 12 unperturbed, 40 pilot Monte Carlo, 4000 formal Monte Carlo, 84 single-factor, six fixed-configuration, 16 selected-down-top indoor and 16 fixed-fleece-jumpsuit indoor records. The formal ensemble contains 1000 independent paired sample identities; the pilot repeats the first ten identities. One fixed-configuration case is rejected by the initial reserve condition. Four equipment-heater activation runs bring the supplied collection to 4178 executions and 4177 electrical completions.

## 3. Generate mission results

```bash
python -m src.run --plan-only
python -m src.run --output-dir reproduced/study/results
python -m scripts.report_evidence --data-dir reproduced/study/results --output-dir reproduced/mission_evidence
python -m scripts.plot_results --data-dir reproduced/mission_evidence --output-dir reproduced/mission_figures
```

Search is performed once per scenario, followed by four controllers on the selected hardware. Explicit scenario or fixed-configuration runs are also supported:

```bash
python -m src.run --scenario 2 --output-dir reproduced/muztagh
python -m src.run --scenario 3 --configuration 2 0 2 --output-dir reproduced/indoor_fixed
```

The tuple is `(battery units, PV units, garment ID)`. Garment IDs 0/1/2/3 denote down jumpsuit / fleece jumpsuit / down top / fleece top. Failed trajectories retain their completed prefixes and failure reason; hardware is held fixed throughout each controller comparison.

## 4. Reproduce configuration and application comparisons

```bash
python -m scripts.reproduce_study --experiment static --output-root reproduced/study
python -m scripts.reproduce_study --experiment catalogue-scaling --output-root reproduced/study
python -m scripts.reproduce_study --experiment indoor --output-root reproduced/study
python -m scripts.reproduce_study --experiment fixed-fleece --output-root reproduced/study
python -m scripts.evaluate_applications --unperturbed-dir reproduced/study/results --indoor-dir reproduced/study/experiments/indoor --output-dir reproduced/study/evaluation/applications
```

`indoor` uses the down top `(2,0,2)` at 25/20/18/15°C; `fixed-fleece` uses the fleece jumpsuit `(8,0,1)` at the same setpoints. Both use the specified temperature waveform and activity schedule. The fixed-fleece-jumpsuit comparison includes an algebraic zero-PWM reference. `static` evaluates `(4,0,0)` and `(7,0,0)` across all three scenarios. `catalogue-scaling` evaluates catalogues of 4/8/16 garments with the same regional pad layout.

The equipment-heater activation comparison uses the Muztagh Ata hardware for a two-hour profile at −18/−20/−22/−18°C in successive half-hour blocks, with 1 MET and zero PV. Its four additional records are separate from the 4174-execution study cohort:

```bash
python -m scripts.reproduce_study --experiment equipment-activation --output-root reproduced/activation
python -m scripts.report_evidence --data-dir reproduced/activation/experiments/equipment_activation --output-dir reproduced/activation-evidence
python -m scripts.plot_results --equipment-dir reproduced/activation/experiments/equipment_activation --output-dir reproduced/activation-figures
```

All four controllers activate the 18-W equipment heater for one hour and complete the two-hour mission. The data package provides these records under `output/experiments/equipment_activation/`; the figure shows temperature, equipment power and SOC.

`data/systemconf/applications.json` defines constant-2026-USD cashflows, sensitivity grids, charging efficiency and building assumptions. Battery depletion is recharged to the initial state of charge (SOC). Building accounting combines 14 hours of wearable operation with a 24-hour building day and a −20°C outdoor design condition. The remaining ten hours use the same occupancy heat gain and reference setpoint in every comparison.

## 5. Run uncertainty comparisons

Run paired Monte Carlo and single-factor simulations:

```bash
python -m scripts.compare --scenario 2 --samples 10 --output-dir reproduced/study/experiments/pilot_mc
python -m scripts.compare --scenario 2 --samples 1000 --output-dir reproduced/study/experiments/formal_mc
python -m scripts.stress_sensitivity --scenario 1 --output-dir reproduced/study/experiments/stress/s1
python -m scripts.stress_sensitivity --scenario 2 --output-dir reproduced/study/experiments/stress/s2
python -m scripts.stress_sensitivity --scenario 3 --output-dir reproduced/study/experiments/stress/s3
```

The formal comparison uses 1000 paired samples and seed 42. Monte Carlo settings use a 0.2-h step, up to 120 prediction steps and independent random streams for temperature, activity, PV and cold-shock events. Unperturbed and single-factor comparisons use a 0.1-h step and a 6-h prediction horizon. RHC forecasts use current observations and scheduled activity. Completion counts and failure reasons report the execution outcomes.

## Model interpretation

- Thermal demand is the model-estimated supplemental heating power. It is allocated across all twelve on-body nodes. The localized thermal-comfort score is defined as modeled heating-demand fulfillment: cap each regional ratio at one, then combine using normalized weights. Missing pads supply zero heat to their assigned regions.
- Configuration minimizes `(-full-activation score, mass, cost)` over garment masks and explicit component bounds, then applies the specified scenario preference.
- The reference energy screen integrates demand-capped heating, electronics and triggered equipment loads, crediting simultaneous PV up to the load. Operation separately allows battery charging.
- All controllers enforce SOC ≥ 0.05 and the remaining mandatory-load reserve. SOC 0.15 is a soft preference; electrical completion and heating-demand fulfillment are reported separately.
- The equipment heater draws 18 W when installed and air temperature is ≤ −20°C.
- RHC optimizes core, distal and other regional groups sequentially, locks their attained values and optimizes the remaining horizon objective.
- The 0.80 line is a retrospective mean-score reference. The down top uses a whole-body equivalent thermal resistance.
- Benchtop records characterize local temperature responses. Regional heat-transfer factors, rental conditions and building properties are specified in the model inputs.

## Licences and citation

Software: Apache-2.0 (`LICENSE`). Authors' data: CC BY 4.0. Retain third-party weather attribution. See [RELEASE.md](RELEASE.md), `release_manifest.json` and archive checksums for contents, source identities and verification details.
