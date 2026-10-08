# BGEMS

BGEMS (Body Grid End-to-End Management System) is a reproducible research package for scenario-driven wearable-heating design and operation. It connects four computational stages through one explicit `SystemConfig`:

1. garment-dependent thermal-demand calculation;
2. finite battery, photovoltaic, and garment configuration search;
3. receding-horizon operation under electrical, service, and reserve constraints;
4. independent trajectory evaluation from realized actions and states.

The repository contains source code, input definitions, tests and examples for this workflow. Software tests verify implementation behavior, not human comfort, physiological safety or field performance. For version `v1.0.0-resubmit`, [RELEASE.md](RELEASE.md) describes the complete data package, restoration and offline reproduction. The GitHub examples are not the full 317-date dataset.

## Repository Roles

### `src/`

`src/` contains flat modules with explicit responsibilities:

- `src/inputs.py`: `Person`, `Garment`, `Device`, `ScenarioProfile`, and `SystemConfig`
- `src/weather.py`: Open-Meteo acquisition plus timezone-aware temperature and reference-PV CSV loading
- `src/modeling.py`: thermal-demand equations, the declared E/W prescription, regional heat-delivery accounting, and the lumped equipment thermal step
- `src/configuration.py`: candidate assembly, diagnostic screens, exhaustive finite search, mass--cost Pareto filtering, and deterministic selection
- `src/operation.py`: scenario construction, forecast policies, electrical transitions, equipment-heater hysteresis, lexicographic service LPs, LP/PWL RHC and one-step LP/PWL simulation
- `src/evaluation.py`: independent completion, electrical, regional service/priority, and reserve auditing
- `src/run.py`: fixed/search orchestration, provenance snapshots, result export, and the four-argument command-line interface

Parameters are represented by the five classes in `inputs.py`. Trajectories and results remain ordinary arrays and dictionaries so that saved evidence can be inspected without a framework-specific object model.

### `data/`

`data/` contains explicit reproducibility inputs:

- `data/weather/temperature/`: timezone-aware temperature series
- `data/weather/pv/`: timezone-aware reference-array PV series

Every source file is declared by `ScenarioProfile`. The runtime rejects missing files, duplicate or overlapping timestamps, uncovered scenario intervals, inconsistent time semantics, and incompatible PV reference ratings.


### `tests/`

`tests/` contains behavior-focused coverage for:

- parameter defaults, mutation, and scenario phase queries;
- thermal-demand and regional-service equations;
- two-objective configuration Pareto/selection and diagnostic-only capability;
- current-action core--distal--other priority, SOC-aware LP operation, equipment hysteresis and terminal reserve;
- independently repeated priority/physical audits and failure classification;
- weather semantics, complete interval coverage, and daylight-saving transitions;
- fixed and search CLI execution, provenance output, and output-directory protection.

### `requirements.txt`

The solver path is `scipy.optimize.linprog(method="highs")` in SciPy 1.17.1. Its embedded HiGHS version is 1.12.0 in the recorded environment; the separately installed `highspy==1.15.1` is not the solver called by this path. Each run records the actual embedded solver version.

## Reproducibility Scope

| Role | Repository authority | Boundary |
|---|---|---|
| Parameter definitions | `src/inputs.py` and `data/systemconf/{subject,scenario_profiles,components}.json` | Scenario-specific pack, control, reserve, comparison, and weather values must be declared explicitly. |
| Equations and algorithms | `src/modeling.py`, `src/configuration.py`, `src/operation.py`, `src/evaluation.py` | Numerical success does not establish physiological or hardware validity. |
| Weather and PV forcing | CSV files declared by `ScenarioProfile` | No interpolation, extrapolation, implicit file search, or gap filling is performed. |
| Execution provenance | `parameters.json` in each output directory | Includes the serialized configuration, source hashes, versions, time semantics, and forecast mode. |
| Candidate and trajectory evidence | `candidates.csv` and `runs/` | Partial failed trajectories are retained; unseen intervals are not imputed. |

## Central Parameter Definitions

The committed JSON registries provide the frozen baseline used by executable runs; constructors and loaders do not invent missing scientific values.

`Person()` defines:

- height: 175 cm;
- weight: 70 kg;
- age: 30 years;
- sex: male;
- twelve regional scoring weights and reference skin temperatures;
- Du Bois body-surface area calculated from the current height and weight.

`Device()` and `data/systemconf/components.json` define:

- standardized battery unit: 60 Wh, 0.25 kg, USD 10, and 10.8 V nominal;
- per-unit heat capacity: 275 J/K;
- per-unit charge/discharge limits: 30 W and 60 W, scaled by the selected battery-unit count;
- effective pack-to-air thermal resistance: 3 K/W;
- standardized PV unit: 20 W, 0.2 kg, USD 20;
- base electronics: 0.5 kg, USD 150, 5 W;
- twelve on-body pad ratings and electro-thermal factors;
- equipment-node pad rating: 18 W;
- shared system-bus limit: 200 W, not scaled by battery-unit count.

These declared model inputs are specified in `data/systemconf/` and loaded by `src/inputs.py`. They are not independently validated physiological or battery-safety limits. The loader rejects omitted or null execution-critical fields rather than substituting implicit values.

`ScenarioProfile` requires an explicit timezone-aware start, phase sequence, initial state, terminal SOC, temperature files, and PV files. Service requirements may be one scalar fraction or twelve regional fractions for each phase.

The E/W prescription uses `58.15 W/m²/MET`. Thermal demand preserves the signed `M - E - W` balance before applying the final nonnegative supplemental-heating bound.

## Environment Setup

Run commands from this directory.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Quick Start: Reproduce Existing Results Offline

First restore the complete result package following [RELEASE.md](RELEASE.md).
The four GitHub examples alone do not contain the full study. From the repository
root, use fresh output directories:

```bash
python -m scripts.report_evidence --output-dir output/reproduced_analysis
python -m scripts.plot_results --data-dir output/reproduced_analysis --output-dir output/reproduced_figures
python -m scripts.pad_calibration_comprehensive --output-dir output/reproduced_pad
```

These commands use the supplied inputs and records without downloading weather
or starting new configuration searches. The expected report covers 317 dates,
308 accepted selections and nine unsuccessful dates.

## Optional: Acquire Weather and PV Data

Weather acquisition is a separate workflow for new inputs, not a prerequisite
for reproducing the supplied results. It requires internet access to OpenStreetMap
Nominatim and Open-Meteo. This example uses separate data and image directories
so existing reproduction inputs are not overwritten. API dates are inclusive.

```bash
python - <<'PY'
from src.weather import WeatherDataProcessor

processor = WeatherDataProcessor()
processor.scrape_and_save(
    location="Harbin",
    start_date="2025-12-01",
    end_date="2025-12-01",
    csv_dir="output/weather/temperature",
    img_dir="output/weather/temperature",
)
processor.scrape_and_save_pv(
    location="Harbin",
    start_date="2025-12-01",
    end_date="2025-12-01",
    csv_dir="output/weather/pv",
    img_dir="output/weather/pv",
    tilt=45.0,
    azimuth=180.0,
    kwp=0.1,  # 100 W reference array
)
PY
```

The acquisition step writes:

- `output/weather/temperature/Harbin_2025-12-01.csv`;
- `output/weather/pv/Harbin_2025-12-01.csv`;
- `output/weather/temperature/Harbin_temperature.png`;
- `output/weather/pv/Harbin_pv.png`.

The temperature CSV contains instantaneous samples with `sample_hold` semantics. The PV CSV contains preceding-hour means with `interval_end_mean` semantics and records the 100 W reference-array rating. These files are direct inputs to `ScenarioProfile`.


## Input Contract

### JSON configuration

The executable configuration authority is:

- `data/systemconf/subject.json`: subject identity and anthropometry;
- `data/systemconf/scenario_profiles.json`: the three scenario profiles, dated phases,
  initial state, search domain, control declarations, and audit tolerances;
- `data/systemconf/components.json`: regional order, garments, heaters, standardized
  battery/PV units, electronics, and backpack equipment.

`src.inputs.load_system_config(...)` converts this fixed schema into exactly five
parameter classes: `Person`, `Garment`, `Device`, `ScenarioProfile`, and
`SystemConfig`. Bare subject filenames resolve under `data/systemconf`; explicit
relative paths resolve from the repository root; absolute paths are retained.
Unknown or missing keys are rejected.

The checked-in profiles declare the execution-critical pack, control, reserve,
service-floor, forecast and solver parameters. Null or omitted execution-critical
fields remain invalid: `python -m src.run` rejects them before creating an output
directory, and the program never substitutes synthetic defaults.

The twelve regional arrays use the order declared in `components.json`: left chest,
right chest, back, pelvis, left upper arm, right upper arm, left lower arm, right
lower arm, left thigh, right thigh, left lower leg, and right lower leg.

### Temperature CSV

Every timestamp must include a UTC offset. Temperature defaults to causal `sample_hold` semantics.

```csv
timestamp,sample_duration_hours,semantics,air_temperature_c
2025-12-01T00:00:00+08:00,1.0,sample_hold,-10.0
```

### PV CSV

PV defaults to `interval_end_mean`: each timestamp closes the preceding interval.

```csv
timestamp,sample_duration_hours,semantics,reference_rating_w,pv_power_w
2025-12-01T01:00:00+08:00,1.0,interval_end_mean,100.0,40.0
```

Candidate PV power is calculated as

```text
reference PV power × pv_rating_w / pv_reference_rating_w × number of PV units
```

The weather acquisition path applies its declared PV conversion efficiency once before writing the CSV. Candidate scaling does not apply that efficiency again.

## Running the Three Stages

### Command-line interface

```bash
python -m src.run \
  --subject subject.json \
  --scenario_type 2 \
  --date 20260101 \
  --output_dir ./output
```

All arguments are optional. Defaults are `subject.json`, scenario type 2
(Muztagh Ata climbing), today's local calendar date in `YYYYMMDD` form, and
`./output/`. `--subject` accepts either a bare filename under `data/systemconf`
or an explicit path. `--help` prints the interface. Before Stage 1, the CLI
geocodes the selected scenario location and fetches temperature and PV data for
the effective date. Current dates use Open-Meteo's forecast endpoint; older
dates use its ERA5 archive endpoint. An unavailable date or API failure is
reported directly and the output directory is not created.
Progress is flushed immediately for configuration loading, each weather request,
candidate preparation, every operationally evaluated candidate, and final search
completion. HTTP requests use a 10-second connection timeout and a 60-second read
timeout; network stalls therefore return a CLI error instead of waiting indefinitely.

One command first acquires the dated weather inputs, then performs the coupled
pipeline:

1. **Weather acquisition.** Temperature and PV records for `--date` are written
   beneath `data/weather/temperature/` and `data/weather/pv/`.
2. **Stage 1 — Modeling.** Weather is aligned to the scenario grid; MET, E/W,
   garment-dependent demand, and regional service declarations are constructed.
3. **Stage 2 — Configuration.** Every declared
   `(battery_units, pv_units, garment_id)` candidate receives mass and acquisition
   cost objectives; full-on capability and aggregate energy/power are diagnostics,
   not acceptance shortcuts or Pareto objectives.
4. **Stage 3 — Operation and independent evaluation.** Hard per-region floors
   precede strict current-action core → distal → other service priority. Three
   sequential LPs lock those group optima; the final soft operating objective
   uses the declared 8-tangent piecewise-linear SOC penalty and a linearized
   service-fulfillment epigraph, so every RHC subproblem is a pure LP solved to
   global optimality by HiGHS. The independently replayed C/E/S/R and priority
   checks precede Pareto acceptance. Numerical failures are retained; no
   approximate rescue is substituted.

The output directory is created when missing and reused when present. Files owned
by the pipeline are replaced for the new run; unrelated files are retained. A successful or resolved execution writes:

| Artifact | Contents |
|---|---|
| `output_configuration.txt` | selected garment, backpack status, battery/PV counts, total mass and acquisition cost |
| `output_operation.pdf` | scenario-phase plots of ambient temperature, SOC, available PV, and modeled thermal service |
| `parameters.json` | loaded configuration, source paths/hashes, package versions, and weather semantics |
| `summary.json` | aggregate status, selected/Pareto configurations, counts, and timings |
| `candidates.csv` | candidate objectives, diagnostics, C/E/S/R outcomes, and evidence paths |
| `runs/` | per-candidate JSON trajectories and state/step CSVs |

CLI exit codes:

- `0`: a configuration was selected successfully;
- `1`: resolved policy or physical failure; no configuration was selected;
- `2`: invalid, missing, or unresolved input/setup evidence;
- `3`: unresolved numerical outcome.


### Date ledger and predeclared paired comparison

`aggregate_sweep.py` never treats candidate counts as sampled dates. Arrange any
completed dated search directories as `output/s{scenario_type}_{YYYYMMDD}/`,
then run:

```bash
python -m scripts.aggregate_sweep --input_root output --output_dir output/date_report
```

`output/date_report/date_ledger.csv` contains one row for each of the 317 planned
scenario--dates, including `not_run`, `error`, `unresolved`, `failed`, and
`success`; `date_summary.json` retains the full denominators. An absent
`summary.json` is **not** silently omitted or marked physically infeasible.
The listed historical dates have not been run by this documentation example.

For a paired design comparison, first produce resolved design-day **search**
results with comparison disabled; do not choose days after inspecting evaluation
results. Example for a declared Harbin design date:

```bash
python -m src.run --scenario_type 1 --date 20251201 \
  --output_dir output/design/s1_20251201
```

Create `output/design/s1_manifest.json` with independent design evidence
(relative paths resolve from the manifest directory):

```json
{
  "scenario_type": 1,
  "subject_id": "reference_adult",
  "design_days": [
    {"date": "20251201", "directory": "s1_20251201"}
  ]
}
```

The design output must be a fully resolved search with identical scientific
settings and executable `src/` hashes; every reference candidate must pass
on **every** listed design date. An unresolved design search, no common
accepted configuration, or design/evaluation date overlap is rejected
without substituting a larger package. Before the evaluation run, explicitly
set `case1.comparison` in `data/systemconf/scenario_profiles.json` to
`{"enabled": true, "design_manifest": "output/design/s1_manifest.json"}`.
The four-argument CLI is unchanged:

```bash
python -m src.run --scenario_type 1 --date 20251202 \
  --output_dir output/comparison/s1_20251202
```

Comparison mode writes `comparison.json`/`comparison.csv`, independent
`adaptive_search/` and `frozen_hardware/` fixed/search evidence, and
`adaptive_myopic.json`/`frozen_myopic.json` action/state traces. It replays
RHC and a one-step LP/PWL solve on each hardware tuple under common actual
forcing,
initial state, hard floors, priority audit, and real mission-end reserve.
Myopic success supplies a causal feasibility witness for that exact
hardware--date; without it, do not call the control comparison a
controller-neutral feasible benchmark. A one-step horizon is an ablation,
**not** a PID/RBC/TACS implementation. Comparison dates and hardware must
be declared before inspection; no formal comparison campaign or stress
domain is implied by this example. Reset `comparison.enabled` to `false`
and its manifest to `null` for ordinary single-date output.

### Run the verification suite

```bash
python -m unittest discover -s tests -v
```

The suite uses synthetic boundary cases where necessary. Passing tests establishes software behavior, not field validity.

### Reproduce frozen paper figures

The two offline postprocessors keep numerical evidence separate from rendering.
They neither fetch weather nor rerun searches/stress simulations. Choose a new
analysis directory; the report and plot entries reject existing output targets:

```bash
python3 -m scripts.report_evidence --output-dir output/review_analysis_next
python3 -m scripts.plot_results --data-dir output/review_analysis_next --output-dir output/review_analysis_next/figures
```

The report writes self-contained `evidence.json`, catalogue JSON and numerical
CSV views. The renderer needs only that evidence JSON and produces five 300-dpi
PNG/vector-PDF pairs: `configuration_outcomes`, `design_pareto_comparison`,
`allocation_operation_evidence`, `stress_mechanism_evidence`, and
`resource_service_reserve`. Both commands write only to their specified output directories.
The configuration figure has three scenario rows and four columns: mission
forcing, battery units, PV units, and garments. Forcing uses all 90/46/181 planned
dates; hardware frequencies use 90/46/172 resolved-success dates, retaining the
nine failed Qinling dates only in the forcing column.

`evidence.json` also includes `seasonal_comparison`: the intersection of accepted
configurations over every planned date, separate mass-first and cost-first fixed
references, per-date resource savings and summary quartiles, search-bound counts,
and ratios of summed PV use to summed electrical load. A failed date cannot be
omitted to obtain full-season coverage. The comparison figure distinguishes daily
resource means from individual feasible configurations; all seasonal references
are retrospective, not independent design/test baselines.

Mission statistics use the configured intervals, not whole days: Harbin
09:00–12:00, Muztagh Ata 06:00–18:00, and Qinling 08:00–22:00, in the configured
Asia/Shanghai timezone. Blue solid lines and pale blue bands show temperature
means and within-mission min–max values. For the latter two scenarios, orange
dashed lines and pale orange bands show available PV power (W) for one nominal
20 W unit, obtained from the cached 100 W reference using the configured 20/100
ratio. This is not selected-array output, controller-used power, or energy in Wh.
Zero-PV intervals remain in mission means; power is not clipped at the nominal
rating. Harbin has no PV forcing axis but retains the zero-unit selection column.

**Numerical-figure style:** use the configuration-outcomes style for future
figures: white background, Times New Roman bold text, no grid, blue `#3498DB`,
black outlines, approximately 1.5pt axes, inward ticks, and the existing garment
hatches. Hide top/right spines except a required PV right axis. Use line styles,
markers, or hatches as well as color: temperature is solid blue, PV dashed orange
`#F39C12`, stress success a blue circle, and stress failure a red cross.
Bands denote min–max ranges, not uncertainty intervals. Retain status distinctions,
labels, and data limits when styling; never clip data to fit the palette or layout.
The consolidated Results renderer explicitly uses Matplotlib Agg so desktop and
headless invocations share the same layout backend. Fonts and library versions
must still match for byte-identical raster output.

The temperature acquisition plots in `WeatherDataProcessor.scrape_and_save`
and `scrape_multiple_locations` read existing `location_YYYY-MM-DD.csv` files
first. Only contiguous runs of missing dates are fetched; a fully cached range
requires neither geocoding nor weather API calls. Existing files are not rewritten.
Both paths reuse the timestamp parser, API-frame conversion and daily CSV writer.
Temperature plots use 200-dpi output, Times New Roman bold and no background grid.
Single-location maximum/minimum/mean curves are red/blue/gold. Multi-location
colors are red, blue, green, orange, purple, brown, pink and gray.

The report checks all 317 dates, configuration/weather hashes and retained actual
forcing. Historical absolute project paths are mapped onto the installed project
without changing archived manifests. Two exact source transitions are recorded:
the known `weather.py` acquisition/plotting version, and an `operation.py`
module-docstring delimiter change with an identical parsed syntax tree.
Their frozen and analysis SHA-256 identities remain distinct. The reader's
temperature/selected-array PV values must agree exactly with a retained trajectory
for every date. All other unknown source changes fail.
This route reproduces derived results and figures; it does not rerun every candidate solve.

The catalogue retains unrounded regional shares and input identities. Resource
objectives are base/garment/battery/PV subtotals, excluding the separately declared
equipment heater and any independent backpack-shell inventory.

Selected metrics, failed-date summaries, search timings, Pareto domains and
stress certificates remain separate numerical views. Missing selected evidence
fails rather than shrinking the denominator. `floor_deficit_wh` differs from
`full_demand_deficit_wh`; regional durations are node·h, and full-search timing
is not online latency. Operation plots use recorded state times and interval
boundaries; no failed trajectory suffix is filled in.

The four existing pad records can be reanalysed independently:

```bash
python3 -m scripts.pad_calibration_comprehensive --output-dir output/resubmit_pad
```

This writes fit/response JSON and a four-panel PNG. Fitted initial sensor
temperature is not measured ambient. `G_app` and `I_ins` are descriptive ratios,
not identified heat-transfer efficiencies.

`scripts/stress_sensitivity` is a separate, opt-in solver replay, not part of
these commands. It requires `--output-dir <new-directory>` and rejects existing
directories; `--root` selects the primary baseline archive. Do not launch it to
regenerate figures from existing records.

Reproduction uses the supplied weather inputs and their recorded checksums.
See `RELEASE.md` for data attribution, package restoration and version binding.
The analysis and plotting workflow requires neither a new API request nor the
manuscript source.


## Execution Output Inventory

For an output path such as `output/`, execution creates:

- `output/output_configuration.txt` and `output/output_operation.pdf`: the two user-facing stage summaries;
- `output/parameters.json`: serialized configuration, software versions, source paths, SHA-256 hashes, time semantics, and forecast mode;
- `output/candidates.csv`: objectives, disposition, independent audit components, residuals, diagnostics, and artifact paths;
- `output/summary.json`: selected configuration, Pareto configurations, outcome counts, timings, and execution scope;
- `output/runs/y{cells}_h{pv}_g{garment}.json`: hardware, realized forcing, trajectory, and independent evaluation evidence;
- matching `_states.csv` and `_steps.csv` files with timezone-aware timestamps.

Candidate counts are deterministic design-domain results, not a reliability sample.
Failed and partial runs remain visible. `summary.json` separates priority-LP,
final-soft-LP, independent audit, mass--cost selection, and total wall-clock
timing scopes;
per-solve LP status, duality/stationarity residuals, group targets and actual
actions appear in each candidate JSON. The group locks use the declared
`solver_tolerance` plus a recorded 1e-9 arithmetic margin; physical SOC and
regional-service floor residuals remain unrelaxed.

## Interpretation Boundaries

- `persistence` is the causal forecast mode. `perfect-preview` deliberately exposes future forcing and must be reported as such.
- Regional thermal fulfillment is a modeled heat-delivery score, not a subjective-comfort or physiological-safety measurement.
- Full-activation capability and energy screens are diagnostics; time-resolved operation and independent evaluation determine candidate acceptance.
- SOC and service violations are not hidden by clipping or solver tolerance.
- The 100 W PV reference array is a weather-series scaling basis, not the physical 20 W candidate PV unit.
- The standardized 60 Wh battery unit does not define cell voltage or pack thermal parameters.
- Formal design-domain, perturbation, or Monte Carlo studies require frozen public configuration files and input data.

## License

Project software is licensed under Apache-2.0; see `LICENSE`. Data licences and third-party attribution are specified in `RELEASE.md`.
