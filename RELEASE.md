# BGEMS v1.0.0-resubmit: code and data

Repository: https://github.com/thuzhangh25/BGEMS

Associated research package: https://doi.org/10.5281/zenodo.23230495

This identifier refers to the combined code-and-data record, not a software-only record. `release_manifest.json` identifies the version, packaged objects and their checksums.

## Package contents

| Object | Contents | Restore location |
|---|---|---|
| `BGEMS-v1.0.0-resubmit-code.tar.gz` | Code, tests, dependencies, configuration, supplied weather, four pad CSVs/README, citation/licensing instructions and four example dates | `BGEMS/` |
| `BGEMS-v1.0.0-resubmit-results.tar.gz` | All primary records retained locally, all stress records, derived result tables/figures and pad-fit outputs | `BGEMS/output/` |
| `frozen_20261004_runs.tar.zst` | Additional candidate trajectories, step records and solver/audit records; original archive bytes preserved | `BGEMS/output/frozen_20261004/` |
| `frozen_20261004_members.csv` | Archive member paths, byte sizes, SHA-256 and storage classification | Keep next to the compressed archive |
| `release_manifest.json`, `SHA256SUMS` | Object identities and code/data version binding | Keep next to downloaded objects |
| `environment.json` | Python/platform and numerical/plotting package versions used for verification | Reference; install dependencies from `requirements.txt` |

The GitHub package is deliberately smaller than the complete Zenodo package. Its examples are Harbin 2025-12-01, Muztagh Ata 2026-06-20, Qinling 2026-02-11 (the three fixed-package stress baseline dates), and Qinling 2026-04-04 (the first unsuccessful date chronologically). They are examples, not the full 317-date study. Locally retained candidate subsets within these examples are not complete candidate archives.

The primary study has 317 planned dates, 308 accepted selections and nine unsuccessful dates. Its ledgers reference 255,960 candidate JSON records: 10,028 are in the primary local records, and 245,932 are in the additional archive. Both data objects are needed for complete candidate coverage. Archived absolute producer paths are provenance strings, not installation locations; `scripts.report_evidence` maps them to the current project and checks source/input identities.

## Restore and reproduce

Download the listed objects into one directory. Verify downloads before extraction:

```bash
# Linux:
sha256sum -c SHA256SUMS
# macOS alternative:
shasum -a 256 -c SHA256SUMS

tar -xzf BGEMS-v1.0.0-resubmit-code.tar.gz
tar -xzf BGEMS-v1.0.0-resubmit-results.tar.gz
```

If using the matching GitHub checkout instead of the code archive, name its directory `BGEMS` before extracting the results beside it. Do not extract onto unrelated work. Install a compatible Python environment from `BGEMS/requirements.txt` as described in README. For matching typography, install Times New Roman through a licensed source and confirm Matplotlib can find it; no font binaries are redistributed.

From `BGEMS/`, regenerate the paper's numerical evidence and five computational Results figures offline:

```bash
python -m scripts.report_evidence --output-dir output/reproduced_analysis
python -m scripts.plot_results --data-dir output/reproduced_analysis --output-dir output/reproduced_figures
python -m scripts.pad_calibration_comprehensive --output-dir output/reproduced_pad
```

Use fresh output directories. These commands use Python and the supplied data; LaTeX is not required. They do not download weather or run a new configuration search. They use all primary ledgers and the retained selected/representative trajectories, so the large additional archive need not be expanded for this result-analysis route. Expected report counts are 317 mission grids, 308 selections and stress certificate classes baseline/thermal/energy = 6/8/10. The pad analysis uses the four supplied CSVs and their experimental description.

To inspect **every** candidate trajectory, additionally restore the archive. From the download directory, with the `zstd` command installed:

```bash
zstd -dc --long=31 frozen_20261004_runs.tar.zst | tar -xf - -C BGEMS/output/frozen_20261004
```

The archive contains relative `s1_YYYYMMDD/runs/...`, `s2_...` and `s3_...` paths. Do not strip a leading path component. Its decompressor needs support for a 1-GiB window; `--long=31` permits this. Allow at least 100 GB of free disk space for the complete expanded package. The archive expands to 86,626,087,278 file-content bytes across 738,412 members, in addition to the retained primary records and filesystem overhead. Streaming selected members is possible, but does not constitute complete restoration. The CSV index specifies each member's size and checksum; avoid changing the original records or replacing their recorded hashes.

Running `src.run.execute`, configuration sweeps or stress simulations is a different workflow that performs new numerical solves. Use separate output directories and the supplied inputs; do not overwrite these records. Figure regeneration alone is not a full candidate rerun. Tests can be run with `python -m unittest discover -s tests -v`.

## Data interpretation

- `data/systemconf/`: subject, component catalogue and scenario settings.
- `data/weather/temperature/`: Celsius samples, held over their stated intervals.
- `data/weather/pv/`: preceding-interval mean power for a 100-W reference array, scaled once by selected panel rating.
- `data/experiments/`: four local pad-temperature CSV records and their experimental description; not measurements of body-delivered efficiency.
- Primary `parameters.json`/`candidates.csv`/`summary.json`: inputs, enumerated candidates and date outcomes. Candidate JSON/step records retain operation, forecast and audit details. Failed prefixes are not completed missions.
- `output/stress_20261006/`: fixed-package perturbation inputs, trajectories and ledger, not a new hardware search.
- `output/review_revision_20261008/`: self-contained evidence and derived CSV views; five computational figures. `output/experiments/` contains pad-fit outputs. New outputs should be regenerated rather than overwriting these reference files.

The static comparisons use the same seasonal dates retrospectively. No alternative-controller campaign, ablation campaign, or external human/battery/field validation is supplied. These are limitations of the study.

## Licences and attribution

- **Code:** Apache License 2.0, full text in `LICENSE`. This applies to the project software, not automatically to third-party dependencies or all data.
- **Authors' data and accompanying data documentation:** Hong Zhang, Xiang Bai, Xinyue Chang, Jiahui Zhang and Xuan Zhang license the supplied experimental CSVs, configuration records, simulation results and derived data under **Creative Commons Attribution 4.0 International**, https://creativecommons.org/licenses/by/4.0/ (legal terms: https://creativecommons.org/licenses/by/4.0/legalcode). Attribute the authors, the version and associated DOI; identify modifications and retain notices. No endorsement is implied.
- **Weather:** weather data supplied through **Open-Meteo**, https://open-meteo.com/, licensed under CC BY 4.0 according to https://open-meteo.com/en/terms and https://open-meteo.com/en/licence. BGEMS organizes the supplied temperature inputs and converts solar/weather inputs into reference-array PV power; acknowledge those transformations and Open-Meteo when reusing them. API service-use limits are separate from the data licence. This is not a new claim of authorship over weather data.
- Dependencies retain their own licences. Times New Roman must be installed from a licensed source; font binaries are not included.

