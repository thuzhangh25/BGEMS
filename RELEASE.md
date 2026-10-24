# BGEMS: An End-to-End Body Grid Configuration and Control Framework for Wearable Thermal Management — code and research data

Version: **v1.1.1-resubmit**. Version-specific archive: https://doi.org/10.5281/zenodo.23607419 . Code: https://github.com/thuzhangh25/BGEMS/tree/v1.1.1-resubmit . The previous version v1.1.0-resubmit remains archived at https://doi.org/10.5281/zenodo.23230495 .

## Presentation-only changes

- Fig. 9(a): move the legend above the bars and adjust canvas width/title spacing.
- Fig. 12: label the reported quantity as net building-electricity savings.
- Synchronize the framework description and release identity.
- Preserve model code, input parameters, weather, benchtop CSVs, all recorded trajectories and numerical results. Reuse the complete data archive byte for byte; only its distribution filename changes.
- The pinned environment describes the inherited research environment, not a claim that simulations were rerun for this release. Historical execution-time source hashes remain unchanged.

This package does not include manuscript sources, response letters or internal review materials. Publication and final journal submission are separate author-controlled steps.

## Study and contents

This collection evaluates a scenario-driven wearable-heating framework in Harbin skiing (2026-01-01), Muztagh Ata mountaineering (2025-07-15) and Qinling indoor work (2025-07-01). Receding horizon control (RHC) is compared with rule-based control (RBC), proportional–integral–derivative (PID) control and thermal adaptive control strategy (TACS), using common hardware, input profiles and mandatory-load reserve constraints.

The main study cohort contains 4174 recorded trajectories: 12 under unperturbed conditions, 40 pilot Monte Carlo, 4000 formal Monte Carlo, 84 single-factor, six fixed-configuration, 16 selected-down-top indoor and 16 fixed-fleece-jumpsuit indoor trajectories. Of these, 4173 complete electrically; one fixed-configuration case is rejected by its initial reserve condition. The formal ensemble contains 1000 independent paired sample identities; the pilot repeats the first ten identities. Catalogue scaling evaluates 960, 1920 and 3840 configurations.

Four additional equipment-heater activation trajectories use a two-hour temperature profile crossing the −20°C threshold. All four controllers complete this profile and consume 18 Wh for equipment heating. The complete collection therefore contains 4178 executions and 4177 electrical completions; the four diagnostic runs are separate from the main study cohort above.

- **Code archive:** source, tests, pinned environment, model inputs, cached weather, benchtop CSVs, 12 unperturbed trajectories and self-contained figure data.
- **Data archive:** 4166 trajectories, comprising the remaining 4162 main-study trajectories and four equipment-heater activation runs, plus catalogue-search records and application calculations.
- **Manifest and checksums:** execution-time source hashes identify the calculations that produced each record; package hashes identify the supplied files and verify archive integrity.

Extract both archives into the same parent directory; both contain a `BGEMS/` root and have disjoint members. Follow `BGEMS/README.md` for installation, numerical reproduction and figure generation. All required weather inputs are supplied. The verified dependency set uses pandas 2 timestamp semantics; use the pinned environment.

## Interpretation

The localized thermal-comfort score measures modeled heating-demand fulfillment. Electrical completion and state of charge (SOC) describe mission execution and remaining energy. The equipment-heater rule uses installation status and ambient temperature. Benchtop CSVs characterize local temperature responses. Rental cashflows and building-energy calculations use the specified price, utilization, thermal and occupancy inputs. Qinling wearable operation lasts 14 hours within a 24-hour building calculation. PV inputs refer to a 100-W array and are scaled once to the selected hardware.

Project repository: https://github.com/thuzhangh25/BGEMS

## Licences and attribution

- **Code:** Apache License 2.0, full text in `LICENSE`. Third-party dependencies retain their own licences.
- **Authors' data and accompanying documentation:** Hong Zhang, Xiang Bai, Xinyue Chang, Jiahui Zhang and Xuan Zhang license the supplied experimental CSVs, configuration records, simulation results and derived data under Creative Commons Attribution 4.0 International: https://creativecommons.org/licenses/by/4.0/ . Attribute the authors and applicable version, identify modifications and retain notices.
- **Weather:** supplied through Open-Meteo, https://open-meteo.com/ , under CC BY 4.0 according to https://open-meteo.com/en/terms and https://open-meteo.com/en/licence . BGEMS organizes temperature inputs, aggregates seasonal daily statistics and converts solar/weather inputs into reference-array PV power. Credit these transformations and Open-Meteo. Service-use limits are separate from data licensing.
- Obtain Times New Roman from a licensed source for figure generation.
