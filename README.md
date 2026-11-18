# Target-Retraining Headroom

Code and saved results for:

> **Target-Retraining Headroom: A Decision-Support Procedure for Retraining Forecasting Models on New Groups of Entities**
> Md. Salehin Seyam. Manuscript submitted for publication.

A forecasting model trained on one group of entities (facilities, clients, sensors, plants) often has higher error on a new group. That increase does not tell you whether retraining on the new group would help. The paper defines **target-retraining headroom**: the relative reduction in target error from retraining the same model family on matched target-group data.

It then gives a decision procedure that:
- separates the apparent cross-entity error from the benefit of retraining;
- compares retraining with the model already in service at equal training budget;
- estimates that benefit from history observed before a decision date, after a calendar-coverage check;
- evaluates retraining against cheaper actions that need no target data (training longer, seasonal blending, classical forecast combinations), under explicit costs.

The study uses four public panels: CMS nursing-home staffing, UCI electricity, freeway traffic, and solar generation.

This repository contains the code that produced every result and the saved per-run and per-entity results from which every reported number, table and figure is computed.

## Repository layout

```
src/                     models (ERM-Stem, FADE, baselines), data windowing, training loop,
                         nursing panel builder
build_scripts/           builders for the electricity, traffic and solar panels
experiments/             experiment, analysis, table and figure scripts
experiments/paper_figures/
                         sources of Figures 1-3 and of the FADE diagram
experiments/results/     saved per-run and per-entity results: the inputs to every
                         reported number, table and figure; entity manifests in
                         manifests_anchors.csv and manifests_temporal.csv
requirements.txt         package versions used for the article
```

## Reproducing the reported numbers, tables and figures

These steps need only this repository; the raw data are not required (except for Figure 2, which plots electricity series).

```bash
pip install -r requirements.txt
cd experiments
python an_decisions.py mae_cat           # decisions against the deployed model (Table 7)
python an_sequential.py mae_cat          # four-action rule, coverage gate, sequential procedure (Tables 8, 9)
python an_actions.py                     # budgets, convergence and actions (Tables 5, 11)
python an_dg.py                          # tuned domain-generalization baselines
python make_supp_tables.py OUT_DIR       # supplementary decision, action and baseline tables
python make_seq_tables.py OUT_DIR        # supplementary sequential-procedure table
python main_table_rows.py dec            # main-text table rows (also: multi, seq)
python make_numbers.py                   # number macros and tables of the remaining analyses
python verify_numbers.py                 # independent recomputations from the results
python fig_components.py                 # Figure 4
python fig_gates.py                      # supplementary gate figures
python paper_figures/fig3_survey.py      # Figure 3
python paper_figures/fig2_overview.py    # Figure 2 (needs the electricity panel)
```

- Analysis scripts write to `experiments/results/`; `make_numbers.py` writes to `Manuscript/generated/` (set `MN_GEN` to choose another folder) together with `MANIFEST.txt`, which lists the results file behind each table.
- Figure 1 and the FADE diagram are TikZ sources in `experiments/paper_figures/`; the FADE implementation schematic is drawn by `src/fig_architecture_color.py`.
- Every analysis script is deterministic: resampling uses fixed seeds, so rerunning it on the saved results reproduces its outputs exactly.

## Data

The raw data are public and are not redistributed here. Training scripts expect them under `cache/` at the repository root (created by you; ignored by git).

| Panel | Source | Place raw files at | Build with |
|---|---|---|---|
| Nursing | CMS Payroll-Based Journal daily nurse staffing, CY2024Q1–CY2025Q4 (eight quarterly files); CMS Provider Information, July 2026 snapshot | `PBJ_dailynursestaffing_CY*.csv` in the repository root; `cache/NH_ProviderInfo.csv` | `python src/build_panel.py`, then `python experiments/nursing_masks.py` |
| Electricity | UCI ElectricityLoadDiagrams20112014, 321-client version of Lai et al. (2018) | `cache/electricity/electricity_lstnet.txt.gz` | `python build_scripts/build_electricity.py` |
| Traffic, solar | Benchmark files of Lai et al. (2018) | `cache/panels/traffic.txt.gz`, `cache/panels/solar.txt.gz` | `python build_scripts/build_panels.py` |

- The PBJ files are published at https://data.cms.gov/quality-of-care/payroll-based-journal-daily-nurse-staffing.
- The Provider Information file used is `NH_ProviderInfo_Jul2026.csv` from the CMS Provider Data Catalog archive release `nursing-homes_2026-08-06.zip` (dataset 4pq5-n9py). With it, 13,387 of the 13,402 analysis facilities match; later snapshots no longer list some of them.
- `src/build_panel.py` on the eight quarterly files gives the 13,402-facility analysis panel, and `experiments/nursing_masks.py` gives its imputation masks; both match the saved `nursing_imputation_masks.npz` and `nursing_cohort_audit.csv` exactly.

Nursing entities are identified by CMS Certification Number. No individual-level records are used.

## Running the experiments

All scripts are in `experiments/` and write to `experiments/results/`. Training runs on CPU; set the number of threads with `EXP_THREADS` (and, where a script parallelizes over entities, workers with `EXP_WORKERS`). Allocations are deterministic given the seeds, and `export_manifests.py` writes them out explicitly.

| Analysis (section of the article) | Training | Analysis |
|---|---|---|
| Anchor splits and seasonal and classical references (5.1–5.2, 7.1) | `anchor.py`, `classical.py`, `dg_extend.py` | `an_anchor.py`, `an_fallbacks.py` |
| Survey of 36 splits, uncertainty levels (5.3) | `survey.py` | `an_survey.py`, `an_uncertainty.py` |
| Model families (5.4) | `family.py` | in `make_numbers.py` |
| Budgets, convergence, joint training and actions without target data (5.5, 7.2) | `joint_finetune.py`, `budget_control.py`, `budget_curve.py`, `anchor_actions.py` (nursing: `nursing_classical.py` first) | `an_budget_dates.py`, `an_actions.py` |
| Input screens and allocation (6.1) | `screen.py` | `an_screen.py` |
| Observed-history estimate, truncation, coverage (6.2) | `temporal.py`, `seasonal.py`, `coverage_crossed.py` | `an_temporal.py`, `an_coverage_check.py`, `an_coverage_measures.py`, `an_calibration.py`, `an_temporal_decision.py` |
| Several decision dates (6.2) | `origins.py` | `an_budget_dates.py` |
| Decisions against the deployed model, choice among actions, sequential procedure (6.3) | `origins_deployed.py`, `origins_sequential.py` | `an_decisions.py`, `an_sequential.py` |
| Number of target entities | `EXP_STEPMATCH=1 python tentities.py` | `an_tentity_sizes.py` |
| Tuned domain-generalization baselines (7.3) | `dg_tuned.py` | `an_dg.py` |
| Full nursing benchmark (Supplement D) | `fullscale.py` | `an_fullscale.py` |
| Learned frequency gates (7.4) | `gate.py` | in `make_numbers.py` |
| Supplementary ablations and sensitivity sweeps | `supplementary_runs.py` | in `make_numbers.py` |

Each training script takes the panels to run as arguments (for example `python origins_deployed.py Nursing`). `anchor_actions.py` can run seeds in separate processes with `EXP_SEEDS=0`, `EXP_SEEDS=1`, ...; the analysis scripts combine the files. `python diff_macros.py OLD NEW` compares two `numbers.tex` files.

## Environment

These are the versions used to produce the results:
- Python 3.12.7
- PyTorch 2.8.0 (CPU)
- NumPy 2.2.6
- pandas 2.3.3
- SciPy 1.16.3
- statsmodels 0.14.6
- PyWavelets 1.8.0
- Matplotlib 3.10.5
- PyMuPDF 1.28.0 (figure check only)

## Note on tools

Parts of the analysis and plotting code were written with the assistance of a generative AI assistant (Claude, Anthropic), as disclosed in the article. The author reviewed the code and its outputs.

## Citation

```
Md. Salehin Seyam. Target-Retraining Headroom: A Decision-Support
Procedure for Retraining Forecasting Models on New Groups of Entities.
Manuscript submitted for publication.
```

## License

MIT; see `LICENSE`.
