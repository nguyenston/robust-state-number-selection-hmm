# Paper reproduction workflows

This directory contains the existing workflows for *Robust State-Number
Selection in Hidden Markov Models*. The Julia package is the sibling submodule
`../StateSpaceDynamics.jl/`;
analysis dependencies are isolated in this directory's environment.

## Layout

```text
notebooks/
├── elephant_movement.ipynb
├── run_spike_trains.jl
├── run_dog_bouthmm.jl
├── Project.toml
├── Manifest.toml
├── helpers/
│   ├── AnimalMovementUtils.jl  # Movement module and includes
│   ├── SpikeTrainHMMUtils.jl   # Spike module and includes
│   ├── DogNotebookUtils.jl     # Dog module and includes
│   ├── FigureExport.jl        # Existing PNG/PDF export and optional trimming
│   ├── movement/              # Data, emissions, fitting, plots, statistics
│   ├── spikes/                # Generation, emission, fitting, plots, statistics
│   └── dogs/                  # Data, fitting, alignment, layout, plots, statistics
├── data/                    # Bundled dog CSVs and local inputs; see data/README.md
├── cache/                   # Local fitted-model and analysis caches; ignored
├── figures/main/            # Figure panels used in the main paper
├── figures/supplementary/   # Figure panels used in the supplement
└── results/                 # Existing numerical summaries
```

The notebook and runners contain setup, paper experiment settings, and workflow
calls. Figure construction and statistics formatting live in the study-specific
helper files, including the movement state-grid export previously embedded in
the notebook. The three original module names are retained so existing serialized
model caches still resolve their emission types. The dog figure and statistics
helpers share the existing pair-count calculation and compute Hungarian alignment
without constructing unused composition figures. No new experiment or selection
algorithm has been introduced.

## Environment and execution

From the repository root, instantiate the notebook environment:

```sh
julia --project=notebooks -e 'using Pkg; Pkg.instantiate()'
```

`StateSpaceDynamics` resolves to the sibling submodule through a relative path;
no separately installed copy of the package is required. The manifest records
the dependency versions for this analysis environment. Use Julia 1.12 for the
recorded environment and existing model caches. Fitting and discrepancy
estimation are stochastic, and this refactor does not add new random seeds or
change the existing random-number handling.

Run the two existing script workflows from the repository root:

```sh
julia --project=notebooks notebooks/run_spike_trains.jl
julia --project=notebooks notebooks/run_dog_bouthmm.jl
```

For the elephant analysis, start Jupyter in `notebooks/`, open
`elephant_movement.ipynb` with a Julia kernel, and run the cells in order. The
first cell activates this directory's environment. Saved cell outputs have
been cleared; the existing paper plots are available under `figures/`.

The spike workflow generates its observations. The elephant and dog workflows
require the local files documented in [data/README.md](data/README.md).
The full experiment settings are retained in the entry points and helper
defaults rather than duplicated into a separate configuration layer.

## Paper figure map

Paths in the last column are relative to `figures/`. PDF and PNG versions are
retained. Multi-panel figures are assembled from these panels by the manuscript.

| Paper output | Entry point | Figure files |
|---|---|---|
| Figure 1, matched/misspecified selection | `run_spike_trains.jl` | `main/spike_trains/{matched,mismatched}/spike_selection_poisson_kl.pdf` |
| Figure 2, three movement specifications | `elephant_movement.ipynb` | `main/elephant_movement/movement_selection_{base,alt1,alt2}.pdf` |
| Figure 3, three-state movement summaries | `elephant_movement.ipynb` | `main/elephant_movement/movement_state_feature_grid_{base,alt1,alt2}_k3.pdf` |
| Figure 4, bout-HMM selection | `run_dog_bouthmm.jl` | `main/dog_biologging/selection_bouthmm.pdf` |
| Figure 5, aligned behavior-pair panels | `run_dog_bouthmm.jl` | `main/dog_biologging/pair_distribution_k1_k10.pdf` |
| Web Figures 1–2, contaminated spike compositions | `run_spike_trains.jl` | `supplementary/spike_trains/spike_composition_mismatched_{bic,acdc_kl}.pdf` |
| Web Figure 3, elephant trajectory | `elephant_movement.ipynb` | `supplementary/elephant_movement/movement_trajectory.pdf` |
| Web Figures 4–5, movement fits at K=2 and K=4 | `elephant_movement.ipynb` | `supplementary/elephant_movement/movement_state_feature_grid_{base,alt1,alt2}_k{2,4}.pdf` |
| Web Figures 6–8, movement fits at K=8 | `elephant_movement.ipynb` | `supplementary/elephant_movement/movement_state_feature_grid_{base,alt1,alt2}_k8.pdf` |

The existing summaries in `results/` contain the spike ground-truth occupancies
used in Table 1 and the dog behavior summaries supporting Table 2. The
plain-language interpretations in Table 2 remain manuscript text; the runners
do not generate that table automatically.

## Details preserved from the original workflows

- Movement uses seed 42 and `K=1:8` for all three emission models. The helper
  defaults split gaps longer than 1.5 hours and retain the likelihood-best fit
  at each K. The paper state grids are exported for K=2, 3, 4, and 8.
- Spikes use the existing five-state firing-rate vectors, 20,000 bins of
  100 ms, 10 fitting restarts per K, and the existing contamination-mode
  generation. Each criterion retains its own best restart at each K.
- Dogs use the original most-observed-plus-two-least-observed dog selection,
  10-timestep feature windows, independent task-bout fitting, `K=1:14`, 20
  restarts, and seed 7777. Criterion-specific fits, full-timeline interpretation,
  and Hungarian alignment across K=1:14 are preserved, including when only
  K=1:10 panels are exported for Figure 5.
- Existing component-driver computations remain in the fitting routines,
  including computations whose diagnostic plots were pruned. Removing random
  draws there could change later fits or discrepancies.
- The dog stacked-bar padding implementation is retained because it controls
  the figure layout. Optional external `magick` and `pdfcrop` programs perform
  the same trimming as before when available.

The paper's ACDC recommendation is an interpretation of the envelope elbow.
The existing statistics also report the candidate with minimum worst-component
discrepancy, which can be a later envelope point. These are different quantities;
the refactor does not substitute the latter for the paper's elbow recommendation
or add an automatic percentage-cutoff rule.

## Caches and existing outputs

The runners use caches by default when present. The analysis file in the dog workflow
and the generated-data caches in the spike workflow retain their original
loading behavior. To force a complete refit or regeneration, move the relevant
study cache directory aside before executing that workflow. Cache files do not
fully encode every analysis parameter, so use a fresh cache when changing the
experiment settings.

## Validation

On November 6, 2026, all pre-existing caches, figures, and summaries were moved
aside before a fresh-refit check in this repository. All 18 elephant notebook
code cells and both study workflows completed successfully, producing 22
PDF/PNG figure pairs and four text summaries. Checkpoint audits verified 24
movement best-fit models, 200 successful spike fits, and 280 successful dog
fits, with finite fitted log-likelihoods throughout. The fresh dog analysis
evaluated all 280 fits, with finite BIC/ICL and best-per-K discrepancies.

All workflows used Julia 1.12.7 and `OPENBLAS_NUM_THREADS=1`. The elephant
notebook ran in a Jupyter Julia kernel. Dog K=1–9 fitting used eight Julia
threads; independent K=10–14 fits used two-thread worker processes calling the
unchanged fitting helper with the original seeds, 20 restarts, 200-iteration
limit, and tolerance. Only fully written fresh checkpoints were passed to the
unchanged dog runner for fresh selection analysis and output generation. No
pre-existing fitted models or selection-analysis results were reused. The
temporary execution/audit harnesses are not part of the repository; the study
source code and experiment settings were not changed.

This checks complete refitting and output generation with the local input
datasets, not the full package test suite or a clean-machine installation/data
acquisition. Fitting and discrepancy estimation retain their original random
number handling, so regenerated figures are not guaranteed to be byte-identical
to earlier runs. Output directories are created automatically when absent.
The repository notebook keeps its saved cell outputs cleared.

### Refactor regression check

On November 7, 2026, the original and refactored workflows were executed in
separate temporary directories against the same existing input files and model
caches, using Julia 1.12.7 and `OPENBLAS_NUM_THREADS=1`. Both elephant notebooks
completed all 18 code cells in Julia Jupyter kernels, and both versions of each
script completed successfully. A seed was set only in the temporary spike
comparison harness to compare stochastic discrepancy evaluations; production
seed handling was not changed.

All 22 PNGs matched pixel-for-pixel, all 22 PDF comparisons matched after
rendering at a maximum dimension of 1600 pixels, and all four text summaries
matched after excluding generation timestamps. Source comparisons also verified
that preprocessing, emission models, fitting/evaluation routines, the Hungarian
assignment algorithm, and plotting-backend padding were preserved. The only
fitting-file change adjusts the movement cache path for its new helper directory.
The recorded environment and package submodule were unchanged.

This was a cached before/after regression check, not another full uncached
refit. Its temporary scripts, executed notebooks, and generated outputs are not
part of the repository; the existing committed paper outputs were left intact.
