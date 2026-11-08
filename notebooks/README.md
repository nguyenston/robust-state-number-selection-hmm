# Paper reproduction workflows

This directory contains the paper workflows for *Robust State-Number
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
│   ├── FigureExport.jl        # PNG/PDF export and optional trimming
│   ├── movement/              # Data, emissions, fitting, plots, statistics
│   ├── spikes/                # Generation, emission, fitting, plots, statistics
│   └── dogs/                  # Data, fitting, alignment, layout, plots, statistics
├── data/                    # Bundled dog CSVs and local inputs; see data/README.md
├── cache/                   # Local fitted-model and analysis caches; ignored
├── figures/main/            # Figure panels used in the main paper
├── figures/supplementary/   # Figure panels used in the supplement
└── results/                 # Numerical summaries
```

The notebook and runners contain setup, paper experiment settings, and workflow
calls. Figure construction and statistics formatting live in the study-specific
helper files.

## Environment and execution

From the repository root, instantiate the notebook environment:

```sh
julia --project=notebooks -e 'using Pkg; Pkg.instantiate()'
```

`StateSpaceDynamics` resolves to the sibling submodule through a relative path;
no separately installed copy of the package is required. The manifest records
the dependency versions for this analysis environment. Use Julia 1.12 for the
recorded environment and model caches. Fitting and discrepancy estimation are
stochastic, so reruns are not guaranteed to produce byte-identical figures.

Run the two script workflows from the repository root:

```sh
julia --project=notebooks notebooks/run_spike_trains.jl
julia --project=notebooks notebooks/run_dog_bouthmm.jl
```

For the elephant analysis, start Jupyter in `notebooks/`, open
`elephant_movement.ipynb` with a Julia kernel, and run the cells in order. The
first cell activates this directory's environment. Paper plots are available
under `figures/`; the notebook is distributed without saved cell outputs.

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

The summaries in `results/` contain the spike ground-truth occupancies
used in Table 1 and the dog behavior summaries supporting Table 2. The
plain-language interpretations in Table 2 remain manuscript text; the runners
do not generate that table automatically.

## Experiment settings

- Movement uses seed 42 and `K=1:8` for all three emission models. The helper
  defaults split gaps longer than 1.5 hours and retain the likelihood-best fit
  at each K. The paper state grids are exported for K=2, 3, 4, and 8.
- Spikes use five-state firing-rate vectors, 20,000 bins of
  100 ms, and 10 fitting restarts per K. The misspecified case includes
  overdispersion and burst contamination. Each criterion retains its own best
  restart at each K.
- Dogs use the most-observed-plus-two-least-observed dog selection,
  10-timestep feature windows, independent task-bout fitting, `K=1:14`, 20
  restarts, and seed 7777. Criterion-specific fits, full-timeline interpretation,
  and Hungarian alignment use K=1:14; Figure 5 exports K=1:10 panels.

The paper's ACDC recommendation is an interpretation of the envelope elbow.
The statistics also report the candidate with minimum worst-component
discrepancy, which can be a later envelope point. These are different quantities;
the numerical minimum is not the paper's elbow recommendation. The workflows
do not apply an automatic percentage-cutoff rule.

## Reproducibility notes

The workflows reuse model caches when present. The dog workflow also caches
selection analysis, and the spike workflow caches generated observations.
To force a complete refit or regeneration, move the relevant
study cache directory aside before executing that workflow. Cache files do not
fully encode every analysis parameter, so use a fresh cache when changing the
experiment settings.

Output directories are created automatically. The workflows export 22 PDF/PNG
figure pairs and four numerical summaries. Optional `magick` and `pdfcrop`
programs trim figure margins when available; installing them can affect output
dimensions. Julia 1.12.7 with `OPENBLAS_NUM_THREADS=1` is a tested configuration.
