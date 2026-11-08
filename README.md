# Robust State-Number Selection in Hidden Markov Models

Analysis workflows and generated figures for the
[manuscript](paper/manuscript.pdf), covering elephant movement, spike trains,
and dog behavior. The notebooks and Julia runners are kept separate from the
StateSpaceDynamics package, which is pinned as a Git submodule. This repository
contains the paper workflows, not additional experiments.

```text
.
├── StateSpaceDynamics.jl/    # Package fork, pinned as a Git submodule
├── notebooks/               # Existing workflows and analysis environment
│   ├── elephant_movement.ipynb
│   ├── run_spike_trains.jl
│   ├── run_dog_bouthmm.jl
│   ├── helpers/
│   ├── data/                # Bundled dog CSVs; other inputs kept locally
│   ├── cache/               # Local caches; ignored
│   ├── figures/             # Main and supplementary figure panels
│   └── results/             # Numerical summaries
└── paper/
    └── manuscript.pdf       # Manuscript PDF only; LaTeX sources stay separate
```

## Setup and execution

Install [Git LFS](https://git-lfs.com/) before cloning:

```sh
git lfs install
git clone --recurse-submodules https://github.com/nguyenston/robust-state-number-selection-hmm.git
cd robust-state-number-selection-hmm
git lfs pull
```

On Nix, enter `nix-shell -p git-lfs` to make the LFS command available for Git
operations without a global installation.
For an existing clone, run `git submodule update --init --recursive` and
`git lfs pull` from the repository root. The bundled dog raw-timestep CSV is
approximately 2 GB and is stored in Git LFS; the small dog label-lookup CSV is
stored in ordinary Git. The submodule uses HTTPS and does not require a GitHub
SSH key for public cloning.

The spike-train workflow generates its synthetic input. The dog workflow uses
the two bundled CSVs. Before running the elephant notebook, obtain the external
GPS input and place it at
`notebooks/data/movement-syndrome-vertebrate-taxa-gps.csv` as described in the
[data instructions](notebooks/data/README.md), including provenance and data
licensing.

Use Julia 1.12 with the recorded analysis environment:

```sh
julia --project=notebooks -e 'using Pkg; Pkg.instantiate()'
julia --project=notebooks notebooks/run_spike_trains.jl
julia --project=notebooks notebooks/run_dog_bouthmm.jl
```

Open `notebooks/elephant_movement.ipynb` in Jupyter with a Julia kernel and run
all cells in order. Its setup cell activates the same environment. The package
resolves locally to `StateSpaceDynamics.jl/`; Julia `using` statements are
unchanged.

See [notebook documentation](notebooks/README.md) for the paper figure map,
experiment settings, and validation scope, and [data instructions](notebooks/data/README.md)
for dataset provenance and the required external elephant GPS input. The two
processed dog CSVs are included; other local datasets and model caches are not
committed.
Move `notebooks/cache/` aside before running to force a fresh refit.

## Validation

Fresh-refit validation completed successfully on November 6, 2026 using Julia
1.12.7. All pre-existing study caches and outputs were moved aside. All 18
elephant notebook code cells and both study workflows completed, regenerating
22 PDF/PNG figure pairs and four numerical summaries. All 200 spike fits and
280 dog fits succeeded; all three movement sweeps produced valid best fits at
every K. The independent higher-K dog fits were parallelized using the existing
helper, and the unchanged runner freshly evaluated all 280 fits and exported
the results. No pre-existing fitted-model or selection-analysis cache was used.

See the [validation details](notebooks/README.md#validation) for execution
settings and scope. This validates the paper workflows with the local datasets,
not the entire package test suite or data acquisition on a clean machine.

## Package version and license

The StateSpaceDynamics submodule is pinned to
[`642089f`](https://github.com/nguyenston/StateSpaceDynamics.jl/commit/642089f4e23b1236073335a0b246b1ec03247243)
on the package fork's `codex/paper-reproduction` branch. Keep the recorded
submodule revision and notebook environment when reproducing the paper.

Code is distributed under [GPL-3.0](LICENSE). Dataset licensing and attribution
are described separately in the [data instructions](notebooks/data/README.md).
