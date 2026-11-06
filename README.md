# Robust State-Number Selection in Hidden Markov Models

Paper-focused analysis repository. Existing notebooks and Julia runners are
kept separate from the StateSpaceDynamics package; no new experiments or
selection algorithms are introduced by this reorganization.

```text
.
├── StateSpaceDynamics.jl/    # Package fork, pinned as a Git submodule
├── notebooks/               # Existing workflows and analysis environment
│   ├── elephant_movement.ipynb
│   ├── run_spike_trains.jl
│   ├── run_dog_bouthmm.jl
│   ├── helpers/
│   ├── data/                # Local inputs; raw datasets ignored
│   ├── cache/               # Local caches; ignored
│   ├── figures/             # Main and supplementary figure panels
│   ├── results/             # Numerical summaries
│   └── docs/
└── paper/
    └── manuscript.pdf       # Manuscript PDF only; LaTeX sources stay separate
```

## Setup and execution

Clone with `git clone --recurse-submodules <repository-url>`, or initialize an
existing clone with `git submodule update --init --recursive`.

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
for required external datasets. Raw data and model caches are not committed.
Move `notebooks/cache/` aside before running to force a fresh refit.

The retained outputs passed the previous cached end-to-end run. The subsequent
uncached run was interrupted for this repository reorganization and is not a
completed validation. A fresh end-to-end run in this layout is still pending.

## Local publication status

This repository was initialized locally; it has no GitHub remote yet. The
package submodule includes a local snapshot commit of the existing package and
test changes. That commit must be pushed to the package fork before publishing
this repository, so that recursive clones can retrieve the pinned revision.
