# Data used by the paper workflows

Local datasets are ignored by Git. The reorganization preserves the existing
files on this machine and their names; it does not add a downloader or a new
preprocessing implementation.

## Synthetic spike trains

`run_spike_trains.jl` generates the observations with the existing generators
in `helpers/SpikeTrainHMMUtils.jl`. No external input file is required.

## Elephant movement

Place the Movebank movement-syndrome GPS CSV at:

```text
notebooks/data/movement-syndrome-vertebrate-taxa-gps.csv
```

The paper cites the Movebank Data Repository record
[10.5441/001/1.hm5nk220](https://doi.org/10.5441/001/1.hm5nk220).
The notebook loads the CSV, computes the existing step/angle features, and
selects the most-observed individual, elephant 8 for the paper input.

## Dog biologging

The paper cites Mendeley Data version 4:
[10.17632/vxhx934tbn.4](https://doi.org/10.17632/vxhx934tbn.4).
The existing runner consumes these processed exports:

```text
notebooks/data/DogMoveMendeley2021/processed/dog_raw_timestep.csv
notebooks/data/DogMoveMendeley2021/processed/dog_label_lookup.csv
```

The historical local directory name `DogMoveMendeley2021` is preserved; it does
not identify the data release used by the paper. The cited release is version 4.

The raw-timestep CSV must contain `trial_index`, `t`, `dog_id`, `test_num`,
the x/y/z columns for `A_back`, `A_neck`, `G_back`, and `G_neck`, `task_label`,
and `behavior_label_1`, `behavior_label_2`, and `behavior_label_3`.
The lookup CSV supplies `label_key`, `code`, and `label`.

The existing helpers construct the windowed features and task bouts from these
CSVs. The MATLAB-to-CSV export code is not present in the original workflow
folder. These two processed CSV inputs are therefore still required separately;
the source download alone is not sufficient to run this analysis. Existing
local MATLAB data and processed CSVs have been preserved.
