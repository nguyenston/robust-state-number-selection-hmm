# Data used by the paper workflows

The two processed dog CSVs used by the existing workflow are included. The
large raw-timestep CSV is stored with Git LFS; install Git LFS and run
`git lfs pull` from the repository root to download it. The small label-lookup
CSV is stored in ordinary Git. Other local datasets remain ignored. No new
downloader or preprocessing implementation has been added.

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

These are the existing processed exports used in the validated paper runs,
not a new preprocessing implementation. The existing helpers construct the
windowed features and task bouts from them. The original MATLAB-to-CSV exporter
is not present, but it is not needed to run the analysis with these bundled
inputs. The original MATLAB file remains local and is not committed.

### Dog dataset attribution

Source: Vehkaoja, Antti; Somppi, Sanni; Törnqvist, Heini; Valldeoriola Cardó,
Anna; Kumpulainen, Pekka; Väätäjä, Heli; Majaranta, Päivi; Surakka, Veikko;
Kujala, Miiamaaria; and Vainio, Outi (2025), *Movement Sensor Dataset for Dog
Behavior Classification*, Mendeley Data, V4,
[doi:10.17632/vxhx934tbn.4](https://doi.org/10.17632/vxhx934tbn.4).

The source record lists [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The included CSVs are processed/reformatted derivatives; the dataset's terms
are separate from the repository's code license. The source authors also
request citation of the 2021 behavior-classification study and 2022 dataset
description listed in the linked dataset record.
