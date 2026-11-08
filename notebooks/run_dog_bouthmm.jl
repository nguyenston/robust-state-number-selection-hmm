using Pkg
Pkg.activate(@__DIR__)

ENV["GKS_ENCODING"] = "utf8"
const SCRIPT_DIR  = @__DIR__
const FIGURES_DIR = joinpath(SCRIPT_DIR, "figures", "main", "dog_biologging")
const DATA_DIR    = joinpath(SCRIPT_DIR, "data", "DogMoveMendeley2021")
const PROCESSED_DIR = joinpath(DATA_DIR, "processed")
mkpath(FIGURES_DIR)

using DataFrames, CSV, Serialization
using StateSpaceDynamics.ACDC

include(joinpath(SCRIPT_DIR, "helpers", "DogNotebookUtils.jl"))
using .DogNotebookUtils

# --- Data loading -----------------------------------------------------------
const DOGMOVE_RAW_CSV    = joinpath(PROCESSED_DIR, "dog_raw_timestep.csv")
const DOGMOVE_LOOKUP_CSV = joinpath(PROCESSED_DIR, "dog_label_lookup.csv")

dog_id_col = CSV.read(DOGMOVE_RAW_CSV, DataFrame; select=[:dog_id])
dog_counts = combine(groupby(dog_id_col, :dog_id), nrow => :n_rows)
dog_id_col = nothing
sort!(dog_counts, [:n_rows, :dog_id])
dog_least_two = Int.(first(dog_counts, min(2, nrow(dog_counts))).dog_id)
dog_most_one  = Int(last(dog_counts, 1).dog_id[1])
const DOG_SELECTOR = unique(vcat([dog_most_one], dog_least_two))
println("DOG_SELECTOR (most + two least): ", DOG_SELECTOR)

loaded = load_dogmove_multi_csv(DOGMOVE_RAW_CSV, DOGMOVE_LOOKUP_CSV;
    window=10, dog_selector=DOG_SELECTOR)
const sdict        = loaded.sdict
const WINDOW       = loaded.WINDOW
const observations = loaded.observations
println("observations size: ", size(observations))

# =============================================================================
# Bout-HMM analysis (per-metric-best) — selection and pair distribution
# =============================================================================
println("\n" * "=" ^ 70)
println("BOUT-HMM ANALYSIS")
println("=" ^ 70)
const BOUT_CACHE_DIR = joinpath(SCRIPT_DIR, "cache", "dog_multi_per_metric_v2", "bouthmm")
mkpath(BOUT_CACHE_DIR)

bout_data = segment_task_bouts(observations, sdict; window=WINDOW, min_windows=2)
println("Segmented $(length(bout_data.sequences)) task bouts (min 2 windows each)")

# Fit + analysis cache for bout-HMM
bouthmm_run = compute_bouthmm_models_allruns(bout_data.sequences;
    K_range=1:14, n_restarts=20, seed=7777, max_iters=200, tol=1e-4,
    cache_dir=BOUT_CACHE_DIR, use_cache=true)

const BOUTHMM_ANALYSIS_CACHE = joinpath(BOUT_CACHE_DIR, "selection_analysis.jls")
res_bhmm = if isfile(BOUTHMM_ANALYSIS_CACHE)
    println("Loaded bout-HMM selection analysis from cache")
    deserialize(BOUTHMM_ANALYSIS_CACHE)
else
    println("Bout-HMM per-metric-best selection (fresh evaluation)")
    r = run_bouthmm_selection_analysis_allruns(bouthmm_run, bout_data.sequences;
        discrepancy=KLDiscrepancy(), n_samples=1)
    serialize(BOUTHMM_ANALYSIS_CACHE, r)
    r
end
println("Bout-HMM selections: BIC=", res_bhmm.selections.bic,
        "  ICL=", res_bhmm.selections.icl,
        "  ACDC-KL=", res_bhmm.selections.acdc)

plot_bouthmm_selection(res_bhmm; figures_dir=FIGURES_DIR)

# Align all K=1:14 fits; Figure 5 exports the existing K=1:10 panels.
K_LIST_BHMM = collect(1:14)
sub_run = DogNotebookUtils._build_run_for_metric_bout(bouthmm_run, res_bhmm.models_per_metric.acdc_kl)
println("\n=== Bout-HMM pair plots (acdc_kl-best) ===")
plot_bouthmm_pairs(sub_run, bout_data, observations, sdict, K_LIST_BHMM;
    window=WINDOW, out_dir=FIGURES_DIR)
println("  Saved bout-HMM pair plots to $FIGURES_DIR")

write_bouthmm_statistics(joinpath(SCRIPT_DIR, "results", "selection_stats_bouthmm.txt"),
    res_bhmm, bout_data, observations, sdict; window=WINDOW)
write_bouthmm_pair_statistics(joinpath(SCRIPT_DIR, "results", "selection_stats_bouthmm_pairs.txt"),
    res_bhmm, bouthmm_run, bout_data, observations, sdict;
    window=WINDOW, alignment_K_list=K_LIST_BHMM)

println("\nDONE")
