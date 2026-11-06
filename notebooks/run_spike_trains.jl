using Pkg
Pkg.activate(@__DIR__)

ENV["GKS_ENCODING"] = "utf8"
const SCRIPT_DIR  = @__DIR__
const FIGURES_DIR = joinpath(SCRIPT_DIR, "figures", "main", "spike_trains")
const CACHE_DIR   = joinpath(SCRIPT_DIR, "cache", "spiketrain_v2c_cont")
mkpath(FIGURES_DIR); mkpath(CACHE_DIR)

using LinearAlgebra, Random, Serialization, Statistics, Printf
using Distributions
using Plots, Measures
using StateSpaceDynamics
using StateSpaceDynamics.ACDC

include(joinpath(SCRIPT_DIR, "helpers", "SpikeTrainHMMUtils.jl"))
using .SpikeTrainHMMUtils

const RATES_V2C = [
    [ 2.0,  2.0,  2.0,  2.0,  2.0],    # SILENT
    [ 2.0, 25.0, 80.0,  2.0, 25.0],    # CHAOTIC
    [25.0, 30.0, 20.0, 35.0, 25.0],    # UP
    [80.0,  5.0, 80.0,  5.0, 80.0],    # BURST_A
    [ 5.0, 80.0,  5.0, 80.0,  5.0],    # BURST_B
]
const LABELS_V2C = ["silent", "chaotic", "up", "burst_a", "burst_b"]

const N_MODES = 10
let rng = MersenneTwister(123)
    global CONT_MODES = [
        [exp(rand(rng, Uniform(log(0.5), log(50.0)))) for _ in 1:5]
        for _ in 1:N_MODES
    ]
end
mode_totals = [sum(m) for m in CONT_MODES]
println("Generated $(N_MODES) random contamination modes (log-uniform [0.5, 50] Hz/neuron):")
@printf("  mode-total Hz: min=%.1f  median=%.1f  max=%.1f\n",
    minimum(mode_totals), median(mode_totals), maximum(mode_totals))
println("  vs GT total Σλ × 10 (per-bin counts): SILENT=1.0 CHAOTIC=13.4 UP=13.5 BURST_B=17.5 BURST_A=25.0")

println("\n=== MATCHED (v2c_cont: Poisson, baseline) ===")
spike_poisson = generate_synthetic_spikes(; seed=42, T=20_000, bin_ms=100,
    cache_dir=CACHE_DIR,
    rates_per_state=RATES_V2C, state_labels=LABELS_V2C, rates_tag="v2c_cont");
describe_ground_truth(spike_poisson)

res_matched = compute_spike_analysis_poisson(spike_poisson;
    K_range=1:10, n_restarts=10, max_iters=200, tol=1e-4,
    use_cache=true, cache_dir=joinpath(CACHE_DIR, "matched"));
println("MATCHED  BIC K=", res_matched.selections.bic, "  ICL K=", res_matched.selections.icl)

plots_matched = get_spike_plots(res_matched, spike_poisson;
    figures_dir=joinpath(FIGURES_DIR, "matched"))
println("\n=== MISMATCHED 2.0× NegBin + 1%/step burst contamination (10-step bursts, 10 modes mid-hot log-uniform [0.5,50]) ===")
spike_cont = generate_synthetic_spikes_negbin(; seed=42, T=20_000, bin_ms=100,
    overdispersion=2.0,
    contamination_rate=0.01,
    contamination_burst_length=10,
    contamination_modes=CONT_MODES,
    cache_dir=CACHE_DIR,
    rates_per_state=RATES_V2C, state_labels=LABELS_V2C, rates_tag="v2c_cont");
describe_ground_truth(spike_cont)
n_cont = sum(spike_cont.contamination_log)
println("contaminated bins: $(n_cont) / $(spike_cont.T)  ($(round(100*n_cont/spike_cont.T; digits=1))%)")
println("avg bins per mode: $(round(n_cont / N_MODES; digits=1))")

res_mismatched = compute_spike_analysis_poisson(spike_cont;
    K_range=1:10, n_restarts=10, max_iters=200, tol=1e-4,
    use_cache=true, cache_dir=joinpath(CACHE_DIR, "mismatched"));
println("MISMATCH  BIC K=", res_mismatched.selections.bic, "  ICL K=", res_mismatched.selections.icl)

plots_mismatched = get_spike_plots(res_mismatched, spike_cont;
    figures_dir=joinpath(FIGURES_DIR, "mismatched"))
for src in (:bic, :acdc_kl)
    plot_spike_composition_grid(res_mismatched, spike_cont;
        K_list=2:9, figures_dir=joinpath(SCRIPT_DIR, "figures", "supplementary", "spike_trains"),
        label="mismatched", model_source=src)
end

write_spike_statistics(joinpath(SCRIPT_DIR, "results", "spike_statistics_v2c_cont.txt"),
    spike_poisson,
    [("matched", res_matched), ("mismatched_burstcontam", res_mismatched)])
println("DONE")
