module SpikeTrainHMMUtils

ENV["GKS_ENCODING"] = "utf8"

using Dates
using LinearAlgebra
using Printf
using Random
using Serialization
using SpecialFunctions: loggamma
using Statistics
using Distributions
using LaTeXStrings
using Plots, Measures
using StatsPlots: groupedbar
using StateSpaceDynamics
using StateSpaceDynamics.ACDC
using StateSpaceDynamics: viterbi

include("FigureExport.jl")

export PoissonEmission
export generate_synthetic_spikes, generate_synthetic_spikes_negbin, describe_ground_truth
export compute_spike_analysis_poisson
export get_spike_plots, write_spike_statistics
export plot_spike_composition_grid

# =============================================================================
# PoissonEmission: multivariate Poisson with one rate per dimension
# =============================================================================

"""
    PoissonEmission

Multivariate Poisson emission with one rate per dimension (independent across dims).

- `output_dim::Int`: number of neurons / channels
- `λ::Vector{T}`: expected counts per bin, length `output_dim`, λ ≥ 0
"""
mutable struct PoissonEmission{T<:Real,V<:AbstractVector{T}} <: StateSpaceDynamics.EmissionModel
    output_dim::Int
    λ::V
end

function PoissonEmission(; output_dim::Int, λ::AbstractVector{T}) where {T<:Real}
    @assert length(λ) == output_dim
    @assert all(λ .>= 0)
    return PoissonEmission(output_dim, λ)
end

# Log-pmf of Poisson: y·log(λ) − λ − loggamma(y+1)
function StateSpaceDynamics.loglikelihood(
    model::PoissonEmission, Y::AbstractMatrix{T}
) where {T<:Real}
    n_obs, d = size(Y)
    @assert d == model.output_dim
    ll = zeros(T, n_obs)
    eps_T = eps(T)
    @inbounds for i in 1:n_obs
        for j in 1:d
            λ = max(model.λ[j], eps_T)
            y = Y[i, j]
            ll[i] += y * log(λ) - λ - loggamma(y + 1)
        end
    end
    return ll
end

# Weighted-mean MLE for Poisson rate, one per dim
function StateSpaceDynamics.fit!(
    model::PoissonEmission, Y::AbstractMatrix{T},
    w::AbstractVector{T} = ones(T, size(Y, 1)),
) where {T<:Real}
    d = size(Y, 2)
    wsum = sum(w)
    eps_T = eps(T)
    for j in 1:d
        model.λ[j] = max(sum(w .* @view(Y[:, j])) / max(wsum, eps_T), eps_T)
    end
    return model
end

function Random.rand(rng::AbstractRNG, model::PoissonEmission; n::Int = 1)
    d = model.output_dim
    T = eltype(model.λ)
    samples = Matrix{T}(undef, n, d)
    for j in 1:d
        p = Poisson(max(model.λ[j], eps(T)))
        for i in 1:n
            samples[i, j] = T(rand(rng, p))
        end
    end
    return samples
end

Random.rand(model::PoissonEmission; kwargs...) =
    rand(Random.default_rng(), model; kwargs...)

"""
Randomized PIT for a discrete (Poisson) distribution:
    ε = F(y − 1; λ) + u · (F(y; λ) − F(y − 1; λ)),   u ∼ U(0, 1)
with F(−1; λ) = 0.  Yields ε ∼ U(0, 1) if y ∼ Poisson(λ).
"""
function StateSpaceDynamics.ACDC._emission_to_driver!(
    ε_out::AbstractVector{T}, emission::PoissonEmission,
    y::AbstractVector{T}, ::Nothing,
) where {T<:Real}
    for d in eachindex(ε_out)
        λ = max(emission.λ[d], eps(T))
        yd = round(Int, y[d])
        p = Poisson(λ)
        F_y  = cdf(p, yd)
        F_ym1 = yd > 0 ? cdf(p, yd - 1) : zero(T)
        u = rand()
        ε_out[d] = clamp(F_ym1 + u * (F_y - F_ym1), eps(T), one(T) - eps(T))
    end
end

# =============================================================================
# Synthetic spike-train generator
# =============================================================================

struct SpikeGroundTruth
    K_true::Int
    rates_per_state::Vector{Vector{Float64}}  # Hz per neuron, per state
    bin_seconds::Float64
    A::Matrix{Float64}
    π₀::Vector{Float64}
    state_labels::Vector{String}
end

"""
    generate_synthetic_spikes(; seed=42, T=10000, bin_ms=10, cache_dir=nothing) -> NamedTuple

Sample a synthetic spike-train sequence from a known Poisson HMM.

Ground truth (K_true=5, D=5):
  1. SILENT:  ~1 Hz uniform
  2. DOWN:    ~5 Hz uniform
  3. UP:      ~25-35 Hz, neuron-specific
  4. BURST_A: neurons {1,3,5} at 80 Hz, {2,4} at 5 Hz
  5. BURST_B: neurons {2,4} at 80 Hz, {1,3,5} at 5 Hz

BURST_A and BURST_B have the same total rate but distinct spatial patterns —
distinguishable only through per-neuron rates, not total counts.
Sticky transitions (A_ii = 0.96). Bin width `bin_ms` ms.
"""
function generate_synthetic_spikes(;
    seed::Int       = 42,
    T::Int          = 10_000,
    bin_ms::Real    = 10.0,
    cache_dir::Union{String,Nothing} = nothing,
    rates_per_state::Union{Nothing,Vector{Vector{Float64}}} = nothing,
    state_labels::Union{Nothing,Vector{String}}             = nothing,
    rates_tag::Union{Nothing,String}                        = nothing,
)
    tag_suffix = isnothing(rates_tag) ? "" : "_$(rates_tag)"
    cache_file = isnothing(cache_dir) ? nothing :
        joinpath(cache_dir, "spikes_seed$(seed)_T$(T)_bin$(bin_ms)$(tag_suffix).jls")
    if !isnothing(cache_file) && isfile(cache_file)
        println("Loading cached spike data: $cache_file")
        return deserialize(cache_file)
    end

    rng = MersenneTwister(seed)
    bin_s = bin_ms / 1000.0   # seconds per bin

    if isnothing(rates_per_state)
        rates_per_state = [
            [ 1.0,  1.0,  1.0,  1.0,  1.0],    # SILENT
            [ 5.0,  5.0,  5.0,  5.0,  5.0],    # DOWN
            [25.0, 30.0, 20.0, 35.0, 25.0],    # UP
            [80.0,  5.0, 80.0,  5.0, 80.0],    # BURST_A
            [ 5.0, 80.0,  5.0, 80.0,  5.0],    # BURST_B
        ]
    end
    K_true = length(rates_per_state)
    D = length(rates_per_state[1])

    A = fill(0.01, K_true, K_true)
    for k in 1:K_true; A[k, k] = 0.96; end
    π₀ = fill(1.0 / K_true, K_true)
    if isnothing(state_labels)
        state_labels = ["silent", "down", "up", "burst_a", "burst_b"]
    end
    gt = SpikeGroundTruth(K_true, rates_per_state, bin_s, A, π₀, state_labels)

    # Sample state sequence
    states = Vector{Int}(undef, T)
    states[1] = rand(rng, Categorical(π₀))
    for t in 2:T
        states[t] = rand(rng, Categorical(A[states[t-1], :]))
    end

    # Sample spike counts per bin, per neuron
    counts = Matrix{Float64}(undef, T, D)
    for t in 1:T
        rates = rates_per_state[states[t]]
        for d in 1:D
            λ = rates[d] * bin_s
            counts[t, d] = rand(rng, Poisson(λ))
        end
    end

    # Observations matrix: D × T (consistent with the HMM convention)
    observations = Matrix(transpose(counts))

    result = (
        seed          = seed,
        T             = T,
        bin_ms        = bin_ms,
        ground_truth  = gt,
        states        = states,
        counts        = counts,          # T × D for convenience
        observations  = observations,    # D × T for HMM fit
    )

    if !isnothing(cache_file)
        mkpath(dirname(cache_file)); serialize(cache_file, result)
        println("Cached spike data to: $cache_file")
    end
    return result
end

"""
    generate_synthetic_spikes_negbin(; seed=42, T=10000, bin_ms=10, overdispersion=2.0, cache_dir=nothing)

**Mismatched case.** Same state structure and mean rates as `generate_synthetic_spikes`,
but spike counts are drawn from NegativeBinomial instead of Poisson. The overdispersion
factor `d` sets variance = d × mean.

When fit with PoissonEmission, the model correctly identifies the mean rates but
cannot capture the extra variance. This is a realistic, mild misspecification — the
kind that arises when neural variability exceeds Poisson (which it usually does).

NegBin(r, p) parametrization: p = 1/d, r = μ/(d − 1).
"""
function generate_synthetic_spikes_negbin(;
    seed::Int           = 42,
    T::Int              = 10_000,
    bin_ms::Real        = 10.0,
    overdispersion::Real = 2.0,
    truncnormal_mix::Real = 0.0,
    contamination_rate::Real = 0.0,
    contamination_burst_length::Int = 1,
    contamination_modes::Union{Nothing,Vector{Vector{Float64}}} = nothing,
    cache_dir::Union{String,Nothing} = nothing,
    rates_per_state::Union{Nothing,Vector{Vector{Float64}}} = nothing,
    state_labels::Union{Nothing,Vector{String}}             = nothing,
    rates_tag::Union{Nothing,String}                        = nothing,
)
    d_factor = Float64(overdispersion)
    w_tn     = Float64(truncnormal_mix)
    cont_rate = Float64(contamination_rate)
    @assert d_factor > 1 "overdispersion must be > 1 (=1 is Poisson)"
    @assert 0 <= w_tn <= 1 "truncnormal_mix must be in [0, 1]"
    @assert 0 <= cont_rate <= 1 "contamination_rate must be in [0, 1]"
    if cont_rate > 0
        @assert !isnothing(contamination_modes) && !isempty(contamination_modes) "contamination_modes must be supplied when contamination_rate > 0"
    end

    tag_suffix  = isnothing(rates_tag) ? "" : "_$(rates_tag)"
    tn_suffix   = w_tn > 0 ? "_tn$(w_tn)" : ""
    cont_suffix = cont_rate > 0 ? "_cont$(cont_rate)L$(contamination_burst_length)x$(length(contamination_modes))" : ""
    cache_file = isnothing(cache_dir) ? nothing :
        joinpath(cache_dir, "spikes_negbin_seed$(seed)_T$(T)_bin$(bin_ms)_od$(d_factor)$(tn_suffix)$(cont_suffix)$(tag_suffix).jls")
    if !isnothing(cache_file) && isfile(cache_file)
        println("Loading cached NegBin spike data: $cache_file")
        return deserialize(cache_file)
    end

    rng = MersenneTwister(seed)
    bin_s = bin_ms / 1000.0

    if isnothing(rates_per_state)
        rates_per_state = [
            [ 1.0,  1.0,  1.0,  1.0,  1.0],    # SILENT
            [ 5.0,  5.0,  5.0,  5.0,  5.0],    # DOWN
            [25.0, 30.0, 20.0, 35.0, 25.0],    # UP
            [80.0,  5.0, 80.0,  5.0, 80.0],    # BURST_A
            [ 5.0, 80.0,  5.0, 80.0,  5.0],    # BURST_B
        ]
    end
    K_true = length(rates_per_state)
    D = length(rates_per_state[1])

    A = fill(0.01, K_true, K_true)
    for k in 1:K_true; A[k, k] = 0.96; end
    π₀ = fill(1.0 / K_true, K_true)
    if isnothing(state_labels)
        state_labels = ["silent", "down", "up", "burst_a", "burst_b"]
    end
    gt = SpikeGroundTruth(K_true, rates_per_state, bin_s, A, π₀, state_labels)

    # Sample state sequence (same process as Poisson case)
    states = Vector{Int}(undef, T)
    states[1] = rand(rng, Categorical(π₀))
    for t in 2:T
        states[t] = rand(rng, Categorical(A[states[t-1], :]))
    end

    # Sample spike counts from a NegBin / TruncatedNormal mixture with matched
    # per-bin mean and variance (var = overdispersion × mean for both).
    # NegBin is right-skewed with a heavy tail; TruncatedNormal-rounded is
    # near-symmetric. The mixture creates a per-state distribution whose shape
    # cannot be matched by a single Poisson, even after the model gets the
    # mean right — and extra components can absorb it by allocating one to
    # the symmetric mode and another to the skewed tail.
    counts = Matrix{Float64}(undef, T, D)
    p_nb = 1.0 / d_factor
    contamination_log = falses(T)
    cont_remaining = 0          # bins left in current contamination burst
    cont_mode_idx  = 0          # which mode is active during the burst
    for t in 1:T
        # Burst-length contamination: at each step, if not currently in a
        # contamination burst, with prob `cont_rate` start a burst of
        # `contamination_burst_length` bins drawing from a randomly-chosen
        # contamination mode. The bin's GT state label is left unchanged —
        # contamination is sample-level noise on top of the HMM.
        if cont_rate > 0 && cont_remaining == 0 && rand(rng) < cont_rate
            cont_remaining = contamination_burst_length
            cont_mode_idx  = rand(rng, 1:length(contamination_modes))
        end

        if cont_remaining > 0
            mode = contamination_modes[cont_mode_idx]
            for dd in 1:D
                μ = mode[dd] * bin_s
                counts[t, dd] = Float64(rand(rng, Poisson(max(μ, eps()))))
            end
            contamination_log[t] = true
            cont_remaining -= 1
            continue
        end

        rates = rates_per_state[states[t]]
        for dd in 1:D
            μ = rates[dd] * bin_s
            σ = sqrt(d_factor * μ)
            from_tn = w_tn > 0 && rand(rng) < w_tn
            if from_tn && μ > 0
                tn = truncated(Normal(μ, σ), 0.0, Inf)
                counts[t, dd] = Float64(max(0, round(Int, rand(rng, tn))))
            else
                r = μ / (d_factor - 1.0)
                r = max(r, 1e-6)
                counts[t, dd] = Float64(rand(rng, NegativeBinomial(r, p_nb)))
            end
        end
    end

    observations = Matrix(transpose(counts))

    result = (
        seed                = seed,
        T                   = T,
        bin_ms              = bin_ms,
        overdispersion      = d_factor,
        truncnormal_mix     = w_tn,
        contamination_rate  = cont_rate,
        contamination_modes = contamination_modes,
        contamination_log   = contamination_log,
        ground_truth        = gt,
        states              = states,
        counts              = counts,
        observations        = observations,
    )

    if !isnothing(cache_file)
        mkpath(dirname(cache_file)); serialize(cache_file, result)
        println("Cached NegBin spike data to: $cache_file")
    end
    return result
end

function describe_ground_truth(io::IO, data)
    gt = data.ground_truth
    is_negbin = hasproperty(data, :overdispersion)
    variant = is_negbin ? "NegativeBinomial (overdispersion=$(data.overdispersion)×)" : "Poisson"
    println(io, "Ground-truth $variant HMM:")
    println(io, "  K_true   = $(gt.K_true)")
    println(io, "  D        = $(length(gt.rates_per_state[1]))  (neurons)")
    println(io, "  T        = $(data.T)  bins")
    println(io, "  bin_ms   = $(data.bin_ms)  ms")
    println(io, "  duration = $(round(data.T * data.bin_ms / 1000; digits=1))  s")
    println(io, "  seed     = $(data.seed)")
    is_negbin && println(io, "  overdispersion = $(data.overdispersion)×  (variance = $(data.overdispersion) × mean)")
    println(io, "  Transition matrix A (A_ii = $(round(gt.A[1,1]; digits=3))):")
    for k in 1:gt.K_true
        @printf(io, "    [%s]\n", join([@sprintf("%.3f", a) for a in gt.A[k, :]], "  "))
    end
    println(io, "  State occupancy (empirical):")
    for k in 1:gt.K_true
        frac = count(==(k), data.states) / data.T * 100
        rates = gt.rates_per_state[k]
        @printf(io, "    State %d [%s]: %5.1f%%  rates (Hz) = [%s]\n",
            k, gt.state_labels[k], frac, join([@sprintf("%.1f", r) for r in rates], ", "))
    end
end

describe_ground_truth(data) = (describe_ground_truth(stdout, data); nothing)

# =============================================================================
# HMM fitting
# =============================================================================

# Parameter counts
# Poisson: λ vector of size D per component → D emission params per K
_n_params_poisson(K::Int, D::Int)  = K * D + K * (K - 1) + (K - 1)
_entropy_class_probs(γ) = -sum(p > 0 ? p * log(p) : 0.0 for p in γ)

function _fit_K_all_restarts(K, observations, emission_factory, n_restarts, max_iters, tol)
    runs = Vector{Tuple{Any,Float64}}()
    for _ in 1:n_restarts
        try
            emissions = [emission_factory() for _ in 1:K]
            A_init = rand(K, K); A_init ./= sum(A_init; dims = 2)
            model_r = HiddenMarkovModel(; K = K, B = emissions, A = A_init, πₖ = fill(1.0/K, K))
            ll_hist = fit!(model_r, observations; max_iters = max_iters, tol = tol)
            finite_ll = filter(isfinite, ll_hist)
            isempty(finite_ll) && continue
            push!(runs, (model_r, maximum(finite_ll)))
        catch
        end
    end
    return runs
end

function _compute_one_variant(data, emission_factory, n_params_fn;
        K_range, n_restarts, max_iters, tol, use_cache, cache_dir,
        variant_name, variant_symbol,
    )
    println("\n" * "="^70)
    println("SPIKE-TRAIN [$variant_name]")
    println("="^70)
    !isnothing(cache_dir) && mkpath(cache_dir)

    observations = data.observations
    D, T_obs = size(observations)
    nK = length(K_range)

    models_bic      = Vector{Union{Nothing,HiddenMarkovModel{Float64}}}(nothing, nK)
    models_icl      = Vector{Union{Nothing,HiddenMarkovModel{Float64}}}(nothing, nK)
    models_kl       = Vector{Union{Nothing,HiddenMarkovModel{Float64}}}(nothing, nK)
    results_kl      = Vector{Union{Nothing,ACDCResult{Float64}}}(nothing, nK)
    all_drivers     = Dict{Int,Vector{Matrix{Float64}}}()
    log_likelihoods = fill(-Inf, nK)
    bic_values      = fill(Inf,  nK)
    icl_values      = fill(Inf,  nK)
    all_runs        = Vector{Vector{NamedTuple}}(undef, nK)
    n_succeeded     = zeros(Int, nK)

    for (i, K) in enumerate(K_range)
        cache_file = isnothing(cache_dir) ? nothing : joinpath(cache_dir, "hmm_K$(K)_allruns.jls")
        runs = Tuple{Any,Float64}[]

        if use_cache && !isnothing(cache_file) && isfile(cache_file)
            try
                cached = deserialize(cache_file)
                runs = cached.runs
                println("  Loaded K=$K all $(length(runs)) restarts from cache")
            catch
                println("  Cache load failed for K=$K; refitting")
            end
        end

        if isempty(runs)
            print("  Fitting K=$K ($(n_restarts) restarts)...")
            runs = _fit_K_all_restarts(K, observations, emission_factory, n_restarts, max_iters, tol)
            isempty(runs) && (println(" all FAILED."); continue)
            !isnothing(cache_file) && use_cache && serialize(cache_file, (runs = runs,))
            println(" $(length(runs))/$n_restarts succeeded.")
        end

        # Evaluate BIC, ICL, ACDC-KL on each restart
        evals = NamedTuple[]
        for (model, ll) in runs
            bic = -2 * ll + n_params_fn(K, D) * log(T_obs)
            γ   = class_probabilities(model, observations)
            H   = _entropy_class_probs(γ)
            icl = bic + 2 * H
            res_kl = component_discrepancies(model, observations, KLDiscrepancy(); n_samples = 1)
            push!(evals, (
                model = model, ll = ll,
                bic = bic, icl = icl,
                res_kl = res_kl,
                max_kl = maximum(res_kl.component_discrepancies),
            ))
        end
        all_runs[i]    = evals
        n_succeeded[i] = length(evals)

        best_bic = evals[argmin([e.bic for e in evals])]
        best_icl = evals[argmin([e.icl for e in evals])]
        best_kl  = evals[argmin([e.max_kl for e in evals])]

        models_bic[i] = best_bic.model
        models_icl[i] = best_icl.model
        models_kl[i]  = best_kl.model

        bic_values[i]      = best_bic.bic
        icl_values[i]      = best_icl.icl
        log_likelihoods[i] = best_bic.ll
        results_kl[i]      = best_kl.res_kl

        drv            = stochastic_drivers(best_bic.model, observations; n_samples = 1)
        all_drivers[K] = drv.ε_pools
    end

    K_bic = K_range[argmin(bic_values)]
    K_icl = K_range[argmin(icl_values)]

    return (
        variant         = variant_symbol,
        variant_name    = variant_name,
        data            = (obs = observations, T = T_obs, D = D),
        models          = models_bic,
        models_per_metric = (bic = models_bic, icl = models_icl,
                             acdc_kl = models_kl),
        results_kl      = results_kl,
        drivers         = all_drivers,
        bic_values      = bic_values,
        icl_values      = icl_values,
        K_range         = K_range,
        selections      = (bic = K_bic, icl = K_icl),
        log_likelihoods = log_likelihoods,
        all_runs        = all_runs,
        n_succeeded     = n_succeeded,
    )
end

compute_spike_analysis_poisson(data; kwargs...) =
    _compute_one_variant(data,
        () -> PoissonEmission(; output_dim = size(data.observations, 1),
                                λ = rand(Uniform(0.05, 1.0), size(data.observations, 1))),
        _n_params_poisson;
        variant_name = "base: Poisson HMM (correctly specified)",
        variant_symbol = :poisson, kwargs...)

# =============================================================================
# Plots
# =============================================================================

function get_spike_plots(res, data; figures_dir = nothing, rho_grid = nothing)
    K_range = res.K_range
    K_bic   = res.selections.bic
    K_icl   = res.selections.icl
    K_true  = data.ground_truth.K_true
    label   = String(res.variant)

    _plain_fmt(x) = (x == 0 ? "0" : abs(x) < 0.01 || abs(x) >= 1e6 ? @sprintf("%.0e", x) : @sprintf("%g", x))
    hires = (
        dpi = 300,
        tickfontsize = 10, guidefontsize = 12, legendfontsize = 10, titlefontsize = 13,
        fontfamily = "Helvetica",
        left_margin = 10Plots.mm, right_margin = 5Plots.mm,
        top_margin = 5Plots.mm, bottom_margin = 24Plots.mm,
        yformatter = _plain_fmt, xformatter = _plain_fmt,
    )

    p_bic = plot(; title = "BIC / ICL",
        xlabel = "K", ylabel = "Score",
        xticks = collect(minimum(K_range):maximum(K_range)),
        legend = :outerbottom, legend_columns = 4,
        hires...)
    if hasproperty(res, :all_runs)
        for (i, K) in enumerate(K_range)
            evals = res.all_runs[i]
            (isnothing(evals) || isempty(evals)) && continue
            xs = fill(K, length(evals))
            scatter!(p_bic, xs, [e.bic for e in evals];
                color = :black, markeralpha = 0.25, markersize = 4,
                markerstrokewidth = 0, label = (i == 1 ? "BIC seeds" : nothing))
            scatter!(p_bic, xs, [e.icl for e in evals];
                color = :royalblue, markeralpha = 0.25, markersize = 4,
                markerstrokewidth = 0, label = (i == 1 ? "ICL seeds" : nothing))
        end
    end
    plot!(p_bic, collect(K_range), res.bic_values;
        marker = :circle, label = "BIC (best)", linewidth = 2.2, markersize = 6, color = :black)
    plot!(p_bic, collect(K_range), res.icl_values;
        marker = :diamond, label = "ICL (best)", linewidth = 2.2, markersize = 6, color = :royalblue)
    vline!(p_bic, [K_true]; linestyle = :dot, color = :seagreen,
        linewidth = 2.5, label = "GT K=$K_true")
    vline!(p_bic, [K_bic]; linestyle = :dash, color = :red,
        label = "BIC K=$K_bic", linewidth = 2.0)
    vline!(p_bic, [K_icl]; linestyle = :dashdot, color = :royalblue,
        label = "ICL K=$K_icl", linewidth = 2.0)

    function _envelope_panel(results_vec, title_str; metric_field::Symbol)
        valid_K     = Int[]
        valid_zeros = Float64[]
        valid_idx   = Int[]
        for (i, K) in enumerate(K_range)
            r = results_vec[i]
            isnothing(r) && continue
            ds = r.component_discrepancies
            all(isfinite, ds) || continue
            mx = maximum(ds)
            mx >= 0 || continue
            push!(valid_K, K); push!(valid_zeros, mx); push!(valid_idx, i)
        end
        p = plot(; title = title_str, xlabel = "K",
            ylabel = "max per-component",
            legend = :outerbottom, legend_columns = 2,
            xticks = collect(minimum(K_range):maximum(K_range)),
            hires...)
        isempty(valid_K) && return p
        node_colors = [palette(:auto)[i] for i in valid_idx]

        if hasproperty(res, :all_runs)
            for (i, K) in enumerate(K_range)
                evals = res.all_runs[i]
                (isnothing(evals) || isempty(evals)) && continue
                vals = [getproperty(e, metric_field) for e in evals]
                vals = filter(v -> isfinite(v) && v >= 0, vals)
                isempty(vals) && continue
                scatter!(p, fill(K, length(vals)), vals;
                    color = :gray, markeralpha = 0.25, markersize = 3,
                    markerstrokewidth = 0,
                    label = (i == 1 ? "all seeds" : nothing))
            end
        end

        plot!(p, valid_K, valid_zeros;
            color = :gray, linewidth = 1.0, alpha = 0.5,
            label = "best per K", primary = true)
        scatter!(p, valid_K, valid_zeros;
            markersize = 5, markerstrokewidth = 0, markercolor = node_colors,
            markeralpha = 0.55, label = nothing)

        env_K = Int[]; env_z = Float64[]; env_colors = []
        best_so_far = Inf
        for (k, z, c) in zip(valid_K, valid_zeros, node_colors)
            if z < best_so_far
                push!(env_K, k); push!(env_z, z); push!(env_colors, c)
                best_so_far = z
            end
        end
        if !isempty(env_K)
            plot!(p, env_K, env_z;
                color = :black, linewidth = 2.5, label = "envelope")
            scatter!(p, env_K, env_z;
                markersize = 8, markerstrokewidth = 1.8,
                markerstrokecolor = :black, markercolor = env_colors, label = nothing)
            # Perpendicular offset in normalized panel-fraction coords so the
            # label sits visually perpendicular to the segment.
            ann_ys = Float64[]
            k_rng    = max(float(maximum(env_K) - minimum(env_K)), 1.0)
            z_rng    = max(maximum(env_z) - minimum(env_z), eps(Float64))
            for j in 1:(length(env_K) - 1)
                k0, k1 = env_K[j], env_K[j+1]
                z0, z1 = env_z[j], env_z[j+1]
                drop_pct = iszero(z0) ? 0.0 : (z0 - z1) / z0 * 100
                dk_n    = (k1 - k0) / k_rng
                dz_n    = (z1 - z0) / z_rng
                mag_n   = sqrt(dk_n^2 + dz_n^2)
                iszero(mag_n) && continue
                perp_x_n = -dz_n / mag_n   # positive for descending env
                perp_y_n =  dk_n / mag_n
                off_n = 0.045
                xm = (k0 + k1) / 2 + perp_x_n * off_n * k_rng
                ym = (z0 + z1) / 2 + perp_y_n * off_n * z_rng
                push!(ann_ys, ym)
                annotate!(p, xm, ym,
                    text(@sprintf("−%.1f%%", drop_pct), 8, :black, :hcenter, :vcenter))
            end
            if !isempty(ann_ys)
                ymin_raw = min(minimum(valid_zeros), minimum(ann_ys))
                ymax_raw = max(maximum(valid_zeros), maximum(ann_ys))
                pad = max((ymax_raw - ymin_raw) * 0.12, eps(Float64))
                plot!(p; ylim = (max(0.0, ymin_raw - pad), ymax_raw + pad))
            end
        end
        vline!(p, [K_true]; linestyle = :dot, color = :seagreen,
            linewidth = 2.0, label = "GT K=$K_true")
        return p
    end

    p_env_kl = _envelope_panel(res.results_kl, "ACDC Envelope (KL)"; metric_field = :max_kl)

    p_bic_kl = plot(p_bic, p_env_kl; layout = (1, 2), size = (1300, 500), margin = 5Plots.mm)
    if !isnothing(figures_dir)
        mkpath(figures_dir)
        _save_png_pdf(p_bic_kl, figures_dir, "spike_selection_$(label)_kl"; trim = true)
    end
    return (kl = p_bic_kl,)
end

# =============================================================================
# Statistics export
# =============================================================================

function write_spike_statistics(filepath::String, data, results_by_label)
    mkpath(dirname(filepath))
    open(filepath, "w") do io
        println(io, "=" ^ 70)
        println(io, "SPIKE-TRAIN HMM STATISTICS")
        println(io, "Generated:  $(Dates.now())")
        println(io, "=" ^ 70)

        println(io, "\n--- Model Legend ---")
        println(io, "  matched    : PoissonEmission on Poisson data (correctly specified)")
        println(io, "  mismatched : PoissonEmission on NegBin data (overdispersion mismatch)")

        println(io)
        describe_ground_truth(io, data)

        for (label, res) in results_by_label
            println(io, "\n" * "─" ^ 70)
            println(io, "[$label] $(res.variant_name)")
            println(io, "─" ^ 70)

            function _fmt_discs(results_vec, i)
                isnothing(results_vec[i]) && return "—"
                join([@sprintf("%.6f", d) for d in results_vec[i].component_discrepancies], "  ")
            end

            K_acdc_kl = let
                best_K = nothing; best_max = Inf
                for (i, K) in enumerate(res.K_range)
                    isnothing(res.results_kl[i]) && continue
                    mx = maximum(res.results_kl[i].component_discrepancies)
                    isfinite(mx) || continue
                    if mx < best_max; best_max = mx; best_K = K; end
                end
                best_K
            end

            @printf(io, "%-5s  %-12s  %-12s  %-12s  KL discs\n",
                "K", "LL", "BIC", "ICL")
            for (i, K) in enumerate(res.K_range)
                ll_s  = isfinite(res.log_likelihoods[i]) ? @sprintf("%.1f", res.log_likelihoods[i]) : "—"
                bic_s = isfinite(res.bic_values[i])      ? @sprintf("%.1f", res.bic_values[i])      : "—"
                icl_s = isfinite(res.icl_values[i])      ? @sprintf("%.1f", res.icl_values[i])      : "—"
                kl_s  = _fmt_discs(res.results_kl, i)
                sel = String[]
                K == res.selections.bic && push!(sel, "← BIC")
                K == res.selections.icl && push!(sel, "← ICL")
                K == K_acdc_kl          && push!(sel, "← ACDC(KL)")
                K == data.ground_truth.K_true && push!(sel, "← GT")
                @printf(io, "K=%-3d  %-12s  %-12s  %-12s  KL[%s]  %s\n",
                    K, ll_s, bic_s, icl_s, kl_s, join(sel, " "))
            end

            # ACDC envelope tables
            function _envelope_table(io, results_vec, name)
                println(io, "\n" * "─" ^ 70)
                println(io, "ACDC envelope — $name")
                println(io, "─" ^ 70)
                zeros_K = Tuple{Int,Float64}[]
                for (i, K) in enumerate(res.K_range)
                    isnothing(results_vec[i]) && continue
                    mx = maximum(results_vec[i].component_discrepancies)
                    isfinite(mx) || continue
                    push!(zeros_K, (K, mx))
                end
                isempty(zeros_K) && (println(io, "  (no valid points)"); return)
                env = Tuple{Int,Float64}[]
                best = Inf
                for (K, z) in zeros_K
                    if z < best; push!(env, (K, z)); best = z; end
                end
                println(io, "  Raw zero-points (per K):")
                for (K, z) in zeros_K
                    on_env = any(e -> e[1] == K, env)
                    @printf(io, "    %s K=%-3d  z = %.4f\n", on_env ? "★" : " ", K, z)
                end
                println(io, "\n  Envelope points (★) and inter-point drops:")
                for (j, (K, z)) in enumerate(env)
                    if j == 1
                        @printf(io, "    K=%-3d  z = %.4f\n", K, z)
                    else
                        K0, z0 = env[j-1]
                        drop = (z0 - z) / z0 * 100
                        @printf(io, "    K=%-3d  z = %.4f   (Δ K=%d→K=%d: −%.1f%%)\n",
                            K, z, K0, K, drop)
                    end
                end
                tail_K, tail_z = env[end]
                @printf(io, "  Last envelope K = %d   (z = %.4f)\n", tail_K, tail_z)
            end
            _envelope_table(io, res.results_kl, "KL")

            # Composition tables (Viterbi-decoded GT → fitted-component breakdown)
            # Logged for three model sources: BIC-best, ICL-best, ACDC-KL-best.
            gt = data.ground_truth
            obs = data.observations
            gt_st = data.states

            # ICL-best is omitted: at GT-K with crisp posteriors, ICL-best
            # tracks BIC-best to within rounding noise (the entropy term is
            # tiny). The two compositions worth comparing are the ones with
            # genuinely different optimisation targets: BIC-best vs ACDC-KL-best.
            sources = if hasproperty(res, :models_per_metric)
                [("BIC-best", res.models_per_metric.bic),
                 ("ACDC-KL-best", res.models_per_metric.acdc_kl)]
            else
                [("BIC-best", res.models)]
            end

            for (name, model_vec) in sources
                println(io, "\n" * "─" ^ 70)
                println(io, "Composition tables — $name fits  (GT state → fitted-component %)")
                println(io, "Components sorted by total per-bin firing rate, low → high.")
                println(io, "─" ^ 70)
                for (i, K) in enumerate(res.K_range)
                    model = model_vec[i]
                    isnothing(model) && continue
                    states = viterbi(model, obs)
                    rates = [_component_total_rate(model, k) for k in 1:K]
                    perm = sortperm(rates)
                    inv_perm = Dict(perm[k] => k for k in 1:K)
                    states_sorted = [inv_perm[s] for s in states]

                    counts = zeros(Float64, K, gt.K_true)
                    for (s, g) in zip(states_sorted, gt_st)
                        s in 1:K || continue
                        g in 1:gt.K_true || continue
                        counts[s, g] += 1
                    end

                    println(io, "\n  K=$K   (components C1=quietest, C$K=burstiest)")
                    comp_hdr = join([@sprintf("%6s", "C$k") for k in 1:K], " ")
                    @printf(io, "    %-10s %s   total\n", "GT state", comp_hdr)
                    for g in 1:gt.K_true
                        total = sum(counts[:, g])
                        pcts = total > 0 ? counts[:, g] ./ total .* 100.0 : zeros(K)
                        pct_strs = join([@sprintf("%5.1f%%", p) for p in pcts], " ")
                        @printf(io, "    %-10s %s   %5d bins\n",
                            String(gt.state_labels[g]), pct_strs, round(Int, total))
                    end
                    rate_strs = join([@sprintf("%6.1f", rates[perm[k]]) for k in 1:K], " ")
                    @printf(io, "    %-10s %s    (sum λ per bin)\n", "rate Σλ:", rate_strs)
                end
            end
        end

        println(io, "\n" * "=" ^ 70)
    end
    println("Statistics written to: $filepath")
end

# =============================================================================
# Composition panels — Viterbi-decoded fitted state vs GT state
# =============================================================================

"""
    _component_total_rate(model, k) -> Float64

Return the sum of per-neuron rates for component k. Used to sort components
into a stable visual order (low rate → silent, high rate → bursty).
"""
function _component_total_rate(model, k)
    em = model.B[k]
    return sum(em.λ)
end

"""
    _composition_panel_gt_to_component(states, gt_states, K, label_names; title="", comp_colors)

Stacked bar: x-axis = GT state, each bar shows the component composition
(percent of bins of that GT state assigned to each fitted component).
"""
function _composition_panel_gt_to_component(states, gt_states, K, label_names; title="", comp_colors)
    K_true = length(label_names)
    counts = zeros(Float64, K, K_true)
    for (s, gt) in zip(states, gt_states)
        s in 1:K || continue
        gt in 1:K_true || continue
        counts[s, gt] += 1
    end
    pct = zeros(Float64, K, K_true)
    for g in 1:K_true
        denom = sum(counts[:, g])
        denom > 0 && (pct[:, g] .= 100.0 .* counts[:, g] ./ denom)
    end
    pct_T = permutedims(pct, (2, 1))
    # Abbreviated labels keep tick text short enough to render unrotated.
    short_labels = map(label_names) do s
        s == "low_a"   ? "LA"   :
        s == "low_b"   ? "LB"   :
        s == "silent"  ? "SIL"  :
        s == "down"    ? "DOWN" :
        s == "chaotic" ? "CHA"  :
        s == "up"      ? "UP"   :
        s == "burst_a" ? "BA"   :
        s == "burst_b" ? "BB"   :
        uppercase(first(s, 4))
    end
    p = groupedbar(1:K_true, pct_T;
        bar_position = :stack, bar_width = 0.85,
        linecolor = :transparent, linewidth = 0,
        palette = comp_colors,
        label = false, title = title,
        xlabel = "Ground-truth state", ylabel = "% of bins",
        ylim = (0, 100), yticks = 0:20:100,
        xticks = (1:K_true, short_labels), xrotation = 0,
        grid = false, legend = false,
        foreground_color_subplot = :black, background_color = :white,
        titlefontsize = 13, tickfontsize = 11, guidefontsize = 12,
        left_margin = 12Plots.mm, right_margin = 4Plots.mm,
        top_margin = 4Plots.mm, bottom_margin = 12Plots.mm)
    return p
end

"""
    plot_spike_composition_grid(res, data; K_list=2:9, figures_dir=nothing, label="")

Build the ground-truth-to-component composition panels used by the paper
supplement, with one panel per K in `K_list`.

Components are reordered within each K by total firing rate (lowest first)
so visual position roughly tracks the SILENT → DOWN → UP → BURST progression.
"""
function plot_spike_composition_grid(res, data; K_list = 2:9,
                                       figures_dir::Union{String,Nothing} = nothing,
                                       label::Union{String,Nothing} = nothing,
                                       model_source::Symbol = :bic)
    K_range  = res.K_range
    gt       = data.ground_truth
    K_true   = gt.K_true
    labels   = String.(gt.state_labels)
    obs      = data.observations
    gt_st    = data.states

    K_keep = collect(K_list)
    K_keep = filter(k -> k in K_range, K_keep)
    isempty(K_keep) && error("K_list has no overlap with res.K_range")

    # Pick which per-metric-best fit to visualise. Default :bic preserves the
    # back-compat reading of res.models. Other options use res.models_per_metric.
    model_vec = if model_source == :bic && !hasproperty(res, :models_per_metric)
        res.models
    else
        @assert hasproperty(res, :models_per_metric) "model_source ≠ :bic requires models_per_metric on res"
        getfield(res.models_per_metric, model_source)
    end

    K_max = maximum(K_keep)
    comp_colors = palette(:tab10, max(K_max, K_true))

    bot_panels = Any[]
    for K in K_keep
        idx = findfirst(==(K), K_range)
        model = model_vec[idx]
        if isnothing(model)
            push!(bot_panels, plot(; framestyle = :none, title = "K=$K (no model)"))
            continue
        end
        states = viterbi(model, obs)

        # Sort components by total firing rate so the leftmost component is the
        # quietest one and the rightmost is burstiest.
        rates = [_component_total_rate(model, k) for k in 1:K]
        perm  = sortperm(rates)
        inv_perm = Dict(perm[k] => k for k in 1:K)
        states_sorted = [inv_perm[s] for s in states]

        title_str = "K=$K" * (K == K_true ? " (GT)" : "")
        push!(bot_panels, _composition_panel_gt_to_component(states_sorted, gt_st, K, labels;
            title = title_str, comp_colors = comp_colors))
    end

    function _legend_row(names, cols; nrows::Int = 2)
        ntotal = length(names)
        ncol_leg = cld(ntotal, nrows)
        p = plot(; xlim = (0, ncol_leg), ylim = (0, nrows),
            framestyle = :none, axis = false, ticks = nothing,
            background_color = :white, legend = false,
            top_margin = 4Plots.mm, bottom_margin = 4Plots.mm,
            left_margin = 6Plots.mm, right_margin = 6Plots.mm)
        for (i, (name, c)) in enumerate(zip(names, cols))
            row = (i - 1) ÷ ncol_leg
            col = (i - 1) % ncol_leg
            x  = col + 0.05
            y  = nrows - row - 0.5
            x1 = x;        x2 = x + 0.20
            y1 = y - 0.18; y2 = y + 0.18
            plot!(p, Shape([x1, x2, x2, x1], [y1, y1, y2, y2]);
                color = c, linecolor = c, label = nothing)
            annotate!(p, x + 0.27, y, text(name, 14, :left, :black))
        end
        return p
    end

    n = length(K_keep)
    legend_names = ["C$(k)" for k in 1:maximum(K_keep)]
    legend_row = _legend_row(legend_names,
        [comp_colors[k] for k in 1:maximum(K_keep)])

    # 3×3 grid for K=2..9 (8 panels + 1 empty cell). Wider panels and squarer
    # overall figure than 2×4.
    ncol  = 3
    nrows = cld(n, ncol)
    padded = copy(bot_panels)
    while length(padded) < ncol * nrows
        push!(padded, plot(; framestyle = :none, axis = false, background_color = :white))
    end

    panel_grid = plot(padded...; layout = (nrows, ncol))

    row_h = 360
    legend_h = 130   # 2 rows of swatches need more vertical room
    total_h = legend_h + row_h * nrows
    leg_frac = legend_h / total_h

    full = plot(legend_row, panel_grid;
        layout = grid(2, 1; heights = [leg_frac, 1.0 - leg_frac]),
        size = (560 * ncol, total_h),
        background_color = :white,
        left_margin   = 6Plots.mm,
        right_margin  = 6Plots.mm,
        top_margin    = 6Plots.mm,
        bottom_margin = 10Plots.mm)

    if !isnothing(figures_dir)
        mkpath(figures_dir)
        suffix = isnothing(label) ? "" : "_" * label
        _save_png_pdf(full, figures_dir, "spike_composition" * suffix * "_$(model_source)")
    end
    return full
end

end  # module SpikeTrainHMMUtils
