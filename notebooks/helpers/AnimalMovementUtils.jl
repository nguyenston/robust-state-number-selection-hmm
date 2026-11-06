module AnimalMovementUtils

# Must be set before GR initializes to fix minus sign / Greek char rendering in PDF
ENV["GKS_ENCODING"] = "utf8"

using CSV
using DataFrames
using Dates
using LinearAlgebra
using Printf
using Random
using Serialization
using Statistics
using Distributions
using LaTeXStrings
using Plots, Measures
using StateSpaceDynamics
using StateSpaceDynamics.ACDC

export haversine_dist, calculate_bearing, prepare_movement_data, get_most_observed_individual
export StepAngleEmission
export CardioidEmission, StepAngleCardEmission
export compute_movement_analysis, compute_movement_analysis_alt, compute_movement_analysis_alt2
export get_movement_plots, get_movement_plots_alt, get_movement_plots_alt2
export feature_xlims_from_res, component_feature_density, movement_state_feature_grid
export write_movement_statistics

# =============================================================================
# GPS feature computation
# =============================================================================

"""Haversine distance in metres between two (lon, lat) points."""
function haversine_dist(lon1, lat1, lon2, lat2)
    R = 6_371_000.0
    dlat = deg2rad(lat2 - lat1)
    dlon = deg2rad(lon2 - lon1)
    a = sin(dlat / 2)^2 + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dlon / 2)^2
    return R * 2 * atan(sqrt(a), sqrt(1 - a))
end

"""Initial compass bearing (radians) from (lon1, lat1) to (lon2, lat2)."""
function calculate_bearing(lon1, lat1, lon2, lat2)
    dlon   = deg2rad(lon2 - lon1)
    lat1_r = deg2rad(lat1)
    lat2_r = deg2rad(lat2)
    y = sin(dlon) * cos(lat2_r)
    x = cos(lat1_r) * sin(lat2_r) - sin(lat1_r) * cos(lat2_r) * cos(dlon)
    return atan(y, x)
end

"""
    prepare_movement_data(df_raw) -> DataFrame

Compute step_length (m) and turn_angle (rad in [-π, π]) for every fix within each
individual's track, then drop rows with missing values. Near-zero steps are clamped
to 1e-5 m to prevent log(0).
"""
function prepare_movement_data(df_raw::DataFrame)
    df = copy(df_raw)
    sort!(df, ["individual-local-identifier", "timestamp"])

    n      = nrow(df)
    steps  = Vector{Union{Float64,Missing}}(missing, n)
    angles = Vector{Union{Float64,Missing}}(missing, n)

    for group in groupby(df, "individual-local-identifier")
        idx = parentindices(group)[1]
        for k in 1:(length(idx) - 1)
            i, ip1 = idx[k], idx[k + 1]
            lon_t,   lat_t   = df[i,   "location-long"], df[i,   "location-lat"]
            lon_tp1, lat_tp1 = df[ip1, "location-long"], df[ip1, "location-lat"]
            any(ismissing, (lon_t, lat_t, lon_tp1, lat_tp1)) && continue

            steps[i] = haversine_dist(lon_t, lat_t, lon_tp1, lat_tp1)

            if k > 1
                im1 = idx[k - 1]
                lon_tm1, lat_tm1 = df[im1, "location-long"], df[im1, "location-lat"]
                if !any(ismissing, (lon_tm1, lat_tm1))
                    b1  = calculate_bearing(lon_tm1, lat_tm1, lon_t, lat_t)
                    b2  = calculate_bearing(lon_t, lat_t, lon_tp1, lat_tp1)
                    ang = b2 - b1
                    ang >  π && (ang -= 2π)
                    ang <= -π && (ang += 2π)
                    angles[i] = ang
                end
            end
        end
    end

    df.step_length = steps
    df.turn_angle  = angles
    df_clean = dropmissing(df, [:step_length, :turn_angle])
    df_clean.step_length = [s <= 1e-6 ? 1e-5 : s for s in df_clean.step_length]
    return df_clean
end

"""
    get_most_observed_individual(df) -> (id::String, df_individual::DataFrame)

Select the individual-local-identifier with the most GPS fixes.
"""
function get_most_observed_individual(df::DataFrame)
    counts  = combine(groupby(df, "individual-local-identifier"), nrow => :count)
    best_id = counts[argmax(counts.count), "individual-local-identifier"]
    df_ind  = filter(row -> row["individual-local-identifier"] == best_id, df)
    println("Selected individual: $best_id  ($(nrow(df_ind)) observations)")
    return String(best_id), df_ind
end

function _timestamp_format()
    return dateformat"yyyy-mm-dd HH:MM:SS.s"
end

function _split_cleaned_individual_trials(
    df_animal::DataFrame; gap_threshold_hours::Float64=1.5
)
    n = nrow(df_animal)
    n == 0 && return DataFrame[], Int[], Float64[]

    ts = DateTime.(df_animal[:, "timestamp"], Ref(_timestamp_format()))
    gap_hours = Dates.value.(diff(ts)) ./ (1000 * 60 * 60)
    break_idxs = findall(>(gap_threshold_hours), gap_hours)

    trials = DataFrame[]
    start_idx = 1
    for b in break_idxs
        stop_idx = b - 1
        if start_idx <= stop_idx
            push!(trials, df_animal[start_idx:stop_idx, :])
        end
        # Rows b and b+1 contain step/angle features that cross the long gap, so
        # exclude them from the trialized fit.
        start_idx = b + 2
    end
    if start_idx <= n
        push!(trials, df_animal[start_idx:n, :])
    end

    return trials, break_idxs, gap_hours[break_idxs]
end

_movement_fit_data(res) = hasproperty(res.data, :obs_trials) ? res.data.obs_trials : res.data.obs

function _movement_states(res, model)
    fit_data = _movement_fit_data(res)
    if fit_data isa Vector
        return vcat(viterbi(model, fit_data)...)
    else
        return viterbi(model, fit_data)
    end
end

function _entropy_class_probs(γ)
    if γ isa Vector
        return sum(-sum(p > 0 ? p * log(p) : 0.0 for p in γ_trial) for γ_trial in γ)
    else
        return -sum(p > 0 ? p * log(p) : 0.0 for p in γ)
    end
end

# =============================================================================
# StepAngleEmission: Gaussian (log-step) ⊕ WrappedCauchy (turn angle)
# =============================================================================

"""
    StepAngleEmission

Composite emission for GPS movement data.

- Column 1: z-scored log step length — modelled as `GaussianEmission(output_dim=1)`.
  Free parameters per component: 1 mean + 1 variance = **2**.
- Column 2: turn angle in radians — modelled as `WrappedCauchyEmission(output_dim=1)`.
  Free parameters per component: 1 μ + 1 ρ = **2**.

Total emission parameters per component: **4** (used in BIC).
"""
mutable struct StepAngleEmission{GE<:GaussianEmission,WC<:WrappedCauchyEmission} <:
               StateSpaceDynamics.EmissionModel
    step::GE
    angle::WC
    output_dim::Int
end

StepAngleEmission(step::GaussianEmission, angle::WrappedCauchyEmission) =
    StepAngleEmission(step, angle, 2)

function StateSpaceDynamics.loglikelihood(
    model::StepAngleEmission, Y::AbstractMatrix{T}
) where {T<:Real}
    return StateSpaceDynamics.loglikelihood(model.step,  @view Y[:, 1:1]) .+
           StateSpaceDynamics.loglikelihood(model.angle, @view Y[:, 2:2])
end

function StateSpaceDynamics.fit!(
    model::StepAngleEmission, Y::AbstractMatrix{T},
    w::AbstractVector{T} = ones(T, size(Y, 1)),
) where {T<:Real}
    StateSpaceDynamics.fit!(model.step,  @view(Y[:, 1:1]), w)
    StateSpaceDynamics.fit!(model.angle, @view(Y[:, 2:2]), w)
    return model
end

function Random.rand(rng::AbstractRNG, model::StepAngleEmission; n::Int = 1)
    return hcat(rand(rng, model.step; n = n), rand(rng, model.angle; n = n))
end

Random.rand(model::StepAngleEmission; kwargs...) =
    rand(Random.default_rng(), model; kwargs...)

function StateSpaceDynamics.ACDC._emission_to_driver!(
    ε_out::AbstractVector{T}, emission::StepAngleEmission,
    y::AbstractVector{T}, ::Nothing,
) where {T<:Real}
    StateSpaceDynamics.ACDC._emission_to_driver!(
        @view(ε_out[1:1]), emission.step,  @view(y[1:1]), nothing)
    StateSpaceDynamics.ACDC._emission_to_driver!(
        @view(ε_out[2:2]), emission.angle, @view(y[2:2]), nothing)
end

# =============================================================================
# CardioidEmission: unimodal circular emission with bounded concentration
# =============================================================================

"""
    CardioidEmission

Cardioid emission model: `f(θ; μ, ρ) = (1/2π)(1 + 2ρ cos(θ − μ))`.

- μ: mean direction (radians)
- ρ: concentration, ρ ∈ [0, 0.5]

Unimodal on the circle. At ρ=0 it is uniform; at ρ=0.5 it touches zero
at the antipode. Unlike WrappedCauchy (bimodal at low ρ), the cardioid
is strictly unimodal for all ρ > 0.

MLE is fully closed-form: μ = circular mean, ρ = min(R̄, 0.5).

Free parameters per dimension: 1 μ + 1 ρ = **2**.
"""
mutable struct CardioidEmission{T<:Real,V<:AbstractVector{T}} <: StateSpaceDynamics.EmissionModel
    output_dim::Int
    μ::V   # mean direction(s), radians
    ρ::V   # concentration(s), 0 ≤ ρ ≤ 0.5
end

function CardioidEmission(; output_dim::Int, μ::AbstractVector{T}, ρ::AbstractVector{T}) where {T<:Real}
    @assert length(μ) == output_dim && length(ρ) == output_dim
    @assert all(0 .<= ρ .<= 0.5) "ρ must be in [0, 0.5]"
    return CardioidEmission(output_dim, μ, ρ)
end

function StateSpaceDynamics.loglikelihood(
    model::CardioidEmission, Y::AbstractMatrix{T}
) where {T<:Real}
    n_obs, d = size(Y)
    ll = zeros(T, n_obs)
    log2pi = log(T(2π))
    @inbounds for i in 1:n_obs
        for j in 1:d
            # f(θ) = (1/2π)(1 + 2ρ cos(θ − μ))
            val = one(T) + T(2) * model.ρ[j] * cos(Y[i, j] - model.μ[j])
            ll[i] += log(max(val, eps(T))) - log2pi
        end
    end
    return ll
end

function StateSpaceDynamics.fit!(
    model::CardioidEmission, Y::AbstractMatrix{T},
    w::AbstractVector{T} = ones(T, size(Y, 1)),
) where {T<:Real}
    d = size(Y, 2)
    wsum = sum(w)
    eps_T = eps(T)
    for j in 1:d
        s = sum(w .* sin.(@view(Y[:, j])))
        c = sum(w .* cos.(@view(Y[:, j])))
        model.μ[j] = atan(s, c)
        R_bar = sqrt(s^2 + c^2) / max(wsum, eps_T)
        model.ρ[j] = clamp(R_bar, zero(T), T(0.5))
    end
    return model
end

function Random.rand(rng::AbstractRNG, model::CardioidEmission; n::Int = 1)
    d = model.output_dim
    T_el = eltype(model.μ)
    samples = Matrix{T_el}(undef, n, d)
    for j in 1:d
        μ_j, ρ_j = model.μ[j], model.ρ[j]
        # Rejection sampling: envelope is uniform on [-π, π), max density = (1+2ρ)/2π
        M = one(T_el) + T_el(2) * ρ_j
        for i in 1:n
            while true
                θ = T_el(2π) * rand(rng, T_el) - T_el(π)
                f = one(T_el) + T_el(2) * ρ_j * cos(θ - μ_j)
                if rand(rng, T_el) * M <= f
                    samples[i, j] = θ
                    break
                end
            end
        end
    end
    return samples
end

Random.rand(model::CardioidEmission; kwargs...) =
    rand(Random.default_rng(), model; kwargs...)

"""
CDF of the cardioid distribution: F(θ; μ, ρ) = (θ − (−π))/2π + (ρ/π)(sin(θ−μ) − sin(−π−μ)).
Normalized so F(−π)=0, F(π)=1.
"""
function _cardioid_cdf(θ::T, μ::T, ρ::T) where {T<:Real}
    # F(θ) = (θ + π)/(2π) + (ρ/π)(sin(θ − μ) + sin(μ + π))
    # Since sin(μ + π) = −sin(μ):
    return (θ + T(π)) / T(2π) + (ρ / T(π)) * (sin(θ - μ) - sin(μ))
end

function StateSpaceDynamics.ACDC._emission_to_driver!(
    ε_out::AbstractVector{T}, emission::CardioidEmission,
    y::AbstractVector{T}, ::Nothing,
) where {T<:Real}
    for d in eachindex(ε_out)
        ε_out[d] = clamp(_cardioid_cdf(y[d], emission.μ[d], emission.ρ[d]), eps(T), one(T) - eps(T))
    end
end

# =============================================================================
# StepAngleCardEmission: Gaussian (log-step) ⊕ Cardioid (turn angle)
# =============================================================================

"""
    StepAngleCardEmission

Composite emission: Gaussian on z-scored log step length + Cardioid on turn angle.
Unimodal circular model with max concentration ρ=0.5.
Free parameters per component: 2 (Gaussian) + 2 (Cardioid) = **4**.
"""
mutable struct StepAngleCardEmission{GE<:GaussianEmission,CD<:CardioidEmission} <:
               StateSpaceDynamics.EmissionModel
    step::GE
    angle::CD
    output_dim::Int
end

StepAngleCardEmission(step::GaussianEmission, angle::CardioidEmission) =
    StepAngleCardEmission(step, angle, 2)

function StateSpaceDynamics.loglikelihood(
    model::StepAngleCardEmission, Y::AbstractMatrix{T}
) where {T<:Real}
    return StateSpaceDynamics.loglikelihood(model.step,  @view Y[:, 1:1]) .+
           StateSpaceDynamics.loglikelihood(model.angle, @view Y[:, 2:2])
end

function StateSpaceDynamics.fit!(
    model::StepAngleCardEmission, Y::AbstractMatrix{T},
    w::AbstractVector{T} = ones(T, size(Y, 1)),
) where {T<:Real}
    StateSpaceDynamics.fit!(model.step,  @view(Y[:, 1:1]), w)
    StateSpaceDynamics.fit!(model.angle, @view(Y[:, 2:2]), w)
    return model
end

function Random.rand(rng::AbstractRNG, model::StepAngleCardEmission; n::Int = 1)
    return hcat(rand(rng, model.step; n = n), rand(rng, model.angle; n = n))
end

Random.rand(model::StepAngleCardEmission; kwargs...) =
    rand(Random.default_rng(), model; kwargs...)

function StateSpaceDynamics.ACDC._emission_to_driver!(
    ε_out::AbstractVector{T}, emission::StepAngleCardEmission,
    y::AbstractVector{T}, ::Nothing,
) where {T<:Real}
    StateSpaceDynamics.ACDC._emission_to_driver!(
        @view(ε_out[1:1]), emission.step,  @view(y[1:1]), nothing)
    StateSpaceDynamics.ACDC._emission_to_driver!(
        @view(ε_out[2:2]), emission.angle, @view(y[2:2]), nothing)
end

# =============================================================================
# BIC parameter counts
# =============================================================================

"""
BIC parameter count for `StepAngleEmission`:
  4 emission params/component (1 μ_G + 1 σ²_G + 1 μ_WC + 1 ρ_WC)
  + K*(K-1) transition + (K-1) initial state.
"""
_n_params_step_angle(K::Int) = K * 4 + K * (K - 1) + (K - 1)

"""
BIC parameter count for a 2D Gaussian emission on `[z_log_step, turn_angle]`:
  5 emission params/component = 2 means + 3 covariance parameters
  + K*(K-1) transition + (K-1) initial state.
"""
_n_params_gaussian_alt(K::Int) = K * 5 + K * (K - 1) + (K - 1)

# =============================================================================
# Model fitting
# =============================================================================

"""
    compute_movement_analysis(df_animal, individual_id, seed; ...) -> NamedTuple

Fit StepAngleEmission HMMs (Gaussian log-step + WrappedCauchy angle) over `K_range`.

BIC uses the correct 4-parameter-per-component count for `StepAngleEmission`.
ACDC and ICL are both computed. Discrepancy: `KLDiscrepancy`.
"""
function compute_movement_analysis(
    df_animal::DataFrame,
    individual_id::String,
    seed::Int;
    K_range    = 2:5,
    n_restarts = 10,
    max_iters  = 200,
    tol        = 1e-4,
    use_cache  = true,
    cache_dir  = nothing,
    split_on_gaps::Bool = true,
    gap_threshold_hours::Float64 = 1.5,
)
    println("\n" * "="^70)
    println("COMPUTATION [$individual_id]: StepAngleEmission HMM")
    println("="^70)

    Random.seed!(seed)
    _cache_dir = isnothing(cache_dir) ?
        joinpath(@__DIR__, "..", "cache", "movement_base_trials_$(replace(individual_id, " " => "_"))") :
        cache_dir
    mkpath(_cache_dir)

    trial_dfs, break_idxs, gap_hours = split_on_gaps ?
        _split_cleaned_individual_trials(df_animal; gap_threshold_hours = gap_threshold_hours) :
        ([df_animal], Int[], Float64[])
    isempty(trial_dfs) && error("No valid movement trials remain after gap splitting.")

    steps_trials  = [log.(disallowmissing(trial[:, :step_length])) for trial in trial_dfs]
    angles_trials = [collect(disallowmissing(trial[:, :turn_angle])) for trial in trial_dfs]
    steps  = vcat(steps_trials...)
    angles = vcat(angles_trials...)

    step_mean, step_std = mean(steps), std(steps)
    observations_trials = [
        Matrix(transpose(hcat((steps_trial .- step_mean) ./ step_std, angles_trial))) for
        (steps_trial, angles_trial) in zip(steps_trials, angles_trials)
    ]
    observations = reduce(hcat, observations_trials)
    fit_data = observations_trials
    T_obs = sum(size(obs, 2) for obs in observations_trials)
    println("  T = $T_obs timesteps across $(length(observations_trials)) trial(s)")
    !isempty(break_idxs) && println("  Split at $(length(break_idxs)) gap(s) > $(gap_threshold_hours)h")

    nK = length(K_range)
    models          = Vector{Union{Nothing,HiddenMarkovModel{Float64}}}(nothing, nK)
    results         = Vector{Union{Nothing,ACDCResult{Float64}}}(nothing, nK)
    all_drivers     = Dict{Int,Vector{Matrix{Float64}}}()
    log_likelihoods = fill(-Inf, nK)
    bic_values      = fill(Inf,  nK)
    icl_values      = fill(Inf,  nK)
    cache_status    = fill(:none, nK)

    for (i, K) in enumerate(K_range)
        cache_file = joinpath(_cache_dir, "hmm_K$(K).jls")
        best_ll    = -Inf
        best_model = nothing

        if use_cache && isfile(cache_file)
            try
                cached     = deserialize(cache_file)
                best_model = cached.model
                best_ll    = Float64(cached.ll)
                cache_status[i] = :loaded
                println("  Loaded K=$K from cache")
            catch
                println("  Cache load failed for K=$K; refitting")
            end
        end

        if isnothing(best_model)
            print("  Fitting K=$K...")
            for _ in 1:n_restarts
                try
                    emissions = [
                        StepAngleEmission(
                            GaussianEmission(1, randn(1), Matrix{Float64}(I(1))),
                            WrappedCauchyEmission(;
                                output_dim = 1,
                                μ = [rand(Uniform(-π, π))],
                                ρ = [0.5],
                            ),
                        ) for _ in 1:K
                    ]
                    A_init = rand(K, K)
                    A_init ./= sum(A_init; dims = 2)
                    model_r = HiddenMarkovModel(; K = K, B = emissions, A = A_init, πₖ = fill(1.0/K, K))
                    ll_hist = fit!(model_r, fit_data; max_iters = max_iters, tol = tol)
                    finite_ll = filter(isfinite, ll_hist)
                    isempty(finite_ll) && continue
                    ll = maximum(finite_ll)
                    if ll > best_ll
                        best_ll = ll; best_model = model_r
                    end
                catch
                end
            end
            isnothing(best_model) && (println(" FAILED."); continue)
            use_cache && serialize(cache_file, (model = best_model, ll = best_ll))
            cache_status[i] = :fitted
            println(" Done. (LL=$(round(best_ll; digits=1)))")
        end

        models[i]          = best_model
        log_likelihoods[i] = best_ll

        results[i]     = component_discrepancies(best_model, fit_data, KLDiscrepancy(); n_samples = 10)
        drv            = stochastic_drivers(best_model, fit_data; n_samples = 10)
        all_drivers[K] = drv.ε_pools

        n_params      = _n_params_step_angle(K)
        bic_values[i] = -2 * best_ll + n_params * log(T_obs)
        γ             = class_probabilities(best_model, fit_data)
        H             = _entropy_class_probs(γ)
        icl_values[i] = bic_values[i] + 2 * H
    end

    K_bic  = K_range[argmin(bic_values)]
    K_icl  = K_range[argmin(icl_values)]

    return (
        individual_id   = individual_id,
        variant         = :step_angle,
        data            = (
            lons   = vcat([trial[:, "location-long"] for trial in trial_dfs]...),
            lats   = vcat([trial[:, "location-lat"] for trial in trial_dfs]...),
            steps  = steps,
            angles = angles,
            obs    = observations,
            obs_trials = observations_trials,
            n_trials = length(observations_trials),
            break_idxs = break_idxs,
            gap_hours = gap_hours,
        ),
        models          = models,
        results         = results,
        drivers         = all_drivers,
        bic_values      = bic_values,
        icl_values      = icl_values,
        K_range         = K_range,
        selections      = (bic = K_bic, icl = K_icl),
        scaling         = (step_m = step_mean, step_s = step_std),
        log_likelihoods = log_likelihoods,
        cache_status    = cache_status,
    )
end

"""
    compute_movement_analysis_alt(df_animal, individual_id, seed; ...) -> NamedTuple

    Fit composite-emission HMMs with movement features `[z_log_step, turn_angle]`
over `K_range`.

This variant uses a plain 2D Gaussian emission on `[z_log_step, turn_angle]`.

BIC uses the correct 5-parameter-per-component count:
- 2 means
- 3 covariance parameters

ACDC and ICL are both computed. Discrepancy: `KLDiscrepancy`.
"""
function compute_movement_analysis_alt(
    df_animal::DataFrame,
    individual_id::String,
    seed::Int;
    K_range    = 2:5,
    n_restarts = 10,
    max_iters  = 200,
    tol        = 1e-4,
    use_cache  = true,
    cache_dir  = nothing,
    split_on_gaps::Bool = true,
    gap_threshold_hours::Float64 = 1.5,
)
    println("\n" * "="^70)
    println("COMPUTATION [$individual_id]: Gaussian(z_log_step, turn_angle) HMM")
    println("="^70)

    Random.seed!(seed)
    _cache_dir = isnothing(cache_dir) ?
        joinpath(@__DIR__, "..", "cache", "movement_gaussian_alt_trials_$(replace(individual_id, " " => "_"))") :
        cache_dir
    mkpath(_cache_dir)

    trial_dfs, break_idxs, gap_hours = split_on_gaps ?
        _split_cleaned_individual_trials(df_animal; gap_threshold_hours = gap_threshold_hours) :
        ([df_animal], Int[], Float64[])
    isempty(trial_dfs) && error("No valid movement trials remain after gap splitting.")

    steps_trials  = [log.(disallowmissing(trial[:, :step_length])) for trial in trial_dfs]
    angles_trials = [collect(disallowmissing(trial[:, :turn_angle])) for trial in trial_dfs]
    steps  = vcat(steps_trials...)
    angles = vcat(angles_trials...)
    step_mean, step_std = mean(steps), std(steps)

    observations_trials = [
        Matrix(transpose(hcat((steps_trial .- step_mean) ./ step_std, angles_trial))) for
        (steps_trial, angles_trial) in zip(steps_trials, angles_trials)
    ]
    observations = reduce(hcat, observations_trials)
    fit_data = observations_trials
    T_obs = sum(size(obs, 2) for obs in observations_trials)
    D_obs = size(observations, 1)
    println("  T = $T_obs timesteps across $(length(observations_trials)) trial(s), D = $D_obs features")
    !isempty(break_idxs) && println("  Split at $(length(break_idxs)) gap(s) > $(gap_threshold_hours)h")

    nK = length(K_range)
    models          = Vector{Union{Nothing,HiddenMarkovModel{Float64}}}(nothing, nK)
    results         = Vector{Union{Nothing,ACDCResult{Float64}}}(nothing, nK)
    all_drivers     = Dict{Int,Vector{Matrix{Float64}}}()
    log_likelihoods = fill(-Inf, nK)
    bic_values      = fill(Inf,  nK)
    icl_values      = fill(Inf,  nK)
    cache_status    = fill(:none, nK)

    for (i, K) in enumerate(K_range)
        cache_file = joinpath(_cache_dir, "hmm_K$(K).jls")
        best_ll    = -Inf
        best_model = nothing

        if use_cache && isfile(cache_file)
            try
                cached     = deserialize(cache_file)
                best_model = cached.model
                best_ll    = Float64(cached.ll)
                cache_status[i] = :loaded
                println("  Loaded K=$K from cache")
            catch
                println("  Cache load failed for K=$K; refitting")
            end
        end

        if isnothing(best_model)
            print("  Fitting K=$K...")
            for _ in 1:n_restarts
                try
                    emissions = [
                        GaussianEmission(;
                            output_dim = 2,
                            μ = randn(2),
                            Σ = Matrix{Float64}(I, 2, 2),
                        ) for _ in 1:K
                    ]
                    A_init = rand(K, K)
                    A_init ./= sum(A_init; dims = 2)
                    model_r = HiddenMarkovModel(; K = K, B = emissions, A = A_init, πₖ = fill(1.0/K, K))
                    ll_hist = fit!(model_r, fit_data; max_iters = max_iters, tol = tol)
                    finite_ll = filter(isfinite, ll_hist)
                    isempty(finite_ll) && continue
                    ll = maximum(finite_ll)
                    if ll > best_ll
                        best_ll = ll; best_model = model_r
                    end
                catch
                end
            end
            isnothing(best_model) && (println(" FAILED."); continue)
            use_cache && serialize(cache_file, (model = best_model, ll = best_ll))
            cache_status[i] = :fitted
            println(" Done. (LL=$(round(best_ll; digits=1)))")
        end

        models[i]          = best_model
        log_likelihoods[i] = best_ll

        results[i]     = component_discrepancies(best_model, fit_data, KLDiscrepancy(); n_samples = 10)
        drv            = stochastic_drivers(best_model, fit_data; n_samples = 10)
        all_drivers[K] = drv.ε_pools

        n_params      = _n_params_gaussian_alt(K)
        bic_values[i] = -2 * best_ll + n_params * log(T_obs)
        γ             = class_probabilities(best_model, fit_data)
        H             = _entropy_class_probs(γ)
        icl_values[i] = bic_values[i] + 2 * H
    end

    K_bic  = K_range[argmin(bic_values)]
    K_icl  = K_range[argmin(icl_values)]

    return (
        individual_id   = individual_id,
        variant         = :gaussian_alt,
        data            = (
            lons   = vcat([trial[:, "location-long"] for trial in trial_dfs]...),
            lats   = vcat([trial[:, "location-lat"] for trial in trial_dfs]...),
            steps  = steps,
            angles = angles,
            obs    = observations,
            obs_trials = observations_trials,
            n_trials = length(observations_trials),
            break_idxs = break_idxs,
            gap_hours = gap_hours,
        ),
        models          = models,
        results         = results,
        drivers         = all_drivers,
        bic_values      = bic_values,
        icl_values      = icl_values,
        K_range         = K_range,
        selections      = (bic = K_bic, icl = K_icl),
        scaling         = (step_m = step_mean, step_s = step_std),
        log_likelihoods = log_likelihoods,
        cache_status    = cache_status,
    )
end

# =============================================================================
# Alt2: StepAngleCardEmission (Gaussian log-step + Cardioid angle)
# =============================================================================

"""
    compute_movement_analysis_alt2(df_animal, individual_id, seed; ...) -> NamedTuple

Fit StepAngleCardEmission HMMs (Gaussian log-step + Cardioid angle) over `K_range`.
Cardioid is unimodal on the circle with max concentration ρ=0.5 — a controlled
misspecification compared to the bimodal WrappedCauchy in the base model.
"""
function compute_movement_analysis_alt2(
    df_animal::DataFrame,
    individual_id::String,
    seed::Int;
    K_range    = 1:8,
    n_restarts = 10,
    max_iters  = 200,
    tol        = 1e-4,
    use_cache  = true,
    cache_dir  = nothing,
    split_on_gaps::Bool = true,
    gap_threshold_hours::Float64 = 1.5,
)
    println("\n" * "="^70)
    println("COMPUTATION [$individual_id]: StepAngleCardEmission HMM (Gaussian + Cardioid)")
    println("="^70)

    Random.seed!(seed)
    _cache_dir = isnothing(cache_dir) ?
        joinpath(@__DIR__, "..", "cache", "movement_cardioid_$(replace(individual_id, " " => "_"))") :
        cache_dir
    mkpath(_cache_dir)

    trial_dfs, break_idxs, gap_hours = split_on_gaps ?
        _split_cleaned_individual_trials(df_animal; gap_threshold_hours = gap_threshold_hours) :
        ([df_animal], Int[], Float64[])
    isempty(trial_dfs) && error("No valid movement trials remain after gap splitting.")

    steps_trials  = [log.(disallowmissing(trial[:, :step_length])) for trial in trial_dfs]
    angles_trials = [collect(disallowmissing(trial[:, :turn_angle])) for trial in trial_dfs]
    steps  = vcat(steps_trials...)
    angles = vcat(angles_trials...)
    step_mean, step_std = mean(steps), std(steps)

    observations_trials = [
        Matrix(transpose(hcat((steps_trial .- step_mean) ./ step_std, angles_trial))) for
        (steps_trial, angles_trial) in zip(steps_trials, angles_trials)
    ]
    observations = reduce(hcat, observations_trials)
    fit_data = observations_trials
    T_obs = sum(size(obs, 2) for obs in observations_trials)
    println("  T = $T_obs timesteps across $(length(observations_trials)) trial(s)")
    !isempty(break_idxs) && println("  Split at $(length(break_idxs)) gap(s) > $(gap_threshold_hours)h")

    nK = length(K_range)
    models          = Vector{Union{Nothing,HiddenMarkovModel{Float64}}}(nothing, nK)
    results         = Vector{Union{Nothing,ACDCResult{Float64}}}(nothing, nK)
    all_drivers     = Dict{Int,Vector{Matrix{Float64}}}()
    log_likelihoods = fill(-Inf, nK)
    bic_values      = fill(Inf,  nK)
    icl_values      = fill(Inf,  nK)
    cache_status    = fill(:none, nK)

    for (i, K) in enumerate(K_range)
        cache_file = joinpath(_cache_dir, "hmm_K$(K).jls")
        best_ll    = -Inf
        best_model = nothing

        if use_cache && isfile(cache_file)
            try
                cached     = deserialize(cache_file)
                best_model = cached.model
                best_ll    = Float64(cached.ll)
                cache_status[i] = :loaded
                println("  Loaded K=$K from cache")
            catch
                println("  Cache load failed for K=$K; refitting")
            end
        end

        if isnothing(best_model)
            print("  Fitting K=$K...")
            for _ in 1:n_restarts
                try
                    emissions = [
                        StepAngleCardEmission(
                            GaussianEmission(1, randn(1), Matrix{Float64}(I(1))),
                            CardioidEmission(;
                                output_dim = 1,
                                μ = [rand(Uniform(-π, π))],
                                ρ = [rand(Uniform(0.0, 0.5))],
                            ),
                        ) for _ in 1:K
                    ]
                    A_init = rand(K, K)
                    A_init ./= sum(A_init; dims = 2)
                    model_r = HiddenMarkovModel(; K = K, B = emissions, A = A_init, πₖ = fill(1.0/K, K))
                    ll_hist = fit!(model_r, fit_data; max_iters = max_iters, tol = tol)
                    finite_ll = filter(isfinite, ll_hist)
                    isempty(finite_ll) && continue
                    ll = maximum(finite_ll)
                    if ll > best_ll
                        best_ll = ll; best_model = model_r
                    end
                catch
                end
            end
            isnothing(best_model) && (println(" FAILED."); continue)
            use_cache && serialize(cache_file, (model = best_model, ll = best_ll))
            cache_status[i] = :fitted
            println(" Done. (LL=$(round(best_ll; digits=1)))")
        end

        models[i]          = best_model
        log_likelihoods[i] = best_ll

        results[i]     = component_discrepancies(best_model, fit_data, KLDiscrepancy(); n_samples = 10)
        drv            = stochastic_drivers(best_model, fit_data; n_samples = 10)
        all_drivers[K] = drv.ε_pools

        n_params      = _n_params_step_angle(K)  # same count: 4 emission params per K
        bic_values[i] = -2 * best_ll + n_params * log(T_obs)
        γ             = class_probabilities(best_model, fit_data)
        H             = _entropy_class_probs(γ)
        icl_values[i] = bic_values[i] + 2 * H
    end

    K_bic  = K_range[argmin(bic_values)]
    K_icl  = K_range[argmin(icl_values)]

    return (
        individual_id   = individual_id,
        variant         = :cardioid,
        data            = (
            lons   = vcat([trial[:, "location-long"] for trial in trial_dfs]...),
            lats   = vcat([trial[:, "location-lat"] for trial in trial_dfs]...),
            steps  = steps,
            angles = angles,
            obs    = observations,
            obs_trials = observations_trials,
            n_trials = length(observations_trials),
            break_idxs = break_idxs,
            gap_hours = gap_hours,
        ),
        models          = models,
        results         = results,
        drivers         = all_drivers,
        bic_values      = bic_values,
        icl_values      = icl_values,
        K_range         = K_range,
        selections      = (bic = K_bic, icl = K_icl),
        scaling         = (step_m = step_mean, step_s = step_std),
        log_likelihoods = log_likelihoods,
        cache_status    = cache_status,
    )
end

# =============================================================================
# Plotting helpers
# =============================================================================

include("FigureExport.jl")

"""
Map an internal `res.variant` symbol to the public base/alt1/alt2 label used
in filenames and statistics output.
"""
function _variant_label(variant::Symbol)
    variant === :step_angle    && return "base"
    variant === :gaussian_alt  && return "alt1"
    variant === :cardioid      && return "alt2"
    return string(variant)
end

_movement_plot_matrix(res) =
    Matrix(transpose(hasproperty(res.data, :obs_plot) ? res.data.obs_plot : res.data.obs))

function _stationary_distribution(model::HiddenMarkovModel)
    A = model.A
    K = size(A, 1)
    M = copy(A') .- I(K)
    M[end, :] .= 1.0
    b = zeros(K); b[end] = 1.0
    π = M \ b
    π = max.(π, 0.0)
    return π ./ sum(π)
end

# Build the ACDC envelope (max per-component KL discrepancy vs K) panel.
# Mirrors the panel produced in SpikeTrainHMMUtils._envelope_panel: per-K
# gray scatter for "best per K", black envelope line, and −xx% drop labels
# positioned above each envelope segment. Auto-expands ylim so labels stay
# within the subplot when y-range is narrow.
function _acdc_envelope_panel(K_range, results_vec, hires_kwargs)
    p = plot(; title = "ACDC Envelope (KL)", xlabel = "K",
        ylabel = "max per-component",
        legend = :outerbottom, legend_columns = 1,
        xticks = collect(minimum(K_range):maximum(K_range)),
        hires_kwargs...)
    vK = Int[]; vZ = Float64[]; vIdx = Int[]
    for (i, K) in enumerate(K_range)
        r = results_vec[i]
        isnothing(r) && continue
        ds = r.component_discrepancies
        all(isfinite, ds) || continue
        mx = maximum(ds)
        mx >= 0 || continue
        push!(vK, K); push!(vZ, mx); push!(vIdx, i)
    end
    isempty(vK) && return p
    node_colors = [palette(:auto)[i] for i in vIdx]
    plot!(p, vK, vZ; color = :gray, linewidth = 1.0, alpha = 0.5,
        label = "best per K")
    scatter!(p, vK, vZ; markersize = 5, markerstrokewidth = 0,
        markercolor = node_colors, markeralpha = 0.55, label = nothing)

    eK = Int[]; eZ = Float64[]; eC = []; bsf = Inf
    for (k, z, c) in zip(vK, vZ, node_colors)
        if z < bsf
            push!(eK, k); push!(eZ, z); push!(eC, c); bsf = z
        end
    end
    isempty(eK) && return p
    plot!(p, eK, eZ; color = :black, linewidth = 2.5, label = "envelope")
    scatter!(p, eK, eZ; markersize = 8, markerstrokewidth = 1.8,
        markerstrokecolor = :black, markercolor = eC, label = nothing)
    ann_ys = Float64[]
    k_rng    = max(float(maximum(eK) - minimum(eK)), 1.0)
    z_rng    = max(maximum(eZ) - minimum(eZ), eps(Float64))
    for j in 1:(length(eK) - 1)
        k0, k1 = eK[j], eK[j+1]
        z0, z1 = eZ[j], eZ[j+1]
        drop_pct = iszero(z0) ? 0.0 : (z0 - z1) / z0 * 100
        dk_n    = (k1 - k0) / k_rng
        dz_n    = (z1 - z0) / z_rng
        mag_n   = sqrt(dk_n^2 + dz_n^2)
        iszero(mag_n) && continue
        perp_x_n = -dz_n / mag_n
        perp_y_n =  dk_n / mag_n
        off_n = 0.045
        xm = (k0 + k1) / 2 + perp_x_n * off_n * k_rng
        ym = (z0 + z1) / 2 + perp_y_n * off_n * z_rng
        push!(ann_ys, ym)
        annotate!(p, xm, ym,
            text(@sprintf("−%.1f%%", drop_pct), 8, :black, :hcenter, :vcenter))
    end
    if !isempty(ann_ys)
        ymin_raw = min(minimum(vZ), minimum(ann_ys))
        ymax_raw = max(maximum(vZ), maximum(ann_ys))
        pad = max((ymax_raw - ymin_raw) * 0.12, eps(Float64))
        ymax_full = ymax_raw + pad
        ymin_full = max(0.0, ymin_raw - pad)
        plot!(p; ylim = (ymin_full, ymax_full))
    end
    return p
end

"""
    get_movement_plots(res, individual_id; figures_dir=nothing) -> NamedTuple

Generate the trajectory and model-selection figures used in the paper.

Returns `(trajectory, selection)`.
"""
function get_movement_plots(res, individual_id::String; figures_dir = nothing, rho_grid = nothing, K_display = nothing)
    lons, lats    = res.data.lons, res.data.lats
    steps, angles = res.data.steps, res.data.angles
    K_bic   = res.selections.bic
    K_icl   = res.selections.icl
    K_range = res.K_range
    available_Ks = [
        K for (i, K) in enumerate(K_range)
        if !isnothing(res.models[i]) && haskey(res.drivers, K)
    ]
    isempty(available_Ks) && error("No fitted models available for plotting.")
    K_show_req = isnothing(K_display) ? K_bic : K_display
    K_show  = K_show_req in available_Ks ? K_show_req :
              (K_bic in available_Ks ? K_bic : first(available_Ks))

    _plain_fmt(x) = (x == 0 ? "0" : abs(x) < 0.01 || abs(x) >= 1e6 ? @sprintf("%.0e", x) : @sprintf("%g", x))
    hires = (
        dpi = 300,
        tickfontsize = 10, guidefontsize = 12, legendfontsize = 10, titlefontsize = 13,
        fontfamily = "Helvetica",
        left_margin = 10Plots.mm, right_margin = 5Plots.mm,
        top_margin = 5Plots.mm, bottom_margin = 24Plots.mm,
        yformatter = _plain_fmt, xformatter = _plain_fmt,
    )

    # ── Trajectory ────────────────────────────────────────────────────────────
    p_traj = plot(
        lons, lats;
        line_z = 1:length(lons), c = :viridis,
        title  = "GPS Trajectory: $individual_id (Time-Colored)",
        xlabel = "Longitude", ylabel = "Latitude",
        aspect_ratio = :equal, legend = false, size = (1000, 800), hires...,
    )

    # ── BIC / ICL + ACDC loss-ρ curve ────────────────────────────────────────
    p_bic = plot(
        collect(K_range), res.bic_values;
        marker = :circle, label = "BIC", linewidth = 2.4, markersize = 6,
        title  = "BIC / ICL", xlabel = "K", ylabel = "Score", color = :black,
        xticks = collect(minimum(K_range):maximum(K_range)),
        legend = :outerbottom, legend_columns = 2,
        hires...,
    )
    plot!(p_bic, collect(K_range), res.icl_values;
        marker = :diamond, label = "ICL", linewidth = 2.4, markersize = 6, color = :royalblue)
    vline!(p_bic, [K_bic]; linestyle = :dash,    color = :red,       label = "BIC K=$K_bic",  linewidth = 2.2)
    vline!(p_bic, [K_icl]; linestyle = :dashdot, color = :royalblue, label = "ICL K=$K_icl",  linewidth = 2.2)

    p_env = _acdc_envelope_panel(K_range, res.results, hires)
    p_selection = plot(p_bic, p_env; layout = (1, 2), size = (1300, 500), margin = 5Plots.mm)

    if !isnothing(figures_dir)
        mkpath(figures_dir)
        _save_png_pdf(p_traj,                 joinpath(figures_dir, "..", "..", "supplementary", "elephant_movement"), "movement_trajectory")
        _save_png_pdf(p_selection,            figures_dir, "movement_selection_base"; trim = true)
    end

    return (trajectory = p_traj, selection = p_selection)
end

"""
    get_movement_plots_alt(res, individual_id; figures_dir=nothing) -> NamedTuple

Generate the paper model-selection figure for the Gaussian alternative.

Returns `(selection,)`.
"""
function get_movement_plots_alt(res, individual_id::String; figures_dir = nothing, rho_grid = nothing, K_display = nothing)
    steps, angles = res.data.steps, res.data.angles
    K_bic   = res.selections.bic
    K_icl   = res.selections.icl
    K_range = res.K_range
    available_Ks = [
        K for (i, K) in enumerate(K_range)
        if !isnothing(res.models[i]) && haskey(res.drivers, K)
    ]
    isempty(available_Ks) && error("No fitted models available for plotting.")
    K_show_req = isnothing(K_display) ? K_bic : K_display
    K_show  = K_show_req in available_Ks ? K_show_req :
              (K_bic in available_Ks ? K_bic : first(available_Ks))
    kidx    = findfirst(==(K_show), collect(K_range))
    model_show = res.models[kidx]

    _plain_fmt(x) = (x == 0 ? "0" : abs(x) < 0.01 || abs(x) >= 1e6 ? @sprintf("%.0e", x) : @sprintf("%g", x))
    hires = (
        dpi = 300,
        tickfontsize = 10, guidefontsize = 12, legendfontsize = 10, titlefontsize = 13,
        fontfamily = "Helvetica",
        left_margin = 10Plots.mm, right_margin = 5Plots.mm,
        top_margin = 5Plots.mm, bottom_margin = 24Plots.mm,
        yformatter = _plain_fmt, xformatter = _plain_fmt,
    )

    p_bic = plot(
        collect(K_range), res.bic_values;
        marker = :circle, label = "BIC", linewidth = 2.4, markersize = 6,
        title  = "BIC / ICL", xlabel = "K", ylabel = "Score", color = :black,
        xticks = collect(minimum(K_range):maximum(K_range)),
        legend = :outerbottom, legend_columns = 2,
        hires...,
    )
    plot!(p_bic, collect(K_range), res.icl_values;
        marker = :diamond, label = "ICL", linewidth = 2.4, markersize = 6, color = :royalblue)
    vline!(p_bic, [K_bic]; linestyle = :dash,    color = :red,       label = "BIC K=$K_bic", linewidth = 2.2)
    vline!(p_bic, [K_icl]; linestyle = :dashdot, color = :royalblue, label = "ICL K=$K_icl", linewidth = 2.2)

    p_env = _acdc_envelope_panel(K_range, res.results, hires)
    p_selection = plot(p_bic, p_env; layout = (1, 2), size = (1300, 500), margin = 5Plots.mm)

    if !isnothing(figures_dir)
        mkpath(figures_dir)
        _save_png_pdf(p_selection,            figures_dir, "movement_selection_alt1"; trim = true)
    end

    return (selection = p_selection,)
end

"""
    get_movement_plots_alt2(res, individual_id; ...) -> NamedTuple

Generate the paper model-selection figure for the Cardioid alternative.
"""
function get_movement_plots_alt2(res, individual_id::String; figures_dir = nothing, rho_grid = nothing, K_display = nothing)
    K_bic   = res.selections.bic
    K_icl   = res.selections.icl
    K_range = res.K_range
    K_show  = isnothing(K_display) ? K_bic : K_display

    steps, angles = res.data.steps, res.data.angles

    _plain_fmt(x) = (x == 0 ? "0" : abs(x) < 0.01 || abs(x) >= 1e6 ? @sprintf("%.0e", x) : @sprintf("%g", x))
    hires = (
        dpi = 300,
        tickfontsize = 10, guidefontsize = 12, legendfontsize = 10, titlefontsize = 13,
        fontfamily = "Helvetica",
        left_margin = 10Plots.mm, right_margin = 5Plots.mm,
        top_margin = 5Plots.mm, bottom_margin = 24Plots.mm,
        yformatter = _plain_fmt, xformatter = _plain_fmt,
    )

    p_bic = plot(
        collect(K_range), res.bic_values;
        marker = :circle, label = "BIC", linewidth = 2.4, markersize = 6,
        title  = "BIC / ICL", xlabel = "K", ylabel = "Score", color = :black,
        xticks = collect(minimum(K_range):maximum(K_range)),
        legend = :outerbottom, legend_columns = 2,
        hires...,
    )
    plot!(p_bic, collect(K_range), res.icl_values;
        marker = :diamond, label = "ICL", linewidth = 2.4, markersize = 6, color = :royalblue)
    vline!(p_bic, [K_bic]; linestyle = :dash,    color = :red,       label = "BIC K=$K_bic",  linewidth = 2.2)
    vline!(p_bic, [K_icl]; linestyle = :dashdot, color = :royalblue, label = "ICL K=$K_icl",  linewidth = 2.2)

    p_env = _acdc_envelope_panel(K_range, res.results, hires)
    p_selection = plot(p_bic, p_env; layout = (1, 2), size = (1300, 500), margin = 5Plots.mm)

    if !isnothing(figures_dir)
        mkpath(figures_dir)
        _save_png_pdf(p_selection,            figures_dir, "movement_selection_alt2"; trim = true)
    end

    return (selection = p_selection,)
end

# =============================================================================
# State-feature grid
# =============================================================================

"""Return per-feature (lo, hi) limits with padding, derived from the observation matrix."""
function feature_xlims_from_res(res; pad_frac = 0.03)
    Y = _movement_plot_matrix(res)
    D = size(Y, 2)
    return [
        let lo = minimum(Y[:, d]), hi = maximum(Y[:, d]), pad = (hi - lo) * pad_frac
            (lo - pad, hi + pad)
        end
        for d in 1:D
    ]
end

"""Evaluate the marginal emission density for component k, feature d at points x."""
function component_feature_density(res, model, k::Int, d::Int, x::AbstractVector{<:Real})
    em = model.B[k]
    if hasproperty(em, :step) && hasproperty(em, :angle)
        d == 1 && return exp.(vec(StateSpaceDynamics.loglikelihood(em.step,  reshape(x, :, 1))))
        d == 2 && return exp.(vec(StateSpaceDynamics.loglikelihood(em.angle, reshape(x, :, 1))))
        return fill(NaN, length(x))
    end
    if hasproperty(em, :μ) && hasproperty(em, :Σ)
        return pdf.(Normal(em.μ[d], sqrt(max(em.Σ[d, d], 1e-12))), x)
    end
    return fill(NaN, length(x))
end

"""
    movement_state_feature_grid(res, K; ...) -> Plot

K × D grid of per-state feature histograms with emission density overlay.
"""
function movement_state_feature_grid(
    res, K;
    feature_names = nothing,
    feature_xlims = nothing,
    bins          = 40,
    color         = :steelblue,
    curve_color   = :crimson,
    left_margin   = 16Plots.mm,
    bottom_margin = 6Plots.mm,
    right_margin  = 4Plots.mm,
    top_margin    = 3Plots.mm,
    panel_w       = 360,
    panel_h       = 260,
)
    kidx  = findfirst(==(K), collect(res.K_range))
    isnothing(kidx) && error("K=$K not in K_range=$(res.K_range)")
    model = res.models[kidx]
    isnothing(model) && error("Model for K=$K is nothing")

    Y      = _movement_plot_matrix(res)
    states = _movement_states(res, model)
    D      = size(Y, 2)

    feat_names = if isnothing(feature_names)
        D == 2 && get(res, :variant, nothing) == :step_angle ? ["z_log_step", "turn_angle"] :
        D == 2 && get(res, :variant, nothing) == :gaussian_alt ? ["z_log_step", "turn_angle"] :
        D == 3 ? ["z_log_step", "sin(angle)", "cos(angle)"] :
        ["feature_$d" for d in 1:D]
    else
        feature_names
    end

    xlims_v = isnothing(feature_xlims) ? feature_xlims_from_res(res) : feature_xlims

    # Layout: split states into 2 columns when K >= 2
    n_state_cols = K >= 2 ? 2 : 1
    n_state_rows = cld(K, n_state_cols)
    grid_cols = D * n_state_cols
    grid_rows = n_state_rows

    p = plot(;
        layout = (grid_rows, grid_cols),
        size = (panel_w * grid_cols, panel_h * grid_rows),
        dpi = 300,
        left_margin = left_margin, right_margin = right_margin,
        top_margin = top_margin, bottom_margin = bottom_margin,
        tickfontsize = 11, guidefontsize = 13,
        titlefontsize = 14, yguidefontsize = 12,
    )

    # Blank out any subplot cells that won't be used by a state
    total_cells = grid_rows * grid_cols
    used = Set{Int}()
    for k in 1:K
        state_row = cld(k, n_state_cols)
        state_col = mod1(k, n_state_cols)
        for d in 1:D
            push!(used, (state_row - 1) * grid_cols + (state_col - 1) * D + d)
        end
    end
    for sp in 1:total_cells
        sp in used && continue
        plot!(p; subplot = sp, framestyle = :none, ticks = nothing,
            grid = false, legend = false, showaxis = false)
    end

    for k in 1:K
        idx = findall(==(k), states)
        # Position in 2-column state layout: row-major
        state_row = cld(k, n_state_cols)
        state_col = mod1(k, n_state_cols)
        for d in 1:D
            # Subplot index: (state_row - 1) * grid_cols + (state_col - 1) * D + d
            sp = (state_row - 1) * grid_cols + (state_col - 1) * D + d
            lo, hi = xlims_v[d]
            if isempty(idx)
                plot!(p; subplot = sp, legend = false, grid = false,
                    framestyle = :box, xlims = (lo, hi))
                annotate!(p, 0.5, 0.5, text("no data", 8, :gray); subplot = sp)
            else
                histogram!(p, Y[idx, d];
                    subplot = sp, bins = bins, normalize = true,
                    alpha = 0.75, color = color, label = nothing, xlims = (lo, hi))
                xg   = collect(range(lo, hi; length = 400))
                dens = component_feature_density(res, model, k, d, xg)
                any(isfinite, dens) && plot!(p, xg, dens;
                    subplot = sp, color = curve_color, linewidth = 1.9, label = nothing)
            end
            # Feature title only on the topmost row
            state_row == 1 && plot!(p; subplot = sp, title = feat_names[d])
            # State label only on the leftmost feature of each state
            d == 1 && plot!(p; subplot = sp, ylabel = "State $k")
            # x-label only on the bottommost row
            plot!(p; subplot = sp,
                xlabel = (state_row == n_state_rows ? "value" : ""), grid = false)
        end
    end
    return p
end

# =============================================================================
# Statistics export
# =============================================================================

"""
    _write_state_data_summary(io, Y_k, scaling)

Write empirical data description for observations `Y_k` (n_k × D matrix) assigned
to one state.  Columns are [z_log_step, turn_angle].
Prints enough information to reconstruct the histogram shapes:
  - step length percentiles (5/25/50/75/95) in metres
  - angle: circular stats (R̄, circular mean) and octant distribution
"""
function _write_state_data_summary(io::IO, Y_k::AbstractMatrix, scaling)
    n_k    = size(Y_k, 1)
    step_k = Y_k[:, 1]   # z-scored log-step
    ang_k  = Y_k[:, 2]   # turn angle (radians)

    # Step-length percentiles in metres
    pcts = [0.05, 0.25, 0.50, 0.75, 0.95]
    step_pct = [exp(quantile(step_k, p) * scaling.step_s + scaling.step_m) for p in pcts]
    @printf(io, "             step (m):  p5=%.0f  p25=%.0f  p50=%.0f  p75=%.0f  p95=%.0f\n",
        step_pct...)

    # Circular statistics
    R_bar     = sqrt(mean(cos.(ang_k))^2 + mean(sin.(ang_k))^2)
    circ_mean = atan(mean(sin.(ang_k)), mean(cos.(ang_k)))

    # Octant distribution: 8 bins of π/4 each, centred at 0, ±π/4, ±π/2, ±3π/4, π
    # Use 4 named bins that matter behaviourally
    frac_fwd  = count(a -> abs(a) < π/4, ang_k) / n_k * 100          # forward
    frac_side = count(a -> π/4 <= abs(a) < 3π/4, ang_k) / n_k * 100  # lateral
    frac_rev  = count(a -> abs(a) >= 3π/4, ang_k) / n_k * 100        # reversal
    @printf(io, "             angle:    R̄=%.3f  circ_mean=%+.3f rad  fwd(<π/4)=%.1f%%  side=%.1f%%  rev(>3π/4)=%.1f%%\n",
        R_bar, circ_mean, frac_fwd, frac_side, frac_rev)
end

"""
    write_movement_statistics(filepath, base_res, alt_res)

Write a text summary of BIC, ICL, ACDC, and state composition for both model variants
to `filepath`.
"""
function write_movement_statistics(filepath::String, base_res, alt_res, extra_results::Pair{String}...)
    mkpath(dirname(filepath))
    individual_id = base_res.individual_id
    open(filepath, "w") do io
        println(io, "=" ^ 70)
        println(io, "MOVEMENT HMM STATISTICS")
        println(io, "Individual: $individual_id")
        println(io, "Generated:  $(Dates.now())")
        println(io, "=" ^ 70)

        println(io, "\n--- Model Legend (used throughout this file and figure filenames) ---")
        println(io, "  base : StepAngleEmission — Gaussian log-step + WrappedCauchy turn angle")
        println(io, "         (correctly specified: circular topology, bimodal at low ρ)")
        println(io, "  alt1 : GaussianEmission(z_log_step, turn_angle) with full 2×2 covariance")
        println(io, "         (topologically wrong: Gaussian cannot wrap around ±π)")
        println(io, "  alt2 : StepAngleCardEmission — Gaussian log-step + Cardioid turn angle")
        println(io, "         (correct topology, limited expressiveness: ρ capped at 0.5, strictly unimodal)")

        steps = base_res.data.steps
        println(io, "\n--- Data Summary ---")
        @printf(io, "Observations:      %d\n", length(steps))
        if hasproperty(base_res.data, :n_trials)
            @printf(io, "Trials:            %d\n", base_res.data.n_trials)
            if !isempty(base_res.data.gap_hours)
                @printf(io, "Split gaps (hr):   %s\n", join([@sprintf("%.2f", g) for g in base_res.data.gap_hours], ", "))
            end
        end
        @printf(io, "Log-step (nats):   mean=%.3f  std=%.3f  min=%.3f  max=%.3f\n",
            mean(steps), std(steps), minimum(steps), maximum(steps))
        @printf(io, "Step (meters):     median=%.1f  max=%.1f\n",
            median(exp.(steps)), maximum(exp.(steps)))

        all_models = [
            ("[base] StepAngleEmission (Gaussian log-step + WrappedCauchy angle)", base_res),
            ("[alt1] GaussianEmission(z_log_step, turn_angle)", alt_res),
        ]
        for (label, res) in extra_results
            push!(all_models, (label, res))
        end

        for (label, res) in all_models
            println(io, "\n" * "─" ^ 70)
            println(io, label)
            println(io, "─" ^ 70)
            @printf(io, "%-5s  %-12s  %-12s  %-12s  KL discrepancies per component\n",
                "K", "LL", "BIC", "ICL")

            # Determine ACDC selection: K with smallest maximum component discrepancy
            K_acdc = let
                best_K = nothing; best_max = Inf
                for (i, K) in enumerate(res.K_range)
                    isnothing(res.results[i]) && continue
                    mx = maximum(res.results[i].component_discrepancies)
                    if mx < best_max; best_max = mx; best_K = K; end
                end
                best_K
            end

            for (i, K) in enumerate(res.K_range)
                ll_s  = isfinite(res.log_likelihoods[i]) ? @sprintf("%.1f", res.log_likelihoods[i]) : "—"
                bic_s = isfinite(res.bic_values[i])      ? @sprintf("%.1f", res.bic_values[i])      : "—"
                icl_s = isfinite(res.icl_values[i])      ? @sprintf("%.1f", res.icl_values[i])      : "—"
                discs = isnothing(res.results[i]) ? "—" :
                    join([@sprintf("%.4f", d) for d in res.results[i].component_discrepancies], "  ")
                sel = String[]
                K == res.selections.bic && push!(sel, "← BIC")
                K == res.selections.icl && push!(sel, "← ICL")
                K == K_acdc             && push!(sel, "← ACDC")
                @printf(io, "K=%-3d  %-12s  %-12s  %-12s  [%s]  %s\n",
                    K, ll_s, bic_s, icl_s, discs, join(sel, " "))
            end

            println(io)
            Y_plot = _movement_plot_matrix(res)  # T × D matrix
            for (i, K) in enumerate(res.K_range)
                model = res.models[i]
                isnothing(model) && continue
                sel_tag = ""
                K == res.selections.bic && (sel_tag *= " [BIC]")
                K == res.selections.icl && (sel_tag *= " [ICL]")
                K == K_acdc             && (sel_tag *= " [ACDC]")
                println(io, "State composition at K=$K$sel_tag:")
                states = _movement_states(res, model)
                T_tot  = length(states)
                for k in 1:K
                    idx_k = findall(==(k), states)
                    frac  = length(idx_k) / T_tot * 100
                    em    = model.B[k]
                    if hasproperty(em, :step) && hasproperty(em, :angle) &&
                       em.step isa GaussianEmission && em.angle isa WrappedCauchyEmission
                        μ_z      = em.step.μ[1]
                        step_log = μ_z * res.scaling.step_s + res.scaling.step_m
                        @printf(io,
                            "  State %-2d: %5.1f%%  step_median≈%7.0fm  WC_μ=%+.3f  WC_ρ=%.3f\n",
                            k, frac, exp(step_log), em.angle.μ[1], em.angle.ρ[1])
                    elseif hasproperty(em, :step) && hasproperty(em, :angle) &&
                       em.step isa GaussianEmission && em.angle isa CardioidEmission
                        μ_z      = em.step.μ[1]
                        step_log = μ_z * res.scaling.step_s + res.scaling.step_m
                        @printf(io,
                            "  State %-2d: %5.1f%%  step_median≈%7.0fm  Card_μ=%+.3f  Card_ρ=%.3f\n",
                            k, frac, exp(step_log), em.angle.μ[1], em.angle.ρ[1])
                    elseif hasproperty(em, :μ) && hasproperty(em, :Σ)
                        μ_vec    = em.μ
                        step_log = μ_vec[1] * res.scaling.step_s + res.scaling.step_m
                        ang_mean = μ_vec[2]
                        σ_step   = sqrt(em.Σ[1, 1])
                        σ_ang    = sqrt(em.Σ[2, 2])
                        ρ_sa     = em.Σ[1, 2] / max(sqrt(max(em.Σ[1, 1] * em.Σ[2, 2], 1e-12)), 1e-12)
                        @printf(io,
                            "  State %-2d: %5.1f%%  step_median≈%7.0fm  angle_mean=%+.3f rad  angle_sd=%.3f\n",
                            k, frac, exp(step_log), ang_mean, σ_ang)
                        @printf(io,
                            "             σ(z_step)=%.3f  corr(z_step,angle)=%.3f\n",
                            σ_step, clamp(ρ_sa, -1.0, 1.0))
                    else
                        @printf(io, "  State %-2d: %5.1f%%\n", k, frac)
                    end
                    # Empirical data description (shared for all emission types)
                    if length(idx_k) > 1
                        _write_state_data_summary(io, Y_plot[idx_k, :], res.scaling)
                    end
                end
                println(io)
            end
        end

        println(io, "\n" * "=" ^ 70)
    end
    println("Statistics written to: $filepath")
end

end  # module AnimalMovementUtils
