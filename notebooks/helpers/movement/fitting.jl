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
        joinpath(@__DIR__, "..", "..", "cache", "movement_base_trials_$(replace(individual_id, " " => "_"))") :
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
        joinpath(@__DIR__, "..", "..", "cache", "movement_gaussian_alt_trials_$(replace(individual_id, " " => "_"))") :
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
        joinpath(@__DIR__, "..", "..", "cache", "movement_cardioid_$(replace(individual_id, " " => "_"))") :
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
