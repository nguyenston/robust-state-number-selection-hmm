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
