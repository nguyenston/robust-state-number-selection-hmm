"""
    compute_bouthmm_models_allruns(sequences; K_range=1:8, seed=42, n_restarts=5, max_iters=200, tol=1e-4, cache_dir, use_cache=true)

Stores ALL `n_restarts` fitted models per K
to `bouthmm_K{K}_allruns.jls` (cached as `runs::Vector{Tuple{model, ll}}`).
Used by `run_bouthmm_selection_analysis_allruns` for per-metric-best filtering.
"""
function compute_bouthmm_models_allruns(sequences; K_range=1:8, seed=42, n_restarts=5, max_iters=200, tol=1e-4, cache_dir, use_cache=true)
    mkpath(cache_dir)
    K_values = collect(K_range)
    nK = length(K_values)
    D_obs = size(sequences[1], 1)

    all_runs = Vector{Vector{Tuple{Any,Float64}}}(undef, nK)
    cache_status = fill(:missing, nK)

    for (i, K) in enumerate(K_values)
        cache_file = joinpath(cache_dir, "bouthmm_K$(K)_allruns.jls")
        if use_cache && isfile(cache_file)
            try
                cached = deserialize(cache_file)
                all_runs[i] = cached.runs
                cache_status[i] = :loaded
                println("Loaded bout-HMM K=$K all $(length(cached.runs)) restarts from cache")
                continue
            catch
                println("Cache load failed for bout-HMM K=$K; refitting")
            end
        end

        print("Fitting bout-HMM K=$K ($(n_restarts) restarts)... ")
        runs = Tuple{Any,Float64}[]
        for r in 1:n_restarts
            try
                rng = MersenneTwister(seed + 10_000 * K + r)
                emissions = [
                    GaussianEmission(D_obs, randn(rng, D_obs), Matrix{Float64}(I, D_obs, D_obs)) for _ in 1:K
                ]
                A_init = rand(rng, K, K); A_init ./= sum(A_init; dims=2)
                π_init = fill(1.0 / K, K)
                model_r = HiddenMarkovModel(; K=K, B=emissions, A=A_init, πₖ=π_init)
                lls = fit!(model_r, sequences; max_iters=max_iters, tol=tol)
                finite_ll = filter(isfinite, lls)
                isempty(finite_ll) && continue
                push!(runs, (model_r, last(finite_ll)))
            catch
                continue
            end
        end
        if isempty(runs)
            cache_status[i] = :failed
            println("all FAILED.")
            all_runs[i] = Tuple{Any,Float64}[]
            continue
        end
        all_runs[i] = runs
        cache_status[i] = :fitted
        serialize(cache_file, (runs = runs,))
        println("$(length(runs))/$n_restarts succeeded.")
    end

    return (; K_range=K_values, all_runs, cache_status, cache_dir)
end

"""
    run_bouthmm_selection_analysis_allruns(model_run, sequences; discrepancy=KLDiscrepancy(), n_samples=10)

Per-metric-best selection over the bout-HMM all-runs sweep. Evaluates BIC,
ICL, and ACDC-KL on every restart at every K (ACDC computed on the
concatenated bout observations, BIC/ICL on per-bout sequences). Returns a
namedtuple compatible with downstream plotting + `models_per_metric` so
both BIC-best and ACDC-KL-best models are accessible.
"""
function run_bouthmm_selection_analysis_allruns(model_run, sequences;
        discrepancy=KLDiscrepancy(), n_samples=10)
    K_values = model_run.K_range
    all_runs = model_run.all_runs
    nK = length(K_values)

    T_total = sum(size(s, 2) for s in sequences)
    D_obs = size(sequences[1], 1)
    obs_concat = hcat(sequences...)

    models_bic = Vector{Union{Nothing,HiddenMarkovModel{Float64}}}(nothing, nK)
    models_icl = Vector{Union{Nothing,HiddenMarkovModel{Float64}}}(nothing, nK)
    models_kl  = Vector{Union{Nothing,HiddenMarkovModel{Float64}}}(nothing, nK)
    results_kl = Vector{Union{Nothing,ACDCResult{Float64}}}(nothing, nK)
    log_likelihoods = fill(-Inf, nK)
    bic_values = fill(Inf, nK)
    icl_values = fill(Inf, nK)
    all_evals  = Vector{Vector{NamedTuple}}(undef, nK)

    for (i, K) in enumerate(K_values)
        runs = all_runs[i]
        (isnothing(runs) || isempty(runs)) && continue

        n_params = K * D_obs + K * D_obs * (D_obs + 1) / 2 + K * (K - 1) + (K - 1)

        evals = NamedTuple[]
        for (model, ll) in runs
            bic = -2 * ll + n_params * log(T_total)
            all_probs = class_probabilities(model, sequences)
            H = sum(-sum(p > 0 ? p * log(p) : 0.0 for p in γ) for γ in all_probs)
            icl = bic + 2 * H
            res_kl = component_discrepancies(model, obs_concat, discrepancy; n_samples=n_samples)
            push!(evals, (
                model = model, ll = ll,
                bic = bic, icl = icl,
                res_kl = res_kl,
                max_kl = maximum(res_kl.component_discrepancies),
            ))
        end
        all_evals[i] = evals

        best_bic = evals[argmin([e.bic for e in evals])]
        best_icl = evals[argmin([e.icl for e in evals])]
        best_kl  = evals[argmin([e.max_kl for e in evals])]

        models_bic[i] = best_bic.model
        models_icl[i] = best_icl.model
        models_kl[i]  = best_kl.model

        bic_values[i] = best_bic.bic
        icl_values[i] = best_icl.icl
        log_likelihoods[i] = best_bic.ll
        results_kl[i] = best_kl.res_kl
    end

    K_bic = K_values[argmin(bic_values)]
    K_icl = K_values[argmin(icl_values)]
    K_acdc = let
        bestK, bestmx = first(K_values), Inf
        for (i, K) in enumerate(K_values)
            isnothing(results_kl[i]) && continue
            mx = maximum(results_kl[i].component_discrepancies)
            isfinite(mx) || continue
            if mx < bestmx; bestmx = mx; bestK = K; end
        end
        bestK
    end

    return (;
        K_range = K_values,
        models = models_bic,
        models_per_metric = (bic = models_bic, icl = models_icl, acdc_kl = models_kl),
        results = results_kl,
        results_kl,
        bic_values, icl_values,
        log_likelihoods,
        all_evals,
        selections = (bic = K_bic, icl = K_icl, acdc = K_acdc),
    )
end
