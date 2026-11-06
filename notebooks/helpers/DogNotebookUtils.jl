module DogNotebookUtils

using CSV
using DataFrames
using LinearAlgebra
using Random
using Serialization
using Statistics
using LaTeXStrings
using Plots
using StatsPlots
using Hungarian
using StateSpaceDynamics
using StateSpaceDynamics.ACDC

export load_dogmove_multi_csv, sensor_table_to_dict, extract_categories,
    extract_task_categories_and_codes, extract_behavior_categories_and_codes,
    extract_categories_and_codes, segment_task_bouts, window_task_code,
    compute_bouthmm_models_allruns, run_bouthmm_selection_analysis_allruns,
    component_label_percentage_from_windows, composition_panel_from_codes,
    label_to_component_panel_from_codes, build_bouthmm_behavior_stacks,
    top_legend_row

function _unwrap_celllike(x)
    if x isa Matrix{Any} && length(x) == 1
        return x[1]
    elseif x isa Vector{Any} && length(x) == 1
        return x[1]
    end
    return x
end

"""
    sensor_table_to_dict(sensor_tbl)

Normalize a dog sensor payload into a `Dict{String,Any}` and unwrap singleton
cell-like containers left over from earlier MATLAB-shaped data structures.
"""
function sensor_table_to_dict(sensor_tbl::Dict)
    return Dict(String(k) => _unwrap_celllike(v) for (k, v) in sensor_tbl)
end

function windowed_mean_std(mat::AbstractMatrix, window::Int)
    window <= 1 && return (mat, zeros(eltype(mat), size(mat)))
    n = size(mat, 1)
    m = n ÷ window
    μ = Matrix{eltype(mat)}(undef, m, size(mat, 2))
    σ = Matrix{eltype(mat)}(undef, m, size(mat, 2))
    for i in 1:m
        r = (window * (i - 1) + 1):(window * i)
        v = @view mat[r, :]
        μ[i, :] .= vec(mean(v; dims=1))
        σ[i, :] .= vec(std(v; dims=1))
    end
    return μ, σ
end


function _annotation_dict_from_codes(names::Vector{String}, codes::AbstractMatrix{<:Integer})
    return Dict{String,Any}(
        "categoryNames" => reshape(Any[n for n in names], :, 1),
        "codes" => Int.(codes),
        "isOrdinal" => false,
        "isProtected" => false,
    )
end

function _canonical_label_names(lookup_df::DataFrame, key::String)
    sub = filter(r -> r.label_key == key && r.code > 0, lookup_df)
    seen = Set{String}()
    out = String[]
    for lbl in String.(sub.label)
        if !(lbl in seen)
            push!(out, lbl)
            push!(seen, lbl)
        end
    end
    if key == "Task"
        out = replace.(out, r"^Task\s+" => "")
    elseif key == "Behavior"
        out = replace.(out, r"^Behavior\s+" => "")
    end
    return out
end

_normalize_label_str(x) = ismissing(x) ? "" : String(x)

function _build_observations(ABack, ANeck, GBack, GNeck, window)
    A = 0.5 .* (ABack .+ ANeck)
    G = 0.5 .* (GBack .+ GNeck)
    A_mean, A_std = windowed_mean_std(A, window)
    G_mean, G_std = windowed_mean_std(G, window)

    eps = 1e-6
    A_std = log.(A_std .+ eps)
    G_std = log.(G_std .+ eps)

    A_mean_z = (A_mean .- mean(A_mean; dims=1)) ./ std(A_mean; dims=1)
    A_std_z = (A_std .- mean(A_std; dims=1)) ./ std(A_std; dims=1)
    G_mean_z = (G_mean .- mean(G_mean; dims=1)) ./ std(G_mean; dims=1)
    G_std_z = (G_std .- mean(G_std; dims=1)) ./ std(G_std; dims=1)
    observations = Matrix(transpose(hcat(A_mean_z, A_std_z, G_mean_z, G_std_z)))

    return (; A_mean_z, A_std_z, G_mean_z, G_std_z, observations)
end

"""
    load_dogmove_multi_csv(raw_csv_path, lookup_csv_path; window=10, dog_selector=:all)

Load and concatenate the processed DogMove CSV exports across one or more dogs,
returning the combined annotations and windowed observation features used by the
multi-dog notebook.
"""
function load_dogmove_multi_csv(raw_csv_path, lookup_csv_path; window=10, dog_selector=:all)
    raw_df = CSV.read(raw_csv_path, DataFrame)
    lookup_df = CSV.read(lookup_csv_path, DataFrame)
    sort!(raw_df, [:trial_index, :t])

    all_dogs = sort(unique(Int.(raw_df.dog_id)))
    selected_dogs = if dog_selector == :all
        all_dogs
    elseif dog_selector isa Integer
        [Int(dog_selector)]
    else
        sort(unique(Int.(collect(dog_selector))))
    end

    dog_set = Set(selected_dogs)
    raw_sel = filter(r -> Int(r.dog_id) in dog_set, raw_df)
    raw_df = nothing  # free the full DataFrame
    nrow(raw_sel) == 0 && error("No rows selected for dog_selector=$(dog_selector)")

    trial_meta = unique(select(raw_sel, [:trial_index, :dog_id, :test_num]))
    sort!(trial_meta, :trial_index)

    # Only store trial metadata — skip per-trial sensor copies to save memory
    DogMoveData = DataFrame(
        DogID=Int.(trial_meta.dog_id),
        TestNum=Int.(trial_meta.test_num),
    )

    ABack = Matrix{Float64}(select(raw_sel, [:A_back_x, :A_back_y, :A_back_z]))
    ANeck = Matrix{Float64}(select(raw_sel, [:A_neck_x, :A_neck_y, :A_neck_z]))
    GBack = Matrix{Float64}(select(raw_sel, [:G_back_x, :G_back_y, :G_back_z]))
    GNeck = Matrix{Float64}(select(raw_sel, [:G_neck_x, :G_neck_y, :G_neck_z]))

    task_names = _canonical_label_names(lookup_df, "Task")
    behavior_names = _canonical_label_names(lookup_df, "Behavior")

    task_map = Dict{String,Int}("Task none" => 0)
    for (i, n) in enumerate(task_names)
        task_map["Task " * n] = i
    end
    behavior_map = Dict{String,Int}("Behavior none" => 0)
    for (i, n) in enumerate(behavior_names)
        behavior_map[n] = i
    end

    for lbl in unique(String.(_normalize_label_str.(raw_sel.task_label)))
        isempty(lbl) && continue
        if !haskey(task_map, lbl)
            push!(task_names, replace(lbl, r"^Task\s+" => ""))
            task_map[lbl] = length(task_names)
        end
    end
    for lbl in unique(vcat(
        String.(_normalize_label_str.(raw_sel.behavior_label_1)),
        String.(_normalize_label_str.(raw_sel.behavior_label_2)),
        String.(_normalize_label_str.(raw_sel.behavior_label_3)),
    ))
        isempty(lbl) && continue
        if lbl != "Behavior none" && !haskey(behavior_map, lbl)
            push!(behavior_names, lbl)
            behavior_map[lbl] = length(behavior_names)
        end
    end

    task_labels = String.(_normalize_label_str.(raw_sel.task_label))
    task_codes = reshape([get(task_map, lbl, 0) for lbl in task_labels], :, 1)
    b1 = String.(_normalize_label_str.(raw_sel.behavior_label_1))
    b2 = String.(_normalize_label_str.(raw_sel.behavior_label_2))
    b3 = String.(_normalize_label_str.(raw_sel.behavior_label_3))
    behavior_codes = hcat(
        [get(behavior_map, lbl, 0) for lbl in b1],
        [get(behavior_map, lbl, 0) for lbl in b2],
        [get(behavior_map, lbl, 0) for lbl in b3],
    )
    raw_sel = nothing  # free filtered DataFrame

    # Only store annotation dicts in sensor_tbl — skip raw sensor matrices to save memory
    sensor_tbl = Dict{String,Any}(
        "Task" => _annotation_dict_from_codes(task_names, task_codes),
        "Behavior" => _annotation_dict_from_codes(behavior_names, behavior_codes),
        "PointEvent" => Dict{String,Any}(
            "categoryNames" => Matrix{Any}(undef, 0, 0),
            "codes" => Matrix{Int}(undef, size(task_codes, 1), 0),
            "isOrdinal" => false,
            "isProtected" => false,
        ),
    )
    sdict = sensor_table_to_dict(sensor_tbl)
    feats = _build_observations(ABack, ANeck, GBack, GNeck, window)
    return merge(
        (; DogMoveData, idx_best=1, sensor_tbl, sdict, WINDOW=window, selected_dog_ids=selected_dogs, selected_trial_count=nrow(trial_meta)),
        feats,
    )
end

extract_categories(col) = col isa Dict ? get(col, "categoryNames", nothing) : nothing

function _annotation_dict(col)
    x = _unwrap_celllike(col)
    x isa Dict && return x
    error("Unsupported annotation object: $(typeof(col))")
end

function _annotation_codes(payload, ncat::Int)
    x = _unwrap_celllike(payload)
    if x isa AbstractMatrix{<:Number}
        if ncat > 0 && size(x, 2) == ncat
            return [argmax(@view x[i, :]) for i in 1:size(x, 1)]
        elseif ncat > 0 && size(x, 1) == ncat
            return [argmax(@view x[:, j]) for j in 1:size(x, 2)]
        elseif size(x, 2) > 1
            return Int.(round.(x))
        else
            return Int.(round.(vec(x)))
        end
    elseif x isa AbstractVector{<:Number}
        return Int.(round.(x))
    elseif x isa Number
        return [Int(round(x))]
    elseif x isa AbstractArray
        out = Int[]
        for y in x
            append!(out, _annotation_codes(y, ncat))
        end
        return out
    else
        return Int[]
    end
end

function _annotation_payload(d::Dict{String,Any})
    payload = nothing
    for k in ("data", "codes", "labels", "values", "value", "idx", "index", "id")
        if haskey(d, k)
            payload = d[k]
            break
        end
    end
    if payload isa Dict
        inner = payload
        for k in ("data", "codes", "labels", "values", "value", "idx", "index", "id")
            if haskey(inner, k)
                payload = inner[k]
                break
            end
        end
    end
    payload === nothing && error("Could not find annotation payload; keys=$(collect(keys(d)))")
    return payload
end

"""
    extract_task_categories_and_codes(col)

Decode task category names and integer task codes from a notebook annotation
payload.
"""
function extract_task_categories_and_codes(col)
    d = _annotation_dict(col)
    names = haskey(d, "categoryNames") ? vec(string.(d["categoryNames"])) : String[]
    payload = _unwrap_celllike(_annotation_payload(d))
    codes = if payload isa AbstractMatrix{<:Number}
        Int.(round.(payload[:, 1]))
    elseif payload isa AbstractVector{<:Number}
        Int.(round.(payload))
    else
        _annotation_codes(payload, length(names))
    end
    return names, codes
end

"""
    extract_behavior_categories_and_codes(col)

Decode behavior category names and integer behavior codes from a notebook
annotation payload.
"""
function extract_behavior_categories_and_codes(col)
    d = _annotation_dict(col)
    names = haskey(d, "categoryNames") ? vec(string.(d["categoryNames"])) : String[]
    payload = _annotation_payload(d)
    codes = _annotation_codes(payload, length(names))
    return names, codes
end

"""
    extract_categories_and_codes(col; label_key=nothing)

Dispatch to the task or behavior decoder based on `label_key`.
"""
function extract_categories_and_codes(col; label_key::Union{Nothing,String}=nothing)
    if label_key == "Task"
        return extract_task_categories_and_codes(col)
    else
        return extract_behavior_categories_and_codes(col)
    end
end

"""
    segment_task_bouts(observations, sensor_dict; window, min_windows=2)

Segment the observation matrix into per-bout sequences, one for each contiguous
block of a non-"none" task.  Returns a NamedTuple with `sequences`,
`bout_task_codes`, `bout_window_ranges`, and `task_names`.
"""
function segment_task_bouts(observations, sensor_dict; window, min_windows=2)
    task_names, raw_codes = extract_task_categories_and_codes(sensor_dict["Task"])
    n_windows = size(observations, 2)
    wt_codes = window_task_code(raw_codes, n_windows, window)

    sequences = Matrix{Float64}[]
    bout_task_codes = Int[]
    bout_window_ranges = UnitRange{Int}[]

    i = 1
    while i <= length(wt_codes)
        code = wt_codes[i]
        if code == 0
            i += 1
            continue
        end
        j = i
        while j <= length(wt_codes) && wt_codes[j] == code
            j += 1
        end
        if j - i >= min_windows
            push!(sequences, observations[:, i:j-1])
            push!(bout_task_codes, code)
            push!(bout_window_ranges, i:j-1)
        end
        i = j
    end
    return (; sequences, bout_task_codes, bout_window_ranges, task_names)
end

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

function component_label_percentage_from_windows(states::Vector{Int}, raw_codes, K::Int, label_names; window, include_none=true)
    n_windows = length(states)
    n_codes = raw_codes isa AbstractMatrix ? size(raw_codes, 1) : length(raw_codes)
    n = min(n_codes, window * n_windows)
    m = n ÷ window
    m_eff = min(m, n_windows)
    m_eff == 0 && return zeros(Float64, 0, K), String[]

    st = states[1:m_eff]
    lb = raw_codes isa AbstractMatrix ? raw_codes[1:(m_eff * window), :] : raw_codes[1:(m_eff * window)]
    lb_all = lb isa AbstractMatrix ? vec(lb) : lb
    valid = include_none ? sort(unique(lb_all)) : sort(unique([c for c in lb_all if c != 0]))
    counts = zeros(Float64, length(valid), K)

    for i in 1:m_eff
        k = st[i]
        r = (window * (i - 1) + 1):(window * i)
        w = lb isa AbstractMatrix ? vec(@view lb[r, :]) : @view lb[r]
        for (j, c) in enumerate(valid)
            counts[j, k] += count(==(c), w)
        end
    end

    pct = zeros(Float64, length(valid), K)
    for k in 1:K
        denom = sum(counts[:, k])
        denom > 0 && (pct[:, k] .= 100.0 .* counts[:, k] ./ denom)
    end

    labels = String[]
    for c in valid
        if c == 0
            push!(labels, "None")
        elseif 1 <= c <= length(label_names)
            push!(labels, string(label_names[c]))
        else
            push!(labels, "Code $c")
        end
    end
    return pct, labels
end

function _normalize_profile(v::AbstractVector{<:Real})
    s = sum(v)
    s <= 0 && return zeros(Float64, length(v))
    return collect(Float64.(v) ./ s)
end

function _cosine_distance(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})
    na = norm(a)
    nb = norm(b)
    if na == 0 || nb == 0
        return 1.0
    end
    return 1.0 - (dot(a, b) / (na * nb))
end

function _cluster_component_profiles_hungarian(records)
    byK = Dict{Int, Vector{Any}}()
    for rec in records
        get!(byK, rec.K, Any[])
        push!(byK[rec.K], rec)
    end
    clusters = Vector{Dict{Symbol, Any}}()
    for K in sort(collect(keys(byK)))
        recs = byK[K]
        assigned_rec = falses(length(recs))
        if !isempty(clusters)
            cost = Matrix{Union{Missing, Float64}}(undef, length(recs), length(clusters))
            for ri in eachindex(recs), ci in eachindex(clusters)
                cl = clusters[ci]
                cost[ri, ci] = K in cl[:Ks] ? missing : _cosine_distance(recs[ri].profile, cl[:centroid])
            end
            assignment, _ = hungarian(cost)
            for ri in eachindex(recs)
                ci = assignment[ri]
                if ci != 0
                    rec = recs[ri]
                    push!(clusters[ci][:members], rec)
                    push!(clusters[ci][:profiles], rec.profile)
                    push!(clusters[ci][:Ks], K)
                    clusters[ci][:centroid] = vec(mean(reduce(hcat, clusters[ci][:profiles]); dims=2))
                    assigned_rec[ri] = true
                end
            end
        end
        for ri in eachindex(recs)
            if !assigned_rec[ri]
                rec = recs[ri]
                push!(clusters, Dict(:members => Any[rec], :profiles => Any[rec.profile], :Ks => Set([K]), :centroid => copy(rec.profile)))
            end
        end
    end
    assignments = Dict{Tuple{Int, Int}, Int}()
    for (cid, cl) in enumerate(clusters), rec in cl[:members]
        assignments[(rec.K, rec.component)] = cid
    end
    return assignments, length(clusters)
end

window_task_code(raw_codes, n_windows, window; pad_to_n_windows=false) = begin
    n_codes = raw_codes isa AbstractMatrix ? size(raw_codes, 1) : length(raw_codes)
    n = min(n_codes, n_windows * window)
    m = n ÷ window
    out = pad_to_n_windows ? fill(0, n_windows) : Vector{Int}(undef, m)
    for i in 1:m
        r = (window * (i - 1) + 1):(window * i)
        w = raw_codes isa AbstractMatrix ? vec(@view raw_codes[r, :]) : @view raw_codes[r]
        vals = unique(w)
        cnts = [count(==(v), w) for v in vals]
        out[i] = vals[argmax(cnts)]
    end
    if pad_to_n_windows && m < n_windows && m > 0
        out[(m + 1):end] .= out[m]
    end
    out
end

"""
    composition_panel_from_codes(states, raw_codes, K, label_names; window, axis_count=K, axis_name="Component", alignment_map=nothing, title="", include_none=false)

Build a component → label stacked bar panel from explicit state assignments and
raw annotation codes, without needing a model or sensor_dict.
"""
function composition_panel_from_codes(states, raw_codes, K, label_names; window, axis_count=K, axis_name="Component", alignment_map=nothing, title="", include_none=false)
    pct, labels = component_label_percentage_from_windows(states, raw_codes, K, label_names; window=window, include_none=include_none)
    labels = [lbl == "None" ? "Label none" : lbl for lbl in labels]
    pct_full = zeros(Float64, size(pct, 1), axis_count)
    if isnothing(alignment_map)
        keep = collect(1:min(K, axis_count))
        pct_full[:, keep] .= pct[:, keep]
    else
        for c in 1:K
            cid = get(alignment_map, (K, c), 0)
            cid > 0 && cid <= axis_count && (pct_full[:, cid] .+= pct[:, c])
        end
    end
    pct_by_component = permutedims(pct_full, (2, 1))
    bar_cols = collect(palette(:tab20, max(length(labels), 1)))
    p = groupedbar(1:axis_count, pct_by_component; bar_position=:stack, bar_width=0.85, alpha=1.0, linecolor=:transparent, linewidth=0, palette=bar_cols, label=false, title=title, xlabel="", ylabel="", ylim=(0, 100), yticks=0:20:100, xlims=(0.5, axis_count + 0.5), xticks=1:axis_count, grid=false, foreground_color_subplot=:black, background_color=:white, legend=false, _panel_style()...)
    return p, labels, bar_cols
end

"""
    label_to_component_panel_from_codes(states, raw_codes, K, label_names; window, axis_count=K, axis_name="Component", alignment_map=nothing, title="", include_none=false)

Build a label → component stacked bar panel from explicit state assignments and
raw annotation codes.
"""
function label_to_component_panel_from_codes(states, raw_codes, K, label_names; window, axis_count=K, axis_name="Component", alignment_map=nothing, title="", include_none=false)
    n_windows = length(states)
    n_codes = raw_codes isa AbstractMatrix ? size(raw_codes, 1) : length(raw_codes)
    n = min(n_codes, window * n_windows)
    m = n ÷ window
    m_eff = min(m, n_windows)
    if m_eff == 0
        p = plot(; title=title, framestyle=:none, axis=false, background_color=:white)
        return p, String[], String[], Any[]
    end

    comps = states[1:m_eff]
    aligned = raw_codes isa AbstractMatrix ? raw_codes[1:(m_eff * window), :] : raw_codes[1:(m_eff * window)]
    a_all = aligned isa AbstractMatrix ? vec(aligned) : aligned
    valid = include_none ? sort(unique(a_all)) : sort(unique(c for c in a_all if c != 0))

    counts = zeros(Float64, length(valid), axis_count)
    for i in 1:m_eff
        comp = comps[i]
        comp_axis = isnothing(alignment_map) ? comp : get(alignment_map, (K, comp), 0)
        (comp_axis <= 0 || comp_axis > axis_count) && continue
        r = (window * (i - 1) + 1):(window * i)
        w = aligned isa AbstractMatrix ? vec(@view aligned[r, :]) : @view aligned[r]
        for (j, c) in enumerate(valid)
            counts[j, comp_axis] += count(==(c), w)
        end
    end

    pct = zeros(Float64, size(counts))
    for j in 1:size(counts, 1)
        denom = sum(counts[j, :])
        denom > 0 && (pct[j, :] .= 100.0 .* counts[j, :] ./ denom)
    end

    labels = String[]
    for c in valid
        if c == 0
            push!(labels, "None")
        elseif 1 <= c <= length(label_names)
            push!(labels, string(label_names[c]))
        else
            push!(labels, "Code $c")
        end
    end

    comp_labels = ["C$(k)" for k in 1:axis_count]
    comp_cols = collect(palette(:tab20, axis_count))
    p = groupedbar(1:length(labels), pct; bar_position=:stack, bar_width=0.85, alpha=1.0, linecolor=:transparent, linewidth=0, palette=comp_cols, label=false, title=title, xlabel="", ylabel="", ylim=(0, 100), yticks=0:20:100, xticks=(1:length(labels), labels), xrotation=20, grid=false, foreground_color_subplot=:black, background_color=:white, legend=false, _panel_style()...)
    return p, labels, comp_labels, comp_cols
end

"""
    build_bouthmm_behavior_stacks(bouthmm_run, bout_data, observations, sensor_dict, K_list; window)

Build behavior composition stacked figures for a bout-HMM, evaluated on both
the bout segments (bout-only) and the full concatenated timeline.
Returns four stacked figures plus alignment metadata.
"""
function build_bouthmm_behavior_stacks(bouthmm_run, bout_data, observations, sensor_dict, K_list; window)
    K_index = Dict(K => i for (i, K) in enumerate(bouthmm_run.K_range))
    behavior_names, behavior_raw = extract_behavior_categories_and_codes(sensor_dict["Behavior"])

    # --- Build cross-K alignment from bout viterbi paths ---
    records = NamedTuple[]
    bout_states_cache = Dict{Int,Vector{Vector{Int}}}()
    task_codes_uniq = sort(unique(bout_data.bout_task_codes))
    n_tasks = length(task_codes_uniq)

    for K in collect(K_list)
        idx = get(K_index, K, 0); idx == 0 && continue
        model = bouthmm_run.models[idx]; isnothing(model) && continue
        paths = viterbi(model, bout_data.sequences)
        bout_states_cache[K] = paths

        counts = zeros(Float64, n_tasks, K)
        for (bi, path) in enumerate(paths)
            tc = bout_data.bout_task_codes[bi]
            ti = findfirst(==(tc), task_codes_uniq)
            for k in path
                counts[ti, k] += 1
            end
        end
        contrib = zeros(Float64, size(counts))
        for t in 1:n_tasks
            denom_t = sum(counts[t, :])
            denom_t > 0 && (contrib[t, :] .= counts[t, :] ./ denom_t)
        end
        for c in 1:K
            rel_c = copy(contrib[:, c])
            denom_c = sum(rel_c)
            denom_c > 0 && (rel_c .= rel_c ./ denom_c)
            push!(records, (K=K, component=c, profile=_normalize_profile(rel_c)))
        end
    end

    alignment_map, axis_count = _cluster_component_profiles_hungarian(records)
    axis_name = "Aligned cluster"

    # --- Build panels for each K ---
    bout_comp_panels = Any[]
    bout_l2c_panels = Any[]
    full_comp_panels = Any[]
    full_l2c_panels = Any[]
    beh_labels_ref = String[]
    beh_cols_ref = Any[]
    comp_labels_ref = String[]
    comp_cols_ref = Any[]

    for K in collect(K_list)
        idx = get(K_index, K, 0); idx == 0 && continue
        model = bouthmm_run.models[idx]; isnothing(model) && continue

        # Bout-only: concatenate bout viterbi paths + matching behavior codes
        paths = get!(bout_states_cache, K) do; viterbi(model, bout_data.sequences); end
        bout_st = vcat(paths...)
        bout_beh = vcat([behavior_raw[((r.start-1)*window+1):min(r.stop*window, length(behavior_raw))] for r in bout_data.bout_window_ranges]...)

        p_bc, bl, bc = composition_panel_from_codes(bout_st, bout_beh, K, behavior_names; window=window, axis_count=axis_count, axis_name=axis_name, alignment_map=alignment_map, title="Bout behavior: K=$K")
        p_bl, _, cl, cc = label_to_component_panel_from_codes(bout_st, bout_beh, K, behavior_names; window=window, axis_count=axis_count, axis_name=axis_name, alignment_map=alignment_map, title="Bout behavior → comp: K=$K")

        # Full-timeline: single-seq viterbi on full observations
        st_full = viterbi(model, observations)
        p_fc, _, _ = composition_panel_from_codes(st_full, behavior_raw, K, behavior_names; window=window, axis_count=axis_count, axis_name=axis_name, alignment_map=alignment_map, title="Full behavior: K=$K")
        p_fl, _, _, _ = label_to_component_panel_from_codes(st_full, behavior_raw, K, behavior_names; window=window, axis_count=axis_count, axis_name=axis_name, alignment_map=alignment_map, title="Full behavior → comp: K=$K")

        push!(bout_comp_panels, p_bc)
        push!(bout_l2c_panels, p_bl)
        push!(full_comp_panels, p_fc)
        push!(full_l2c_panels, p_fl)
        beh_labels_ref = bl
        beh_cols_ref = bc
        comp_labels_ref = cl
        comp_cols_ref = cc
    end

    p_beh_legend = top_legend_row(beh_labels_ref, beh_cols_ref)
    p_comp_legend = top_legend_row(comp_labels_ref, comp_cols_ref)

    bout_behavior_composition = _stack_with_legend(p_beh_legend, bout_comp_panels)
    bout_behavior_to_component = _stack_with_legend(p_comp_legend, bout_l2c_panels)
    full_behavior_composition = _stack_with_legend(p_beh_legend, full_comp_panels)
    full_behavior_to_component = _stack_with_legend(p_comp_legend, full_l2c_panels)

    # The K-values associated with each panel (filters out K's whose model is missing).
    rendered_K = Int[K for K in collect(K_list)
                    if let idx = get(K_index, K, 0)
                        idx != 0 && !isnothing(bouthmm_run.models[idx])
                    end]

    return (; bout_behavior_composition, bout_behavior_to_component,
            full_behavior_composition, full_behavior_to_component,
            bout_comp_panels, bout_l2c_panels, full_comp_panels, full_l2c_panels,
            p_beh_legend, p_comp_legend, rendered_K,
            alignment_map, axis_count, axis_name)
end

"""
    top_legend_row(labels, cols; nrows=2)

Create the compact legend strip used above the stacked cross-`K` panels.
"""
function top_legend_row(labels::Vector{String}, cols; nrows::Int=2)
    if isempty(labels)
        return plot(; framestyle=:none, axis=false, size=(_FIG_COL2, _LEGEND_H), background_color=:white)
    end
    ncol = cld(length(labels), nrows)
    p = plot(; xlim=(0, ncol), ylim=(0, nrows), axis=false, framestyle=:none, size=(_FIG_COL2, _LEGEND_H + 30 * (nrows - 1)), background_color=:white)
    for (i, lab) in enumerate(labels)
        row = (i - 1) ÷ ncol
        col = (i - 1) % ncol
        x = col + 0.08
        y = nrows - row - 0.5
        x1 = x
        x2 = x + 0.16
        y1 = y - 0.11
        y2 = y + 0.11
        plot!(p, Shape([x1, x2, x2, x1], [y1, y1, y2, y2]); color=cols[i], linecolor=cols[i], label=nothing)
        annotate!(p, x + 0.22, y, text(lab, 31, :left, :black))
    end
    return p
end

# Paper-quality defaults: 600 dpi, readable fonts at print size.
# At final column width (~3.5–7"), these font sizes render at ~6–8pt.
_default_hires() = (
    dpi=600,
    tickfontsize=23,
    guidefontsize=30,
    legendfontsize=23,
    titlefontsize=30,
)

# Per-panel defaults for individual bar charts (embedded in stacked figures).
# Fonts cumulative 1.3× × 1.3× ≈ 1.69× over original (18/22/24 → 31/38/42).
# bottom_margin scaled to match the larger rotated xticks.
_panel_style() = (
    dpi=600,
    tickfontsize=31,
    guidefontsize=38,
    titlefontsize=42,
    bottom_margin=30Plots.mm,
    top_margin=10Plots.mm,
    left_margin=25Plots.mm,
)

# Standard widths at 600 dpi (inches → px).
# Most journals: single-col ≈ 3.3", 1.5-col ≈ 5", double-col ≈ 7".
const _FIG_COL1  = 2000   # single-column (~3.3")
const _FIG_COL15 = 3000   # 1.5-column  (~5")
const _FIG_COL2  = 4200   # double-column (~7")
const _FIG_PANEL_H = 886  # linear-fit slope: per-row px contribution that gives plotarea_h ≈ 139.5 mm (matches K=1..10)
const _LEGEND_H   = 700   # legend row px contribution (bumped for thicker legend)
# h = _LEGEND_H + _FIG_PANEL_H × nrows
# Calibrated so plotarea_h stays constant across nrows (= K=1..10's 139.5 mm).
# This is NOT the actual rendered legend pixel height (Plots inflates the legend's
# allocated space to ~640 px regardless), it's the linear constant that, combined
# with _FIG_PANEL_H, produces the right total h for consistent cell rendering.
# Measured: at nrows=5, h=4722, plotarea=139.5 mm; at nrows=2, h=2066, plotarea=139.5 mm.
const _STACK_MARGIN_OUTER = 6Plots.mm   # outer figure edge clearance only (was 34mm — was bleeding into per-subplot top/bottom margins and inflating inter-panel gaps)

# Reference subplot height used to override Plots' default per-cell padding
# calculation (which scales padding by actual subplot height). Pinning to the
# K=1..10 panel-row size makes padding nrows-independent.
const _REF_PADDING_HEIGHT = 3000   # plain pixel count; bigger than _FIG_PANEL_H so title+tick auto-padding gets enough vertical space

# Monkey-patch Plots' GRBackend min-padding for OUR stacked-bar panels only.
# The bouthmm composition and pair-distribution panels all use
# `groupedbar(...; bar_position=:stack, ...)`. For those subplots, we want
# padding to be independent of the actual subplot height so cells render
# consistently across figures with different nrows. For ALL OTHER subplots
# (BIC/ICL, ACDC, envelope, etc.), we fall back to the original Plots
# behavior so they aren't affected.
function _is_our_stacked_bar(sp::Plots.Subplot)
    for s in sp.series_list
        if get(s, :seriestype, nothing) == :bar &&
           get(s, :bar_position, nothing) == :stack
            return true
        end
    end
    return false
end

function Plots._update_min_padding!(sp::Plots.Subplot{Plots.GRBackend})
    if !_is_our_stacked_bar(sp)
        # Fall back to upstream Plots behavior for non-stacked-bar panels —
        # verbatim copy of gr.jl::_update_min_padding!(sp::Subplot{GRBackend}).
        dpi = sp.plt[:thickness_scaling]
        width, height = sp_size = Plots.get_size(sp)
        padding = (
            left = Ref(2Plots.mm + sp[:left_margin]),
            top = Ref(2Plots.mm + sp[:top_margin]),
            right = Ref(2Plots.mm + sp[:right_margin]),
            bottom = Ref(2Plots.mm + sp[:bottom_margin]),
        )
        if (title = sp[:title]) != ""
            Plots.gr_set_font(Plots.titlefont(sp), sp)
            l = last(Plots.gr_text_size(title))
            padding.top[] += 1Plots.mm + height * l * Plots.px
        end
        xaxis, yaxis, zaxis = axes = sp[:xaxis], sp[:yaxis], sp[:zaxis]
        xticks, yticks, zticks = Plots.get_ticks.(Ref(sp), axes)
        if Plots.gr_is3d(sp)
            m = 0Plots.mm
            for (ax, tc) in ((xaxis, xticks), (yaxis, yticks))
                isempty(first(tc)) && continue
                rot = ax[:rotation]
                Plots.gr_set_tickfont(sp, ax;
                    halign = (:left, :hcenter, :right)[sign(rot) + 2],
                    valign = ax[:mirror] ? :bottom : :top)
                l = 0.01 + last(Plots.gr_get_ticks_size(tc, rot))
                m = max(m, 1Plots.mm + height * l * Plots.px)
            end
            if m > 0Plots.mm
                (xaxis[:mirror] || yaxis[:mirror]) && (padding.top[] += m)
                (!xaxis[:mirror] || !yaxis[:mirror]) && (padding.bottom[] += m)
            end
        else
            for (ax, tc, (a, b)) in
                ((xaxis, xticks, (:top, :bottom)), (yaxis, yticks, (:right, :left)))
                if !isempty(first(tc))
                    isy = ax[:letter] === :y
                    Plots.gr_set_tickfont(sp, ax)
                    ts = Plots.gr_get_ticks_size(tc, ax[:rotation])
                    l = 0.01 + (isy ? first(ts) : last(ts))
                    padding[ax[:mirror] ? a : b][] += 1Plots.mm + sp_size[isy ? 1 : 2] * l * Plots.px
                end
                if (guide = Plots.get_guide(ax)) != ""
                    Plots.gr_set_font(Plots.guidefont(ax), sp)
                    l = last(Plots.gr_text_size(guide))
                    padding[Plots.mirrored(ax, a) ? a : b][] += 1Plots.mm + height * l * Plots.px
                end
            end
        end
        return sp.minpad = (
            dpi * padding.left[],
            dpi * padding.top[],
            dpi * padding.right[],
            dpi * padding.bottom[],
        )
    end

    # Patched path: stacked-bar panels get padding based on a fixed reference
    # subplot height, independent of actual nrows.
    dpi = sp.plt[:thickness_scaling]
    width = first(Plots.get_size(sp))
    height = _REF_PADDING_HEIGHT
    sp_size = (width, height)
    padding = (
        left = Ref(2Plots.mm + sp[:left_margin]),
        top = Ref(2Plots.mm + sp[:top_margin]),
        right = Ref(2Plots.mm + sp[:right_margin]),
        bottom = Ref(2Plots.mm + sp[:bottom_margin]),
    )
    if (title = sp[:title]) != ""
        Plots.gr_set_font(Plots.titlefont(sp), sp)
        l = last(Plots.gr_text_size(title))
        padding.top[] += 1Plots.mm + height * l * Plots.px
    end
    xaxis, yaxis, zaxis = axes = sp[:xaxis], sp[:yaxis], sp[:zaxis]
    xticks, yticks, zticks = Plots.get_ticks.(Ref(sp), axes)
    for (ax, tc, (a, b)) in
        ((xaxis, xticks, (:top, :bottom)), (yaxis, yticks, (:right, :left)))
        if !isempty(first(tc))
            isy = ax[:letter] === :y
            Plots.gr_set_tickfont(sp, ax)
            ts = Plots.gr_get_ticks_size(tc, ax[:rotation])
            l = 0.01 + (isy ? first(ts) : last(ts))
            padding[ax[:mirror] ? a : b][] += 1Plots.mm + sp_size[isy ? 1 : 2] * l * Plots.px
        end
        if (guide = Plots.get_guide(ax)) != ""
            Plots.gr_set_font(Plots.guidefont(ax), sp)
            l = last(Plots.gr_text_size(guide))
            padding[Plots.mirrored(ax, a) ? a : b][] += 1Plots.mm + height * l * Plots.px
        end
    end
    return sp.minpad = (
        dpi * padding.left[],
        dpi * padding.top[],
        dpi * padding.right[],
        dpi * padding.bottom[],
    )
end

"""
    _stack_with_legend(legend, panels; w=_FIG_COL2, force_2col=false)

Assemble a stacked figure with a legend row on top.
Uses 2-column grid when panels > 5 (or always when `force_2col=true`)
to keep total height manageable.
"""
function _stack_with_legend(legend, panels; w=_FIG_COL2, force_2col=false)
    np = length(panels)
    if np == 1
        # Trivial single-panel case
        h = _LEGEND_H + _FIG_PANEL_H
        leg_frac = _LEGEND_H / h
        hs = [leg_frac, 1.0 - leg_frac]
        return plot(legend, panels...; layout=(2, 1), size=(w, h), heights=hs)
    else
        # 2-col paired layout: pair adjacent panels into one 2-col subplot
        # per row, then stack rows with dynamic heights=[leg_frac, ...] so
        # the legend always gets _LEGEND_H pixels regardless of total
        # figure size. Per-panel margins set by `_panel_style()` are
        # preserved (no top/bottom_margin override here).
        padded = copy(panels)
        isodd(np) && push!(padded, plot(; framestyle=:none, axis=false, background_color=:white))
        nrows = cld(np, 2)
        paired = [plot(padded[2i-1], padded[2i]; layout=(1, 2)) for i in 1:nrows]
        h = _LEGEND_H + _FIG_PANEL_H * nrows
        leg_frac = _LEGEND_H / h
        hs = vcat([leg_frac], fill((1.0 - leg_frac) / nrows, nrows))
        return plot(legend, paired...; layout=grid(nrows + 1, 1, heights=hs), size=(w, h))
    end
end

end  # module DogNotebookUtils
