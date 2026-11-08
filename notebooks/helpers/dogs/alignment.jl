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
"""Compute the existing task-profile Hungarian alignment across the full supplied K range."""
function align_bouthmm_components(bouthmm_run, bout_data, K_list)
    K_index = Dict(K => i for (i, K) in enumerate(bouthmm_run.K_range))

    # --- Build cross-K alignment from bout viterbi paths ---
    records = NamedTuple[]
    task_codes_uniq = sort(unique(bout_data.bout_task_codes))
    n_tasks = length(task_codes_uniq)

    for K in collect(K_list)
        idx = get(K_index, K, 0); idx == 0 && continue
        model = bouthmm_run.models[idx]; isnothing(model) && continue
        paths = viterbi(model, bout_data.sequences)

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

    return (; alignment_map, axis_count, axis_name)
end

function _build_run_for_metric_bout(model_run, models_for_metric)
    return (; K_range=model_run.K_range,
              models=models_for_metric,
              log_likelihoods=fill(NaN, length(model_run.K_range)),
              cache_status=fill(:loaded, length(model_run.K_range)),
              cache_dir="")
end


function _bouthmm_pair_counts(model_run, observations, sdict, alignment, K; window)
    K_index_bp = Dict(K => i for (i, K) in enumerate(model_run.K_range))
    beh_raw_mat_bp = sdict["Behavior"]["codes"]
    beh_cat_names_bp = extract_categories(sdict["Behavior"])
    bhmm_amap_bp = alignment.alignment_map
    bhmm_ac_bp = alignment.axis_count
    bhmm_an_bp = alignment.axis_name
    idx = get(K_index_bp, K, 0)
    idx == 0 && return nothing
    mdl = model_run.models[idx]
    isnothing(mdl) && return nothing

    st = viterbi(mdl, observations)
    n_samples = size(beh_raw_mat_bp, 1)
    m_eff = min(length(st), n_samples ÷ window)
    m_eff == 0 && return nothing

    beh_valid = sort(unique(c for c in beh_raw_mat_bp if c != 0))
    beh_labels = [1 <= c <= length(beh_cat_names_bp) ?
                  string(beh_cat_names_bp[c]) : "Code $c" for c in beh_valid]
    n_beh = length(beh_valid); n_with_none = n_beh + 1
    labels_full = vcat(beh_labels, ["(pure)"])
    beh_code_to_idx = Dict(c => j for (j, c) in enumerate(beh_valid))

    pair_counts = zeros(Float64, n_with_none, n_with_none, bhmm_ac_bp)
    for wi in 1:m_eff
        k_raw = st[wi]
        k_al = get(bhmm_amap_bp, (K, k_raw), 0)
        k_al == 0 && continue
        r_start = window * (wi - 1) + 1
        r_end = min(window * wi, n_samples)
        code_counts = Dict{Int,Int}()
        for si in r_start:r_end, col in 1:size(beh_raw_mat_bp, 2)
            c = beh_raw_mat_bp[si, col]
            c == 0 && continue
            code_counts[c] = get(code_counts, c, 0) + 1
        end
        present = collect(keys(code_counts))
        if isempty(present)
            pair_counts[n_with_none, n_with_none, k_al] += 1
            continue
        elseif length(present) == 1
            ai = beh_code_to_idx[present[1]]
            pair_counts[ai, n_with_none, k_al] += 1
            pair_counts[n_with_none, ai, k_al] += 1
            continue
        end
        for ci in present, cj in present
            ci >= cj && continue
            pair_counts[beh_code_to_idx[ci], beh_code_to_idx[cj], k_al] += 1
        end
    end
    return (; pair_counts, labels_full, n_with_none, axis_count=bhmm_ac_bp, axis_name=bhmm_an_bp)
end
