"""Write the existing selection, envelope, and behavior-composition summaries."""
function write_bouthmm_statistics(bout_stats_path, res_bhmm, bout_data, observations, sdict; window)
    mkpath(dirname(bout_stats_path))
    open(bout_stats_path, "w") do io
        println(io, "=" ^ 70)
        println(io, "DOG MULTI BOUT-HMM — PER-METRIC-BEST SELECTION")
        println(io, "Generated: $(Dates.now())")
        println(io, "=" ^ 70)
        println(io, "\n--- Data ---")
        println(io, "  $(length(bout_data.sequences)) task bouts (min 2 windows each)")
        println(io, "  Σ windows across bouts: $(sum(size(s,2) for s in bout_data.sequences))")
        println(io, "  D obs = $(size(bout_data.sequences[1], 1))")
        println(io, "  Task category breakdown:")
        for tc in sort(unique(bout_data.bout_task_codes))
            nm = bout_data.task_names[tc]
            cnt = count(==(tc), bout_data.bout_task_codes)
            @printf(io, "    %-15s : %d bouts\n", nm, cnt)
        end

        println(io, "\n--- Selections ---")
        println(io, "  K_range  = ", collect(res_bhmm.K_range))
        println(io, "  BIC K     = ", res_bhmm.selections.bic)
        println(io, "  ICL K     = ", res_bhmm.selections.icl)
        println(io, "  ACDC-KL K = ", res_bhmm.selections.acdc)

        println(io, "\n" * "─" ^ 70)
        println(io, "BIC / ICL / ACDC table (per-metric-best per K)")
        println(io, "─" ^ 70)
        @printf(io, "%-5s  %-12s  %-14s  %-14s  KL per component\n",
            "K", "LL(BIC-best)", "BIC", "ICL")
        for (i, K) in enumerate(res_bhmm.K_range)
            ll = res_bhmm.log_likelihoods[i]
            bic = res_bhmm.bic_values[i]
            icl = res_bhmm.icl_values[i]
            kl_str = isnothing(res_bhmm.results_kl[i]) ? "—" :
                join([@sprintf("%.4f", d) for d in res_bhmm.results_kl[i].component_discrepancies], "  ")
            sel = String[]
            K == res_bhmm.selections.bic  && push!(sel, "← BIC")
            K == res_bhmm.selections.icl  && push!(sel, "← ICL")
            K == res_bhmm.selections.acdc && push!(sel, "← ACDC(KL)")
            @printf(io, "K=%-3d  %-12s  %-14s  %-14s  KL[%s]  %s\n",
                K,
                isfinite(ll)  ? @sprintf("%.1f", ll)  : "—",
                isfinite(bic) ? @sprintf("%.1f", bic) : "—",
                isfinite(icl) ? @sprintf("%.1f", icl) : "—",
                kl_str, join(sel, " "))
        end

        println(io, "\n" * "─" ^ 70)
        println(io, "ACDC envelope — KL")
        println(io, "─" ^ 70)
        zeros_K = Tuple{Int,Float64}[]
        for (i, K) in enumerate(res_bhmm.K_range)
            isnothing(res_bhmm.results_kl[i]) && continue
            mx = maximum(res_bhmm.results_kl[i].component_discrepancies)
            isfinite(mx) && mx > 0 || continue
            push!(zeros_K, (K, mx))
        end
        env = Tuple{Int,Float64}[]; bsf = Inf
        for (K, z) in zeros_K
            if z < bsf; push!(env, (K, z)); bsf = z; end
        end
        println(io, "  Raw zero-points (per K):")
        for (K, z) in zeros_K
            @printf(io, "    %s K=%-3d  z = %.4f\n",
                any(e -> e[1] == K, env) ? "★" : " ", K, z)
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

        # Per-cluster behavior composition under both metric-best fits at the
        # ACDC-selected K and the BIC/ICL-selected K.
        K_show = sort(unique([res_bhmm.selections.acdc, res_bhmm.selections.bic]))
        sources = [("BIC-best",     res_bhmm.models_per_metric.bic),
                   ("ACDC-KL-best", res_bhmm.models_per_metric.acdc_kl)]
        beh_cat_names = extract_categories(sdict["Behavior"])
        beh_raw_mat = sdict["Behavior"]["codes"]

        for (sname, model_vec) in sources
            println(io, "\n" * "═" ^ 70)
            println(io, "[$sname] Behavior label → component composition (rows sum to 100%)")
            println(io, "═" ^ 70)
            for K in K_show
                idx = findfirst(==(K), res_bhmm.K_range)
                isnothing(idx) && continue
                mdl = model_vec[idx]
                isnothing(mdl) && continue
                st = viterbi(mdl, observations)
                n_smp = size(beh_raw_mat, 1)
                m_eff = min(length(st), n_smp ÷ window)
                beh_valid = sort(unique(c for c in beh_raw_mat if c != 0))
                counts = zeros(Float64, length(beh_valid), K)
                for j in 1:m_eff
                    comp = st[j]
                    r = (window * (j - 1) + 1):(window * j)
                    w = vec(@view beh_raw_mat[r, :])
                    for (li, c) in enumerate(beh_valid)
                        counts[li, comp] += count(==(c), w)
                    end
                end
                rowpct = zeros(Float64, length(beh_valid), K)
                for li in 1:length(beh_valid)
                    denom = sum(counts[li, :])
                    denom > 0 && (rowpct[li, :] .= 100.0 .* counts[li, :] ./ denom)
                end
                println(io, "\n  K=$K")
                comp_hdr = join([@sprintf("%6s", "C$k") for k in 1:K], " ")
                @printf(io, "    %-22s %s   total\n", "behavior", comp_hdr)
                for (li, c) in enumerate(beh_valid)
                    lname = 1 <= c <= length(beh_cat_names) ? string(beh_cat_names[c]) : "Code$c"
                    tot = sum(counts[li, :])
                    pct_strs = join([@sprintf("%5.1f%%", p) for p in rowpct[li, :]], " ")
                    @printf(io, "    %-22s %s   %6d windows\n", lname, pct_strs, round(Int, tot))
                end
            end
        end
    end
    println("Wrote bout-HMM stats to $bout_stats_path")
end

function _write_pair_stats(io, model_vec, model_run, bout_data, K_list, observations, sdict;
                            window, top_n=15,
                            alignment_K_list=nothing)
    sub_run = _build_run_for_metric_bout(model_run, model_vec)
    # Align across the same full K range used for the paper pair plots.
    align_K_list = isnothing(alignment_K_list) ? K_list : alignment_K_list
    alignment = align_bouthmm_components(sub_run, bout_data, align_K_list)
    ac = alignment.axis_count
    an = alignment.axis_name

    for K in K_list
        pd = _bouthmm_pair_counts(sub_run, observations, sdict, alignment, K; window=window); isnothing(pd) === true && continue
        println(io, "\n" * "─" ^ 70)
        println(io, "K=$K   pair statistics ($(an)-aligned, $(ac) clusters)")
        println(io, "─" ^ 70)

        # 1) Build flat list of (label_a, label_b, total_count, cluster_pcts)
        pair_rows = NamedTuple[]
        for ai in 1:pd.n_with_none, bi in (ai + 1):pd.n_with_none
            tot = sum(pd.pair_counts[ai, bi, :])
            tot < 50 && continue
            pcts = pd.pair_counts[ai, bi, :] ./ tot .* 100
            la = pd.labels_full[ai]; lb = pd.labels_full[bi]
            pair_label = la == "(pure)" ? "$lb (pure)" :
                          lb == "(pure)" ? "$la (pure)" : "$la + $lb"
            push!(pair_rows, (label=pair_label, total=Int(tot), pcts=pcts))
        end

        # 2) Per-cluster top-N pairs (sorted by pcts[c] descending)
        println(io, "\n  Per-cluster top $top_n pairs (by % of pair's bins in that cluster):")
        for c in 1:ac
            sorted = sort(pair_rows; by = r -> -r.pcts[c])
            println(io, "\n    $(an) $c:")
            shown = 0
            for r in sorted
                r.pcts[c] < 5.0 && break
                shown += 1; shown > top_n && break
                @printf(io, "      %5.1f%%  %-40s  (%d windows total)\n",
                    r.pcts[c], r.label, r.total)
            end
            shown == 0 && println(io, "      (none above 5%)")
        end

        # 3) Top-N most-frequent pairs overall (with their dominant cluster)
        println(io, "\n  Top $(top_n) most frequent pairs overall:")
        top_overall = sort(pair_rows; by = r -> -r.total)
        for r in first(top_overall, top_n)
            top_c = argmax(r.pcts)
            cluster_str = "$(an) $top_c"
            @printf(io, "    %6d windows  %-40s  ->  %-12s  (%5.1f%%)\n",
                r.total, r.label, cluster_str, r.pcts[top_c])
        end
    end
end


"""Write behavior-pair summaries with the same alignment as the paper panels."""
function write_bouthmm_pair_statistics(pair_stats_path, res_bhmm, bouthmm_run, bout_data, observations, sdict; window, alignment_K_list)
    mkpath(dirname(pair_stats_path))
    open(pair_stats_path, "w") do io
        println(io, "=" ^ 70)
        println(io, "DOG MULTI BOUT-HMM — BEHAVIOR PAIR STATISTICS")
        println(io, "Generated: $(Dates.now())")
        println(io, "=" ^ 70)
        println(io, "\n  Pairs: ordered combinations of two behaviors co-occurring")
        println(io, "  within the same window (or one behavior + (pure) marker for")
        println(io, "  windows where exactly one behavior is annotated).")
        println(io, "  Filter: only pairs with > 50 windows shown.")

        # Use the full paper alignment range, but report only the selected K values.
        K_show = sort(unique([res_bhmm.selections.acdc, res_bhmm.selections.bic, 4, 5, 7]))
        sources = [("BIC-best",     res_bhmm.models_per_metric.bic),
                   ("ACDC-KL-best", res_bhmm.models_per_metric.acdc_kl)]
        for (sname, model_vec) in sources
            println(io, "\n" * "═" ^ 70)
            println(io, "[$sname] fits — pair statistics at K=$(K_show)")
            println(io, "═" ^ 70)
            _write_pair_stats(io, model_vec, bouthmm_run, bout_data, K_show, observations, sdict;
                window=window, alignment_K_list=alignment_K_list)
        end
    end
    println("Wrote pair stats to $pair_stats_path")
end
