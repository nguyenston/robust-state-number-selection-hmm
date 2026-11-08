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
