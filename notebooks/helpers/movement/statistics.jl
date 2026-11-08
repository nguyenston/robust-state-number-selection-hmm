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
