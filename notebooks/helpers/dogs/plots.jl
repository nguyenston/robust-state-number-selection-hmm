_plain_fmt(x) = (x == 0 ? "0" : abs(x) < 0.01 || abs(x) >= 1e6 ?
    @sprintf("%.1e", x) : @sprintf("%g", x))
_selection_style() = (
    dpi=300, tickfontsize=10, guidefontsize=12, legendfontsize=10, titlefontsize=13,
    fontfamily="Helvetica",
    left_margin=10Plots.mm, right_margin=5Plots.mm,
    top_margin=5Plots.mm, bottom_margin=24Plots.mm,
    yformatter=_plain_fmt, xformatter=_plain_fmt,
)

"""Export the paper bout-HMM BIC/ICL and envelope panels."""
function plot_bouthmm_selection(res_bhmm; figures_dir)
    hires = _selection_style()
    Kr = res_bhmm.K_range
    Kbic = res_bhmm.selections.bic
    Kicl = res_bhmm.selections.icl

    p_b_bic = plot(; title="BIC / ICL", xticks=collect(minimum(Kr):maximum(Kr)),
        legend=:outerbottom, legend_columns=3,
        xlabel="K", ylabel="Score", hires...)
    for (i, K) in enumerate(Kr)
        evals = res_bhmm.all_evals[i]
        (isnothing(evals) || isempty(evals)) && continue
        xs = fill(K, length(evals))
        scatter!(p_b_bic, xs, [e.bic for e in evals];
            color=:black, markeralpha=0.25, markersize=4, markerstrokewidth=0,
            label=(i == 1 ? "BIC seeds" : nothing))
        scatter!(p_b_bic, xs, [e.icl for e in evals];
            color=:royalblue, markeralpha=0.25, markersize=4, markerstrokewidth=0,
            label=(i == 1 ? "ICL seeds" : nothing))
    end
    plot!(p_b_bic, collect(Kr), res_bhmm.bic_values;
        marker=:circle, label="BIC (best)", linewidth=2.2, markersize=6, color=:black)
    plot!(p_b_bic, collect(Kr), res_bhmm.icl_values;
        marker=:diamond, label="ICL (best)", linewidth=2.2, markersize=6, color=:royalblue)
    vline!(p_b_bic, [Kbic]; linestyle=:dash, color=:red, linewidth=2.0,
        label="BIC K=$Kbic")
    vline!(p_b_bic, [Kicl]; linestyle=:dashdot, color=:royalblue, linewidth=2.0,
        label="ICL K=$Kicl")

    p_b_env = plot(; title="ACDC Envelope (KL)",
        xlabel="K", ylabel="max per-component",
        legend=:outerbottom, legend_columns=2,
        xticks=collect(minimum(Kr):maximum(Kr)),
        hires...)
    for (i, K) in enumerate(Kr)
        evals = res_bhmm.all_evals[i]
        (isnothing(evals) || isempty(evals)) && continue
        vs = filter(v -> isfinite(v) && v >= 0, [e.max_kl for e in evals])
        isempty(vs) && continue
        scatter!(p_b_env, fill(K, length(vs)), vs;
            color=:gray, markeralpha=0.25, markersize=3, markerstrokewidth=0,
            label=(i == 1 ? "all seeds" : nothing))
    end
    vK, vZ, vIdx = Int[], Float64[], Int[]
    for (i, K) in enumerate(Kr)
        isnothing(res_bhmm.results_kl[i]) && continue
        mx = maximum(res_bhmm.results_kl[i].component_discrepancies)
        isfinite(mx) && mx >= 0 || continue
        push!(vK, K); push!(vZ, mx); push!(vIdx, i)
    end
    nc = [palette(:auto)[i] for i in vIdx]
    if !isempty(vK)
        plot!(p_b_env, vK, vZ; color=:gray, linewidth=1.0, alpha=0.5,
            label="best per K")
        scatter!(p_b_env, vK, vZ; markersize=5, markerstrokewidth=0,
            markercolor=nc, markeralpha=0.55, label=nothing)
        eK, eZ, eC = let
            kk = Int[]; zz = Float64[]; cc = Any[]; bsf = Inf
            for (k, z, c) in zip(vK, vZ, nc)
                if z < bsf; push!(kk, k); push!(zz, z); push!(cc, c); bsf = z; end
            end
            kk, zz, cc
        end
        plot!(p_b_env, eK, eZ; color=:black, linewidth=2.5, label="envelope")
        scatter!(p_b_env, eK, eZ; markersize=8, markerstrokewidth=1.8,
            markerstrokecolor=:black, markercolor=eC, label=nothing)
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
            annotate!(p_b_env, xm, ym,
                text(@sprintf("−%.1f%%", drop_pct), 8, :black, :hcenter, :vcenter))
        end
        # Expand ylim so labels stay inside the envelope panel — dog's tight
        # y-range otherwise spills annotations into adjacent subplots.
        if !isempty(ann_ys)
            ymin_raw = min(minimum(vZ), minimum(ann_ys))
            ymax_raw = max(maximum(vZ), maximum(ann_ys))
            pad = max((ymax_raw - ymin_raw) * 0.12, eps(Float64))
            ymax_full = ymax_raw + pad
            ymin_full = max(0.0, ymin_raw - pad)
            plot!(p_b_env; ylim=(ymin_full, ymax_full))
        end
    end

    p_b_sel = plot(p_b_bic, p_b_env;
        layout=(1, 2), size=(1400, 540), margin=5Plots.mm)
    _save_png_pdf(p_b_sel, figures_dir, "selection_bouthmm"; trim=true)
    println("Saved bout-HMM selection plot")
    return p_b_sel
end

function plot_bouthmm_pairs(model_run_for_metric, bout_data, observations, sdict, K_list;
                                    window, out_dir)
    mkpath(out_dir)
    alignment = align_bouthmm_components(model_run_for_metric, bout_data, K_list)
    bhmm_ac_bp = alignment.axis_count

    # Pair-distribution panels for the paper, K=1:10.
    # Use the same explicit label-shortening map as the original notebook so
    # text fits under the 45° rotated x-axis without truncation artifacts.
    function short_pair_label(label)
        replace(label,
            "Synchronization" => "Sync",
            "Lying chest"     => "Lying",
            "Carrying object" => "Carry obj",
            "Standing"        => "Stand",
            "Walking"         => "Walk",
            "Trotting"        => "Trot",
            "Sitting"         => "Sit",
            "Sniffing"        => "Sniff",
            "Playing"         => "Play",
            "Panting"         => "Pant",
            "Jumping"         => "Jump",
            "Shaking"         => "Shake",
            "Galloping"       => "Gallop",
            "Drinking"        => "Drink",
            "Eating"          => "Eat",
            "Bowing"          => "Bow",
            "Pacing"          => "Pace",
            "Tugging"         => "Tug",
            "(pure)"          => "pure",
        )
    end
    pair_comp_cols = [
        RGB(0.16, 0.38, 0.60), RGB(0.75, 0.30, 0.22), RGB(0.20, 0.55, 0.32),
        RGB(0.73, 0.58, 0.18), RGB(0.52, 0.33, 0.64), RGB(0.18, 0.62, 0.62),
        RGB(0.84, 0.47, 0.65), RGB(0.40, 0.40, 0.40),
        RGB(0.90, 0.60, 0.30), RGB(0.30, 0.40, 0.70),
        RGB(0.60, 0.20, 0.10), RGB(0.10, 0.30, 0.40),
        RGB(0.95, 0.85, 0.45), RGB(0.55, 0.75, 0.35),
    ][1:bhmm_ac_bp]
    pair_comp_labels = ["C$k" for k in 1:bhmm_ac_bp]

    pair_dist_panels = Any[]
    for K in K_list
        pair_data = _bouthmm_pair_counts(model_run_for_metric, observations, sdict, alignment, K; window=window)
        isnothing(pair_data) && continue
        pair_labels_local = String[]
        pair_pcts = Matrix{Float64}(undef, 0, bhmm_ac_bp)
        for ai in 1:pair_data.n_with_none, bi in (ai + 1):pair_data.n_with_none
            tot = sum(pair_data.pair_counts[ai, bi, :])
            tot < 50 && continue   # show only common pairs (reduce label crowding)
            la = short_pair_label(pair_data.labels_full[ai])
            lb = short_pair_label(pair_data.labels_full[bi])
            pair_label = (la == "Sync" && lb == "pure") ? "nothing" : "$(la) + $(lb)"
            push!(pair_labels_local, pair_label)
            pair_pcts = vcat(pair_pcts, (100.0 .* pair_data.pair_counts[ai, bi, :] ./ tot)')
        end
        isempty(pair_labels_local) && continue
        p = groupedbar(1:length(pair_labels_local), pair_pcts;
            bar_position=:stack, bar_width=0.85, alpha=1.0,
            linecolor=:transparent, linewidth=0, palette=pair_comp_cols,
            label=false, title="K=$K", xlabel="", ylabel="", ylim=(0, 100),
            yticks=0:20:100, xticks=(1:length(pair_labels_local), pair_labels_local),
            xrotation=45, grid=false, foreground_color_subplot=:black,
            background_color=:white, legend=false,
            left_margin=4Plots.mm, right_margin=4Plots.mm,
            bottom_margin=14Plots.mm,
            DogNotebookUtils._panel_style()...)
        push!(pair_dist_panels, p)
    end

    if !isempty(pair_dist_panels)
        p_legend = top_legend_row(pair_comp_labels, pair_comp_cols; nrows=2)
        rendered_K_pd = collect(K_list)[1:length(pair_dist_panels)]
        for (k_lo, k_hi) in ((1, 10),)
            sub_pd = findall(K -> k_lo <= K <= k_hi, rendered_K_pd)
            isempty(sub_pd) && continue
            fig = DogNotebookUtils._stack_with_legend(p_legend, pair_dist_panels[sub_pd])
            savefig(fig, joinpath(out_dir, "pair_distribution_k$(k_lo)_k$(k_hi).png"))
            savefig(fig, joinpath(out_dir, "pair_distribution_k$(k_lo)_k$(k_hi).pdf"))
        end
    end
end
