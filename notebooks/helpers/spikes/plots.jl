# =============================================================================
# Plots
# =============================================================================

function get_spike_plots(res, data; figures_dir = nothing)
    K_range = res.K_range
    K_bic   = res.selections.bic
    K_icl   = res.selections.icl
    K_true  = data.ground_truth.K_true
    label   = String(res.variant)

    _plain_fmt(x) = (x == 0 ? "0" : abs(x) < 0.01 || abs(x) >= 1e6 ? @sprintf("%.0e", x) : @sprintf("%g", x))
    hires = (
        dpi = 300,
        tickfontsize = 10, guidefontsize = 12, legendfontsize = 10, titlefontsize = 13,
        fontfamily = "Helvetica",
        left_margin = 10Plots.mm, right_margin = 5Plots.mm,
        top_margin = 5Plots.mm, bottom_margin = 24Plots.mm,
        yformatter = _plain_fmt, xformatter = _plain_fmt,
    )

    p_bic = plot(; title = "BIC / ICL",
        xlabel = "K", ylabel = "Score",
        xticks = collect(minimum(K_range):maximum(K_range)),
        legend = :outerbottom, legend_columns = 4,
        hires...)
    if hasproperty(res, :all_runs)
        for (i, K) in enumerate(K_range)
            evals = res.all_runs[i]
            (isnothing(evals) || isempty(evals)) && continue
            xs = fill(K, length(evals))
            scatter!(p_bic, xs, [e.bic for e in evals];
                color = :black, markeralpha = 0.25, markersize = 4,
                markerstrokewidth = 0, label = (i == 1 ? "BIC seeds" : nothing))
            scatter!(p_bic, xs, [e.icl for e in evals];
                color = :royalblue, markeralpha = 0.25, markersize = 4,
                markerstrokewidth = 0, label = (i == 1 ? "ICL seeds" : nothing))
        end
    end
    plot!(p_bic, collect(K_range), res.bic_values;
        marker = :circle, label = "BIC (best)", linewidth = 2.2, markersize = 6, color = :black)
    plot!(p_bic, collect(K_range), res.icl_values;
        marker = :diamond, label = "ICL (best)", linewidth = 2.2, markersize = 6, color = :royalblue)
    vline!(p_bic, [K_true]; linestyle = :dot, color = :seagreen,
        linewidth = 2.5, label = "GT K=$K_true")
    vline!(p_bic, [K_bic]; linestyle = :dash, color = :red,
        label = "BIC K=$K_bic", linewidth = 2.0)
    vline!(p_bic, [K_icl]; linestyle = :dashdot, color = :royalblue,
        label = "ICL K=$K_icl", linewidth = 2.0)

    function _envelope_panel(results_vec, title_str; metric_field::Symbol)
        valid_K     = Int[]
        valid_zeros = Float64[]
        valid_idx   = Int[]
        for (i, K) in enumerate(K_range)
            r = results_vec[i]
            isnothing(r) && continue
            ds = r.component_discrepancies
            all(isfinite, ds) || continue
            mx = maximum(ds)
            mx >= 0 || continue
            push!(valid_K, K); push!(valid_zeros, mx); push!(valid_idx, i)
        end
        p = plot(; title = title_str, xlabel = "K",
            ylabel = "max per-component",
            legend = :outerbottom, legend_columns = 2,
            xticks = collect(minimum(K_range):maximum(K_range)),
            hires...)
        isempty(valid_K) && return p
        node_colors = [palette(:auto)[i] for i in valid_idx]

        if hasproperty(res, :all_runs)
            for (i, K) in enumerate(K_range)
                evals = res.all_runs[i]
                (isnothing(evals) || isempty(evals)) && continue
                vals = [getproperty(e, metric_field) for e in evals]
                vals = filter(v -> isfinite(v) && v >= 0, vals)
                isempty(vals) && continue
                scatter!(p, fill(K, length(vals)), vals;
                    color = :gray, markeralpha = 0.25, markersize = 3,
                    markerstrokewidth = 0,
                    label = (i == 1 ? "all seeds" : nothing))
            end
        end

        plot!(p, valid_K, valid_zeros;
            color = :gray, linewidth = 1.0, alpha = 0.5,
            label = "best per K", primary = true)
        scatter!(p, valid_K, valid_zeros;
            markersize = 5, markerstrokewidth = 0, markercolor = node_colors,
            markeralpha = 0.55, label = nothing)

        env_K = Int[]; env_z = Float64[]; env_colors = []
        best_so_far = Inf
        for (k, z, c) in zip(valid_K, valid_zeros, node_colors)
            if z < best_so_far
                push!(env_K, k); push!(env_z, z); push!(env_colors, c)
                best_so_far = z
            end
        end
        if !isempty(env_K)
            plot!(p, env_K, env_z;
                color = :black, linewidth = 2.5, label = "envelope")
            scatter!(p, env_K, env_z;
                markersize = 8, markerstrokewidth = 1.8,
                markerstrokecolor = :black, markercolor = env_colors, label = nothing)
            # Perpendicular offset in normalized panel-fraction coords so the
            # label sits visually perpendicular to the segment.
            ann_ys = Float64[]
            k_rng    = max(float(maximum(env_K) - minimum(env_K)), 1.0)
            z_rng    = max(maximum(env_z) - minimum(env_z), eps(Float64))
            for j in 1:(length(env_K) - 1)
                k0, k1 = env_K[j], env_K[j+1]
                z0, z1 = env_z[j], env_z[j+1]
                drop_pct = iszero(z0) ? 0.0 : (z0 - z1) / z0 * 100
                dk_n    = (k1 - k0) / k_rng
                dz_n    = (z1 - z0) / z_rng
                mag_n   = sqrt(dk_n^2 + dz_n^2)
                iszero(mag_n) && continue
                perp_x_n = -dz_n / mag_n   # positive for descending env
                perp_y_n =  dk_n / mag_n
                off_n = 0.045
                xm = (k0 + k1) / 2 + perp_x_n * off_n * k_rng
                ym = (z0 + z1) / 2 + perp_y_n * off_n * z_rng
                push!(ann_ys, ym)
                annotate!(p, xm, ym,
                    text(@sprintf("−%.1f%%", drop_pct), 8, :black, :hcenter, :vcenter))
            end
            if !isempty(ann_ys)
                ymin_raw = min(minimum(valid_zeros), minimum(ann_ys))
                ymax_raw = max(maximum(valid_zeros), maximum(ann_ys))
                pad = max((ymax_raw - ymin_raw) * 0.12, eps(Float64))
                plot!(p; ylim = (max(0.0, ymin_raw - pad), ymax_raw + pad))
            end
        end
        vline!(p, [K_true]; linestyle = :dot, color = :seagreen,
            linewidth = 2.0, label = "GT K=$K_true")
        return p
    end

    p_env_kl = _envelope_panel(res.results_kl, "ACDC Envelope (KL)"; metric_field = :max_kl)

    p_bic_kl = plot(p_bic, p_env_kl; layout = (1, 2), size = (1300, 500), margin = 5Plots.mm)
    if !isnothing(figures_dir)
        mkpath(figures_dir)
        _save_png_pdf(p_bic_kl, figures_dir, "spike_selection_$(label)_kl"; trim = true)
    end
    return (kl = p_bic_kl,)
end

# =============================================================================
# Composition panels — Viterbi-decoded fitted state vs GT state
# =============================================================================

"""
    _component_total_rate(model, k) -> Float64

Return the sum of per-neuron rates for component k. Used to sort components
into a stable visual order (low rate → silent, high rate → bursty).
"""
function _component_total_rate(model, k)
    em = model.B[k]
    return sum(em.λ)
end

"""
    _composition_panel_gt_to_component(states, gt_states, K, label_names; title="", comp_colors)

Stacked bar: x-axis = GT state, each bar shows the component composition
(percent of bins of that GT state assigned to each fitted component).
"""
function _composition_panel_gt_to_component(states, gt_states, K, label_names; title="", comp_colors)
    K_true = length(label_names)
    counts = zeros(Float64, K, K_true)
    for (s, gt) in zip(states, gt_states)
        s in 1:K || continue
        gt in 1:K_true || continue
        counts[s, gt] += 1
    end
    pct = zeros(Float64, K, K_true)
    for g in 1:K_true
        denom = sum(counts[:, g])
        denom > 0 && (pct[:, g] .= 100.0 .* counts[:, g] ./ denom)
    end
    pct_T = permutedims(pct, (2, 1))
    # Abbreviated labels keep tick text short enough to render unrotated.
    short_labels = map(label_names) do s
        s == "low_a"   ? "LA"   :
        s == "low_b"   ? "LB"   :
        s == "silent"  ? "SIL"  :
        s == "down"    ? "DOWN" :
        s == "chaotic" ? "CHA"  :
        s == "up"      ? "UP"   :
        s == "burst_a" ? "BA"   :
        s == "burst_b" ? "BB"   :
        uppercase(first(s, 4))
    end
    p = groupedbar(1:K_true, pct_T;
        bar_position = :stack, bar_width = 0.85,
        linecolor = :transparent, linewidth = 0,
        palette = comp_colors,
        label = false, title = title,
        xlabel = "Ground-truth state", ylabel = "% of bins",
        ylim = (0, 100), yticks = 0:20:100,
        xticks = (1:K_true, short_labels), xrotation = 0,
        grid = false, legend = false,
        foreground_color_subplot = :black, background_color = :white,
        titlefontsize = 13, tickfontsize = 11, guidefontsize = 12,
        left_margin = 12Plots.mm, right_margin = 4Plots.mm,
        top_margin = 4Plots.mm, bottom_margin = 12Plots.mm)
    return p
end

"""
    plot_spike_composition_grid(res, data; K_list=2:9, figures_dir=nothing, label="")

Build the ground-truth-to-component composition panels used by the paper
supplement, with one panel per K in `K_list`.

Components are reordered within each K by total firing rate (lowest first)
so visual position roughly tracks the SILENT → DOWN → UP → BURST progression.
"""
function plot_spike_composition_grid(res, data; K_list = 2:9,
                                       figures_dir::Union{String,Nothing} = nothing,
                                       label::Union{String,Nothing} = nothing,
                                       model_source::Symbol = :bic)
    K_range  = res.K_range
    gt       = data.ground_truth
    K_true   = gt.K_true
    labels   = String.(gt.state_labels)
    obs      = data.observations
    gt_st    = data.states

    K_keep = collect(K_list)
    K_keep = filter(k -> k in K_range, K_keep)
    isempty(K_keep) && error("K_list has no overlap with res.K_range")

    # Pick which per-metric-best fit to visualise. Default :bic preserves the
    # back-compat reading of res.models. Other options use res.models_per_metric.
    model_vec = if model_source == :bic && !hasproperty(res, :models_per_metric)
        res.models
    else
        @assert hasproperty(res, :models_per_metric) "model_source ≠ :bic requires models_per_metric on res"
        getfield(res.models_per_metric, model_source)
    end

    K_max = maximum(K_keep)
    comp_colors = palette(:tab10, max(K_max, K_true))

    bot_panels = Any[]
    for K in K_keep
        idx = findfirst(==(K), K_range)
        model = model_vec[idx]
        if isnothing(model)
            push!(bot_panels, plot(; framestyle = :none, title = "K=$K (no model)"))
            continue
        end
        states = viterbi(model, obs)

        # Sort components by total firing rate so the leftmost component is the
        # quietest one and the rightmost is burstiest.
        rates = [_component_total_rate(model, k) for k in 1:K]
        perm  = sortperm(rates)
        inv_perm = Dict(perm[k] => k for k in 1:K)
        states_sorted = [inv_perm[s] for s in states]

        title_str = "K=$K" * (K == K_true ? " (GT)" : "")
        push!(bot_panels, _composition_panel_gt_to_component(states_sorted, gt_st, K, labels;
            title = title_str, comp_colors = comp_colors))
    end

    function _legend_row(names, cols; nrows::Int = 2)
        ntotal = length(names)
        ncol_leg = cld(ntotal, nrows)
        p = plot(; xlim = (0, ncol_leg), ylim = (0, nrows),
            framestyle = :none, axis = false, ticks = nothing,
            background_color = :white, legend = false,
            top_margin = 4Plots.mm, bottom_margin = 4Plots.mm,
            left_margin = 6Plots.mm, right_margin = 6Plots.mm)
        for (i, (name, c)) in enumerate(zip(names, cols))
            row = (i - 1) ÷ ncol_leg
            col = (i - 1) % ncol_leg
            x  = col + 0.05
            y  = nrows - row - 0.5
            x1 = x;        x2 = x + 0.20
            y1 = y - 0.18; y2 = y + 0.18
            plot!(p, Shape([x1, x2, x2, x1], [y1, y1, y2, y2]);
                color = c, linecolor = c, label = nothing)
            annotate!(p, x + 0.27, y, text(name, 14, :left, :black))
        end
        return p
    end

    n = length(K_keep)
    legend_names = ["C$(k)" for k in 1:maximum(K_keep)]
    legend_row = _legend_row(legend_names,
        [comp_colors[k] for k in 1:maximum(K_keep)])

    # 3×3 grid for K=2..9 (8 panels + 1 empty cell). Wider panels and squarer
    # overall figure than 2×4.
    ncol  = 3
    nrows = cld(n, ncol)
    padded = copy(bot_panels)
    while length(padded) < ncol * nrows
        push!(padded, plot(; framestyle = :none, axis = false, background_color = :white))
    end

    panel_grid = plot(padded...; layout = (nrows, ncol))

    row_h = 360
    legend_h = 130   # 2 rows of swatches need more vertical room
    total_h = legend_h + row_h * nrows
    leg_frac = legend_h / total_h

    full = plot(legend_row, panel_grid;
        layout = grid(2, 1; heights = [leg_frac, 1.0 - leg_frac]),
        size = (560 * ncol, total_h),
        background_color = :white,
        left_margin   = 6Plots.mm,
        right_margin  = 6Plots.mm,
        top_margin    = 6Plots.mm,
        bottom_margin = 10Plots.mm)

    if !isnothing(figures_dir)
        mkpath(figures_dir)
        suffix = isnothing(label) ? "" : "_" * label
        _save_png_pdf(full, figures_dir, "spike_composition" * suffix * "_$(model_source)")
    end
    return full
end
