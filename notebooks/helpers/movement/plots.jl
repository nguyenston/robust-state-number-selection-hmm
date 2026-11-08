function _movement_selection_plot(res, hires)
    K_range = res.K_range
    K_bic = res.selections.bic
    K_icl = res.selections.icl
    p_bic = plot(
        collect(K_range), res.bic_values;
        marker = :circle, label = "BIC", linewidth = 2.4, markersize = 6,
        title  = "BIC / ICL", xlabel = "K", ylabel = "Score", color = :black,
        xticks = collect(minimum(K_range):maximum(K_range)),
        legend = :outerbottom, legend_columns = 2,
        hires...,
    )
    plot!(p_bic, collect(K_range), res.icl_values;
        marker = :diamond, label = "ICL", linewidth = 2.4, markersize = 6, color = :royalblue)
    vline!(p_bic, [K_bic]; linestyle = :dash,    color = :red,       label = "BIC K=$K_bic",  linewidth = 2.2)
    vline!(p_bic, [K_icl]; linestyle = :dashdot, color = :royalblue, label = "ICL K=$K_icl",  linewidth = 2.2)

    p_env = _acdc_envelope_panel(K_range, res.results, hires)
    p_selection = plot(p_bic, p_env; layout = (1, 2), size = (1300, 500), margin = 5Plots.mm)

    return p_selection
end

# =============================================================================
# Plotting helpers
# =============================================================================

_movement_plot_matrix(res) =
    Matrix(transpose(hasproperty(res.data, :obs_plot) ? res.data.obs_plot : res.data.obs))

# Build the ACDC envelope (max per-component KL discrepancy vs K) panel.
# Mirrors the panel produced in SpikeTrainHMMUtils._envelope_panel: per-K
# gray scatter for "best per K", black envelope line, and −xx% drop labels
# positioned above each envelope segment. Auto-expands ylim so labels stay
# within the subplot when y-range is narrow.
function _acdc_envelope_panel(K_range, results_vec, hires_kwargs)
    p = plot(; title = "ACDC Envelope (KL)", xlabel = "K",
        ylabel = "max per-component",
        legend = :outerbottom, legend_columns = 1,
        xticks = collect(minimum(K_range):maximum(K_range)),
        hires_kwargs...)
    vK = Int[]; vZ = Float64[]; vIdx = Int[]
    for (i, K) in enumerate(K_range)
        r = results_vec[i]
        isnothing(r) && continue
        ds = r.component_discrepancies
        all(isfinite, ds) || continue
        mx = maximum(ds)
        mx >= 0 || continue
        push!(vK, K); push!(vZ, mx); push!(vIdx, i)
    end
    isempty(vK) && return p
    node_colors = [palette(:auto)[i] for i in vIdx]
    plot!(p, vK, vZ; color = :gray, linewidth = 1.0, alpha = 0.5,
        label = "best per K")
    scatter!(p, vK, vZ; markersize = 5, markerstrokewidth = 0,
        markercolor = node_colors, markeralpha = 0.55, label = nothing)

    eK = Int[]; eZ = Float64[]; eC = []; bsf = Inf
    for (k, z, c) in zip(vK, vZ, node_colors)
        if z < bsf
            push!(eK, k); push!(eZ, z); push!(eC, c); bsf = z
        end
    end
    isempty(eK) && return p
    plot!(p, eK, eZ; color = :black, linewidth = 2.5, label = "envelope")
    scatter!(p, eK, eZ; markersize = 8, markerstrokewidth = 1.8,
        markerstrokecolor = :black, markercolor = eC, label = nothing)
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
        annotate!(p, xm, ym,
            text(@sprintf("−%.1f%%", drop_pct), 8, :black, :hcenter, :vcenter))
    end
    if !isempty(ann_ys)
        ymin_raw = min(minimum(vZ), minimum(ann_ys))
        ymax_raw = max(maximum(vZ), maximum(ann_ys))
        pad = max((ymax_raw - ymin_raw) * 0.12, eps(Float64))
        ymax_full = ymax_raw + pad
        ymin_full = max(0.0, ymin_raw - pad)
        plot!(p; ylim = (ymin_full, ymax_full))
    end
    return p
end

"""
    get_movement_plots(res, individual_id; figures_dir=nothing) -> NamedTuple

Generate the trajectory and model-selection figures used in the paper.

Returns `(trajectory, selection)`.
"""
function get_movement_plots(res, individual_id::String; figures_dir = nothing)
    lons, lats    = res.data.lons, res.data.lats
    K_bic   = res.selections.bic
    K_icl   = res.selections.icl
    K_range = res.K_range
    _plain_fmt(x) = (x == 0 ? "0" : abs(x) < 0.01 || abs(x) >= 1e6 ? @sprintf("%.0e", x) : @sprintf("%g", x))
    hires = (
        dpi = 300,
        tickfontsize = 10, guidefontsize = 12, legendfontsize = 10, titlefontsize = 13,
        fontfamily = "Helvetica",
        left_margin = 10Plots.mm, right_margin = 5Plots.mm,
        top_margin = 5Plots.mm, bottom_margin = 24Plots.mm,
        yformatter = _plain_fmt, xformatter = _plain_fmt,
    )

    # ── Trajectory ────────────────────────────────────────────────────────────
    p_traj = plot(
        lons, lats;
        line_z = 1:length(lons), c = :viridis,
        title  = "GPS Trajectory: $individual_id (Time-Colored)",
        xlabel = "Longitude", ylabel = "Latitude",
        aspect_ratio = :equal, legend = false, size = (1000, 800), hires...,
    )

    # ── BIC / ICL and ACDC envelope ─────────────────────────────────────────
    p_selection = _movement_selection_plot(res, hires)

    if !isnothing(figures_dir)
        mkpath(figures_dir)
        _save_png_pdf(p_traj,                 joinpath(figures_dir, "..", "..", "supplementary", "elephant_movement"), "movement_trajectory")
        _save_png_pdf(p_selection,            figures_dir, "movement_selection_base"; trim = true)
    end

    return (trajectory = p_traj, selection = p_selection)
end

"""
    get_movement_plots_alt(res, individual_id; figures_dir=nothing) -> NamedTuple

Generate the paper model-selection figure for the Gaussian alternative.

Returns `(selection,)`.
"""
function get_movement_plots_alt(res, individual_id::String; figures_dir = nothing)
    K_bic   = res.selections.bic
    K_icl   = res.selections.icl
    K_range = res.K_range
    _plain_fmt(x) = (x == 0 ? "0" : abs(x) < 0.01 || abs(x) >= 1e6 ? @sprintf("%.0e", x) : @sprintf("%g", x))
    hires = (
        dpi = 300,
        tickfontsize = 10, guidefontsize = 12, legendfontsize = 10, titlefontsize = 13,
        fontfamily = "Helvetica",
        left_margin = 10Plots.mm, right_margin = 5Plots.mm,
        top_margin = 5Plots.mm, bottom_margin = 24Plots.mm,
        yformatter = _plain_fmt, xformatter = _plain_fmt,
    )

    p_selection = _movement_selection_plot(res, hires)

    if !isnothing(figures_dir)
        mkpath(figures_dir)
        _save_png_pdf(p_selection,            figures_dir, "movement_selection_alt1"; trim = true)
    end

    return (selection = p_selection,)
end

"""
    get_movement_plots_alt2(res, individual_id; ...) -> NamedTuple

Generate the paper model-selection figure for the Cardioid alternative.
"""
function get_movement_plots_alt2(res, individual_id::String; figures_dir = nothing)
    K_bic   = res.selections.bic
    K_icl   = res.selections.icl
    K_range = res.K_range


    _plain_fmt(x) = (x == 0 ? "0" : abs(x) < 0.01 || abs(x) >= 1e6 ? @sprintf("%.0e", x) : @sprintf("%g", x))
    hires = (
        dpi = 300,
        tickfontsize = 10, guidefontsize = 12, legendfontsize = 10, titlefontsize = 13,
        fontfamily = "Helvetica",
        left_margin = 10Plots.mm, right_margin = 5Plots.mm,
        top_margin = 5Plots.mm, bottom_margin = 24Plots.mm,
        yformatter = _plain_fmt, xformatter = _plain_fmt,
    )

    p_selection = _movement_selection_plot(res, hires)

    if !isnothing(figures_dir)
        mkpath(figures_dir)
        _save_png_pdf(p_selection,            figures_dir, "movement_selection_alt2"; trim = true)
    end

    return (selection = p_selection,)
end

# =============================================================================
# State-feature grid
# =============================================================================

"""Return per-feature (lo, hi) limits with padding, derived from the observation matrix."""
function feature_xlims_from_res(res; pad_frac = 0.03)
    Y = _movement_plot_matrix(res)
    D = size(Y, 2)
    return [
        let lo = minimum(Y[:, d]), hi = maximum(Y[:, d]), pad = (hi - lo) * pad_frac
            (lo - pad, hi + pad)
        end
        for d in 1:D
    ]
end

"""Evaluate the marginal emission density for component k, feature d at points x."""
function component_feature_density(res, model, k::Int, d::Int, x::AbstractVector{<:Real})
    em = model.B[k]
    if hasproperty(em, :step) && hasproperty(em, :angle)
        d == 1 && return exp.(vec(StateSpaceDynamics.loglikelihood(em.step,  reshape(x, :, 1))))
        d == 2 && return exp.(vec(StateSpaceDynamics.loglikelihood(em.angle, reshape(x, :, 1))))
        return fill(NaN, length(x))
    end
    if hasproperty(em, :μ) && hasproperty(em, :Σ)
        return pdf.(Normal(em.μ[d], sqrt(max(em.Σ[d, d], 1e-12))), x)
    end
    return fill(NaN, length(x))
end

"""
    movement_state_feature_grid(res, K; ...) -> Plot

K × D grid of per-state feature histograms with emission density overlay.
"""
function movement_state_feature_grid(
    res, K;
    feature_names = nothing,
    feature_xlims = nothing,
    bins          = 40,
    color         = :steelblue,
    curve_color   = :crimson,
    left_margin   = 16Plots.mm,
    bottom_margin = 6Plots.mm,
    right_margin  = 4Plots.mm,
    top_margin    = 3Plots.mm,
    panel_w       = 360,
    panel_h       = 260,
)
    kidx  = findfirst(==(K), collect(res.K_range))
    isnothing(kidx) && error("K=$K not in K_range=$(res.K_range)")
    model = res.models[kidx]
    isnothing(model) && error("Model for K=$K is nothing")

    Y      = _movement_plot_matrix(res)
    states = _movement_states(res, model)
    D      = size(Y, 2)

    feat_names = if isnothing(feature_names)
        D == 2 && get(res, :variant, nothing) == :step_angle ? ["z_log_step", "turn_angle"] :
        D == 2 && get(res, :variant, nothing) == :gaussian_alt ? ["z_log_step", "turn_angle"] :
        D == 3 ? ["z_log_step", "sin(angle)", "cos(angle)"] :
        ["feature_$d" for d in 1:D]
    else
        feature_names
    end

    xlims_v = isnothing(feature_xlims) ? feature_xlims_from_res(res) : feature_xlims

    # Layout: split states into 2 columns when K >= 2
    n_state_cols = K >= 2 ? 2 : 1
    n_state_rows = cld(K, n_state_cols)
    grid_cols = D * n_state_cols
    grid_rows = n_state_rows

    p = plot(;
        layout = (grid_rows, grid_cols),
        size = (panel_w * grid_cols, panel_h * grid_rows),
        dpi = 300,
        left_margin = left_margin, right_margin = right_margin,
        top_margin = top_margin, bottom_margin = bottom_margin,
        tickfontsize = 11, guidefontsize = 13,
        titlefontsize = 14, yguidefontsize = 12,
    )

    # Blank out any subplot cells that won't be used by a state
    total_cells = grid_rows * grid_cols
    used = Set{Int}()
    for k in 1:K
        state_row = cld(k, n_state_cols)
        state_col = mod1(k, n_state_cols)
        for d in 1:D
            push!(used, (state_row - 1) * grid_cols + (state_col - 1) * D + d)
        end
    end
    for sp in 1:total_cells
        sp in used && continue
        plot!(p; subplot = sp, framestyle = :none, ticks = nothing,
            grid = false, legend = false, showaxis = false)
    end

    for k in 1:K
        idx = findall(==(k), states)
        # Position in 2-column state layout: row-major
        state_row = cld(k, n_state_cols)
        state_col = mod1(k, n_state_cols)
        for d in 1:D
            # Subplot index: (state_row - 1) * grid_cols + (state_col - 1) * D + d
            sp = (state_row - 1) * grid_cols + (state_col - 1) * D + d
            lo, hi = xlims_v[d]
            if isempty(idx)
                plot!(p; subplot = sp, legend = false, grid = false,
                    framestyle = :box, xlims = (lo, hi))
                annotate!(p, 0.5, 0.5, text("no data", 8, :gray); subplot = sp)
            else
                histogram!(p, Y[idx, d];
                    subplot = sp, bins = bins, normalize = true,
                    alpha = 0.75, color = color, label = nothing, xlims = (lo, hi))
                xg   = collect(range(lo, hi; length = 400))
                dens = component_feature_density(res, model, k, d, xg)
                any(isfinite, dens) && plot!(p, xg, dens;
                    subplot = sp, color = curve_color, linewidth = 1.9, label = nothing)
            end
            # Feature title only on the topmost row
            state_row == 1 && plot!(p; subplot = sp, title = feat_names[d])
            # State label only on the leftmost feature of each state
            d == 1 && plot!(p; subplot = sp, ylabel = "State $k")
            # x-label only on the bottommost row
            plot!(p; subplot = sp,
                xlabel = (state_row == n_state_rows ? "value" : ""), grid = false)
        end
    end
    return p
end


"""Export the existing main and supplementary state-feature grids for all three movement models."""
function export_movement_state_grids(analysis_results, analysis_results_alt, analysis_results_alt2; figures_dir)
    base_xlims   = feature_xlims_from_res(analysis_results)
    alt1_xlims   = feature_xlims_from_res(analysis_results_alt)
    alt2_xlims   = feature_xlims_from_res(analysis_results_alt2)

    for K_show in (2, 3, 4, 8)
        p_base = movement_state_feature_grid(
            analysis_results, K_show;
            feature_names = ["z_log_step", "turn_angle"],
            feature_xlims = base_xlims,
            color = :steelblue,
        )
        AnimalMovementUtils._save_png_pdf(p_base, joinpath(figures_dir, K_show == 3 ? "main" : "supplementary", "elephant_movement"), "movement_state_feature_grid_base_k$(K_show)")

        p_alt1 = movement_state_feature_grid(
            analysis_results_alt, K_show;
            feature_names = ["z_log_step", "turn_angle"],
            feature_xlims = alt1_xlims,
            color = :darkorange,
        )
        AnimalMovementUtils._save_png_pdf(p_alt1, joinpath(figures_dir, K_show == 3 ? "main" : "supplementary", "elephant_movement"), "movement_state_feature_grid_alt1_k$(K_show)")

        p_alt2 = movement_state_feature_grid(
            analysis_results_alt2, K_show;
            feature_names = ["z_log_step", "turn_angle"],
            feature_xlims = alt2_xlims,
            color = :seagreen,
        )
        AnimalMovementUtils._save_png_pdf(p_alt2, joinpath(figures_dir, K_show == 3 ? "main" : "supplementary", "elephant_movement"), "movement_state_feature_grid_alt2_k$(K_show)")
    end
    return nothing
end
