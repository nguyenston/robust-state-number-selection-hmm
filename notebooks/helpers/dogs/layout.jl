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
const _FIG_COL2  = 4200   # double-column (~7")
const _FIG_PANEL_H = 886  # linear-fit slope: per-row px contribution that gives plotarea_h ≈ 139.5 mm (matches K=1..10)
const _LEGEND_H   = 700   # legend row px contribution (bumped for thicker legend)
# h = _LEGEND_H + _FIG_PANEL_H × nrows
# Calibrated so plotarea_h stays constant across nrows (= K=1..10's 139.5 mm).
# This is NOT the actual rendered legend pixel height (Plots inflates the legend's
# allocated space to ~640 px regardless), it's the linear constant that, combined
# with _FIG_PANEL_H, produces the right total h for consistent cell rendering.
# Measured: at nrows=5, h=4722, plotarea=139.5 mm; at nrows=2, h=2066, plotarea=139.5 mm.

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
    _stack_with_legend(legend, panels; w=_FIG_COL2)

Assemble a stacked figure with a legend row on top.
Uses two columns for multiple panels to keep total height manageable.
"""
function _stack_with_legend(legend, panels; w=_FIG_COL2)
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
