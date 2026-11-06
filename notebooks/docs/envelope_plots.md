# ACDC Envelope Plots: Problem, Design, and Reproducible Construction

This guide explains how to construct and interpret the **ACDC Envelope (KL)**
plots used by the example notebooks. It covers the modeling problem, every
visual layer, axis scales, legend placement, the envelope algorithm, edge
cases, and complete Julia example code.

Current implementations are in:

- [`AnimalMovementUtils.jl::_acdc_envelope_panel`](../helpers/AnimalMovementUtils.jl);
- [`SpikeTrainHMMUtils.jl::_envelope_panel`](../helpers/SpikeTrainHMMUtils.jl); and
- [`run_dog_bouthmm.jl`](../run_dog_bouthmm.jl),
  where the dog Bout-HMM version is written inline.

For the component-level KL calculation, see
[`kl_discrepancy.md`](kl_discrepancy.md).

---

## 1. Problem description

Suppose one model family is fitted with several candidate numbers of latent
states or mixture components:

```math
K \in \{K_{\min},\ldots,K_{\max}\}.
```

For every candidate `K`, ACDC measures the discrepancy of each fitted
component from its expected reference behavior. In these examples the
component statistic is a KL divergence. Define

```math
z_K = \max_{1 \le j \le K} D_{\mathrm{KL},j}(K).
```

Thus, `z_K` is the discrepancy of the worst component at model size `K`.
Large decreases as `K` increases suggest that the added component resolved
structure that a smaller model could not represent. Small decreases suggest
diminishing returns. The useful model size is often near the last large drop
before a low-improvement tail.

Two complications motivate the envelope:

1. EM fits at different values of `K` are independent. A larger model can
   land in a worse local optimum, so the observed `z_K` curve can rise.
2. Multiple restarts at one `K` can produce different values. The plot should
   show this instability without allowing it to obscure the selected result.

The plot therefore shows the restart distribution and best-per-`K` path in
the background, then emphasizes only new cumulative record lows.

---

## 2. Mathematical definition of the envelope

Process candidates in increasing `K`. Retain `(K,z_K)` only when `z_K`
is strictly smaller than every earlier value:

```math
(K,z_K)\text{ is on the envelope}
\quad\Longleftrightarrow\quad
z_K < \min_{K' < K} z_{K'}.
```

The first valid point is retained automatically. Connecting retained points
produces the black monotone envelope.

This is a **cumulative-best or record-low curve**. It is not a convex hull,
smoothed fit, regression line, or rolling mean. A skipped `K` means that its
worst-component discrepancy did not improve on the previous record.

Between consecutive envelope points `(K_0,z_0)` and `(K_1,z_1)`, label

```math
\operatorname{drop}(\%) = 100\frac{z_0-z_1}{z_0}.
```

`K_1` need not equal `K_0+1`. The percentage is relative to the previous
**envelope point**, not necessarily the immediately preceding candidate.


---

## 3. Detailed plot specification

Draw the layers from background to foreground:

| Layer | Appearance | Purpose |
|---|---|---|
| All restarts, if available | Gray dots, size 3, alpha 0.25, no border | Shows optimization variability at each `K` |
| Best-per-`K` path | Gray line, width 1.0, alpha 0.5 | Shows every selected value, including upward bumps |
| Best-per-`K` nodes | Per-`K` colored dots, size 5, alpha 0.55, no border | Links `K` colors to the companion loss plot |
| Envelope path | Solid black line, width 2.5 | Emphasizes cumulative record lows |
| Envelope nodes | Per-`K` colored dots, size 8, black border width 1.8 | Identifies candidates that establish new records |
| Drop labels | Black `−X.X%` text, size 8 | Quantifies each envelope improvement |
| Ground-truth line, optional | Seagreen dotted vertical line, width 2.0 | Marks known `K_true` in synthetic examples |

The marker hierarchy is intentional: restart points are smallest, selected
points are medium-sized, and envelope points are largest and outlined.

### 3.1 Axes and scales

The current notebook code uses:

| Property | Setting | Rationale |
|---|---|---|
| Title | `"ACDC Envelope (KL)"` | Names the diagnostic and discrepancy family |
| x label | `"K"` | Number of states/components |
| x scale | **Linear** (`:identity`) | One tick step means one added component |
| x ticks | Every integer in the candidate range | Fractional state counts are meaningless |
| y label | `"max per-component"` | Displays `maximum(component_discrepancies)` |
| y scale | **Linear** (`:identity`) | This is the current repository default |
| y values | Raw, nonnegative KL discrepancies | Do not transform the input values |
| y limits | Data-dependent and padded; lower padding clipped at zero | Keeps labels inside without displaying negative KL |

On the linear y-axis, the same absolute difference has the same visual size
everywhere. Percentage annotations provide the relative-change view.

Do not call this axis logarithmic unless `yscale = :log10` is enabled. An
older version of this document described a log-y plot, but the current source
uses a linear y-axis and the label `"max per-component"`.

### 3.2 Legend location

Place the legend **outside and below the plotting area**:

```julia
legend = :outerbottom
```

An in-panel legend can cover restart clouds, the descending curve, or drop
labels. Reserve approximately `24Plots.mm` of bottom margin.

Use two columns for `best per K` and `envelope`, three if restart points are
present, and up to four when a ground-truth line is also present. The
pipeline-specific functions sometimes wrap these entries into one or two
columns to fit a multi-panel figure; the location should remain
`:outerbottom`.

Only line series should create the `best per K` and `envelope` entries.
Their overlaid colored scatter series must use `label = nothing` to prevent
duplicate legend entries.

---

## 4. Inputs and validation

| Input | Form | Meaning |
|---|---|---|
| `K_range` | Sorted integer vector/range | Candidate model sizes |
| `results_vec` | One result or `nothing` per `K` | Selected ACDC result |
| `result.component_discrepancies` | Numeric vector | Component-level KL values |
| `restart_values`, optional | Vector of numeric vectors | Maximum component KL from every restart |
| `K_true`, optional | Integer | Known ground-truth state count |

`K_range` and `results_vec` must have equal lengths and be ordered by
increasing `K`. For each valid result, calculate

```julia
z_K = maximum(result.component_discrepancies)
```

Skip a candidate when its result is `nothing`, its discrepancy vector is
empty, any selected discrepancy is nonfinite, or its maximum is negative.
Filter restart points individually so one invalid restart does not remove the
other restart values at that `K`.

Do not replace a failed fit with zero. Zero would look like a perfect fit and
incorrectly dominate all later envelope points.

---

## 5. Construction procedure

1. **Summarize candidates.** Build parallel arrays of valid `K` values,
   maximum discrepancies, and original sweep indices. The indices preserve
   consistent colors across panels.
2. **Create the empty panel.** Set both scales explicitly to `:identity`,
   force integer ticks, and put the legend below the axes. Return this correctly
   labeled empty panel if no valid data remain.
3. **Draw restarts first.** Use small translucent gray points. Label only the
   first nonempty group so the legend gets one entry.
4. **Draw the selected path.** Connect every valid selected value with a thin
   gray line and overlay medium per-`K` colored nodes. Preserve upward bumps:
   they reveal optimization instability.
5. **Extract record lows.** Initialize `best_so_far = Inf` and retain a point
   only when `z < best_so_far`. Use a strict comparison; `<=` would turn
   equal plateaus into false improvements.
6. **Draw the envelope.** Use a thick black line with large, black-outlined,
   per-`K` colored nodes.
7. **Annotate drops.** Place each `−X.X%` label near its segment midpoint,
   offset perpendicular to the segment in normalized panel coordinates.
8. **Expand y limits.** Include annotation y coordinates in the limit
   calculation, add about 12% padding, and clip the lower limit at zero.
9. **Add references and export.** Draw an optional ground-truth line last.
   Save PNG at 300 dpi and PDF for vector output.

For label placement on linear axes, normalize the segment direction:

```math
\Delta k_n = \frac{K_1-K_0}{K_{\max}-K_{\min}},\qquad
\Delta z_n = \frac{z_1-z_0}{z_{\max}-z_{\min}}.
```

Normalize `(Δk_n, Δz_n)`, rotate it 90 degrees, and offset the arithmetic
midpoint by approximately 0.045 of the panel span. Normalization prevents the
offset from changing arbitrarily with numerical units.

---

## 6. Complete Julia example

This code accepts any result object with a `component_discrepancies` property,
including named tuples.

```julia
using Plots
using Printf

function acdc_envelope_panel(
    K_range,
    results_vec;
    restart_values = nothing,
    K_true = nothing,
    title = "ACDC Envelope (KL)",
)
    Ks = Int.(collect(K_range))
    length(Ks) == length(results_vec) ||
        throw(ArgumentError("K_range and results_vec must have equal lengths"))
    issorted(Ks) || throw(ArgumentError("K_range must be sorted increasingly"))

    valid_K = Int[]
    valid_z = Float64[]
    valid_idx = Int[]
    for (i, K) in enumerate(Ks)
        result = results_vec[i]
        isnothing(result) && continue
        discrepancies = Float64.(result.component_discrepancies)
        isempty(discrepancies) && continue
        all(isfinite, discrepancies) || continue
        z = maximum(discrepancies)
        z >= 0 || continue
        push!(valid_K, K)
        push!(valid_z, z)
        push!(valid_idx, i)
    end

    nlegend = 2 + !isnothing(restart_values) + !isnothing(K_true)
    hires = (
        dpi = 300,
        tickfontsize = 10,
        guidefontsize = 12,
        legendfontsize = 10,
        titlefontsize = 13,
        fontfamily = "Helvetica",
        left_margin = 10Plots.mm,
        right_margin = 5Plots.mm,
        top_margin = 5Plots.mm,
        bottom_margin = 24Plots.mm,
    )

    p = plot(;
        title = title,
        xlabel = "K",
        ylabel = "max per-component",
        xscale = :identity,
        yscale = :identity,
        xticks = collect(minimum(Ks):maximum(Ks)),
        legend = :outerbottom,
        legend_columns = min(nlegend, 4),
        hires...,
    )
    isempty(valid_K) && return p
    node_colors = [palette(:auto)[i] for i in valid_idx]

    # Background restart cloud.
    if !isnothing(restart_values)
        length(restart_values) == length(Ks) ||
            throw(ArgumentError("restart_values must have one entry per K"))
        used_restart_label = false
        for (i, K) in enumerate(Ks)
            values = filter(
                z -> isfinite(z) && z >= 0,
                Float64.(restart_values[i]),
            )
            isempty(values) && continue
            scatter!(
                p, fill(K, length(values)), values;
                color = :gray, markeralpha = 0.25, markersize = 3,
                markerstrokewidth = 0,
                label = used_restart_label ? nothing : "all restarts",
            )
            used_restart_label = true
        end
    end

    # Selected result for every valid K, including nonmonotone bumps.
    plot!(
        p, valid_K, valid_z;
        color = :gray, linewidth = 1.0, alpha = 0.5,
        label = "best per K",
    )
    scatter!(
        p, valid_K, valid_z;
        markersize = 5, markerstrokewidth = 0,
        markercolor = node_colors, markeralpha = 0.55, label = nothing,
    )

    # Cumulative record-low envelope.
    env_K = Int[]
    env_z = Float64[]
    env_colors = Any[]
    best_so_far = Inf
    for (K, z, color) in zip(valid_K, valid_z, node_colors)
        if z < best_so_far
            push!(env_K, K)
            push!(env_z, z)
            push!(env_colors, color)
            best_so_far = z
        end
    end

    plot!(
        p, env_K, env_z;
        color = :black, linewidth = 2.5, label = "envelope",
    )
    scatter!(
        p, env_K, env_z;
        markersize = 8, markerstrokewidth = 1.8,
        markerstrokecolor = :black, markercolor = env_colors,
        label = nothing,
    )

    # Drop labels offset perpendicular to segments in normalized linear space.
    annotation_y = Float64[]
    K_span = max(float(maximum(env_K) - minimum(env_K)), 1.0)
    z_span = max(maximum(env_z) - minimum(env_z), eps(Float64))
    for j in 1:(length(env_K) - 1)
        K0, K1 = env_K[j], env_K[j + 1]
        z0, z1 = env_z[j], env_z[j + 1]
        drop_percent = iszero(z0) ? 0.0 : 100 * (z0 - z1) / z0

        dK_n = (K1 - K0) / K_span
        dz_n = (z1 - z0) / z_span
        magnitude = hypot(dK_n, dz_n)
        iszero(magnitude) && continue
        perpendicular_x = -dz_n / magnitude
        perpendicular_y = dK_n / magnitude

        offset = 0.045
        x_text = (K0 + K1) / 2 + perpendicular_x * offset * K_span
        y_text = (z0 + z1) / 2 + perpendicular_y * offset * z_span
        push!(annotation_y, y_text)
        annotate!(
            p, x_text, y_text,
            text(@sprintf("−%.1f%%", drop_percent),
                 8, :black, :hcenter, :vcenter),
        )
    end

    if !isempty(annotation_y)
        y_min = min(minimum(valid_z), minimum(annotation_y))
        y_max = max(maximum(valid_z), maximum(annotation_y))
        padding = max(0.12 * (y_max - y_min), eps(Float64))
        plot!(p; ylim = (max(0.0, y_min - padding), y_max + padding))
    end

    if !isnothing(K_true)
        vline!(
            p, [K_true];
            linestyle = :dot, color = :seagreen, linewidth = 2.0,
            label = "GT K=$K_true",
        )
    end
    return p
end

# Demonstration data: one selected result per K.
K_range = 1:8
max_discrepancy = [6.3, 3.5, 3.8, 1.9, 0.72, 0.64, 0.63, 0.66]
results = [
    (component_discrepancies = [0.55z, 0.82z, z],)
    for z in max_discrepancy
]
restart_values = [
    [0.98z, z, 1.08z, 1.17z]
    for z in max_discrepancy
]

p_envelope = acdc_envelope_panel(
    K_range, results;
    restart_values = restart_values,
    K_true = 5,
)
savefig(p_envelope, "acdc_envelope_example.png")
savefig(p_envelope, "acdc_envelope_example.pdf")
```

Here `K=3` is excluded from the envelope because 3.8 is worse than the record
3.5 at `K=2`. `K=8` is excluded because 0.66 does not improve on 0.63 at
`K=7`. Both remain visible on the gray best-per-`K` path.

---

## 7. Companion ACDC loss plot

The notebooks commonly place the envelope beside an ACDC loss-versus-threshold
panel. For component discrepancies `D_j(K)` and threshold `ρ`, the loss is

```math
R^\rho(K) = \sum_j \max(0,D_j(K)-\rho).
```

Use a linear `ρ` x-axis, a linear ACDC-loss y-axis, one colored line per
valid `K` with width about 2.0, and an `:outerbottom` legend with four or
five columns. Choose a `ρ` grid covering the per-`K` maximum discrepancies,
with approximately 8% padding.

```julia
function acdc_loss_panel(K_range, results_vec)
    valid = [
        (K, results_vec[i])
        for (i, K) in enumerate(K_range)
        if !isnothing(results_vec[i]) &&
           all(isfinite, results_vec[i].component_discrepancies)
    ]

    p = plot(;
        title = "ACDC Loss (KL)", xlabel = "ρ", ylabel = "ACDC Loss",
        xscale = :identity, yscale = :identity,
        legend = :outerbottom,
        legend_columns = max(1, min(length(valid), 5)),
        bottom_margin = 24Plots.mm,
    )
    isempty(valid) && return p

    maxima = [maximum(r.component_discrepancies) for (_, r) in valid]
    low, high = extrema(maxima)
    padding = max(0.08 * (high - low), 0.02 * max(low, eps(Float64)))
    rho_grid = range(low - padding, high + padding; length = 300)

    for (K, result) in valid
        losses = [
            sum(max(0.0, d - rho) for d in result.component_discrepancies)
            for rho in rho_grid
        ]
        plot!(p, rho_grid, losses; label = "K=$K", linewidth = 2.0)
    end
    return p
end

p_loss = acdc_loss_panel(K_range, results)
p_combined = plot(
    p_loss, p_envelope;
    layout = (1, 2), size = (1300, 500), margin = 5Plots.mm,
)
savefig(p_combined, "acdc_loss_and_envelope.png")
```

The envelope can instead be paired with BIC/ICL. In that panel, use a linear
x-axis with integer `K` ticks, a linear score y-axis, and an outside-bottom
legend.

---

## 8. Optional log-y envelope

Use log-y only when raw discrepancies span several orders of magnitude and
small values are unreadable. It is not the current notebook default.

A correct log-y variant must:

1. retain only strictly positive y values;
2. set `yscale = :log10`;
3. label y as `"max per-component (log10 scale)"`;
4. keep `K` linear with integer ticks;
5. use `sqrt(z0*z1)`, the geometric mean, for a segment's visual y midpoint;
6. compute perpendicular offsets in `log10(z)` coordinates and transform the
   result back with `10^value`.

Merely adding `yscale = :log10` to the linear annotation code makes labels
unevenly spaced and no longer visually perpendicular to their segments.

---

## 9. Interpretation and common mistakes

### Interpretation

- A large drop means an added component fixed substantial misspecification.
- A run with no envelope nodes means later candidates did not beat the record.
- A tail of small drops suggests diminishing returns, estimator noise, or
  contamination fitting rather than new structural states.
- A wide restart cloud indicates optimization instability.
- An upward gray-path bump is preserved deliberately; the black envelope
  ignores it, while the supporting path reveals it.

The elbow is a diagnostic, not proof of the correct `K`. Interpret it with
BIC/ICL, restart stability, component occupancy, and domain knowledge. 

### Common mistakes

1. Taking the mean instead of `maximum(component_discrepancies)`.
2. Sorting by discrepancy instead of scanning in increasing `K`.
3. Using `<=`, which treats equal plateaus as improvements.
4. Drawing only the black envelope and hiding unstable selected fits.
5. Replacing failed fits with zero rather than skipping them.
6. Allowing fractional x ticks for integer-valued `K`.
7. Putting the legend over the data instead of at `:outerbottom`.
8. Calling the linear y-axis logarithmic.
9. Mixing total loss, mean KL, and maximum KL among plot layers.
10. Computing drops between adjacent candidates instead of envelope nodes.

---

## 10. Verification checklist

- [ ] `K_range` is sorted and aligned with `results_vec`.
- [ ] Every plotted value is the maximum component KL discrepancy.
- [ ] Missing/nonfinite fits are skipped, not converted to zero.
- [ ] Envelope values strictly decrease from left to right.
- [ ] Skipped candidates remain visible on the gray path.
- [ ] Drop labels use the previous envelope node as their baseline.
- [ ] x is linear and every candidate `K` has an integer tick.
- [ ] y is linear and labeled `max per-component` by default.
- [ ] The legend is outside below the panel with no duplicate markers.
- [ ] Annotation text remains inside the panel.
- [ ] The same `K` uses the same color in envelope and loss panels.
- [ ] PNG and PDF exports are readable at their intended size.
