using Pkg
Pkg.activate(@__DIR__)

ENV["GKS_ENCODING"] = "utf8"
const SCRIPT_DIR  = @__DIR__
const FIGURES_DIR = joinpath(SCRIPT_DIR, "figures", "main", "dog_biologging")
const CACHE_DIR   = joinpath(SCRIPT_DIR, "cache", "dog_multi_per_metric_v2", "hmm")
const DATA_DIR    = joinpath(SCRIPT_DIR, "data", "DogMoveMendeley2021")
const PROCESSED_DIR = joinpath(DATA_DIR, "processed")
mkpath(FIGURES_DIR); mkpath(CACHE_DIR)

using DataFrames, Statistics, LinearAlgebra, Random, Serialization, Printf, Dates
using Distributions, CSV
using LaTeXStrings
using Plots, Measures
using StatsPlots
using StateSpaceDynamics
using StateSpaceDynamics.ACDC
using StateSpaceDynamics: viterbi

include(joinpath(SCRIPT_DIR, "helpers", "DogNotebookUtils.jl"))
using .DogNotebookUtils
include(joinpath(SCRIPT_DIR, "helpers", "FigureExport.jl"))

# --- Data loading -----------------------------------------------------------
const DOGMOVE_RAW_CSV    = joinpath(PROCESSED_DIR, "dog_raw_timestep.csv")
const DOGMOVE_LOOKUP_CSV = joinpath(PROCESSED_DIR, "dog_label_lookup.csv")

dog_id_col = CSV.read(DOGMOVE_RAW_CSV, DataFrame; select=[:dog_id])
dog_counts = combine(groupby(dog_id_col, :dog_id), nrow => :n_rows)
dog_id_col = nothing
sort!(dog_counts, [:n_rows, :dog_id])
dog_least_two = Int.(first(dog_counts, min(2, nrow(dog_counts))).dog_id)
dog_most_one  = Int(last(dog_counts, 1).dog_id[1])
const DOG_SELECTOR = unique(vcat([dog_most_one], dog_least_two))
println("DOG_SELECTOR (most + two least): ", DOG_SELECTOR)

loaded = load_dogmove_multi_csv(DOGMOVE_RAW_CSV, DOGMOVE_LOOKUP_CSV;
    window=10, dog_selector=DOG_SELECTOR)
const sdict        = loaded.sdict
const WINDOW       = loaded.WINDOW
const observations = loaded.observations
println("observations size: ", size(observations))

_plain_fmt(x) = (x == 0 ? "0" : abs(x) < 0.01 || abs(x) >= 1e6 ?
    @sprintf("%.1e", x) : @sprintf("%g", x))
const hires = (
    dpi=300, tickfontsize=10, guidefontsize=12, legendfontsize=10, titlefontsize=13,
    fontfamily="Helvetica",
    left_margin=10Plots.mm, right_margin=5Plots.mm,
    top_margin=5Plots.mm, bottom_margin=24Plots.mm,
    yformatter=_plain_fmt, xformatter=_plain_fmt,
)

# =============================================================================
# Bout-HMM analysis (per-metric-best) — selection and pair distribution
# =============================================================================
println("\n" * "=" ^ 70)
println("BOUT-HMM ANALYSIS")
println("=" ^ 70)
const BOUT_CACHE_DIR = joinpath(SCRIPT_DIR, "cache", "dog_multi_per_metric_v2", "bouthmm")
const BOUT_FIGURES_DIR = FIGURES_DIR
mkpath(BOUT_CACHE_DIR)

bout_data = segment_task_bouts(observations, sdict; window=WINDOW, min_windows=2)
println("Segmented $(length(bout_data.sequences)) task bouts (min 2 windows each)")

# Fit + analysis cache for bout-HMM
bouthmm_run = compute_bouthmm_models_allruns(bout_data.sequences;
    K_range=1:14, n_restarts=20, seed=7777, max_iters=200, tol=1e-4,
    cache_dir=BOUT_CACHE_DIR, use_cache=true)

const BOUTHMM_ANALYSIS_CACHE = joinpath(BOUT_CACHE_DIR, "selection_analysis.jls")
res_bhmm = if isfile(BOUTHMM_ANALYSIS_CACHE)
    println("Loaded bout-HMM selection analysis from cache")
    deserialize(BOUTHMM_ANALYSIS_CACHE)
else
    println("Bout-HMM per-metric-best selection (fresh evaluation)")
    r = run_bouthmm_selection_analysis_allruns(bouthmm_run, bout_data.sequences;
        discrepancy=KLDiscrepancy(), n_samples=1)
    serialize(BOUTHMM_ANALYSIS_CACHE, r)
    r
end
println("Bout-HMM selections: BIC=", res_bhmm.selections.bic,
        "  ICL=", res_bhmm.selections.icl,
        "  ACDC-KL=", res_bhmm.selections.acdc)

# Bout-HMM selection plot (BIC/ICL + ρ-curve + envelope, with per-seed scatter)
let
    Kr = res_bhmm.K_range
    Kbic = res_bhmm.selections.bic
    Kicl = res_bhmm.selections.icl
    Kacdc = res_bhmm.selections.acdc

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
    _save_png_pdf(p_b_sel, BOUT_FIGURES_DIR, "selection_bouthmm"; trim=true)
    println("Saved bout-HMM selection plot")
end

# Pair-plot helper: builds the paper's aligned pair-distribution panels.
using StateSpaceDynamics: viterbi
function _build_bouthmm_pair_plots(model_run_for_metric, bout_data, observations, sdict, K_list;
                                    window, out_dir, tag)
    mkpath(out_dir)
    bhmm_stacks = build_bouthmm_behavior_stacks(model_run_for_metric, bout_data, observations,
        sdict, K_list; window=window)

    K_index_bp = Dict(K => i for (i, K) in enumerate(model_run_for_metric.K_range))
    beh_raw_mat_bp = sdict["Behavior"]["codes"]
    beh_cat_names_bp = extract_categories(sdict["Behavior"])
    bhmm_amap_bp = bhmm_stacks.alignment_map
    bhmm_ac_bp   = bhmm_stacks.axis_count
    bhmm_an_bp   = bhmm_stacks.axis_name

    function compute_pair_counts_bp(K)
        idx = get(K_index_bp, K, 0)
        idx == 0 && return nothing
        mdl = model_run_for_metric.models[idx]
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
        pair_data = compute_pair_counts_bp(K)
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

function _build_run_for_metric_bout(model_run, models_for_metric)
    return (; K_range=model_run.K_range,
              models=models_for_metric,
              log_likelihoods=fill(NaN, length(model_run.K_range)),
              cache_status=fill(:loaded, length(model_run.K_range)),
              cache_dir="")
end

K_LIST_BHMM = collect(1:14)
for (tag, models) in [("acdc_kl", res_bhmm.models_per_metric.acdc_kl)]
    println("\n=== Bout-HMM pair plots ($tag-best) ===")
    sub_run = _build_run_for_metric_bout(bouthmm_run, models)
    out_dir = BOUT_FIGURES_DIR
    _build_bouthmm_pair_plots(sub_run, bout_data, observations, sdict, K_LIST_BHMM;
        window=WINDOW, out_dir=out_dir, tag=tag)
    println("  Saved bout-HMM pair plots to $out_dir")
end

# Bout-HMM stats text (BIC/ICL/ACDC table + envelope + per-cluster behavior breakdown)
bout_stats_path = joinpath(SCRIPT_DIR, "results", "selection_stats_bouthmm.txt")
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
            m_eff = min(length(st), n_smp ÷ WINDOW)
            beh_valid = sort(unique(c for c in beh_raw_mat if c != 0))
            counts = zeros(Float64, length(beh_valid), K)
            for j in 1:m_eff
                comp = st[j]
                r = (WINDOW * (j - 1) + 1):(WINDOW * j)
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

# --- Bout-HMM pair statistics (text dump) -----------------------------------
# Reports for each (BIC-best, ACDC-KL-best) and selected K's:
#   1. Per-cluster top behavior pairs (which pair concentrates in which cluster)
#   2. Per-pair cluster distribution (which clusters does this pair land in)
function _write_pair_stats(io, model_vec, model_run, K_list, observations, sdict;
                            window, alignment_mode=:hungarian, top_n=15,
                            alignment_K_list=nothing)
    sub_run = (; K_range=model_run.K_range,
                 models=model_vec,
                 log_likelihoods=fill(NaN, length(model_run.K_range)),
                 cache_status=fill(:loaded, length(model_run.K_range)),
                 cache_dir="")
    # Use the wider alignment_K_list (e.g., 1:8) so the alignment matches the
    # composition plots; default to K_list if not given.
    align_K_list = isnothing(alignment_K_list) ? K_list : alignment_K_list
    bhmm_stacks = build_bouthmm_behavior_stacks(sub_run, bout_data, observations,
        sdict, align_K_list; window=window)
    K_index = Dict(K => i for (i, K) in enumerate(model_run.K_range))
    beh_raw_mat = sdict["Behavior"]["codes"]
    beh_cat_names = extract_categories(sdict["Behavior"])
    amap = bhmm_stacks.alignment_map
    ac   = bhmm_stacks.axis_count
    an   = bhmm_stacks.axis_name

    function pair_counts_for(K)
        idx = get(K_index, K, 0); idx == 0 && return nothing
        mdl = model_vec[idx]; isnothing(mdl) && return nothing
        st = viterbi(mdl, observations)
        n_smp = size(beh_raw_mat, 1)
        m_eff = min(length(st), n_smp ÷ window)
        m_eff == 0 && return nothing

        beh_valid = sort(unique(c for c in beh_raw_mat if c != 0))
        beh_labels = [1 <= c <= length(beh_cat_names) ?
            string(beh_cat_names[c]) : "Code $c" for c in beh_valid]
        n_beh = length(beh_valid); n_with_none = n_beh + 1
        labels_full = vcat(beh_labels, ["(pure)"])
        beh_code_to_idx = Dict(c => j for (j, c) in enumerate(beh_valid))
        pair_counts = zeros(Float64, n_with_none, n_with_none, ac)
        for wi in 1:m_eff
            k_raw = st[wi]; k_al = get(amap, (K, k_raw), 0); k_al == 0 && continue
            r_start = window * (wi - 1) + 1
            r_end = min(window * wi, n_smp)
            code_counts = Dict{Int,Int}()
            for si in r_start:r_end, col in 1:size(beh_raw_mat, 2)
                c = beh_raw_mat[si, col]; c == 0 && continue
                code_counts[c] = get(code_counts, c, 0) + 1
            end
            present = collect(keys(code_counts))
            if isempty(present)
                pair_counts[n_with_none, n_with_none, k_al] += 1
            elseif length(present) == 1
                ai = beh_code_to_idx[present[1]]
                pair_counts[ai, n_with_none, k_al] += 1
                pair_counts[n_with_none, ai, k_al] += 1
            else
                for ci in present, cj in present
                    ci >= cj && continue
                    pair_counts[beh_code_to_idx[ci], beh_code_to_idx[cj], k_al] += 1
                end
            end
        end
        return (; pair_counts, labels_full, n_with_none, axis_count=ac, axis_name=an)
    end

    for K in K_list
        pd = pair_counts_for(K); isnothing(pd) === true && continue
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

pair_stats_path = joinpath(SCRIPT_DIR, "results", "selection_stats_bouthmm_pairs.txt")
open(pair_stats_path, "w") do io
    println(io, "=" ^ 70)
    println(io, "DOG MULTI BOUT-HMM — BEHAVIOR PAIR STATISTICS")
    println(io, "Generated: $(Dates.now())")
    println(io, "=" ^ 70)
    println(io, "\n  Pairs: ordered combinations of two behaviors co-occurring")
    println(io, "  within the same window (or one behavior + (pure) marker for")
    println(io, "  windows where exactly one behavior is annotated).")
    println(io, "  Filter: only pairs with > 50 windows shown.")

    # Use K_LIST_BHMM=1:8 alignment to match the composition/pair plots; only
    # WRITE stats for the K's of interest (4, 5, 6, 7, 8) inside the writer.
    K_show = sort(unique([res_bhmm.selections.acdc, res_bhmm.selections.bic, 4, 5, 7]))
    sources = [("BIC-best",     res_bhmm.models_per_metric.bic),
               ("ACDC-KL-best", res_bhmm.models_per_metric.acdc_kl)]
    for (sname, model_vec) in sources
        println(io, "\n" * "═" ^ 70)
        println(io, "[$sname] fits — pair statistics at K=$(K_show)")
        println(io, "═" ^ 70)
        _write_pair_stats(io, model_vec, bouthmm_run, K_show, observations, sdict;
            window=WINDOW, alignment_K_list=K_LIST_BHMM)
    end
end
println("Wrote pair stats to $pair_stats_path")

println("\nDONE")
