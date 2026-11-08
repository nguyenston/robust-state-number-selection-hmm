# =============================================================================
# Synthetic spike-train generator
# =============================================================================

struct SpikeGroundTruth
    K_true::Int
    rates_per_state::Vector{Vector{Float64}}  # Hz per neuron, per state
    bin_seconds::Float64
    A::Matrix{Float64}
    π₀::Vector{Float64}
    state_labels::Vector{String}
end

"""
    generate_synthetic_spikes(; seed=42, T=10000, bin_ms=10, cache_dir=nothing) -> NamedTuple

Sample a synthetic spike-train sequence from a known Poisson HMM.

Ground truth (K_true=5, D=5):
  1. SILENT:  ~1 Hz uniform
  2. DOWN:    ~5 Hz uniform
  3. UP:      ~25-35 Hz, neuron-specific
  4. BURST_A: neurons {1,3,5} at 80 Hz, {2,4} at 5 Hz
  5. BURST_B: neurons {2,4} at 80 Hz, {1,3,5} at 5 Hz

BURST_A and BURST_B have the same total rate but distinct spatial patterns —
distinguishable only through per-neuron rates, not total counts.
Sticky transitions (A_ii = 0.96). Bin width `bin_ms` ms.
"""
function generate_synthetic_spikes(;
    seed::Int       = 42,
    T::Int          = 10_000,
    bin_ms::Real    = 10.0,
    cache_dir::Union{String,Nothing} = nothing,
    rates_per_state::Union{Nothing,Vector{Vector{Float64}}} = nothing,
    state_labels::Union{Nothing,Vector{String}}             = nothing,
    rates_tag::Union{Nothing,String}                        = nothing,
)
    tag_suffix = isnothing(rates_tag) ? "" : "_$(rates_tag)"
    cache_file = isnothing(cache_dir) ? nothing :
        joinpath(cache_dir, "spikes_seed$(seed)_T$(T)_bin$(bin_ms)$(tag_suffix).jls")
    if !isnothing(cache_file) && isfile(cache_file)
        println("Loading cached spike data: $cache_file")
        return deserialize(cache_file)
    end

    rng = MersenneTwister(seed)
    bin_s = bin_ms / 1000.0   # seconds per bin

    if isnothing(rates_per_state)
        rates_per_state = [
            [ 1.0,  1.0,  1.0,  1.0,  1.0],    # SILENT
            [ 5.0,  5.0,  5.0,  5.0,  5.0],    # DOWN
            [25.0, 30.0, 20.0, 35.0, 25.0],    # UP
            [80.0,  5.0, 80.0,  5.0, 80.0],    # BURST_A
            [ 5.0, 80.0,  5.0, 80.0,  5.0],    # BURST_B
        ]
    end
    K_true = length(rates_per_state)
    D = length(rates_per_state[1])

    A = fill(0.01, K_true, K_true)
    for k in 1:K_true; A[k, k] = 0.96; end
    π₀ = fill(1.0 / K_true, K_true)
    if isnothing(state_labels)
        state_labels = ["silent", "down", "up", "burst_a", "burst_b"]
    end
    gt = SpikeGroundTruth(K_true, rates_per_state, bin_s, A, π₀, state_labels)

    # Sample state sequence
    states = Vector{Int}(undef, T)
    states[1] = rand(rng, Categorical(π₀))
    for t in 2:T
        states[t] = rand(rng, Categorical(A[states[t-1], :]))
    end

    # Sample spike counts per bin, per neuron
    counts = Matrix{Float64}(undef, T, D)
    for t in 1:T
        rates = rates_per_state[states[t]]
        for d in 1:D
            λ = rates[d] * bin_s
            counts[t, d] = rand(rng, Poisson(λ))
        end
    end

    # Observations matrix: D × T (consistent with the HMM convention)
    observations = Matrix(transpose(counts))

    result = (
        seed          = seed,
        T             = T,
        bin_ms        = bin_ms,
        ground_truth  = gt,
        states        = states,
        counts        = counts,          # T × D for convenience
        observations  = observations,    # D × T for HMM fit
    )

    if !isnothing(cache_file)
        mkpath(dirname(cache_file)); serialize(cache_file, result)
        println("Cached spike data to: $cache_file")
    end
    return result
end

"""
    generate_synthetic_spikes_negbin(; seed=42, T=10000, bin_ms=10, overdispersion=2.0, cache_dir=nothing)

**Mismatched case.** Same state structure and mean rates as `generate_synthetic_spikes`,
but spike counts are drawn from NegativeBinomial instead of Poisson. The overdispersion
factor `d` sets variance = d × mean.

When fit with PoissonEmission, the model correctly identifies the mean rates but
cannot capture the extra variance. This is a realistic, mild misspecification — the
kind that arises when neural variability exceeds Poisson (which it usually does).

NegBin(r, p) parametrization: p = 1/d, r = μ/(d − 1).
"""
function generate_synthetic_spikes_negbin(;
    seed::Int           = 42,
    T::Int              = 10_000,
    bin_ms::Real        = 10.0,
    overdispersion::Real = 2.0,
    truncnormal_mix::Real = 0.0,
    contamination_rate::Real = 0.0,
    contamination_burst_length::Int = 1,
    contamination_modes::Union{Nothing,Vector{Vector{Float64}}} = nothing,
    cache_dir::Union{String,Nothing} = nothing,
    rates_per_state::Union{Nothing,Vector{Vector{Float64}}} = nothing,
    state_labels::Union{Nothing,Vector{String}}             = nothing,
    rates_tag::Union{Nothing,String}                        = nothing,
)
    d_factor = Float64(overdispersion)
    w_tn     = Float64(truncnormal_mix)
    cont_rate = Float64(contamination_rate)
    @assert d_factor > 1 "overdispersion must be > 1 (=1 is Poisson)"
    @assert 0 <= w_tn <= 1 "truncnormal_mix must be in [0, 1]"
    @assert 0 <= cont_rate <= 1 "contamination_rate must be in [0, 1]"
    if cont_rate > 0
        @assert !isnothing(contamination_modes) && !isempty(contamination_modes) "contamination_modes must be supplied when contamination_rate > 0"
    end

    tag_suffix  = isnothing(rates_tag) ? "" : "_$(rates_tag)"
    tn_suffix   = w_tn > 0 ? "_tn$(w_tn)" : ""
    cont_suffix = cont_rate > 0 ? "_cont$(cont_rate)L$(contamination_burst_length)x$(length(contamination_modes))" : ""
    cache_file = isnothing(cache_dir) ? nothing :
        joinpath(cache_dir, "spikes_negbin_seed$(seed)_T$(T)_bin$(bin_ms)_od$(d_factor)$(tn_suffix)$(cont_suffix)$(tag_suffix).jls")
    if !isnothing(cache_file) && isfile(cache_file)
        println("Loading cached NegBin spike data: $cache_file")
        return deserialize(cache_file)
    end

    rng = MersenneTwister(seed)
    bin_s = bin_ms / 1000.0

    if isnothing(rates_per_state)
        rates_per_state = [
            [ 1.0,  1.0,  1.0,  1.0,  1.0],    # SILENT
            [ 5.0,  5.0,  5.0,  5.0,  5.0],    # DOWN
            [25.0, 30.0, 20.0, 35.0, 25.0],    # UP
            [80.0,  5.0, 80.0,  5.0, 80.0],    # BURST_A
            [ 5.0, 80.0,  5.0, 80.0,  5.0],    # BURST_B
        ]
    end
    K_true = length(rates_per_state)
    D = length(rates_per_state[1])

    A = fill(0.01, K_true, K_true)
    for k in 1:K_true; A[k, k] = 0.96; end
    π₀ = fill(1.0 / K_true, K_true)
    if isnothing(state_labels)
        state_labels = ["silent", "down", "up", "burst_a", "burst_b"]
    end
    gt = SpikeGroundTruth(K_true, rates_per_state, bin_s, A, π₀, state_labels)

    # Sample state sequence (same process as Poisson case)
    states = Vector{Int}(undef, T)
    states[1] = rand(rng, Categorical(π₀))
    for t in 2:T
        states[t] = rand(rng, Categorical(A[states[t-1], :]))
    end

    # Sample spike counts from a NegBin / TruncatedNormal mixture with matched
    # per-bin mean and variance (var = overdispersion × mean for both).
    # NegBin is right-skewed with a heavy tail; TruncatedNormal-rounded is
    # near-symmetric. The mixture creates a per-state distribution whose shape
    # cannot be matched by a single Poisson, even after the model gets the
    # mean right — and extra components can absorb it by allocating one to
    # the symmetric mode and another to the skewed tail.
    counts = Matrix{Float64}(undef, T, D)
    p_nb = 1.0 / d_factor
    contamination_log = falses(T)
    cont_remaining = 0          # bins left in current contamination burst
    cont_mode_idx  = 0          # which mode is active during the burst
    for t in 1:T
        # Burst-length contamination: at each step, if not currently in a
        # contamination burst, with prob `cont_rate` start a burst of
        # `contamination_burst_length` bins drawing from a randomly-chosen
        # contamination mode. The bin's GT state label is left unchanged —
        # contamination is sample-level noise on top of the HMM.
        if cont_rate > 0 && cont_remaining == 0 && rand(rng) < cont_rate
            cont_remaining = contamination_burst_length
            cont_mode_idx  = rand(rng, 1:length(contamination_modes))
        end

        if cont_remaining > 0
            mode = contamination_modes[cont_mode_idx]
            for dd in 1:D
                μ = mode[dd] * bin_s
                counts[t, dd] = Float64(rand(rng, Poisson(max(μ, eps()))))
            end
            contamination_log[t] = true
            cont_remaining -= 1
            continue
        end

        rates = rates_per_state[states[t]]
        for dd in 1:D
            μ = rates[dd] * bin_s
            σ = sqrt(d_factor * μ)
            from_tn = w_tn > 0 && rand(rng) < w_tn
            if from_tn && μ > 0
                tn = truncated(Normal(μ, σ), 0.0, Inf)
                counts[t, dd] = Float64(max(0, round(Int, rand(rng, tn))))
            else
                r = μ / (d_factor - 1.0)
                r = max(r, 1e-6)
                counts[t, dd] = Float64(rand(rng, NegativeBinomial(r, p_nb)))
            end
        end
    end

    observations = Matrix(transpose(counts))

    result = (
        seed                = seed,
        T                   = T,
        bin_ms              = bin_ms,
        overdispersion      = d_factor,
        truncnormal_mix     = w_tn,
        contamination_rate  = cont_rate,
        contamination_modes = contamination_modes,
        contamination_log   = contamination_log,
        ground_truth        = gt,
        states              = states,
        counts              = counts,
        observations        = observations,
    )

    if !isnothing(cache_file)
        mkpath(dirname(cache_file)); serialize(cache_file, result)
        println("Cached NegBin spike data to: $cache_file")
    end
    return result
end

function describe_ground_truth(io::IO, data)
    gt = data.ground_truth
    is_negbin = hasproperty(data, :overdispersion)
    variant = is_negbin ? "NegativeBinomial (overdispersion=$(data.overdispersion)×)" : "Poisson"
    println(io, "Ground-truth $variant HMM:")
    println(io, "  K_true   = $(gt.K_true)")
    println(io, "  D        = $(length(gt.rates_per_state[1]))  (neurons)")
    println(io, "  T        = $(data.T)  bins")
    println(io, "  bin_ms   = $(data.bin_ms)  ms")
    println(io, "  duration = $(round(data.T * data.bin_ms / 1000; digits=1))  s")
    println(io, "  seed     = $(data.seed)")
    is_negbin && println(io, "  overdispersion = $(data.overdispersion)×  (variance = $(data.overdispersion) × mean)")
    println(io, "  Transition matrix A (A_ii = $(round(gt.A[1,1]; digits=3))):")
    for k in 1:gt.K_true
        @printf(io, "    [%s]\n", join([@sprintf("%.3f", a) for a in gt.A[k, :]], "  "))
    end
    println(io, "  State occupancy (empirical):")
    for k in 1:gt.K_true
        frac = count(==(k), data.states) / data.T * 100
        rates = gt.rates_per_state[k]
        @printf(io, "    State %d [%s]: %5.1f%%  rates (Hz) = [%s]\n",
            k, gt.state_labels[k], frac, join([@sprintf("%.1f", r) for r in rates], ", "))
    end
end

describe_ground_truth(data) = (describe_ground_truth(stdout, data); nothing)
