# =============================================================================
# StepAngleEmission: Gaussian (log-step) ⊕ WrappedCauchy (turn angle)
# =============================================================================

"""
    StepAngleEmission

Composite emission for GPS movement data.

- Column 1: z-scored log step length — modelled as `GaussianEmission(output_dim=1)`.
  Free parameters per component: 1 mean + 1 variance = **2**.
- Column 2: turn angle in radians — modelled as `WrappedCauchyEmission(output_dim=1)`.
  Free parameters per component: 1 μ + 1 ρ = **2**.

Total emission parameters per component: **4** (used in BIC).
"""
mutable struct StepAngleEmission{GE<:GaussianEmission,WC<:WrappedCauchyEmission} <:
               StateSpaceDynamics.EmissionModel
    step::GE
    angle::WC
    output_dim::Int
end

StepAngleEmission(step::GaussianEmission, angle::WrappedCauchyEmission) =
    StepAngleEmission(step, angle, 2)

function StateSpaceDynamics.loglikelihood(
    model::StepAngleEmission, Y::AbstractMatrix{T}
) where {T<:Real}
    return StateSpaceDynamics.loglikelihood(model.step,  @view Y[:, 1:1]) .+
           StateSpaceDynamics.loglikelihood(model.angle, @view Y[:, 2:2])
end

function StateSpaceDynamics.fit!(
    model::StepAngleEmission, Y::AbstractMatrix{T},
    w::AbstractVector{T} = ones(T, size(Y, 1)),
) where {T<:Real}
    StateSpaceDynamics.fit!(model.step,  @view(Y[:, 1:1]), w)
    StateSpaceDynamics.fit!(model.angle, @view(Y[:, 2:2]), w)
    return model
end

function Random.rand(rng::AbstractRNG, model::StepAngleEmission; n::Int = 1)
    return hcat(rand(rng, model.step; n = n), rand(rng, model.angle; n = n))
end

Random.rand(model::StepAngleEmission; kwargs...) =
    rand(Random.default_rng(), model; kwargs...)

function StateSpaceDynamics.ACDC._emission_to_driver!(
    ε_out::AbstractVector{T}, emission::StepAngleEmission,
    y::AbstractVector{T}, ::Nothing,
) where {T<:Real}
    StateSpaceDynamics.ACDC._emission_to_driver!(
        @view(ε_out[1:1]), emission.step,  @view(y[1:1]), nothing)
    StateSpaceDynamics.ACDC._emission_to_driver!(
        @view(ε_out[2:2]), emission.angle, @view(y[2:2]), nothing)
end

# =============================================================================
# CardioidEmission: unimodal circular emission with bounded concentration
# =============================================================================

"""
    CardioidEmission

Cardioid emission model: `f(θ; μ, ρ) = (1/2π)(1 + 2ρ cos(θ − μ))`.

- μ: mean direction (radians)
- ρ: concentration, ρ ∈ [0, 0.5]

Unimodal on the circle. At ρ=0 it is uniform; at ρ=0.5 it touches zero
at the antipode. Unlike WrappedCauchy (bimodal at low ρ), the cardioid
is strictly unimodal for all ρ > 0.

MLE is fully closed-form: μ = circular mean, ρ = min(R̄, 0.5).

Free parameters per dimension: 1 μ + 1 ρ = **2**.
"""
mutable struct CardioidEmission{T<:Real,V<:AbstractVector{T}} <: StateSpaceDynamics.EmissionModel
    output_dim::Int
    μ::V   # mean direction(s), radians
    ρ::V   # concentration(s), 0 ≤ ρ ≤ 0.5
end

function CardioidEmission(; output_dim::Int, μ::AbstractVector{T}, ρ::AbstractVector{T}) where {T<:Real}
    @assert length(μ) == output_dim && length(ρ) == output_dim
    @assert all(0 .<= ρ .<= 0.5) "ρ must be in [0, 0.5]"
    return CardioidEmission(output_dim, μ, ρ)
end

function StateSpaceDynamics.loglikelihood(
    model::CardioidEmission, Y::AbstractMatrix{T}
) where {T<:Real}
    n_obs, d = size(Y)
    ll = zeros(T, n_obs)
    log2pi = log(T(2π))
    @inbounds for i in 1:n_obs
        for j in 1:d
            # f(θ) = (1/2π)(1 + 2ρ cos(θ − μ))
            val = one(T) + T(2) * model.ρ[j] * cos(Y[i, j] - model.μ[j])
            ll[i] += log(max(val, eps(T))) - log2pi
        end
    end
    return ll
end

function StateSpaceDynamics.fit!(
    model::CardioidEmission, Y::AbstractMatrix{T},
    w::AbstractVector{T} = ones(T, size(Y, 1)),
) where {T<:Real}
    d = size(Y, 2)
    wsum = sum(w)
    eps_T = eps(T)
    for j in 1:d
        s = sum(w .* sin.(@view(Y[:, j])))
        c = sum(w .* cos.(@view(Y[:, j])))
        model.μ[j] = atan(s, c)
        R_bar = sqrt(s^2 + c^2) / max(wsum, eps_T)
        model.ρ[j] = clamp(R_bar, zero(T), T(0.5))
    end
    return model
end

function Random.rand(rng::AbstractRNG, model::CardioidEmission; n::Int = 1)
    d = model.output_dim
    T_el = eltype(model.μ)
    samples = Matrix{T_el}(undef, n, d)
    for j in 1:d
        μ_j, ρ_j = model.μ[j], model.ρ[j]
        # Rejection sampling: envelope is uniform on [-π, π), max density = (1+2ρ)/2π
        M = one(T_el) + T_el(2) * ρ_j
        for i in 1:n
            while true
                θ = T_el(2π) * rand(rng, T_el) - T_el(π)
                f = one(T_el) + T_el(2) * ρ_j * cos(θ - μ_j)
                if rand(rng, T_el) * M <= f
                    samples[i, j] = θ
                    break
                end
            end
        end
    end
    return samples
end

Random.rand(model::CardioidEmission; kwargs...) =
    rand(Random.default_rng(), model; kwargs...)

"""
CDF of the cardioid distribution: F(θ; μ, ρ) = (θ − (−π))/2π + (ρ/π)(sin(θ−μ) − sin(−π−μ)).
Normalized so F(−π)=0, F(π)=1.
"""
function _cardioid_cdf(θ::T, μ::T, ρ::T) where {T<:Real}
    # F(θ) = (θ + π)/(2π) + (ρ/π)(sin(θ − μ) + sin(μ + π))
    # Since sin(μ + π) = −sin(μ):
    return (θ + T(π)) / T(2π) + (ρ / T(π)) * (sin(θ - μ) - sin(μ))
end

function StateSpaceDynamics.ACDC._emission_to_driver!(
    ε_out::AbstractVector{T}, emission::CardioidEmission,
    y::AbstractVector{T}, ::Nothing,
) where {T<:Real}
    for d in eachindex(ε_out)
        ε_out[d] = clamp(_cardioid_cdf(y[d], emission.μ[d], emission.ρ[d]), eps(T), one(T) - eps(T))
    end
end

# =============================================================================
# StepAngleCardEmission: Gaussian (log-step) ⊕ Cardioid (turn angle)
# =============================================================================

"""
    StepAngleCardEmission

Composite emission: Gaussian on z-scored log step length + Cardioid on turn angle.
Unimodal circular model with max concentration ρ=0.5.
Free parameters per component: 2 (Gaussian) + 2 (Cardioid) = **4**.
"""
mutable struct StepAngleCardEmission{GE<:GaussianEmission,CD<:CardioidEmission} <:
               StateSpaceDynamics.EmissionModel
    step::GE
    angle::CD
    output_dim::Int
end

StepAngleCardEmission(step::GaussianEmission, angle::CardioidEmission) =
    StepAngleCardEmission(step, angle, 2)

function StateSpaceDynamics.loglikelihood(
    model::StepAngleCardEmission, Y::AbstractMatrix{T}
) where {T<:Real}
    return StateSpaceDynamics.loglikelihood(model.step,  @view Y[:, 1:1]) .+
           StateSpaceDynamics.loglikelihood(model.angle, @view Y[:, 2:2])
end

function StateSpaceDynamics.fit!(
    model::StepAngleCardEmission, Y::AbstractMatrix{T},
    w::AbstractVector{T} = ones(T, size(Y, 1)),
) where {T<:Real}
    StateSpaceDynamics.fit!(model.step,  @view(Y[:, 1:1]), w)
    StateSpaceDynamics.fit!(model.angle, @view(Y[:, 2:2]), w)
    return model
end

function Random.rand(rng::AbstractRNG, model::StepAngleCardEmission; n::Int = 1)
    return hcat(rand(rng, model.step; n = n), rand(rng, model.angle; n = n))
end

Random.rand(model::StepAngleCardEmission; kwargs...) =
    rand(Random.default_rng(), model; kwargs...)

function StateSpaceDynamics.ACDC._emission_to_driver!(
    ε_out::AbstractVector{T}, emission::StepAngleCardEmission,
    y::AbstractVector{T}, ::Nothing,
) where {T<:Real}
    StateSpaceDynamics.ACDC._emission_to_driver!(
        @view(ε_out[1:1]), emission.step,  @view(y[1:1]), nothing)
    StateSpaceDynamics.ACDC._emission_to_driver!(
        @view(ε_out[2:2]), emission.angle, @view(y[2:2]), nothing)
end
