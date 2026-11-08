# =============================================================================
# PoissonEmission: multivariate Poisson with one rate per dimension
# =============================================================================

"""
    PoissonEmission

Multivariate Poisson emission with one rate per dimension (independent across dims).

- `output_dim::Int`: number of neurons / channels
- `λ::Vector{T}`: expected counts per bin, length `output_dim`, λ ≥ 0
"""
mutable struct PoissonEmission{T<:Real,V<:AbstractVector{T}} <: StateSpaceDynamics.EmissionModel
    output_dim::Int
    λ::V
end

function PoissonEmission(; output_dim::Int, λ::AbstractVector{T}) where {T<:Real}
    @assert length(λ) == output_dim
    @assert all(λ .>= 0)
    return PoissonEmission(output_dim, λ)
end

# Log-pmf of Poisson: y·log(λ) − λ − loggamma(y+1)
function StateSpaceDynamics.loglikelihood(
    model::PoissonEmission, Y::AbstractMatrix{T}
) where {T<:Real}
    n_obs, d = size(Y)
    @assert d == model.output_dim
    ll = zeros(T, n_obs)
    eps_T = eps(T)
    @inbounds for i in 1:n_obs
        for j in 1:d
            λ = max(model.λ[j], eps_T)
            y = Y[i, j]
            ll[i] += y * log(λ) - λ - loggamma(y + 1)
        end
    end
    return ll
end

# Weighted-mean MLE for Poisson rate, one per dim
function StateSpaceDynamics.fit!(
    model::PoissonEmission, Y::AbstractMatrix{T},
    w::AbstractVector{T} = ones(T, size(Y, 1)),
) where {T<:Real}
    d = size(Y, 2)
    wsum = sum(w)
    eps_T = eps(T)
    for j in 1:d
        model.λ[j] = max(sum(w .* @view(Y[:, j])) / max(wsum, eps_T), eps_T)
    end
    return model
end

function Random.rand(rng::AbstractRNG, model::PoissonEmission; n::Int = 1)
    d = model.output_dim
    T = eltype(model.λ)
    samples = Matrix{T}(undef, n, d)
    for j in 1:d
        p = Poisson(max(model.λ[j], eps(T)))
        for i in 1:n
            samples[i, j] = T(rand(rng, p))
        end
    end
    return samples
end

Random.rand(model::PoissonEmission; kwargs...) =
    rand(Random.default_rng(), model; kwargs...)

"""
Randomized PIT for a discrete (Poisson) distribution:
    ε = F(y − 1; λ) + u · (F(y; λ) − F(y − 1; λ)),   u ∼ U(0, 1)
with F(−1; λ) = 0.  Yields ε ∼ U(0, 1) if y ∼ Poisson(λ).
"""
function StateSpaceDynamics.ACDC._emission_to_driver!(
    ε_out::AbstractVector{T}, emission::PoissonEmission,
    y::AbstractVector{T}, ::Nothing,
) where {T<:Real}
    for d in eachindex(ε_out)
        λ = max(emission.λ[d], eps(T))
        yd = round(Int, y[d])
        p = Poisson(λ)
        F_y  = cdf(p, yd)
        F_ym1 = yd > 0 ? cdf(p, yd - 1) : zero(T)
        u = rand()
        ε_out[d] = clamp(F_ym1 + u * (F_y - F_ym1), eps(T), one(T) - eps(T))
    end
end
