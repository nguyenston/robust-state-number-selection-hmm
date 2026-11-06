# ACDC — Per-Component KL Discrepancy

This document describes how the per-component value `D_KL` reported on the
y-axis of the ACDC envelope panel ("max per-component") is computed.
The full pipeline lives in [src/ACDC/](../../StateSpaceDynamics.jl/src/ACDC/) — primarily
[ACDCInterface.jl](../../StateSpaceDynamics.jl/src/ACDC/ACDCInterface.jl) and
[ACDCAdapters.jl](../../StateSpaceDynamics.jl/src/ACDC/ACDCAdapters.jl).

The discrepancy answers: *for each fitted component k, how far is the
distribution of that component's stochastic drivers from the reference
uniform ``U([0,1]^D)``?* Under correct specification of the emission
family **and** correct K, every component's drivers should be uniform on
the unit hypercube. Any departure is a fit defect attributable to
component k. The envelope plot summarises this across K by reporting
``\max_k D_{KL,k}`` — the worst-fit component at each K.

---

## Pipeline overview

For a single fitted model at a given K:

```
data y_t              ┐
posterior γ_{k,t}     │
emission B_k          ├──► stochastic_drivers ──► ε_pools[k] ∈ [0,1]^{D × n_k}
n_samples replicates  ┘                            (one matrix per component)

ε_pools[k]  ──► probit transform ──► k-NN density estimate ──► D_KL,k
```

Then per K the envelope uses ``\max_k D_{KL,k}``; across K the monotone
envelope is extracted and plotted (see [envelope_plots.md](envelope_plots.md)).

---

## Step 1 — Stochastic drivers per component

The per-component driver pool ``\varepsilon_{(k)} \in [0,1]^{D \times n_k}``
is built in
[ACDCAdapters.jl::stochastic_drivers(::HiddenMarkovModel, ...)](../../StateSpaceDynamics.jl/src/ACDC/ACDCAdapters.jl).

For each timestep t and each replicate s ∈ 1..n_samples:

1. Compute posterior class probabilities ``\gamma_{k,t} = P(z_t = k \mid y_{1:T})``
   via forward-backward (`StateSpaceDynamics.class_probabilities`).
2. Sample an "active" state ``z_t \sim \mathrm{Cat}(\gamma_{\cdot,t})``.
3. Apply the emission-family CDF chain to ``y_t`` under emission ``B_{z_t}``:
   - **Gaussian**: whiten via Cholesky, ``\eta = L^{-1}(y_t - \mu)``,
     then ``\varepsilon_d = \Phi(\eta_d)`` per dimension
     ([_emission_to_driver!::GaussianEmission](../../StateSpaceDynamics.jl/src/ACDC/ACDCAdapters.jl#L484)).
   - **Poisson**: randomized discrete CDF
     ``\varepsilon_d = F_{\lambda}(y_d - 1) + u \cdot (F_{\lambda}(y_d) - F_{\lambda}(y_d - 1))``,
     ``u \sim U(0,1)``, restoring continuity over the count support
     ([_poisson_cdf_randomized](../../StateSpaceDynamics.jl/src/ACDC/ACDCAdapters.jl#L72)).
   - **Bernoulli**, **WrappedCauchy**, **Beta**, etc.: their respective
     CDFs. The randomized step is used for any discrete component.
4. Push the resulting ``\varepsilon \in [0,1]^D`` into the driver pool for
   the sampled state: `ε_lists[z_t]` gets one column.

After the full sweep, ``\varepsilon_{(k)}`` is a `D × n_k` matrix with
``n_k \approx \mathrm{usage}_k \cdot N \cdot n_{\mathrm{samples}}``, and
``\mathrm{usage}_k = \frac{1}{N} \sum_t \gamma_{k,t}`` is reported
alongside in `StochasticDriverResult.usage`.

**Why the CDF transform?** If the data really came from emission ``B_k``,
then ``F_{B_k}(y_t)`` is uniform on ``[0,1]^D`` (probability integral
transform). Misfit shows up as departure from uniformity in
``\varepsilon_{(k)}``. The transformation is component-specific —
different components can fail in different ways, and each pool isolates
its own component's misfit.

**Multi-trial HMMs**: when `data` is a vector of trials, the per-trial
pools are concatenated by state and `usage` is the time-weighted average
across trials.

---

## Step 2 — KL discrepancy from ``\varepsilon_{(k)}``

For each component k independently, the per-pool discrepancy is computed
in [compute_discrepancy(::KLDiscrepancy, samples)](../../StateSpaceDynamics.jl/src/ACDC/ACDCInterface.jl#L317).

We want ``D_{KL}(P \,\|\, U([0,1]^D))`` where ``P`` is the empirical
distribution of the columns of ``\varepsilon_{(k)}``. Direct k-NN
density estimation on a bounded hypercube suffers from boundary bias
(samples near 0 or 1 have asymmetric neighborhoods). To sidestep this,
we apply the **probit transform** to the entire reference frame
simultaneously:

### 2a. Probit transform

```
ε̃ = clamp(ε, eps, 1 - eps)        # avoid -Inf/+Inf at the edges
z  = Φ^{-1}(ε̃)                     # samples now in R^D
```

Under the null (drivers truly uniform), ``z`` is then standard normal
``\mathcal{N}(0, I_D)``. The reference distribution changes from
``U([0,1]^D)`` to ``\mathcal{N}(0, I_D)``, and the KL divergence is
invariant under invertible transformations, so

```math
D_{KL}\big( P \,\|\, U([0,1]^D) \big)
  = D_{KL}\big( \Phi^{-1}_\sharp P \,\|\, \mathcal{N}(0, I_D) \big)
```

But the right-hand side estimator can run on an unbounded domain with no
edge effects.

### 2b. Decompose into two cross-entropies

```math
D_{KL}(P \,\|\, Q) = \mathbb{E}_P[\log p] - \mathbb{E}_P[\log q]
```

- ``\mathbb{E}_P[\log p]`` (entropy term) is estimated from samples
  using a k-NN density estimator.
- ``\mathbb{E}_P[\log q]`` with ``Q = \mathcal{N}(0, I_D)`` is computed
  in closed form from the second moment of the samples.

### 2c. ``\mathbb{E}_P[\log p]`` — Kozachenko-Leonenko k-NN estimator

Implemented in
[_mean_log_pdf_knn_multivariate](../../StateSpaceDynamics.jl/src/ACDC/ACDCInterface.jl#L587).

For each sample ``z_i``, let ``\rho_k(z_i)`` be the Euclidean distance
to its k-th nearest neighbor (excluding itself). Then

```math
\log \hat p(z_i)
  = \psi(k) - \psi(N) - \log c_D - D \cdot \log \rho_k(z_i)
```

where:
- ``\psi`` is the digamma function (bias correction for the finite-sample
  k-NN density estimate)
- ``c_D = \pi^{D/2} / \Gamma(D/2 + 1)`` is the volume of the unit ball
  in ``\mathbb{R}^D``
- ``N`` is the pool size (``n_k`` columns in ``\varepsilon_{(k)}``)
- ``k`` defaults to 5 (`KLDiscrepancy(; k_neighbors=5)`)

The k-NN search uses a `KDTree` (`NearestNeighbors.jl`) for
``O(N \log N)`` complexity. ``\mathbb{E}_P[\log p]`` is the average of
``\log \hat p(z_i)`` across the pool.

### 2d. ``\mathbb{E}_P[\log q]`` under ``\mathcal{N}(0, I_D)``

For ``q(z) = (2\pi)^{-D/2} \exp(-\|z\|^2 / 2)``:

```math
\mathbb{E}_P[\log q]
  = -\tfrac{1}{2}\Big( D \log(2\pi) + \mathbb{E}_P[\|z\|^2] \Big)
```

So we just need the sample mean of ``\sum_d z_{i,d}^2`` across the pool —
one inner product per sample, computed in closed form (no estimator).

### 2e. Final value

```math
\hat D_{KL,k} = \max\Big(0,\; \widehat{\mathbb{E}_P[\log p]} - \mathbb{E}_P[\log q]\Big)
```

The `max(0, ·)` clip guards against k-NN noise pushing the estimate
slightly negative for very-near-uniform pools (KL is non-negative in
truth; only the estimator can be negative on finite samples).

---

## Step 3 — Per-component result

`component_discrepancies` returns an
[ACDCResult](../../StateSpaceDynamics.jl/src/ACDC/ACDCInterface.jl#L78) carrying:

```julia
struct ACDCResult{T<:Real}
    K::Int
    component_discrepancies::Vector{T}   # length K, the D_KL,k values
    usage::Vector{T}                     # length K, time-weighted occupancy
end
```

The plots use `maximum(component_discrepancies)` as the per-K envelope
value. The `usage` field is *not* used to weight the discrepancy — a
rarely-visited component contributes its full D_KL regardless of how
often it fires. This is intentional: rare components are more likely to
expose misfit (they're "harder" to fit because they have less data), and
ACDC's selection signal comes from the worst component, not the
average.

If you want a usage-weighted aggregate (e.g. an "expected per-bin
contribution to misfit"), it would be
``\sum_k \mathrm{usage}_k \cdot D_{KL,k}`` — but that's not what's
plotted.

---

## Choice of k

`k_neighbors = 5` is the default. Trade-offs:

- **Smaller k (k=1..3)**: lower bias on smooth densities, but high
  variance — single-neighbor distances are noisy. Tends to inflate
  ``D_{KL}`` estimates on small pools.
- **Larger k (k=10..20)**: lower variance, but more bias on densities
  with structure on scale ``< \rho_k``. The k-NN estimator effectively
  smooths over a ball of radius ``\rho_k``.

For pool sizes ``n_k \sim 10^3`` and ``D \le 5``, k=5 sits in the
sweet spot. For very small pools (n_k < 30) the estimator becomes
unreliable; the code returns `Inf` if ``n_k < k+1`` (warned at runtime).

---

## Reading the envelope y-axis

Concretely, a value of `D_KL,k = 0.7` on the envelope panel means:
for that K's worst-fit component, the k-NN density estimate of the
driver pool diverges from a standard normal by ``\approx 0.7`` nats
in KL.

- `D_KL ≈ 0` (typically ≤ 0.05): component drivers are visually
  indistinguishable from N(0, I). The component is a clean fit.
- `D_KL ≈ 0.1–0.5`: noticeable departure — usually a mean/variance
  shift, slight tail mismatch, or one neuron / feature being off.
- `D_KL > 1`: gross mismatch — the component is probably absorbing
  data from a different emission shape than what it was parameterised
  for.

These are nats. Multiply by ``\log_2(e) \approx 1.443`` for bits.

---

## What happens at high K

As K increases, the model has more components to allocate. Three
generic patterns:

1. **Clean K=K\***: at the right K, every component fits a coherent
   subset of the data and ``\max_k D_{KL,k}`` drops sharply. This is
   the elbow signal.
2. **Past K=K\* (over-parametrised, structureless)**: extra components
   sit on "leftover" residual structure. Each individual ``D_{KL,k}``
   may drop a little, but the *max* stays roughly flat — there's
   always at least one component absorbing something the model
   architecture can't represent cleanly.
3. **Past K=K\* (over-parametrised, contamination-rich)**: extra
   components carve up genuine off-family contamination clusters. The
   *max* keeps dropping in small increments as the model finds
   coherent contamination sub-modes (see e.g.
   [spike-train statistics](../results/spike_statistics_v2c_cont.txt)).
   The total improvement is small relative
   to the K\* elbow.

The envelope plot encodes (1) as a sharp drop, (2) as a flat tail, and
(3) as a slowly-descending tail with single-digit per-step drops.

---

## Sanity checks

Useful properties to spot-check when interpreting an envelope:

- ``D_{KL,k} = 0`` for all k under correctly-specified synthetic data
  fitted at K=K\* (modulo k-NN noise; expect order ``10^{-2}`` on
  typical pool sizes).
- ``D_{KL,k}`` invariant under (a) Gaussian rotation/scaling of the
  pool — because the probit-transformed reference is rotationally
  symmetric and we use Euclidean distance; (b) re-ordering of state
  labels — components are computed independently.
- ``D_{KL,k}`` decreases monotonically with the pool size at fixed
  fit quality — finite-sample noise has order ``1/n_k`` (bias) plus
  ``1/\sqrt{n_k}`` (variance). Components with rare usage are noisier.

If the envelope shows a `D_KL` floor much higher than expected from the
above (e.g. `0.05` instead of `~10^{-2}`) on synthetic data with
matching emission family, the suspects in order of likelihood are:
sample-size limitations on rare components, probit-clamp eps too
aggressive, or genuine misfit in the emission parametrisation.
