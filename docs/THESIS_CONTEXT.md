## Research question and theoretical connection

The thesis considers the autonomous ODE `dx/dt = h_θ(x)`, where `h_θ` is a feedforward ReLU network with time-independent parameters. Its flow `Φ_θ(t, x)` induces a realization map from a finite-dimensional parameter space into an L² function space. Local parametric redundancy measures the dimension of nearby parameter sets representing the same function.

The main result establishes continuous differentiability and locally constant derivative rank on an open, dense set of full measure under its boundedness, subanalyticity, and analytic-density assumptions. Where the continuous differentiability and constant-rank hypotheses hold, the Constant Rank Theorem identifies the local level-set dimension with the derivative's kernel dimension. For the experiments' `r` network parameters, this gives:

$$
\text{local redundancy dimension} = r - \text{rank} Df(\theta).
$$

This motivates approximating the derivative's rank and singular spectrum over randomly sampled parameter vectors. Numerical ranks depend on input sampling, integration error, and a singular-value tolerance; their interpretation relies on these approximations as well as the main results local hypotheses.

## Comparison convention

The general augmented map is defined as

$$
f_{\mathrm{aug}}(\theta)
= [x \mapsto V_1\Phi_\theta(1,V_2x)],
$$

with **fixed** output matrix `V₁` and input embedding `V₂`. With `V₁ = I_n`, this can be compared to the non-augmented map

$$
\widehat f(\theta) = [x \mapsto \Phi_\theta(1,x)].
$$

The experiment therefore changes which initial states are observed while keeping the governing dynamics and parameter vector fixed:

| Quantity | Non-augmented | Augmented |
| --- | --- | --- |
| Vector field and ODE state | Same `h_θ: Rⁿ → Rⁿ` | Same `h_θ: Rⁿ → Rⁿ` |
| Differentiated parameters | Network parameters `θ` | The same network parameters `θ` |
| Input domain in this script | Radius-`ρ` ball in `Rⁿ` | Radius-`ρ` ball in `Rⁿ⁻¹` |
| Initial state | `x` | `(x, 0)` |
| Output | Full state `Φ_θ(1, x)` | Full state `Φ_θ(1, (x, 0))` |
| Integral measure | Lebesgue volume of the `n`-dimensional ball | Lebesgue volume of the `(n-1)`-dimensional ball |

The thesis uses `n = 2`, augmented physical input dimension `p = 1`, and `V₂ = (1, 0)ᵀ`: a disk is compared with an interval embedded in the same two-dimensional state space. The script extends this convention to `n_dim` by appending one zero coordinate. Neither the embedding nor the identity output map contributes trainable parameters.

The empirical question is whether this domain restriction creates additional parameter redundancies beyond those already present in the vector field. Separate volume normalization approximates each realization map in its own L² space; singular-value magnitudes consequently reflect different domain measures as well as the dynamics.

Parameter directions leaving the vector field unchanged belong to both realization derivatives' kernels almost everywhere. If their dimension is `d(θ)`, both ranks are bounded by `r - d(θ)`. This bound can be estimated using the rank of a sampled feedforward-network Jacobian. The exported `rank_ff` is this **finite-sample numerical proxy**; sampling and tolerance mean it is not a certified upper bound for every computed flow rank.

## Experiments and observations

The experiments use fan-in-dependent uniform parameter sampling, without parameter training, and compare four configurations alongside a standard feed-forward one: RN (non-augmented ReLU), RA (augmented ReLU), GN (non-augmented GELU), and GA (augmented GELU). Flow-domain radius `ρ = 50` is chosen based on numerical tests to reduce effects from unvisited ReLU regions.

For the deeper ReLU networks, the results show that RN usually attains the vector-field rank bound, while RA often has lower rank, suggesting additional local redundancy. For the sampled GELU configurations, a full numerical parameter rank can be observed. The one-hidden-layer comparison shows nearly coincident RN/RA mean ranks, so the observed augmentation effect depends on the architecture.

The proposed explanation is that ReLU parameter perturbations can affect only particular piecewise-affine regions. Initial states restricted to an embedded subspace may generate trajectories that never visit some of those regions, leaving further parameter directions unobservable. The thesis presents this as a plausible mechanism for the numerical observations.

## Numerical approximation and spectral summary

The outputs summarize spectra through eleven averaged percentile coordinates. The supplied script's `plot.csv` uses mean index percentiles paired with reversed mean **natural-log singular-value percentiles**, and the current plot preserves that calculation. The rightmost index percentile equals rank for each positive-rank spectrum, so its average shows mean numerical rank. See the [README's output description](../README.md#outputs-and-figures) for the exact logarithmic convention and how zero-rank samples are handled. The additional `mean_spectrum.csv` contains fixed-index arithmetic averages as a separate export.

## Reproduction details

The thesis context documents the original research contribution. Configuration, reporting, plots, and regression checks added during maintenance remain identified separately in [maintenance notes](REFACTORING.md).
