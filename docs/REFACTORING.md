# Maintenance notes

The original script is preserved byte-for-byte as `archive/numerical_experiments.py` (SHA-256 `e61b96b94abc448159b2d870f1eec16436cd218d13b98525a6a344cde39a3dc3`). Its vector field and numerical methods live in `rank_simulations/`; later configuration, CLI, reporting, and plotting additions live in `experiment_tools/`.

Preserved: network architecture, ReLU/GELU functions, uniform ball sampling, fan-in parameter sampling, shared samples across iterations, explicit Euler, segmented state/parameter sensitivities, segment composition, full-state augmented outputs, direct Jacobian SVD, volume normalization, original large defaults, and the six-list return format. Named public helper imports remain available through `numerical_experiments.py`.

Intentional maintenance changes:

- Move editable constants into `defaults.toml`; allow partial TOML files and optional CLI overrides.
- Expose the original hardcoded segment count of ten and reject incomplete integrations when it does not divide the step count.
- Make tensor dtype/device explicit to retain float64 without relying on a global default dtype.
- Run the original variational method eagerly on CPU and retain optional CUDA graph compilation on CUDA.
- Introduce an explicit seed and record effective settings, versions, source hashes, and runtime.
- Add non-augmented flow ranks to the original rank table; retain augmented and feedforward columns.
- Retain the original percentile/log CSV and add an arithmetic mean at each singular index as an additional data export.
- Save the rank histogram and the original mean percentile/log curve automatically, and support regeneration from saved tables. The curve's x endpoint gives mean rank; its y coordinates are already natural logarithms.
- Export requested GELU results and apply the documented rank tolerance when reporting them.
- Handle zero-rank spectra without taking the logarithm of an empty/zero array.
- Use PyTorch's public compiler reset API in place of the original private reset call.

Verification: 29 tests pass on Python 3.12 with PyTorch 2.14.1+cpu, including independent direct-autodiff comparisons for both activations and three segment counts, ambient-dtype-independent seeded sampling, configuration validation, zero ranks, and GELU export. The plot regression verifies both mean percentile coordinates, mean-rank endpoints for differing ranks, and regeneration without a fixed-index mean table. Ruff checks pass and pip reports no broken dependencies. The small 12-iteration CPU demo generated both plots, which were visually inspected; its manifest's source checksums match the maintained code.
