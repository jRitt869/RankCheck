"""The original Monte Carlo loop with settings supplied by the caller."""

from typing import TYPE_CHECKING

import torch

from .model import NeuralVectorField
from .numerics import (
    compute_jac,
    compute_n_ball_volume,
    eval_funcs,
    generate_random_parameters,
    numerical_rank_from_svals,
    sample_uniform_ball,
)

if TYPE_CHECKING:
    from experiment_tools.config import ExperimentSettings


def resolve_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu or auto")
    return (
        "cuda"
        if requested == "auto" and torch.cuda.is_available()
        else ("cpu" if requested == "auto" else requested)
    )


def run_monte_carlo_experiment(settings: "ExperimentSettings | None" = None) -> tuple[list, ...]:
    """Retain the six-list return format, sampling laws, Euler scheme, and scaling."""
    if settings is None:
        from experiment_tools.config import load_settings

        settings = load_settings()
    settings.validate()
    device = resolve_device(settings.device)
    torch.manual_seed(settings.seed)
    print(f"Using device: {device}; float64; seed: {settings.seed}", flush=True)
    volume = compute_n_ball_volume(settings.rho_val, settings.n_dim).to(device)
    volume_aug = (
        compute_n_ball_volume(settings.rho_val, settings.n_dim - 1).to(device)
        if settings.compute_aug
        else None
    )
    if (
        not torch.isfinite(volume).item()
        or volume.item() <= 0
        or (
            volume_aug is not None
            and (not torch.isfinite(volume_aug).item() or volume_aug.item() <= 0)
        )
    ):
        raise ValueError("The domain volume is outside float64 range; reduce dimension or radius")
    dt = torch.tensor(settings.t_end / settings.steps, device=device, dtype=torch.float64)
    model = NeuralVectorField(
        settings.n_dim, settings.depth_val, settings.latent_val, dtype=torch.float64
    ).to(device=device, dtype=torch.float64)
    model_gelu = (
        NeuralVectorField(
            settings.n_dim,
            settings.depth_val,
            settings.latent_val,
            activation=False,
            dtype=torch.float64,
        ).to(device=device, dtype=torch.float64)
        if settings.compute_gelu
        else None
    )
    samples = sample_uniform_ball(
        settings.num_samples, settings.n_dim, settings.rho_val, device=device
    )
    samples_ff = sample_uniform_ball(
        settings.num_samples_ff, settings.n_dim, settings.rho_val * 2, device=device
    )
    samples_aug = None
    if settings.compute_aug:
        embedded = sample_uniform_ball(
            settings.num_samples, settings.n_dim - 1, settings.rho_val, device=device
        )
        samples_aug = torch.cat([embedded, embedded.new_zeros((len(embedded), 1))], dim=1)
    step, ff = eval_funcs(model, compile_steps=settings.compile)
    step_gelu, ff_gelu = (
        eval_funcs(model_gelu, compile_steps=settings.compile)
        if model_gelu is not None
        else (None, None)
    )
    parameters = generate_random_parameters(model, settings.iterations, device)
    flow, aug, ranks_ff, flow_gelu, aug_gelu, ranks_ff_gelu = [], [], [], [], [], []

    def spectrum(field, var_step, params, initial, measure):
        jacobian = compute_jac(
            lambda x, jx, jp: var_step(x, jx, jp, params, dt),
            field,
            params,
            initial,
            t_end=settings.t_end,
            steps=settings.steps,
            N=settings.segments,
        )
        scale = torch.sqrt(
            torch.tensor(settings.num_samples, device=device, dtype=torch.float64) / measure
        )
        values = torch.linalg.svdvals(jacobian / scale)
        if not torch.isfinite(values).all().item():
            raise ValueError(
                "Nonfinite singular values; inspect Euler step size and parameter scales"
            )
        return values

    def thresholded(values):
        threshold = max(settings.atol, settings.rtol * values.max().item())
        return (values * (values > threshold)).detach().cpu().numpy()

    def ff_rank(function, params):
        jacobian = function(samples_ff, params).flatten(0, 1)
        scale = torch.sqrt(
            torch.tensor(settings.num_samples_ff, device=device, dtype=torch.float64)
            / (2**settings.n_dim * volume)
        )
        return numerical_rank_from_svals(
            torch.linalg.svdvals(jacobian / scale), settings.rtol, settings.atol
        )

    for iteration in range(settings.iterations):
        params = {key: parameter[iteration] for key, parameter in parameters.items()}
        flow.append(thresholded(spectrum(model, step, params, samples, volume)))
        if samples_aug is not None:
            aug.append(thresholded(spectrum(model, step, params, samples_aug, volume_aug)))
        if model_gelu is not None:
            # Keep the original unthresholded GELU spectra; reporting applies rank tolerances.
            flow_gelu.append(
                spectrum(model_gelu, step_gelu, params, samples, volume).detach().cpu().numpy()
            )
            if samples_aug is not None:
                aug_gelu.append(
                    spectrum(model_gelu, step_gelu, params, samples_aug, volume_aug)
                    .detach()
                    .cpu()
                    .numpy()
                )
        if device == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        ranks_ff.append(ff_rank(ff, params))
        if model_gelu is not None:
            ranks_ff_gelu.append(ff_rank(ff_gelu, params))
        if device == "cuda" and settings.compile:
            torch.cuda.empty_cache()
            torch.compiler.reset()
        print(f"step: {iteration + 1} / {settings.iterations}", flush=True)
    return flow, aug, ranks_ff, flow_gelu, aug_gelu, ranks_ff_gelu
