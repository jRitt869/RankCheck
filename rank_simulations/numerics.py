"""Euler integration, variational Jacobians, and the original sampling laws."""

from collections.abc import Callable

import torch
from torch import nn
from torch.func import functional_call, jacrev, vmap

Parameters = dict[str, torch.Tensor]


def solve_ode_fixed_step(
    func: Callable, p: Parameters, x0: torch.Tensor, t_end: float = 1.0, steps: int = 20
) -> tuple[torch.Tensor, torch.Tensor]:
    """Explicit Euler, retaining the vector-field evaluation points."""
    dt = x0.new_tensor(t_end / steps)
    x, points = x0, []
    for _ in range(steps):
        points.append(x)
        x = x + dt * func(p, x)
    return x, torch.stack(points)


def sample_uniform_ball(
    num_samples: int, n: int, rho: float, device: str = "cpu", dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    """Uniform directions with radii rho * U**(1/n), as in the thesis script."""
    directions = torch.randn(num_samples, n, device=device, dtype=dtype)
    directions = directions / torch.linalg.vector_norm(directions, dim=1, keepdim=True)
    radii = torch.rand(num_samples, 1, device=device, dtype=dtype).pow(1 / n) * rho
    return directions * radii


def compute_n_ball_volume(rho_val: float, n_dim: int) -> torch.Tensor:
    """Preserve the even/odd ball-volume formula in explicit float64."""
    if n_dim % 2 == 0:
        k = n_dim // 2
        denominator = torch.prod(torch.arange(1, k + 1, dtype=torch.float64))
        return torch.pi**k / denominator * rho_val**n_dim
    k = (n_dim - 1) // 2
    denominator = torch.prod(2 * torch.arange(1, k + 1, dtype=torch.float64) + 1)
    return 2 ** (k + 1) * torch.pi**k / denominator * rho_val**n_dim


def eval_funcs(model: nn.Module, *, compile_steps: bool = True) -> tuple[Callable, Callable]:
    """Original jacrev/vmap sensitivity steps; CUDA graph compilation is optional."""

    def euler_variational_step(x_curr, J_x, J_p, params, dt):
        df_dx = jacrev(lambda x: functional_call(model, params, x))(x_curr)
        df_dp_dict = jacrev(lambda p: functional_call(model, p, x_curr))(params)
        df_dp = torch.cat([value.flatten(1) for value in df_dp_dict.values()], dim=-1)
        return J_x + dt * df_dx @ J_x, J_p + dt * (df_dx @ J_p + df_dp)

    step = vmap(
        vmap(euler_variational_step, in_dims=(0, 0, 0, None, None)), in_dims=(0, 0, 0, None, None)
    )
    if compile_steps and next(model.parameters()).is_cuda:
        step = torch.compile(step, backend="cudagraphs", fullgraph=True)

    def ff_func(x, params):
        derivative = jacrev(lambda p: functional_call(model, p, x))(params)
        return torch.cat([value.flatten(1) for value in derivative.values()], dim=-1)

    return step, vmap(ff_func, in_dims=(0, None))


def compute_jac(
    compiled_var_step: Callable,
    model: nn.Module,
    params: Parameters,
    x_0: torch.Tensor,
    t_end: float = 1.0,
    steps: int = 4000,
    N: int = 2,
) -> torch.Tensor:
    """Compose the original N segment sensitivities into a parameter Jacobian.

    Rows concatenate the full state output for every input sample. No augmented
    output coordinate is projected away, matching the supplied thesis script.
    """
    if steps < 1 or N < 1 or steps % N:
        raise ValueError("steps must be positive and divisible by the positive segment count N")
    batch_size, state_dim = x_0.shape
    dt = x_0.new_tensor(t_end / steps)
    steps_per_segment = steps // N
    num_params = sum(parameter.numel() for parameter in params.values())
    segment_starts = x_0.new_zeros((N, batch_size, state_dim))
    x = x_0.clone()
    with torch.no_grad():
        for segment in range(N):
            segment_starts[segment] = x
            for _ in range(steps_per_segment):
                x = x + dt * functional_call(model, params, x)

    J_x = (
        torch.eye(state_dim, device=x_0.device, dtype=x_0.dtype)
        .expand(N, batch_size, state_dim, state_dim)
        .clone()
    )
    J_p = x_0.new_zeros((N, batch_size, state_dim, num_params))
    x_var = segment_starts.clone()
    batched_field = vmap(vmap(lambda state: functional_call(model, params, state)))
    for _ in range(steps_per_segment):
        with torch.no_grad():
            if x_0.is_cuda:
                torch.compiler.cudagraph_mark_step_begin()
            J_x, J_p = compiled_var_step(x_var.clone(), J_x.clone(), J_p.clone())
            x_var = x_var + dt * batched_field(x_var)

    total = J_p[0]
    for segment in range(1, N):
        total = J_p[segment] + torch.bmm(J_x[segment], total)
    return total.flatten(0, 1)


def numerical_rank_from_svals(svals: torch.Tensor, rtol: float = 1e-12, atol: float = 1e-10) -> int:
    threshold = max(atol, float(rtol * svals.max().item()))
    return int((svals > threshold).sum().item())


def get_parameter_initialization_scale(key: str, params_ref: Parameters) -> torch.Tensor:
    """Use 1/sqrt(fan-in) for weights and their corresponding biases."""
    parts = key.split(".")
    weight = params_ref[key] if parts[2] == "weight" else params_ref[f"net.{parts[1]}.weight"]
    return weight.new_tensor(1 / weight.shape[1]).sqrt()


def generate_random_parameters(model: nn.Module, iterations: int, device: str) -> Parameters:
    reference = {key: parameter.detach().clone() for key, parameter in model.named_parameters()}
    return {
        key: (torch.rand((iterations, *parameter.shape), device=device, dtype=torch.float64) - 0.5)
        * (2 * get_parameter_initialization_scale(key, reference))
        for key, parameter in reference.items()
    }
