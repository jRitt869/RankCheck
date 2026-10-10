import math

import numpy as np
import pytest
import torch
from torch.func import functional_call, jacrev

from experiment_tools.config import load_settings
from rank_simulations.experiment import run_monte_carlo_experiment
from rank_simulations.model import NeuralVectorField
from rank_simulations.numerics import (
    compute_jac,
    compute_n_ball_volume,
    eval_funcs,
    solve_ode_fixed_step,
)


@pytest.mark.parametrize("segments", [1, 2, 4])
@pytest.mark.parametrize("activation", [True, False])
def test_composed_variational_jacobian_matches_direct_autodiff(
    segments: int, activation: bool
) -> None:
    torch.manual_seed(7)
    model = NeuralVectorField(2, 1, 2, activation=activation).double()
    parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    initial = torch.tensor([[0.1, 0.2], [0.4, -0.1], [-0.3, 0.2]], dtype=torch.float64)
    dt = initial.new_tensor(0.25)
    step, _ = eval_funcs(model, compile_steps=False)
    composed = compute_jac(
        lambda x, jx, jp: step(x, jx, jp, parameters, dt),
        model,
        parameters,
        initial,
        steps=4,
        N=segments,
    )

    def final_state(params):
        return solve_ode_fixed_step(
            lambda p, x: functional_call(model, p, x), params, initial, steps=4
        )[0]

    direct = jacrev(final_state)(parameters)
    flattened = torch.cat([value.flatten(2) for value in direct.values()], dim=-1).flatten(0, 1)
    torch.testing.assert_close(composed, flattened, rtol=1e-11, atol=1e-12)
    assert composed.dtype == torch.float64


@pytest.mark.parametrize("dimension,expected", [(1, 4), (2, 4 * math.pi), (3, 32 * math.pi / 3)])
def test_ball_volume_matches_known_domains(dimension: int, expected: float) -> None:
    assert compute_n_ball_volume(2, dimension).item() == pytest.approx(expected)


def test_seeded_simulation_does_not_depend_on_or_change_default_dtype() -> None:
    settings = load_settings(
        overrides={
            "num_samples": 4,
            "num_samples_ff": 4,
            "depth_val": 1,
            "latent_val": 2,
            "rho_val": 0.5,
            "steps": 2,
            "segments": 1,
            "iterations": 1,
            "device": "cpu",
            "compile": False,
        }
    )
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        first = run_monte_carlo_experiment(settings)
        assert torch.get_default_dtype() == torch.float32
        torch.set_default_dtype(torch.float64)
        second = run_monte_carlo_experiment(settings)
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(previous)
    for first_values, second_values in zip(first, second, strict=True):
        np.testing.assert_array_equal(np.asarray(first_values), np.asarray(second_values))
