"""Original entry point, now with TOML settings and optional CLI overrides."""

from importlib import import_module

from experiment_tools.cli import main

# Preserve named public helpers while allowing --help/--show-settings without PyTorch.
_PUBLIC_HELPERS = {
    "NeuralVectorField": "rank_simulations.model",
    "solve_ode_fixed_step": "rank_simulations.numerics",
    "sample_uniform_ball": "rank_simulations.numerics",
    "compute_jac": "rank_simulations.numerics",
    "compute_n_ball_volume": "rank_simulations.numerics",
    "eval_funcs": "rank_simulations.numerics",
    "numerical_rank_from_svals": "rank_simulations.numerics",
    "get_parameter_initialization_scale": "rank_simulations.numerics",
    "generate_random_parameters": "rank_simulations.numerics",
    "compute_percentile_stats": "rank_simulations.statistics",
    "run_monte_carlo_experiment": "rank_simulations.experiment",
    "save_results": "experiment_tools.reporting",
}


def __getattr__(name: str):
    if name not in _PUBLIC_HELPERS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(_PUBLIC_HELPERS[name]), name)
    globals()[name] = value
    return value


if __name__ == "__main__":
    raise SystemExit(main())
