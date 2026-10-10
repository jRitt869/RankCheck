"""CLI configuration and output orchestration added during maintenance."""

import argparse
import hashlib
import json
import platform
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from .config import DEFAULT_CONFIG, load_settings, save_settings

INTEGER_OPTIONS = {
    "n_dim": (
        ["--n-dim", "--input-dim"],
        "ODE state dimension (augmented input has one fewer coordinate)",
    ),
    "depth_val": (["--depth-val", "--depth"], "Number of hidden layers"),
    "latent_val": (["--latent-val", "--latent-dim"], "Width of hidden layers"),
    "num_samples_ff": (["--num-samples-ff"], "Input samples for the feedforward rank bound"),
    "num_samples": (["--num-samples"], "Input samples for each flow Jacobian"),
    "steps": (["--steps"], "Euler integration steps"),
    "segments": (["--segments"], "Sensitivity segments; must divide steps"),
    "iterations": (["--iterations"], "Number of parameter samples"),
    "seed": (["--seed"], "Random seed"),
}
FLOAT_OPTIONS = {
    "rho_val": (["--rho-val", "--radius"], "Radius of the input ball"),
    "t_end": (["--t-end"], "Terminal time; default time-one map"),
    "rtol": (["--rtol"], "Relative numerical rank tolerance"),
    "atol": (["--atol"], "Absolute numerical rank tolerance"),
}
BOOLEAN_OPTIONS = {
    "compute_aug": (["--compute-aug", "--augmented"], "Include the augmented-input experiment"),
    "compute_gelu": (["--compute-gelu", "--gelu"], "Also compute and export GELU results"),
    "compile": (["--compile"], "Use the original CUDA graph backend when running on CUDA"),
    "plots": (["--plots"], "Automatically save both figures"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monte Carlo Jacobian rank experiments for neural ODEs"
    )
    parser.add_argument("--config", type=Path, help="Optional TOML overrides for defaults.toml")
    for options, converter in ((INTEGER_OPTIONS, int), (FLOAT_OPTIONS, float)):
        for name, (flags, help_text) in options.items():
            parser.add_argument(*flags, dest=name, type=converter, default=None, help=help_text)
    for name, (flags, help_text) in BOOLEAN_OPTIONS.items():
        parser.add_argument(
            *flags, dest=name, action=argparse.BooleanOptionalAction, default=None, help=help_text
        )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument(
        "--output-dir", default=None, help="Directory for CSV, NPZ, settings, and PNG outputs"
    )
    parser.add_argument(
        "--show-settings",
        action="store_true",
        help="Print effective settings and exit without simulating",
    )
    parser.add_argument(
        "--plot-only", type=Path, help="Regenerate figures from an existing output directory"
    )
    return parser


def _fingerprint(path: Path) -> dict[str, str]:
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    overrides = {
        name: getattr(args, name)
        for name in (*INTEGER_OPTIONS, *FLOAT_OPTIONS, *BOOLEAN_OPTIONS, "device", "output_dir")
    }
    try:
        settings = load_settings(args.config, overrides)
        if args.show_settings:
            print(json.dumps(asdict(settings), indent=2))
            return 0
        if args.plot_only:
            from .plots import plot_results

            destination = settings.output_dir if args.output_dir is not None else None
            plot_results(args.plot_only, destination)
            print(f"Saved both figures to {destination or args.plot_only}")
            return 0
        from rank_simulations.experiment import resolve_device, run_monte_carlo_experiment

        from .reporting import save_results

        started_at = datetime.now(timezone.utc).isoformat()
        started = perf_counter()
        print(f"Loaded defaults from {DEFAULT_CONFIG.name}; settings validated", flush=True)
        results = run_monte_carlo_experiment(settings)
        simulation_seconds = perf_counter() - started
        output = Path(settings.output_dir)
        save_results(
            *results[:3], output, plots=settings.plots, rtol=settings.rtol, atol=settings.atol
        )
        if settings.compute_gelu:
            save_results(
                *results[3:],
                output / "gelu",
                plots=settings.plots,
                rtol=settings.rtol,
                atol=settings.atol,
            )
        save_settings(settings, output / "run_settings.toml")
        project = Path(__file__).resolve().parent.parent
        code = [
            project / "numerical_experiments.py",
            *sorted((project / "rank_simulations").glob("*.py")),
            *sorted((project / "experiment_tools").glob("*.py")),
        ]
        manifest = {
            "started_at_utc": started_at,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "simulation_seconds": simulation_seconds,
            "settings": asdict(settings),
            "device": resolve_device(settings.device),
            "effective_compile": settings.compile and resolve_device(settings.device) == "cuda",
            "dtype": "float64",
            "parameter_count": (
                settings.n_dim * settings.latent_val
                + settings.latent_val
                + (settings.depth_val - 1) * (settings.latent_val**2 + settings.latent_val)
                + settings.latent_val * settings.n_dim
                + settings.n_dim
            ),
            "python": platform.python_version(),
            "versions": {
                package: version(package)
                for package in (
                    "torch",
                    "numpy",
                    "pandas",
                    *(("matplotlib",) if settings.plots else ()),
                )
            },
            "defaults": _fingerprint(DEFAULT_CONFIG),
            "config": _fingerprint(args.config) if args.config else None,
            "code": [
                {**_fingerprint(path), "file": path.relative_to(project).as_posix()}
                for path in code
            ],
            "method": "Explicit Euler and composed segment variational Jacobians; full state output retained",
            "augmentation": "Sample n_dim-1 coordinates and append a zero coordinate; same state dimension and parameters",
            "spectrum_mean": "Arithmetic mean at each fixed singular index, including zeros",
            "singular_spectrum": "Mean index percentiles vs mean reversed natural-log singular-value percentiles from plot.csv; endpoint is mean rank among positive spectra",
            "rank_histogram": "rank_flow vs rank_aug; rank_flow vs rank_ff when augmentation is disabled",
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"Saved numerical results{' and both figures' if settings.plots else ''} to {output}")
    return 0
