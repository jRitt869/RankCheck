"""TOML defaults and strictly validated CLI overrides, without importing PyTorch."""

import json
import math
import tomllib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "defaults.toml"


@dataclass(frozen=True)
class ExperimentSettings:
    compute_aug: bool
    compute_gelu: bool
    n_dim: int
    rho_val: float
    depth_val: int
    latent_val: int
    num_samples_ff: int
    num_samples: int
    t_end: float
    steps: int
    segments: int
    iterations: int
    rtol: float
    atol: float
    seed: int
    device: str
    compile: bool
    output_dir: str
    plots: bool

    def validate(self) -> None:
        for name in ("compute_aug", "compute_gelu", "compile", "plots"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be true or false")
        for name in (
            "n_dim",
            "depth_val",
            "latent_val",
            "num_samples_ff",
            "num_samples",
            "steps",
            "segments",
            "iterations",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("rho_val", "t_end", "rtol", "atol"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            if value < 0 or (name in ("rho_val", "t_end") and value == 0):
                raise ValueError(
                    f"{name} must be {'positive' if name in ('rho_val', 't_end') else 'nonnegative'}"
                )
        if self.compute_aug and self.n_dim < 2:
            raise ValueError("Augmentation requires n_dim >= 2")
        if self.steps % self.segments:
            raise ValueError("steps must be divisible by segments so integration reaches t_end")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("seed must be an integer between 0 and 9223372036854775807")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu, or cuda")
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ValueError("output_dir must be a nonempty path string")


def load_settings(
    config: Path | None = None, overrides: dict[str, Any] | None = None
) -> ExperimentSettings:
    """Precedence: shipped defaults < optional TOML file < explicit CLI options."""
    expected = {field.name for field in fields(ExperimentSettings)}
    values = {}
    for path in (DEFAULT_CONFIG, config):
        if path is None:
            continue
        with path.open("rb") as source:
            supplied = tomllib.load(source)
        if unknown := set(supplied) - expected:
            raise ValueError(f"Unknown settings in {path}: {', '.join(sorted(unknown))}")
        values.update(supplied)
    supplied = {key: value for key, value in (overrides or {}).items() if value is not None}
    if unknown := set(supplied) - expected:
        raise ValueError(f"Unknown overrides: {', '.join(sorted(unknown))}")
    values.update(supplied)
    if missing := expected - set(values):
        raise ValueError(f"Missing defaults: {', '.join(sorted(missing))}")
    settings = ExperimentSettings(**values)
    settings.validate()
    return settings


def save_settings(settings: ExperimentSettings, destination: Path) -> None:
    """Write a replayable TOML file containing all effective settings."""
    lines = []
    for key, value in asdict(settings).items():
        literal = json.dumps(value) if isinstance(value, (str, bool)) else repr(value)
        lines.append(f"{key} = {literal}")
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
