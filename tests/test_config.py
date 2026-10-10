from dataclasses import asdict
from pathlib import Path

import pytest

from experiment_tools.cli import build_parser, main
from experiment_tools.config import load_settings, save_settings


def test_toml_and_cli_precedence_and_false_override(tmp_path: Path) -> None:
    path = tmp_path / "custom.toml"
    path.write_text("iterations = 4\ncompute_aug = true\nplots = true\n", encoding="utf-8")
    settings = load_settings(path, {"iterations": 2, "compute_aug": False, "plots": False})
    assert settings.iterations == 2
    assert settings.compute_aug is False
    assert settings.plots is False
    assert settings.steps == 500
    assert settings.num_samples == 40000


@pytest.mark.parametrize(
    "override",
    [
        {"steps": 501},
        {"n_dim": 1},
        {"num_samples": 0},
        {"iterations": True},
        {"rho_val": float("nan")},
        {"atol": -1},
        {"seed": -1},
        {"compile": "false"},
        {"device": "other"},
        {"output_dir": ""},
    ],
)
def test_invalid_settings_fail_before_simulation(override: dict) -> None:
    with pytest.raises(ValueError):
        load_settings(overrides=override)


def test_unknown_toml_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("iteration = 4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown settings"):
        load_settings(path)


def test_effective_settings_can_be_replayed(tmp_path: Path) -> None:
    settings = load_settings(
        overrides={"output_dir": str(tmp_path / "folder with spaces"), "seed": 17}
    )
    path = tmp_path / "run_settings.toml"
    save_settings(settings, path)
    assert asdict(load_settings(path)) == asdict(settings)


def test_cli_aliases_and_boolean_overrides() -> None:
    parsed = build_parser().parse_args(
        [
            "--iterations",
            "3",
            "--radius",
            "2",
            "--depth",
            "1",
            "--no-augmented",
            "--gelu",
            "--no-compile",
            "--no-plots",
        ]
    )
    assert parsed.compute_aug is False
    assert parsed.compute_gelu is True
    assert parsed.compile is False
    assert parsed.plots is False
    assert parsed.depth_val == 1
    assert parsed.rho_val == 2


def test_show_settings_does_not_run_simulation(capsys) -> None:
    assert main(["--show-settings", "--iterations", "7"]) == 0
    assert '"iterations": 7' in capsys.readouterr().out
