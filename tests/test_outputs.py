import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from experiment_tools.cli import main
from experiment_tools.plots import plot_results
from experiment_tools.reporting import save_results


def test_spectrum_plot_uses_mean_log_percentile_coordinates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    save_results(
        [[1.6, 0.4, 0], [0.4, 0, 0]],
        [[0.9, 0.3, 0.1], [0.3, 0.1, 0]],
        [3, 3],
        tmp_path,
        plots=False,
    )
    percentiles = pd.read_csv(tmp_path / "plot.csv")
    fraction = np.linspace(0, 1, 11)
    np.testing.assert_allclose(percentiles.x_flow, 1 + 0.5 * fraction)
    np.testing.assert_allclose(percentiles.y_flow, np.log(0.8) - fraction * np.log(2))
    np.testing.assert_allclose(percentiles.x_aug, 1 + 1.5 * fraction)
    np.testing.assert_allclose(percentiles.y_aug, 1.5 * np.log(3) * (1 - fraction) - np.log(10))
    # Plot regeneration depends on the original percentile table, not fixed-index means.
    (tmp_path / "mean_spectrum.csv").unlink()
    figures = []
    original_savefig = Figure.savefig

    def capture_figure(figure: Figure, *args, **kwargs) -> None:
        figures.append(figure)
        original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_figure)
    plot_results(tmp_path)
    axis = figures[1].axes[0]
    assert axis.get_yscale() == "linear"
    for name, curve in zip(("flow", "aug"), axis.lines, strict=True):
        np.testing.assert_allclose(curve.get_xdata(), percentiles[f"x_{name}"])
        np.testing.assert_allclose(curve.get_ydata(), percentiles[f"y_{name}"])
    assert [text.get_text() for text in axis.texts] == ["Mean rank = 1.50", "Mean rank = 2.50"]


def test_true_index_mean_keeps_zeros_and_rank_columns(tmp_path: Path) -> None:
    tables = save_results(
        [[4, 2, 0], [2, 0, 0]], [[3, 0, 0], [1, 0, 0]], [3, 3], tmp_path, plots=False
    )
    np.testing.assert_array_equal(tables["spectrum"].mean_flow, [3, 1, 0])
    assert tables["ranks"].rank_flow.tolist() == [2, 1]
    assert tables["ranks"].rank_aug.tolist() == [1, 1]
    assert tables["ranks"].rank_ff.tolist() == [3, 3]
    assert pd.read_csv(tmp_path / "plot.csv").shape == (11, 4)
    with np.load(tmp_path / "spectra.npz") as archive:
        np.testing.assert_array_equal(archive["flow"], [[4, 2, 0], [2, 0, 0]])


def test_zero_rank_and_non_augmented_results_are_plottable(tmp_path: Path) -> None:
    save_results([[0, 0], [0, 0]], [], [0, 0], tmp_path)
    assert pd.read_csv(tmp_path / "Ranks.csv").rank_flow.tolist() == [0, 0]
    assert "rank_aug" not in pd.read_csv(tmp_path / "Ranks.csv")
    for name in ("rank_histogram.png", "singular_spectrum.png"):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG")


def test_small_cli_run_exports_gelu_and_both_plots(tmp_path: Path) -> None:
    config = tmp_path / "small.toml"
    config.write_text(
        "n_dim = 2\nrho_val = 0.5\ndepth_val = 1\nlatent_val = 2\n"
        "num_samples = 4\nnum_samples_ff = 8\niterations = 2\nsteps = 4\nsegments = 2\n"
        'device = "cpu"\ncompile = false\ncompute_gelu = true\n',
        encoding="utf-8",
    )
    output = tmp_path / "run"
    assert main(["--config", str(config), "--output-dir", str(output)]) == 0
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["dtype"] == "float64"
    assert manifest["effective_compile"] is False
    assert manifest["parameter_count"] == 12
    assert manifest["settings"]["iterations"] == 2
    for folder in (output, output / "gelu"):
        assert pd.read_csv(folder / "Ranks.csv").shape == (2, 3)
        for name in ("rank_histogram.png", "singular_spectrum.png"):
            assert (folder / name).read_bytes().startswith(b"\x89PNG")
    replay = tmp_path / "replotted"
    assert main(["--plot-only", str(output), "--output-dir", str(replay)]) == 0
    assert (replay / "rank_histogram.png").exists()
