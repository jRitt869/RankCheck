"""Static scientific figures saved beside the experiment data."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_results(input_dir: str | Path, output_dir: str | Path | None = None) -> None:
    """Regenerate both figures from saved tables without running a simulation."""
    source = Path(input_dir)
    output = Path(output_dir) if output_dir is not None else source
    ranks = pd.read_csv(source / "Ranks.csv")
    spectrum = pd.read_csv(source / "plot.csv")
    required = {"x_flow", "y_flow"}
    if "rank_aug" in ranks:
        required.update({"x_aug", "y_aug"})
    if ranks.empty or "rank_flow" not in ranks or spectrum.empty or not required <= set(spectrum):
        raise ValueError("Plots require current Ranks.csv and percentile coordinates in plot.csv")
    output.mkdir(parents=True, exist_ok=True)
    y_name = "rank_aug" if "rank_aug" in ranks else "rank_ff"
    x_values, y_values = ranks["rank_flow"].to_numpy(), ranks[y_name].to_numpy()
    if not np.isfinite(np.concatenate([x_values, y_values])).all():
        raise ValueError("Rank data must be finite")
    maximum = int(max(x_values.max(), y_values.max()))
    edges = np.arange(-0.5, maximum + 1.5, 1)
    counts, _, _ = np.histogram2d(x_values, y_values, bins=(edges, edges))
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    mesh = ax.pcolormesh(
        edges,
        edges,
        np.ma.masked_equal(counts.T, 0),
        cmap="Blues",
        vmin=0,
        vmax=max(1, counts.max()),
    )
    ax.plot(
        [-0.5, maximum + 0.5],
        [-0.5, maximum + 0.5],
        "--",
        color="#64748b",
        lw=1,
        label="Equal rank",
    )
    if np.count_nonzero(counts) <= 30:
        for x, y in np.argwhere(counts > 0):
            ax.text(
                x,
                y,
                str(int(counts[x, y])),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if counts[x, y] > counts.max() / 2 else "#0f172a",
            )
    ax.set(
        xlabel="Non-augmented flow rank",
        ylabel="Augmented flow rank" if y_name == "rank_aug" else "Feedforward Jacobian rank",
        title=f"Numerical rank comparison • {len(ranks)} parameter samples",
        aspect="equal",
        xlim=(edges[0], edges[-1]),
        ylim=(edges[0], edges[-1]),
    )
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.legend(loc="upper left", fontsize=9)
    bar = fig.colorbar(mesh, ax=ax, label="Number of parameter samples")
    bar.locator = MaxNLocator(nbins=6, integer=True)
    bar.update_ticks()
    fig.savefig(output / "rank_histogram.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    has_curve = False
    for name, title, color in (
        ("flow", "Non-augmented flow", "#0369a1"),
        ("aug", "Augmented flow", "#b45309"),
    ):
        x_column, y_column = f"x_{name}", f"y_{name}"
        if x_column not in spectrum or y_column not in spectrum:
            continue
        coordinates = spectrum[[x_column, y_column]].to_numpy(dtype=float)
        finite = np.isfinite(coordinates).all(axis=1)
        if not finite.any():
            continue
        has_curve = True
        indices, log_values = coordinates[finite].T
        ax.plot(
            indices,
            log_values,
            "o-",
            markersize=3,
            color=color,
            label=title,
        )
        ax.annotate(
            f"Mean rank = {indices[-1]:.2f}",
            xy=(indices[-1], log_values[-1]),
            xytext=(-6, -18) if name == "flow" else (8, 10),
            textcoords="offset points",
            ha="right" if name == "flow" else "left",
            fontsize=8,
            color=color,
        )
    if not has_curve:
        ax.text(
            0.5,
            0.5,
            "No positive singular values for percentile averaging",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    ax.set(
        xlabel="Mean percentile of singular value indices",
        ylabel="Mean percentile of ln(singular value)",
        title="Mean singular spectrum percentiles across parameter samples",
    )
    # Values in plot.csv have already been logged; keep the y-axis linear.
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.grid(alpha=0.25)
    if has_curve:
        ax.legend()
    fig.savefig(output / "singular_spectrum.png", dpi=180)
    plt.close(fig)
