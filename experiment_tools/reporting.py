"""Save original CSV summaries, plus per-index spectra and complete rank tables."""

from pathlib import Path

import numpy as np
import pandas as pd

from rank_simulations.statistics import compute_percentile_stats


def _spectra(values: list, name: str, iterations: int | None = None) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if (
        result.ndim != 2
        or not result.size
        or (iterations is not None and len(result) != iterations)
    ):
        raise ValueError(f"{name} must contain one equally sized spectrum per parameter sample")
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError(f"{name} contains nonfinite or negative singular values")
    if (np.diff(result, axis=1) > 0).any():
        raise ValueError(f"{name} must contain descending singular values")
    return result


def save_results(
    sVals_flow: list,
    sVals_aug: list,
    sRanks_ff: list,
    output_dir: str | Path = ".",
    *,
    plots: bool = True,
    rtol: float = 1e-12,
    atol: float = 1e-10,
) -> dict[str, pd.DataFrame]:
    """Keep Ranks.csv/plot.csv; add the missing flow ranks and a true mean spectrum."""
    flow = _spectra(sVals_flow, "flow")
    aug = _spectra(sVals_aug, "augmented", len(flow)) if len(sVals_aug) else None
    if aug is not None and aug.shape[1] != flow.shape[1]:
        raise ValueError("Full and augmented spectra must have the same singular index count")
    ff = np.asarray(sRanks_ff)
    if ff.shape != (len(flow),) or not np.isfinite(ff).all() or (ff < 0).any() or (ff % 1).any():
        raise ValueError("Feedforward ranks must be nonnegative integers, one per parameter sample")
    if not np.isfinite([rtol, atol]).all() or min(rtol, atol) < 0:
        raise ValueError("Rank tolerances must be finite and nonnegative")
    modes = {"flow": flow, **({"aug": aug} if aug is not None else {})}
    ranks = {
        name: (values > np.maximum(atol, rtol * values.max(axis=1, keepdims=True))).sum(axis=1)
        for name, values in modes.items()
    }
    # Preserve the two original columns first when augmentation is enabled.
    rank_table = pd.DataFrame(
        {
            **({"rank_aug": ranks["aug"]} if aug is not None else {}),
            "rank_ff": ff.astype(int),
            "rank_flow": ranks["flow"],
        }
    )
    spectrum_table = pd.DataFrame({"index": np.arange(1, flow.shape[1] + 1)})
    legacy = {}
    for name, values in modes.items():
        spectrum_table[f"mean_{name}"] = values.mean(axis=0)
        spectrum_table[f"std_{name}"] = values.std(axis=0)
        lines = []
        for row in values:
            threshold = max(atol, rtol * row.max())
            positive = row[row > threshold]
            lines.append({"x": np.arange(1, len(positive) + 1), "y": positive})
        mean_x, _, mean_y, _ = compute_percentile_stats(lines)
        legacy[f"x_{name}"] = mean_x
        legacy[f"y_{name}"] = mean_y
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rank_table.to_csv(output / "Ranks.csv", index=False)
    spectrum_table.to_csv(output / "mean_spectrum.csv", index=False)
    pd.DataFrame(legacy).to_csv(output / "plot.csv", index=False)
    np.savez_compressed(output / "spectra.npz", **modes, rank_ff=ff.astype(int))
    if plots:
        from .plots import plot_results

        plot_results(output)
    return {"ranks": rank_table, "spectrum": spectrum_table}
