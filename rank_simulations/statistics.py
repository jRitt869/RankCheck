"""The original percentile summary, with explicit handling of zero-rank spectra."""

import numpy as np


def compute_percentile_stats(line_plots: list[dict]) -> tuple[np.ndarray, ...]:
    """Mean/std of index and reversed log-value percentiles (0,10,...,100).
    Zero-rank iterations have no positive log spectrum and are omitted here.
    """
    percentiles = np.arange(0, 110, 10)
    positive = [plot for plot in line_plots if len(plot["y"])]
    if not positive:
        empty = np.full(len(percentiles), np.nan)
        return empty.copy(), empty.copy(), empty.copy(), empty.copy()
    px = np.array([np.percentile(plot["x"], percentiles) for plot in positive])
    py = np.array([np.flip(np.percentile(np.log(plot["y"]), percentiles)) for plot in positive])
    return px.mean(axis=0), px.std(axis=0), py.mean(axis=0), py.std(axis=0)
