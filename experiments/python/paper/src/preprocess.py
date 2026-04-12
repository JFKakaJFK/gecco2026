import numpy as np
import pandas as pd

N_GRID_POINTS = 300


def interpolate_onto_time_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolates each fold's NMSE curve onto a common log-spaced time grid so
    that a median + percentile band can be computed across folds.

    Each group (log_config, dataset, fold, template_depth) is treated as a
    monotonically decreasing step function. Linear interpolation is used;
    values before the first measurement hold the first value, and values after
    the last hold the last.
    """
    group_keys = ["log_config", "dataset", "fold", "template_depth"]

    t_min = df["total_time_seconds"].min()
    t_max = df["total_time_seconds"].max()
    time_grid = np.linspace(t_min, t_max, N_GRID_POINTS)

    rows = []
    for keys, group in df.groupby(group_keys):
        group = group.sort_values("total_time_seconds")
        nmse_interp = np.interp(time_grid, group["total_time_seconds"], group["nmse"])

        for t, nmse in zip(time_grid, nmse_interp):
            rows.append(dict(zip(group_keys, keys)) | {"time_grid_seconds": t, "nmse": nmse})

    return pd.DataFrame(rows)
