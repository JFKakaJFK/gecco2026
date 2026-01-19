import os
from ast import literal_eval
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from src.plot.config import CONFIGS

KERNEL_ORDER = [
    "cpu",
    "baseline",
    "restrict",
    "shared_memory",
    "block_reduce",
    "single_kernel",
    "single_kernel_fmaf",
    "single_kernel_inplace",
]

sns.set_theme(context="notebook", style="whitegrid")


def get_modifiers(
    conn: duckdb.DuckDBPyConnection,
    modifier_query: str | None = None,
    modifier_labels: list[str] | None = None,
    default_query: str | None = None,
    allow_none: bool = False,
    sort_key: Callable[[tuple], Any] | None = None,
):
    if modifier_query is not None:
        assert modifier_labels is not None, "modifier_labels must be provided"

        modifiers = sorted(
            map(
                literal_eval,
                set(
                    str(m)
                    for m, *_ in conn.sql(
                        f"SELECT {modifier_query} FROM results"
                    ).fetchall()
                    if m is not None
                ),
            ),
            key=sort_key,
        )

        assert len(modifiers) > 0, "No modifiers found in database"
        num_modifiers = len(modifiers[0])
        assert num_modifiers == len(modifier_labels), (
            f"Expected {num_modifiers} labels got {len(modifier_labels)}"
        )

        return modifiers, num_modifiers, modifier_query

    if default_query is not None:
        modifiers = sorted(
            [m for m, *_ in conn.sql(default_query).fetchall() if m is not None]
        )

        return modifiers, 0, None

    if allow_none:
        return None, 0, "['']"

    raise ValueError("Invalid modifier configuration")


def plot_convergence(
    plot_dir: Path,
    conn: duckdb.DuckDBPyConnection,
    kernels: list[str] | None = None,
    kernel_query: str = "kernel",
    row_modifier_query: str | None = None,
    row_modifier_labels: list[str] | None = None,
    row_sort_key: Callable[[tuple], Any] | None = None,
    col_modifier_query: str | None = None,
    col_modifier_labels: list[str] | None = None,
    col_sort_key: Callable[[tuple], Any] | None = None,
    unit_query: str = "run",
    y_var: str = "objectives[1]",
    y_agg: Literal["MIN", "MAX"] = "MIN",
    y_label: str = "Objective Value",
    metrics=("evaluations", "total_time_seconds", "generation"),
    metric_labels=("Evaluations", "Time [s]", "Generations"),
    ylog: bool = False,
    nsamples: int = 25,
    ymin: float | str | None = None,
    ymax: float | str | None = None,
):
    row_modifiers, num_row_modifiers, row_modifier_query = get_modifiers(
        conn,
        modifier_query=row_modifier_query,
        modifier_labels=row_modifier_labels,
        allow_none=True,
        sort_key=row_sort_key,
    )

    col_modifiers, num_col_modifiers, col_modifier_query = get_modifiers(
        conn,
        modifier_query=col_modifier_query,
        modifier_labels=col_modifier_labels,
        default_query="SELECT DISTINCT(problem) FROM results",
        sort_key=col_sort_key,
    )

    if kernels is None:
        kernels = sorted(
            [
                d
                for d, *_ in conn.sql("SELECT DISTINCT(kernel) FROM results").fetchall()
                if d is not None
            ]
        )

    if num_row_modifiers == 0 or num_col_modifiers == 0 or len(kernels) == 0:
        print("Not enough data?")
        return

    nrows = len(row_modifiers) if row_modifiers else 1
    ncols = len(col_modifiers)

    # Sort devices by predetermined order
    kernel_order_index = {k: i for i, k in enumerate(KERNEL_ORDER)}
    kernels = sorted(kernels, key=lambda k: kernel_order_index.get(k, float("inf")))

    hues = sns.color_palette(n_colors=len(kernels))
    palette = {d: h for d, h in zip(kernels, hues, strict=True)}

    global_ymin, global_ymax = ymin, ymax

    pbar = tqdm(leave=False, total=(2 * len(metrics) * nrows * ncols), ascii=True)

    for log_x in [False, True]:
        for metric, metric_label in zip(metrics, metric_labels, strict=True):
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(5 * ncols, 3 * nrows),
                sharey=False,
                constrained_layout=True,
            )

            if isinstance(axes, plt.Axes):
                axes = np.array([axes])
            axes = axes.reshape(nrows, ncols)

            for row_mi, row_modifier in enumerate(
                row_modifiers if row_modifiers is not None else [None]
            ):
                for col_mi, col_modifier in enumerate(col_modifiers):
                    ax = axes[row_mi, col_mi]

                    if row_modifier is not None:
                        xlim = conn.execute(
                            f"""
                                SELECT MIN({metric}) as xmin, MAX({metric}) as xmax FROM results
                                WHERE 
                                    {kernel_query} IN $1 AND
                                    {col_modifier_query} = $2 AND
                                    {row_modifier_query} = $3
                                LIMIT 1
                                """,
                            [kernels, col_modifier, row_modifier],
                        ).fetchone()

                        assert xlim is not None

                        if xlim[0] is None or xlim[1] is None:
                            print(
                                f"NO RUNS FOR {row_modifier}? skipping {col_modifier}..."
                            )

                            ax.set_axis_off()  # hide the plot
                            continue
                    else:
                        xlim = conn.execute(
                            f"""
                                SELECT MIN({metric}) as xmin, MAX({metric}) as xmax FROM results
                                WHERE 
                                    {kernel_query} IN $1 AND
                                    {col_modifier_query} = $2
                                LIMIT 1
                                """,
                            [kernels, col_modifier],
                        ).fetchone()

                    # print(metric, row_modifier, col_modifier, xlim)

                    df = None
                    x_samples = (
                        np.logspace(
                            np.log10(xlim[0]) if xlim[0] > 0 else -1,
                            np.log10(xlim[1]) if xlim[1] > 0 else 0,
                            nsamples,
                        )
                        if log_x
                        else np.linspace(*xlim, nsamples, endpoint=True)
                    )

                    for x in tqdm(x_samples, leave=False, ascii=True):
                        q = f"""
                            SELECT
                                {kernel_query} as kernel,
                                {unit_query} as unit,
                                {x}::DOUBLE as metric,
                                -- aggregate is needed to find the best result up to `x`
                                {y_agg}({y_var}) as value,
                            FROM results
                            WHERE 
                                {kernel_query} in $1 AND
                                {col_modifier_query} = $2 AND
                                {metric}::DOUBLE <= {x}::DOUBLE
                                {f"AND {row_modifier_query} = $3" if row_modifier else ""}
                            GROUP BY ALL
                            """
                        try:
                            part = conn.execute(
                                q,
                                [kernels, col_modifier, row_modifier]
                                if row_modifier
                                else [kernels, col_modifier],
                            ).df()
                        except Exception as e:
                            print(q)
                            raise e

                        if df is None:
                            df = part
                        else:
                            df = pd.concat([df, part], ignore_index=True)

                    assert df is not None

                    df = df.sort_values(
                        by=["kernel", "unit", "metric"],
                        ignore_index=True,
                    )

                    if kernels is not None:
                        df = df[df[kernel_query].isin(kernels)]

                    sns.lineplot(
                        df,
                        x="metric",
                        y="value",
                        hue="kernel",
                        palette=palette,
                        # units="unit",
                        # estimator=None,
                        estimator=np.median,
                        errorbar=("pi", 50),
                        err_kws=dict(linewidth=0),
                        legend=None,
                        ax=ax,
                    )

                    if row_mi == 0:
                        # Show column modifier as title
                        col_str = ""
                        if col_modifier is not None and col_modifier_labels:
                            col_str = "\n".join(
                                f"{label} = {value}"
                                for label, value in zip(
                                    col_modifier_labels, col_modifier, strict=True
                                )
                            )
                        ax.set_title(
                            f"{col_str if col_str else col_modifier}", fontsize=10
                        )

                    ax.set_xlabel("")

                    if row_mi == len(row_modifiers) - 1:
                        ax.set_xlabel(metric_label)

                    if col_mi == 0:
                        modifier_str = ""
                        if row_modifier is not None:
                            modifier_str = (
                                "\n".join(
                                    f"{label} = {value}"
                                    for label, value in zip(
                                        row_modifier_labels, row_modifier, strict=True
                                    )
                                )
                                + "\n"
                            )
                        ax.set_ylabel(modifier_str + y_label)
                    else:
                        ax.set_ylabel("")

                    q = 0.975
                    if global_ymin == "auto":
                        ymin = df["value"].quantile(1 - q)
                        if np.isfinite(ymin):  # and np.isfinite(ymax):
                            ax.set_ylim(ymin=ymin)  # , ymax=ymax)
                    elif isinstance(ymin, (int, float)):
                        ax.set_ylim(ymin=ymin)

                    if global_ymax == "auto":
                        ymax = df["value"].quantile(q)
                        if np.isfinite(ymax):
                            ax.set_ylim(ymax=ymax)
                    elif isinstance(ymin, (int, float)):
                        ax.set_ylim(ymax=ymax)

                    if ylog:
                        ax.set_yscale("symlog")

                    if log_x:
                        ax.set_xscale("symlog")

                    ax.set_xlim(xmin=x_samples[0], xmax=xlim[1])

                    pbar.update(1)

            handles = [
                axes.flat[0].plot([], [], label=k, color=palette[k])[0] for k in kernels
            ]
            fig.legend(
                handles=handles,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                borderaxespad=0.0,
                frameon=False,
            )

            log_str = "_logx" if log_x else ""

            if not plot_dir.is_dir():
                os.makedirs(plot_dir, exist_ok=True)

            for fmt in ["pdf", "png"]:
                fig.savefig(
                    plot_dir
                    / f"convergence_{metric_label.replace(' ', '-').replace('/', '|')}{log_str}.{fmt}",
                    dpi=600,
                    # transparent=True,
                    bbox_inches="tight",
                )

    pbar.close()


def plot_fraction_target_reached(
    plot_dir: Path,
    conn: duckdb.DuckDBPyConnection,
    kernels: list[str] | None = None,
    kernel_query: str = "kernel",
    row_modifier_query: str | None = None,
    row_modifier_labels: list[str] | None = None,
    col_modifier_query: str | None = None,
    col_modifier_labels: list[str] | None = None,
    unit_query: str = "run",
):
    row_modifiers, num_row_modifiers, row_modifier_query = get_modifiers(
        conn,
        modifier_query=row_modifier_query,
        modifier_labels=row_modifier_labels,
        allow_none=True,
    )

    col_modifiers, num_col_modifiers, col_modifier_query = get_modifiers(
        conn,
        modifier_query=col_modifier_query,
        modifier_labels=col_modifier_labels,
        default_query="SELECT DISTINCT(problem) FROM results",
    )

    if kernels is None:
        kernels = sorted(
            [
                d
                for d, *_ in conn.sql("SELECT DISTINCT(kernel) FROM results").fetchall()
                if d is not None
            ]
        )

    if num_row_modifiers == 0 or num_col_modifiers == 0 or len(kernels) == 0:
        print("Not enough data?")
        return

    nrows = len(row_modifiers) if row_modifiers else 1
    ncols = len(col_modifiers)

    # Sort devices by predetermined order
    kernel_order_index = {k: i for i, k in enumerate(KERNEL_ORDER)}
    kernels = sorted(kernels, key=lambda k: kernel_order_index.get(k, float("inf")))

    hues = sns.color_palette(n_colors=len(kernels))
    palette = {d: h for d, h in zip(kernels, hues, strict=True)}

    pbar = tqdm(leave=False, total=(nrows * ncols), ascii=True)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4 * ncols, 5 * nrows),
        sharey=True,
        constrained_layout=True,
    )

    axes = np.atleast_2d(axes)

    if axes.shape != (nrows, ncols):
        axes = axes.reshape(nrows, ncols)

    for row_mi, row_modifier in enumerate(
        row_modifiers if row_modifiers is not None else [None]
    ):
        for col_mi, col_modifier in enumerate(col_modifiers):
            ax = axes[row_mi, col_mi]

            q = f"""
                SELECT 
                    kernel,
                    SUM(CASE WHEN status = 'TargetReached' THEN 1 ELSE 0 END)
                        / SUM(CASE WHEN status != 'Running' THEN 1 ELSE 0 END) AS reached
                FROM results
                WHERE
                    {col_modifier_query} = $1
                    {f"AND {row_modifier_query} = $2" if row_modifier is not None else ""}
                GROUP BY kernel
            """

            df = conn.execute(
                q,
                [col_modifier, row_modifier]
                if row_modifier is not None
                else [col_modifier],
            ).df()

            # df = df.set_index("kernel").reindex(kernels, fill_value=0.0).reset_index()

            assert df is not None

            sns.barplot(
                df,
                x="kernel",
                y="reached",
                hue="kernel",
                order=kernels,
                palette=palette,
                legend=None,
                ax=ax,
            )

            # Rotate x-tick labels
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

            if row_mi == 0:
                # Show column modifier as title
                col_str = ""
                if col_modifier is not None and col_modifier_labels:
                    col_str = "\n".join(
                        f"{label} = {value}"
                        for label, value in zip(
                            col_modifier_labels, col_modifier, strict=True
                        )
                    )
                ax.set_title(f"{col_str if col_str else col_modifier}", fontsize=10)

            ax.set_xlabel("")

            if row_mi == len(row_modifiers) - 1:
                ax.set_xlabel("Kernel Version")

            if col_mi == 0:
                modifier_str = ""
                if row_modifier is not None:
                    modifier_str = (
                        "\n".join(
                            f"{label} = {value}"
                            for label, value in zip(
                                row_modifier_labels, row_modifier, strict=True
                            )
                        )
                        + "\n"
                    )
                ax.set_ylabel(modifier_str + "Target reached [%]")
            else:
                ax.set_ylabel("")

            pbar.update(1)

    handles = [axes.flat[0].plot([], [], label=k, color=palette[k])[0] for k in kernels]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
        frameon=False,
    )

    if not plot_dir.is_dir():
        os.makedirs(plot_dir, exist_ok=True)

    for fmt in ["pdf", "png"]:
        fig.savefig(
            plot_dir / f"target_reached.{fmt}",
            dpi=600,
            # transparent=True,
            bbox_inches="tight",
        )

    pbar.close()


def plot(output_dir: Path):
    print("Starting plot creation...")

    db_path = output_dir / "experiments.duckdb"
    plot_dir = output_dir / "plots"

    conn = duckdb.connect(db_path)

    for cfg in tqdm(CONFIGS.values(), ascii=True):
        plot_convergence(
            plot_dir / cfg.name,
            conn,
            kernels=cfg.kernels,
            y_var=cfg.y_var,
            y_agg=cfg.y_agg,
            y_label=cfg.y_label,
            unit_query=cfg.unit_query,
            row_modifier_query=cfg.row_modifier_query,
            row_modifier_labels=cfg.row_modifier_labels,
            row_sort_key=cfg.row_sort_key,
            col_modifier_query=cfg.col_modifier_query,
            col_modifier_labels=cfg.col_modifier_labels,
            col_sort_key=cfg.col_sort_key,
            metrics=cfg.metrics,
            metric_labels=cfg.metric_labels,
            ylog=cfg.ylog,
            ymin=cfg.ymin,
            ymax=cfg.ymax,
            nsamples=cfg.nsamples,
        )

    conn.close()
