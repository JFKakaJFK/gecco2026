import json
import os
import pathlib
from ast import literal_eval
from typing import Literal

import duckdb
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sns.set_theme(context="notebook", style="whitegrid")


def plot_convergence_so(
    plot_dir: pathlib.Path,
    conn: duckdb.DuckDBPyConnection,
    problems: list[str] | None = None,
    methods: list[str] | None = None,
    method_query: str = "method_name",
    modifier_query: str | None = None,
    modifier_labels: list[str] | None = None,
    unit_query: str = "run",
    y_var: str = "objectives[1]",
    y_agg: Literal["MIN", "MAX"] = "MIN",
    y_label: str = "Objective Value",
    metrics=("evaluations", "total_time_seconds", "generation"),
    metric_labels=("Evaluations", "Time [s]", "Generations"),
    ylog: bool = False,
    show_generation_boundaries: bool = False,
    nsamples: int = 25,
    ymin: float | str | None = None,
    ymax: float | str | None = None,
    **kwargs,
):
    if methods is None:
        methods = sorted(
            [
                m
                for m, *_ in conn.sql(
                    f"SELECT DISTINCT({method_query}) FROM results"
                ).fetchall()
                if m is not None
            ]
        )

    if modifier_query is not None:
        assert modifier_labels is not None

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
            )
        )

        assert len(modifiers) > 0
        num_modifiers = len(modifiers[0])
        assert num_modifiers == len(modifier_labels)

        methods_actual = conn.sql(
            "SELECT DISTINCT(method_name) FROM results"
        ).fetchall()

        methods_with_modifiers = set(
            conn.sql(
                f"SELECT format('{{}};{{}};{{}}', method_name, {method_query}, {modifier_query}) FROM results"
            ).fetchall()
        )

        # not every method might have the same modifiers, and modifiers might split methods into more versions than initially present (e.g. same methods, problem dim as modifier) - so the best check to avoid accidentally merging multiple methods into the same line is to check that there is a combination for at least each method...
        if len(methods) == len(methods_actual):
            assert len(methods_with_modifiers) >= len(methods_actual), (
                methods_actual,
                methods_with_modifiers,
            )
    else:
        modifiers = None
        num_modifiers = 0
        modifier_query = "['']"

    assert methods is not None

    if problems is None:
        problems = sorted(
            [
                p
                for p, *_ in conn.sql(
                    "SELECT DISTINCT(problem_name) FROM results"
                ).fetchall()
                if p is not None
            ]
        )

    if len(methods) == 0 or len(problems) == 0:
        print("Not enough data?")
        return

    nrows = len(modifiers) if modifiers else 1
    ncols = len(problems)

    methods = sorted(methods)
    hues = sns.color_palette(n_colors=len(methods))
    palette = {m: h for m, h in zip(methods, hues)}

    global_ymin, global_ymax = ymin, ymax

    for log_x in [False, True]:
        for metric, metric_label in zip(metrics, metric_labels):
            fig, axes = plt.subplots(
                nrows=nrows,
                ncols=ncols,
                figsize=(5 * ncols, 3 * nrows),
                sharey=False,
                # sharex="col",
            )
            if isinstance(axes, plt.Axes):
                axes = np.array([axes])
            axes = axes.reshape(nrows, ncols)

            for mi, modifier in enumerate(
                modifiers if modifiers is not None else [None]
            ):
                for pi, problem_name in enumerate(problems):
                    ax = axes[mi, pi]

                    if modifier is not None:
                        xlim = conn.execute(
                            f"""
                            SELECT MIN({metric}) as xmin, MAX({metric}) as xmax FROM results
                            WHERE problem_name = $1
                            AND {modifier_query} = $2
                            LIMIT 1
                            """,
                            [problem_name, modifier],
                        ).fetchone()
                        assert xlim is not None
                        if xlim[0] is None or xlim[1] is None:
                            print(f"NO RUNS FOR {modifier}? skipping {problem_name}...")

                            ax.set_axis_off()  # hide the plot
                            continue
                    else:
                        xlim = conn.execute(
                            f"""
                            SELECT MIN({metric}) as xmin, MAX({metric}) as xmax FROM results
                            WHERE problem_name = $1
                            LIMIT 1
                            """,
                            [problem_name],
                        ).fetchone()

                    print(metric, modifier, problem_name, xlim)

                    # annoyingly depending on the metric, not all runs have the entries at the same metric value (e.g. time is noisy) - instead the following is done:
                    # - pick linearly/logspaced sampling points for the desired metric
                    # - get the data for each sampling point by using the best y value recorded where the metric was not above the sampling point
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
                    for x in tqdm(x_samples):
                        q = f"""
                        SELECT
                            {method_query} as method,
                            {unit_query} as unit,
                            {x}::DOUBLE as metric,
                            -- aggregate is needed to find the best result up to `x`
                            {y_agg}({y_var}) as value,
                        FROM results
                        WHERE problem_name = $1
                            AND {metric}::DOUBLE <= {x}::DOUBLE
                            {f"AND {modifier_query} = $2" if modifier else ""}
                        GROUP BY ALL
                        """
                        try:
                            part = conn.execute(
                                q,
                                [problem_name, modifier]
                                if modifier
                                else [problem_name],
                            ).df()
                        except Exception as e:
                            print(q)
                            raise e

                        if df is None:
                            df = part
                        else:
                            df = pd.concat([df, part], ignore_index=True)

                    assert df is not None
                    df = df.sort_values(  # sorting needs to happen when all data is in the df...
                        by=["method", "unit", "metric"],
                        ignore_index=True,
                    )

                    sns.lineplot(
                        df,
                        x="metric",
                        y="value",
                        hue="method",
                        palette=palette,
                        # units="unit",
                        # estimator=None,
                        estimator=np.median,
                        errorbar=("pi", 50),
                        err_kws=dict(linewidth=0),
                        legend=None,
                        ax=ax,
                    )

                    if mi == 0:
                        ax.set_title(problem_name)
                    ax.set_xlabel("")
                    if mi == len(modifiers) - 1:
                        ax.set_xlabel(metric_label)

                    if pi == 0:
                        modifier_str = ""
                        if modifier is not None:
                            modifier_str = (
                                "\n".join(
                                    f"{label} = {value}"
                                    for label, value in zip(modifier_labels, modifier)
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

                    if show_generation_boundaries:
                        q = f"""
                        WITH unit_min_values AS (
                            SELECT
                                {method_query} AS method,
                                {unit_query} AS unit,
                                MIN({metric}) AS min_value,
                                generation
                            FROM results
                            WHERE problem_name = $1 AND generation > 0
                                {f"AND {modifier_query} = $2" if modifier else ""}
                            GROUP BY ALL
                        )
                        SELECT
                            method,
                            generation,
                            quantile(min_value, 0.5) AS median_value
                        FROM unit_min_values
                        GROUP BY ALL
                        ORDER BY method, generation
                        """
                        try:
                            for method, generation, mvalue in conn.execute(
                                q,
                                [problem_name, modifier]
                                if modifier
                                else [problem_name],
                            ).fetchall():
                                # print(method, generation, mvalue)
                                ax.axvline(mvalue, color=palette[method], lw=0.75)
                        except Exception as e:
                            print(q)
                            raise e

            handles = [
                axes.flat[0].plot([], [], label=m, color=palette[m])[0] for m in methods
            ]
            fig.legend(
                handles=handles,
                loc="center right",
                # mode="expand",
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


def plot_scalability(
    conn: duckdb.DuckDBPyConnection,
    problems: list[str] | None = None,
    methods: list[str] | None = None,
    metrics=("evaluations", "total_time_seconds", "generation"),
    log_x: bool = True,
    log_y: bool = True,
    show_success_rates: bool = False,
    alpha: float = 0.9,
):
    sr_thresholds = [
        (0.9, "solid"),
        (0.5, "dashed"),
        (0.1, "dotted"),
        (0.0, (0, (1, 10))),
    ]

    metric_names = dict(
        evaluations="Evaluations",
        total_time_seconds="Time [s]",
        generation="Generations",
    )

    if methods is None:
        methods = sorted(
            [
                m
                for m, *_ in conn.sql(
                    "SELECT DISTINCT(method_name) AS method FROM results"
                ).fetchall()
                if m is not None
            ]
        )

    if problems is None:
        problems = sorted(
            [
                p
                for p, *_ in conn.sql(
                    "SELECT DISTINCT(problem_name) AS problem FROM results"
                ).fetchall()
                if p is not None
            ]
        )

    if len(methods) == 0 or len(problems) == 0:
        print("Not enough data?")
        return

    nrows = len(metrics)
    ncols = len(problems)
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(4 * ncols, 3 * nrows), sharex="col"
    )
    if isinstance(axes, plt.Axes):
        axes = np.array([axes])
    axes = axes.reshape(nrows, ncols)

    methods = sorted(methods)
    hues = sns.color_palette(n_colors=len(methods))  # , palette="colorblind")

    method_handles = dict()
    for mi, metric in enumerate(metrics):
        for pi, problem_name in enumerate(problems):
            ax = axes[mi, pi]

            num_runs = conn.execute(
                "SELECT count(DISTINCT run) FROM results WHERE problem_name = $1",
                [problem_name],
            ).fetchone()[0]

            sr_ax = None
            if show_success_rates:
                sr_ax = ax.twinx()

            all_dims = []
            for i, method in enumerate(methods):
                hue = hues[i]

                if method not in method_handles:
                    method_handles[method] = ax.plot([], [], label=method, color=hue)[0]

                df = conn.execute(
                    f"""
                    SELECT dims, count(status)::DOUBLE / {float(num_runs)} AS success_rate, quantile({metric}, 0.1) AS lower, median({metric}) AS median, quantile({metric}, 0.9) AS upper FROM results
                    WHERE problem_name = $1 AND method_name = $2 AND status = 'TargetReached'
                    GROUP BY dims
                    ORDER BY dims ASC
                """,
                    [problem_name, method],
                ).df()

                # pyplot expects the errors to be positive deviations only...
                df["lower"] = df["median"] - df["lower"]
                df["upper"] = df["upper"] - df["median"]

                if show_success_rates:
                    sr_ax.plot(
                        df["dims"], df["success_rate"], color=hue, lw=0.75, alpha=0.5
                    )

                all_dims += df["dims"].tolist()

                start, end = 0, 0
                for sr_threshold, ls in sr_thresholds:
                    start = end
                    while (
                        end < len(df) and df["success_rate"].iloc[end] >= sr_threshold
                    ):
                        end += 1

                    if start >= end:
                        # skip empy
                        continue

                    if start > 0:
                        # plot the connecting line between two success rate thresholds
                        ax.plot(
                            df["dims"].iloc[start - 1 : start + 1],
                            df["median"].iloc[start - 1 : start + 1],
                            color=hue,
                            ls=ls,
                            alpha=alpha,
                        )

                    if sr_threshold > 0.0:
                        ax.errorbar(
                            df["dims"].iloc[start:end],
                            df["median"].iloc[start:end],
                            yerr=(
                                df["lower"].iloc[start:end],
                                df["upper"].iloc[start:end],
                            ),
                            color=hue,
                            capsize=5,
                            ls=ls,
                            alpha=alpha,
                        )
                    else:
                        ax.plot(
                            df["dims"].iloc[start:end],
                            df["median"].iloc[start:end],
                            color=hue,
                            ls=ls,
                            alpha=alpha,
                        )

            if log_y:
                ax.set_yscale("log")
            if log_x:
                ax.set_xscale("log")
            sdims = sorted(set(all_dims))
            if not sdims:
                continue
            ax.set_xticks(sdims)
            # make sure all ticks are shown
            ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
            assert len(sdims) > 0, sdims
            ax.set_xlim(xmin=sdims[0] - 1)  # , xmax=sdims[-1] + 1)

            if mi == 0:
                ax.set_title(problem_name)
            elif mi == nrows - 1:
                ax.set_xlabel("Dimensions")

            if pi == 0:
                ax.set_ylabel(metric_names.get(metric, metric))

            if show_success_rates:
                sr_ax.set_ylim((0, 1))
                sr_ax.set_ylabel("Success Rate")

    # add the thresholds to the legend
    sr_handles = []
    for i, (sr_threshold, ls) in enumerate(sr_thresholds):
        is_first = i == 0
        prev = 1.0 if is_first else sr_thresholds[i - 1][0]
        is_zero = sr_threshold == 0.0
        label = f"{'(['[is_first]}{prev}, {sr_threshold}{'])'[is_zero]}"

        sr_handles.append(
            axes.flat[0].plot([], [], ls=ls, color="black", label=label)[0]
        )
    l1 = fig.legend(
        title="Success Rate",
        handles=sr_handles,
        loc="center right",
        # ncols=len(sr_thresholds),
        # mode="expand",
        borderaxespad=0.0,
        frameon=False,
    )
    fig.legend(
        handles=list(method_handles.values()),
        loc="lower center",
        ncols=min(len(problems), len(methods)),
        mode="expand",
        borderaxespad=0.0,
        frameon=False,
    )
    fig.add_artist(l1)

    return fig
