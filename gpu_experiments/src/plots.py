import os
from ast import literal_eval
from pathlib import Path
from typing import Literal

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

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


def plot_convergence(
    plot_dir: Path,
    conn: duckdb.DuckDBPyConnection,
    kernels: list[str] | None = None,
    kernel_query: str = "kernel",
    row_modifier_query: str | None = None,
    row_modifier_labels: list[str] | None = None,
    col_modifier_query: str | None = None,
    col_modifier_labels: list[str] | None = None,
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
    if row_modifier_query is not None:
        assert row_modifier_labels is not None

        row_modifiers = sorted(
            map(
                literal_eval,
                set(
                    str(m)
                    for m, *_ in conn.sql(
                        f"SELECT {row_modifier_query} FROM results"
                    ).fetchall()
                    if m is not None
                ),
            )
        )

        assert len(row_modifiers) > 0
        num_row_modifiers = len(row_modifiers[0])
        assert num_row_modifiers == len(row_modifier_labels)
    else:
        row_modifiers = None
        num_row_modifiers = 0
        row_modifier_query = "['']"

    if col_modifier_query is not None:
        assert col_modifier_labels is not None

        col_modifiers = sorted(
            map(
                literal_eval,
                set(
                    str(m)
                    for m, *_ in conn.sql(
                        f"SELECT {col_modifier_query} FROM results"
                    ).fetchall()
                    if m is not None
                ),
            )
        )

        assert len(col_modifiers) > 0
        num_col_modifiers = len(col_modifiers[0])
        assert num_col_modifiers == len(col_modifier_labels)
    else:
        col_modifiers = sorted(
            [
                p
                for p, *_ in conn.sql(
                    "SELECT DISTINCT(problem) FROM results"
                ).fetchall()
                if p is not None
            ]
        )

        col_modifier_labels = []
        num_col_modifiers = 0

    if kernels is None:
        kernels = sorted(
            [
                d
                for d, *_ in conn.sql("SELECT DISTINCT(kernel) FROM results").fetchall()
                if d is not None
            ]
        )

    if len(col_modifiers) == 0 or len(col_modifiers) == 0 or len(kernels) == 0:
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
                                WHERE {col_modifier_query} = $1
                                AND {row_modifier_query} = $2
                                LIMIT 1
                                """,
                            [col_modifier, row_modifier],
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
                                WHERE {col_modifier_query} = $1
                                LIMIT 1
                                """,
                            [col_modifier],
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
                            WHERE {col_modifier_query} = $1
                                AND {metric}::DOUBLE <= {x}::DOUBLE
                                {f"AND {row_modifier_query} = $2" if row_modifier else ""}
                            GROUP BY ALL
                            """
                        try:
                            part = conn.execute(
                                q,
                                [col_modifier, row_modifier]
                                if row_modifier
                                else [col_modifier],
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

                    # if row_mi == 0:
                    #     ax.set_title(col_modifier)

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
                            f"{col_modifier if not col_str else col_str}", fontsize=10
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
                # mode="expand",
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


# def plot_gpu_kernels(output_dir: Path, conn: duckdb.DuckDBPyConnection):
#     df = conn.execute(
#         """
#         SELECT
#             device,
#             population_size,
#             num_observations,
#             AVG(evaluations / total_time_seconds) AS evals_per_sec
#         FROM results
#         WHERE num_observations IN (10000, 100000)
#         GROUP BY device, population_size, num_observations
#         ORDER BY population_size, num_observations, device
#         """
#     ).fetchdf()

#     # Now plot using seaborn
#     plt.figure(figsize=(6, 12))
#     sns.set_style("whitegrid")

#     # Get unique number of observations for subplots
#     num_observations_list = sorted(df["num_observations"].unique())
#     num_obs = len(num_observations_list)

#     # Create subplots: 1 row, N columns (one per num_observations)
#     fig, axes = plt.subplots(1, num_obs, figsize=(6 * num_obs, 5), sharey=True)

#     if num_obs == 1:
#         axes = [axes]

#     y_ticks = [1e4, 1e5, 1e6]

#     for ax, obs in zip(axes, num_observations_list):
#         subset = df[df["num_observations"] == obs]
#         sns.barplot(
#             data=subset,
#             x="population_size",
#             y="evals_per_sec",
#             hue="device",
#             palette="tab10",
#             ax=ax,
#         )
#         ax.set_title(f"Num Observations = {obs}")
#         ax.set_xlabel("Population Size")
#         ax.set_ylabel("")  # Only label first subplot
#         ax.set_yscale("log")  # Set logarithmic y-axis
#         ax.set_yticks(y_ticks)  # Force ticks at 10^4, 10^5, 10^6
#         ax.tick_params(axis="y", which="major", labelsize=10)

#     axes[0].set_ylabel("Evaluations/sec")  # y-axis label only once
#     plt.tight_layout()

#     plt.savefig(output_dir / "kernel.png")


def plot(output_dir: Path):
    print("Starting plot creation...")

    db_path = output_dir / "experiments.duckdb"
    plot_dir = output_dir / "plots"

    conn = duckdb.connect(db_path)

    plot_convergence(
        plot_dir,
        conn,
        y_agg="MAX",
        ymin="auto",  # cuts of the lower 2.5% of data points
        ymax=None,
        # SQL can be used to derive values...
        y_var="1 - objectives[1]::DOUBLE / var_y::DOUBLE",
        y_label="$R^2$ Train",
        unit_query="format('{}.{}', fold, run)",
        # metrics=["evaluations / dims::DOUBLE", "total_time_seconds / dims::DOUBLE"],
        # metric_labels=[r"$\frac{Evaluations}{Dimensions}$", "Time/Dimensions [s]"],
        row_modifier_query="[template_height::STRING,operator_set::STRING]",
        row_modifier_labels=["Height", "Operators"],
        col_modifier_query="[num_observations::STRING, population_size::STRING]",
        col_modifier_labels=["#Observations", "#Individiuals"],
        # show_generation_boundaries=True,
        nsamples=100,
    )

    conn.close()

    print("Finished plot creation!")


# def plot_cpu_gpu(datetime: str, dataset_sizes: dict[str, dict[str, int]]):
#     path = Path("experiments/results/") / datetime

#     records = []

#     # Get information from result files
#     for file in path.glob("*-*-pop*-iter*.csv"):
#         df = pd.read_csv(file)

#         # Extract metadata
#         dataset, device, population, iteration = file.stem.split("-")

#         population = int(population.replace("pop", ""))
#         iteration = int(iteration.replace("iter", ""))

#         # Select converged row
#         converged = df[df["status"] == "Converged"]
#         if converged.empty:
#             continue

#         row = converged.iloc[0]

#         evals_per_sec = row["evaluations"] / row["eval_time_seconds"]

#         records.append(
#             {
#                 "dataset": dataset,
#                 "device": device,
#                 "population": population,
#                 "iteration": iteration,
#                 "evals_per_sec": evals_per_sec,
#             }
#         )

#     df_all = pd.DataFrame(records)

#     # Faceted grouped bar plot of evals per sec (CPU vs GPU)
#     g = sns.catplot(
#         data=df_all,
#         x="population",
#         y="evals_per_sec",
#         hue="device",
#         col="dataset",
#         kind="bar",
#         height=4,
#         aspect=1,
#         errorbar=("ci", 95),
#     )

#     g.set_axis_labels("Population Size", "Evaluations per Second")
#     g.set_titles("Dataset: {col_name}")
#     g._legend.set_title("Device")

#     plt.yscale("log")

#     plt.tight_layout()
#     plt.savefig(f"{path}/evals_per_second_barplot.png", dpi=300)

#     # Speedup plot (GPU / CPU)
#     df_mean = df_all.groupby(["dataset", "device", "population"], as_index=False).agg(
#         evals_per_sec=("evals_per_sec", "mean")
#     )

#     cpu = df_mean[df_mean["device"] == "cpu"]
#     gpu = df_mean[df_mean["device"] == "gpu"]

#     df_speedup = pd.merge(
#         cpu, gpu, on=["dataset", "population"], suffixes=("_cpu", "_gpu")
#     )

#     df_speedup["speedup"] = (
#         df_speedup["evals_per_sec_gpu"] / df_speedup["evals_per_sec_cpu"]
#     )

#     plt.figure(figsize=(8, 5))
#     sns.lineplot(
#         data=df_speedup, x="population", y="speedup", hue="dataset", marker="o"
#     )

#     plt.xscale("log", base=2)
#     plt.axhline(1.0, linestyle="--")
#     plt.xlabel("Population Size")
#     plt.ylabel("Speedup (GPU / CPU)")
#     plt.title("Acceleration Speedup vs Population Size")
#     plt.tight_layout()
#     plt.savefig(f"{path}/speedup_plot.png", dpi=300)


# def plot_kernel_versions(datetime: str):
#     records = []

#     path = Path("experiments/results/") / datetime

#     for file in path.glob("*-*-pop*-obs*-iter*.csv"):
#         df = pd.read_csv(file)

#         # Extract metadata
#         dataset, kernel_version, population, observations, iterations = file.stem.split(
#             "-"
#         )

#         population = int(population.replace("pop", ""))
#         observations = int(observations.replace("obs", ""))
#         iterations = int(iterations.replace("iter", ""))

#         converged = df[df["status"] == "Converged"]
#         if converged.empty:
#             continue

#         row = converged.iloc[0]
#         threads_per_sec = (row["evaluations"] * observations * population) / row[
#             "eval_time_seconds"
#         ]

#         records.append(
#             {
#                 "dataset": dataset,
#                 "version": kernel_version,
#                 "observations": observations,
#                 "iteration": iterations,
#                 "threads_per_sec": threads_per_sec,
#             }
#         )

#     df_all = pd.DataFrame(records)

#     df_mean = df_all.groupby(["version", "observations"], as_index=False).agg(
#         threads_per_sec=("threads_per_sec", "mean")
#     )

#     versions = df_all["version"].unique()
#     palette = sns.color_palette("tab10", n_colors=len(versions))
#     color_map = {v: palette[i] for i, v in enumerate(versions)}

#     sns.scatterplot(
#         data=df_all,
#         x="observations",
#         y="threads_per_sec",
#         hue="version",
#         palette=color_map,
#         alpha=0.7,
#     )

#     for version in df_mean["version"].unique():
#         subset = df_mean[df_mean["version"] == version]
#         sns.regplot(
#             data=subset,
#             x="observations",
#             y="threads_per_sec",
#             scatter=False,
#             logx=True,
#             color=color_map[version],
#             label=f"{version} trend",
#         )

#     plt.xscale("log")
#     plt.xlabel("Number of Observations (log scale)")
#     plt.ylabel("Threads per Second")
#     plt.title("Kernel Versions vs Number of Observations")
#     plt.tight_layout()
#     plt.savefig(f"{path}/kernel_scaling.png", dpi=300)

#     # Pivot baseline
#     baseline = df_mean[df_mean["version"] == "baseline"][
#         ["observations", "threads_per_sec"]
#     ]
#     baseline = baseline.rename(columns={"threads_per_sec": "baseline_threads"})

#     # Merge baseline back
#     df_relative = pd.merge(df_mean, baseline, on="observations")
#     df_relative["relative_perf"] = (
#         df_relative["threads_per_sec"] / df_relative["baseline_threads"]
#     )

#     # Plot
#     plt.figure(figsize=(8, 5))
#     sns.lineplot(
#         data=df_relative,
#         x="observations",
#         y="relative_perf",
#         hue="version",
#         marker="o",
#     )

#     plt.xscale("log")
#     plt.xlabel("Number of Observations (log scale)")
#     plt.ylabel("Relative Performance vs Baseline")
#     plt.title("Kernel Versions Relative Performance")
#     # plt.axhline(1.0, linestyle="--", color="gray", label="Baseline")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f"{path}/kernel_relative_performance.png", dpi=300)
#     # plt.show()
