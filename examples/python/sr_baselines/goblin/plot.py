import os

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sns.set_theme(style="whitegrid")

RESULT_DIR = "../results"
# RESULT_DIR = "../raw"

PLOT_DIR = "../plots"

if __name__ == "__main__":
    os.makedirs(PLOT_DIR, exist_ok=True)

    # load data
    with duckdb.connect(":memory:") as conn:
        conn.sql(
            f"""
            CREATE OR REPLACE VIEW results AS SELECT * FROM
                read_csv('{RESULT_DIR}/*.csv', union_by_name=true)
            """
        )

        # make some plots
        methods = sorted(
            conn.execute("SELECT DISTINCT alg_name FROM results")
            .df()["alg_name"]
            .tolist()
        )
        palette = {
            m: c for m, c in zip(methods, sns.color_palette(n_colors=len(methods)))
        }

        metrics = ["r2_train", "r2_test", "time_seconds"]
        metric_labels = [r"$R^2$ Train", r"$R^2$ Test", "Time [s]"]

        for max_evals in tqdm(
            conn.execute("SELECT DISTINCT max_evals FROM results").df()["max_evals"]
        ):
            # for max_evals in df["max_evals"].unique():
            # df_subset = df[df["max_evals"] == max_evals]

            fig, axes = plt.subplots(
                nrows=2,
                ncols=len(metrics),
                figsize=(18, 10),
                sharex="col",  # , sharey="row"
            )
            fig.suptitle(
                f"{max_evals} Evaluations", fontsize=16, fontweight="bold", y=0.995
            )

            for row_idx, linear_scaling in enumerate([True, False]):
                # df_row = df_subset[df_subset["linear_scaling"] == linear_scaling]
                df_row = conn.execute(
                    "SELECT * FROM results WHERE max_evals = $1 AND linear_scaling = $2",
                    [max_evals, linear_scaling],
                ).df()

                for col_idx, (metric, metric_label) in enumerate(
                    zip(metrics, metric_labels)
                ):
                    ax = axes[row_idx, col_idx]

                    sns.pointplot(
                        data=df_row,
                        x=metric,
                        y="problem_name",
                        hue="alg_name",
                        units="run",
                        estimator=np.median,
                        errorbar=("ci", 95),
                        # err_kws=dict(linewidth=0.75),
                        palette=palette,
                        dodge=0.3 if df_row["alg_name"].nunique() > 1 else False,
                        log_scale=metric == "time_seconds",
                        # alpha=0.75,
                        linestyle="none",
                        legend=False,
                        ax=ax,
                    )

                    if row_idx == 0:
                        # ax.set_title(metric_label)
                        # ax.set_xticks([])
                        ax.set_xlabel("")
                    else:
                        ax.set_xlabel(metric_label)

                    if col_idx == 0:
                        ax.set_ylabel(
                            ("Linear Scaling\n" if linear_scaling else "") + "Dataset",
                            fontsize=10,
                            fontweight="bold",
                        )
                    else:
                        ax.set_ylabel("")
                        ax.set_yticklabels([])

                    if metric != "time_seconds":
                        ax.axvline(
                            0.0,
                            zorder=0,
                            color="black",
                            linestyle="dashed",
                            lw=0.75,
                            alpha=0.5,
                        )
                        # ax.set_xlim(-1.0, 1.0)
                        ax.set_xlim(0.0, 1.0)

                    ax.xaxis.grid(False)
                    ax.yaxis.grid(True)

            sns.despine(fig, left=True, bottom=True)

            fig.legend(
                handles=[
                    ax.plot([], [], label=m, color=palette[m])[0] for m in methods
                ],
                loc="lower center",
                ncols=min(len(metrics), len(methods)),
                mode="expand",
                borderaxespad=0.0,
                frameon=False,
            )

            for fmt in ["pdf", "png"]:
                fig.savefig(
                    f"{PLOT_DIR}/{max_evals}evals.{fmt}",
                    dpi=600,
                    bbox_inches="tight",
                    # transparent=True,
                )
