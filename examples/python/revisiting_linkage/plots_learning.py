import pathlib
from operator import sub
from urllib.parse import quote

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygom
import seaborn as sns
from autorank import autorank, create_report, latex_table, plot_stats
from matplotlib.patches import Rectangle
from pygom import *
from rliable import library as rly
from rliable import metrics, plot_utils
from seaborn._core.typing import PaletteSpec
from sklearn import metrics as skm
from tqdm import tqdm

from src.config import c, extract, instantiate, load_config
from src.data import prepare_problem, problem_info
from src.plots import plot_convergence_so
from src.postprocessing import (
    load_results,
    rliable_convergence_score_dict,
    rliable_score_dict,
)
from src.run import compute_run_path, run_tasks

sns.set_theme(
    context="paper",
    style="ticks",
    font_scale=1.5,
    rc={
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    },
)

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42


RESULT_DIR = pathlib.Path("results") / "learning"
LOG_DIR = RESULT_DIR / "raw"
PARQUET_DIR = RESULT_DIR / "processed"
PLOT_DIR = RESULT_DIR / "plots"

# TODO rliable/baycomp comparison
# Per problem comparison


def method2name(m):
    return {"$MI_{mask\\ inactive}$": "$MI_{masked}$"}.get(m, m)


def pop_plot(
    conn,
    pdir,
    generations=(1, 2, 5, 10, 20),
    problem_query=r"format('{}{}', problem_name, IF(linear_scaling, '_LS', ''))",
    run_query=r"format('{}.{}', fold, run)",
):
    for problem, *_ in conn.sql(
        f"SELECT DISTINCT({problem_query}) AS problem FROM populations ORDER BY problem"
    ).fetchall():
        # if "exp" not in problem:
        #     continue
        for method, *_ in conn.execute(
            f"SELECT DISTINCT(method_name) AS method FROM populations WHERE {problem_query} = $1 ORDER BY method",
            [problem],
        ).fetchall():
            for ri, (run, *_) in list(
                enumerate(
                    conn.execute(
                        f"SELECT DISTINCT({run_query}) AS run FROM populations WHERE {problem_query} = $1 AND method_name = $2 ORDER BY run",
                        [problem, method],
                    ).fetchall()
                )
            )[:3]:
                # TODO population size for IMS??

                gens_available = [
                    g
                    for g, *_ in conn.execute(
                        f"""
                    SELECT
                     DISTINCT(current_population_generation) AS gen
                    FROM populations
                    WHERE {problem_query} = $1
                      AND method_name = $2
                      AND {run_query} = $3
                    ORDER BY gen
                    """,
                        [problem, method, run],
                    ).fetchall()
                ]

                actual_generations = sorted(
                    set(
                        [
                            min(gens_available, key=lambda g: abs(g0 - g))
                            for g0 in generations
                        ]
                    )
                )
                print(actual_generations, generations, gens_available)

                nrows = len(actual_generations)
                fig, axes = plt.subplots(
                    nrows=nrows,
                    ncols=3,
                    squeeze=False,
                    figsize=(12, nrows * 4),
                    gridspec_kw=dict(width_ratios=[1, 10, 10]),
                )

                for gi, generation in enumerate(actual_generations):
                    f_ax, d_ax, c_ax = axes[gi, :]
                    cbar_ax = axes[gi, :].ravel().tolist()

                    f_ax.set_ylabel(generation)

                    pop_df = conn.execute(
                        f"""
                        SELECT
                            discrete,
                            discrete_active,
                            continuous,
                            continuous_active,
                            expressions,
                            nmse_train
                        FROM populations
                        WHERE {problem_query} = $1
                          AND method_name = $2
                          AND {run_query} = $3
                          AND current_population_generation::UBIGINT = $4
                        ORDER BY nmse_train DESC
                        """,
                        [problem, method, run, generation],
                    ).df()

                    popsize = len(pop_df)
                    if popsize == 0:
                        print(problem, method, run, generation)
                        continue
                    fitnesses = pop_df["nmse_train"].values.reshape(popsize, 1)

                    print(problem, method, run, generation)
                    sns.heatmap(fitnesses, ax=f_ax)

                    d_active = np.array(
                        pop_df["discrete_active"].tolist(), dtype=np.bool_
                    )
                    d_values = np.array(pop_df["discrete"].tolist(), dtype=np.uint16)
                    sns.heatmap(
                        d_values,
                        cmap="tab20",
                        alpha=0.2,
                        ax=d_ax,
                        cbar=False,
                    )
                    sns.heatmap(d_values, mask=~d_active, cmap="tab20", ax=d_ax)

                    d_ax.set_title("Best: " + pop_df["expressions"][0])

                    c_active = np.array(
                        pop_df["continuous_active"].tolist(), dtype=np.bool_
                    )
                    c_values = np.array(pop_df["continuous"].tolist(), dtype=np.float64)
                    sns.heatmap(c_values, alpha=0.2, ax=c_ax, cbar=False)
                    sns.heatmap(c_values, mask=~c_active, ax=c_ax)

                odir = (
                    pdir
                    / "runs"
                    / quote(problem, safe=' (){}$_+-"')
                    / quote(method, safe=' (){}$_+-"')
                )
                odir.mkdir(parents=True, exist_ok=True)
                fig.savefig(
                    odir / f"{run}.pdf", dpi=600, bbox_inches="tight", transparent=True
                )

                # exit()


def main():
    preprocess = False
    with load_results(
        LOG_DIR,
        # enable pre-processing the .csv logs into .parquet files
        preprocess=preprocess,
        parquet_dir=PARQUET_DIR,
    ) as conn:
        load_results(
            LOG_DIR,
            file_pattern="subset_stats",
            table_name="fos_stats",
            types=dict(
                population_size="INTEGER",
                cluster="INTEGER",
                similarity="DOUBLE[][]",
                subsets="UINTEGER[][]",
                usage_count="UINTEGER[]",
                evaluation_rate="DOUBLE[]",
                acceptance_rate="DOUBLE[]",
                avg_improvement="DOUBLE[]",
                solution_activation_rate="DOUBLE[]",
                variables_activation_rate="DOUBLE[]",
            ),
            # enable pre-processing the .csv logs into .parquet files
            preprocess=preprocess,
            parquet_dir=PARQUET_DIR / "fos_stats",
            conn=conn,
        )
        load_results(
            LOG_DIR,
            file_pattern="population_stats",
            table_name="populations",
            # enable pre-processing the .csv logs into .parquet files
            preprocess=preprocess,
            parquet_dir=PARQUET_DIR / "populations",
            conn=conn,
        )
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        pop_plot(conn, PLOT_DIR)


if __name__ == "__main__":
    main()
