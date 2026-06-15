import pathlib
import re
from urllib.parse import quote

import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygom
import seaborn as sns
from matplotlib.colors import AsinhNorm, Normalize, SymLogNorm
from matplotlib.lines import Line2D
from pygom import *
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg._isolve.lsqr import eps
from src.config import c, instantiate
from src.data import prepare_problem, problem_info
from src.plots import plot_convergence_so
from src.postprocessing import load_results
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

REPEATS_PER_DATASET = 30
NUM_FOLDS = 5

REPEATS_PER_FOLD = REPEATS_PER_DATASET // NUM_FOLDS

RESULT_DIR = pathlib.Path("results") / "linkage_details"
RESULT_DIR = pathlib.Path("results") / "linkage_details_large"
DATA_DIR = RESULT_DIR / "data"
LOG_DIR = RESULT_DIR / "raw"
PARQUET_DIR = RESULT_DIR / "processed"
PLOT_DIR = RESULT_DIR / "plots"

SEED = 42


def method2name(m: str) -> str:
    return {
        "$MI_{mask\\ inactive}$": "$MI_{masked}$",
        "Node (peter)": "#Common Subfunctions",
    }.get(m, m)


def problem2name(p: str) -> str:
    return {
        "Concrete Compressive Strength": "Concrete",
    }.get(p, p)


m_order = {
    m: i
    for i, m in enumerate(
        [
            "Random",
            "$MI$",
            "$MI_{adjusted}$",
            "$MI_{masked}$",
            "Node",
            "Node (static)",
        ]
    )
}


def analyze_subset_stats(
    conn,
    odir,
    problem_query="problem_name",
    cmap="Blues",
    shared_norms=None,  # "problem",
    where_query: str = r"method_name SIMILAR TO '.*mask.*' AND linear_scaling::BOOL",
):
    def fmt_name(name):
        rm = re.match(r"^GP-GOMEA \((.+)\)$", name)
        n = rm.group(1) if rm else name
        n = method2name(n)
        return "Node / Node (static)" if n == "Node" else n

    problems = sorted(
        [
            p
            for p, *_ in conn.sql(
                f"SELECT DISTINCT({problem_query}) FROM fos_stats WHERE {where_query}"
            ).fetchall()
        ]
    )

    methods = sorted(
        [
            m
            for m, *_ in conn.execute(
                f"SELECT DISTINCT(method_name) FROM fos_stats WHERE {where_query}",
            ).fetchall()
        ]
    )

    assert len(methods) == 1

    method = methods[0]

    generations = [5, 10, 20]

    nrows = len(generations)
    ncols = len(problems)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=True,
        sharey=True,
        figsize=(ncols * 3.3, nrows * 3),
        gridspec_kw=dict(wspace=0.1, hspace=0.1),
    )

    def compute_average_similarity(problem, generation):
        stats = conn.execute(
            f"""
            SELECT
                population_size,
                cluster,
                similarity,
                subsets,
                usage_count,
                evaluation_rate,
                acceptance_rate,
                avg_improvement,
                solution_activation_rate,
                variables_activation_rate,
                format('{{}}.{{}}', fold, seed) as run
            FROM fos_stats
            WHERE
                {problem_query} = $1
                AND method_name = $2
                AND generation = $3
                AND {where_query}
            ORDER BY run
            """,
            [problem, method, generation],
        ).df()

        assert len(stats) > 0

        _similarities = [
            np.array([r.tolist() for r in s]) for s in (stats["similarity"])
        ]

        assert len(_similarities) >= 1

        avg_similarity = _similarities[0]
        for s in _similarities[1:]:
            avg_similarity += s
        avg_similarity /= len(_similarities)

        return avg_similarity

    mask = np.eye(31, dtype=np.bool_)

    def compute_norm(similarities):
        vmin, vmax = np.inf, -np.inf
        ix = ~mask
        for s in similarities:
            vmin = min(vmin, np.min(s[ix]))
            vmax = max(vmax, np.max(s[ix]))
        return Normalize(vmin=vmin, vmax=vmax)

    method_name = fmt_name(method)

    heatmap_kw = dict(
        square=True,
        linewidths=0,
    )

    def add_cbar(cax, norm, **kwargs):
        cbar = cm.ScalarMappable(cmap=cmap, norm=norm)
        kw = dict(
            shrink=0.6,
            aspect=30,
            drawedges=False,
        )
        for k, v in kwargs.items():
            kw[k] = v
        cb = fig.colorbar(
            cbar,
            ax=cax,
            **kw,
        )
        cb.outline.set_linewidth(0.0)
        cb.ax.tick_params(
            labelsize="xx-small",
            pad=1,
            length=2,
            width=1,
        )

    def add_ticks(ax, sim):
        ticks, ticklabels = zip(*[(t + 0.5, str(t)) for t in range(0, sim.shape[0], 5)])
        ax.set_xticks(ticks, labels=ticklabels, fontsize="xx-small")
        ax.set_yticks(ticks, labels=ticklabels, fontsize="xx-small")
        ax.tick_params(
            axis="both",
            which="major",
            bottom=True,
            left=True,
            length=2,
            width=1,
        )

    for c, problem in enumerate(problems):
        psims = [compute_average_similarity(problem, g) for g in generations]
        pnorm = compute_norm(psims)

        for r, generation in enumerate(generations):
            ax = axes[r, c]
            sim = psims[r]

            if shared_norms == "problem":
                sns.heatmap(
                    sim,
                    mask=mask,
                    cmap=cmap,
                    norm=pnorm,
                    ax=ax,
                    cbar=False,
                    **heatmap_kw,
                )
            else:
                sns.heatmap(
                    sim,
                    mask=mask,
                    cmap=cmap,
                    ax=ax,
                    cbar=False,
                    # cbar=True,
                    # cbar_kws=dict(
                    #     shrink=0.6,
                    #     aspect=30,
                    #     drawedges=False,
                    #     labelsize="xx-small",
                    #     # pad=1,
                    #     length=2,
                    #     width=1,
                    # ),
                    **heatmap_kw,
                )
                add_cbar(axes[r, c], compute_norm([sim]), location="right", pad=0.0175)

            add_ticks(ax, sim)

            if r == 0:
                ax.set_title(problem2name(problem))

            ax.set_ylabel("")
            if c == 0:
                ax.set_ylabel(f"Generation {generation}")

            ax.set_xlabel("")

        if shared_norms == "problem":
            add_cbar(axes[:, c], pnorm, location="right", pad=0.0175)

    for i in range(len(problems) - 1):
        ptl = axes[0, i].get_position()
        ptr = axes[0, i + 1].get_position()
        pbl = axes[-1, i].get_position()
        yr = ptl.y1 - pbl.y0
        p = 0.02

        x = (ptl.x1 + ptr.x0) / 2 + 0.0075
        fig.add_artist(
            Line2D(
                [x, x],
                [
                    pbl.y0 - p * yr,
                    ptl.y1 + p * yr,
                ],
                transform=fig.transFigure,
                color="black",
                # ls="--",
                alpha=0.75,
                lw=2,
            )
        )

    fig.supxlabel(method_name, y=0.05)

    odir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        odir / "similarities_problems.pdf",
        dpi=600,
        bbox_inches="tight",
        transparent=True,
    )

    plt.close(fig)


def main():
    preprocess = False
    with load_results(
        LOG_DIR,
        file_pattern="stats",
        # enable pre-processing the .csv logs into .parquet files
        preprocess=preprocess,
        parquet_dir=PARQUET_DIR / "stats",
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

        PLOT_DIR.mkdir(parents=True, exist_ok=True)

        with sns.axes_style("white"):
            analyze_subset_stats(conn, PLOT_DIR / "stats_presentation")


if __name__ == "__main__":
    main()
