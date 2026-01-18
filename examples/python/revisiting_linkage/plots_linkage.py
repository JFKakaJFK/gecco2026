import pathlib
from operator import sub

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygom
import seaborn as sns
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


RESULT_DIR = pathlib.Path("results") / "linkage"  #  "linkage_wrong_init"
LOG_DIR = RESULT_DIR / "raw"
PARQUET_DIR = RESULT_DIR / "processed"
PLOT_DIR = RESULT_DIR / "plots"

# TODO rliable/baycomp comparison
# Per problem comparison


def method2name(m):
    return {"$MI_{mask\\ inactive}$": "$MI_{masked}$"}.get(m, m)


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

# colorblind-friendly colors from https://www.nature.com/articles/nmeth.1618.pdf
PALETTE = {
    "Random": "#E69F00",  # orange
    "$MI$": "#009E73",  # bluish green
    "$MI_{adjusted}$": "#CC79A7",  # reddish purple
    "$MI_{masked}$": "#D55E00",  # vermillion red
    "Node": "#0072B2",  # blue
    "Node (static)": "#56B4E9",  # sky blue
    # "#F0E442", # yellow
}


def custom_problem_convergence_plot(
    conn,
    plot_dir,
    y: str = "1.0 - nmse_train",
    y_agg: str = "MAX",
    ylabel: str = "$R^2$ Train",
    metrics=["evaluations", "total_time_seconds"],
    metric_labels=["Evaluations", "Approximate Runtime [s]"],
    run_expr: str = r"format('{}.{}', fold, seed)",
    where_query: str = r"method_name NOT SIMILAR TO '.*(\*|any|all).*'",
    problem_query: str = "problem_name",
    modifier_query: str = r"format('H={}{}', template_height, IF(linear_scaling, ' LS', ''))",
    num_samples: int = 100,
):
    progress = tqdm(total=1)
    pdir = plot_dir / "custom"
    pdir.mkdir(parents=True, exist_ok=True)

    methods = [
        m
        for m, *_ in conn.sql(
            f"SELECT DISTINCT(method_name) AS method FROM results WHERE {where_query} ORDER by method"
        ).fetchall()
    ]

    problems = sorted(
        [
            p
            for p, *_ in conn.sql(
                f"SELECT DISTINCT({problem_query}) AS problem FROM results WHERE {where_query} ORDER by problem"
            ).fetchall()
        ]
    )

    modifiers = sorted(
        [
            m
            for m, *_ in conn.sql(
                f"SELECT DISTINCT({modifier_query}) AS modifier FROM results WHERE {where_query} ORDER by modifier"
            ).fetchall()
        ]
    )

    algorithms = sorted(set([method2name(m) for m in methods]))
    keys = {m: m_order.get(m, len(m_order) + i) for i, m in enumerate(algorithms)}
    algorithms = sorted(set(algorithms), key=lambda m: keys[m])

    hues = sns.color_palette(  #
        "colorblind", n_colors=len(algorithms)
    )
    palette = {a: h for a, h in zip(algorithms, hues)}

    palette = PALETTE

    nrows = len(modifiers)
    ncols = len(problems)

    fig, main_axes = plt.subplots(
        nrows=len(metrics),
        ncols=1,
        figsize=(4 * ncols, 2.75 * nrows * len(metrics)),
        layout="constrained",
        gridspec_kw=dict(hspace=0.05),
    )

    gs = main_axes[0].get_subplotspec().get_gridspec()

    for midx, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        subfig = fig.add_subfigure(gs[midx, :])

        axes = subfig.subplots(nrows=nrows, ncols=ncols, squeeze=False)
        for row, modifier in enumerate(modifiers):
            for col, problem in enumerate(problems):
                ax = axes[row, col]

                all_dfs = []
                for method in methods:
                    alg = method2name(method)
                    hue = palette[alg]

                    xlim = conn.execute(
                        f"""
                    SELECT
                        -- MIN({metric}::DOUBLE),
                        quantile({metric}::DOUBLE, 0.01),
                        MAX({metric}::DOUBLE)
                    FROM results
                    WHERE {modifier_query} = $1
                      AND {problem_query} = $2
                      AND method_name = $3
                      AND {where_query}
                    """,
                        [modifier, problem, method],
                    ).fetchone()

                    first_done_x, *_ = conn.execute(
                        f"""
                    SELECT
                        MIN({metric}::DOUBLE)
                    FROM results
                    WHERE {modifier_query} = $1
                      AND {problem_query} = $2
                      AND method_name = $3
                      AND {where_query}
                      AND status != 'Running'
                      AND status != 'Aborted'
                    """,
                        [modifier, problem, method],
                    ).fetchone()
                    first_done_y = (
                        conn.execute(
                            f"""
                            SELECT
                                {y_agg}({y}) AS value,
                                {run_expr} AS run
                            FROM results
                            WHERE {modifier_query} = $1
                            AND {problem_query} = $2
                            AND method_name = $3
                            AND {metric}::DOUBLE <= {first_done_x}::DOUBLE
                            AND {where_query}
                            GROUP BY all
                            """,
                            [modifier, problem, method],
                        )
                        .df()["value"]
                        .median()
                    )

                    df = pd.concat(
                        [
                            conn.execute(
                                f"""
                        SELECT
                            {y_agg}({y}) AS value,
                            {run_expr} AS run,
                            {x}::DOUBLE AS metric,
                        FROM results
                        WHERE {modifier_query} = $1
                          AND {problem_query} = $2
                          AND method_name = $3
                          AND {metric}::DOUBLE <= {x}::DOUBLE
                          AND {where_query}
                        GROUP BY all
                        """,
                                [modifier, problem, method],
                            ).df()
                            for x in np.linspace(*xlim, num_samples)
                        ],
                        ignore_index=True,
                    )

                    sns.lineplot(
                        df,
                        x="metric",
                        y="value",
                        color=hue,
                        # units="unit",
                        # estimator=None,
                        lw=1.5,
                        estimator=np.median,
                        errorbar=("pi", 50),
                        err_kws=dict(linewidth=0),
                        legend=False,
                        ax=ax,
                    )

                    if first_done_x < xlim[1]:
                        ax.scatter(
                            first_done_x,
                            first_done_y,
                            color=hue,
                            marker="x",
                            s=100,
                            # lw=0.5,
                            zorder=0,
                        )

                    all_dfs.append(df)

                adf = pd.concat(all_dfs, ignore_index=True)
                truncation_ratio = 0.05
                ylim = (
                    adf["value"].quantile(truncation_ratio / 2),
                    min(
                        1.0, adf["value"].max()
                    ),  # adf["value"].quantile(1.0 - truncation_ratio / 2),
                )
                ax.set_ylim(*ylim)

                if row == 0:  # and midx == 0:
                    ax.set_title(problem)

                ax.set_xlabel(
                    # metric_label if row + 1 == nrows else
                    ""
                )
                ax.set_ylabel(ylabel + "\n" + modifier if col == 0 else "")

                sns.despine(ax=ax)

                ax.grid(
                    visible=True,
                    which="major",
                    axis="both",
                    lw=0.5,
                    color="black",
                    alpha=0.2,
                )

        subfig.supxlabel(metric_label, y=-0.0225)

    labels, handles = [], []
    for alg in algorithms:
        h, *_ = main_axes.flat[0].plot([], [], color=palette[alg], lw=3)
        labels.append(alg)
        handles.append(h)

    for a in main_axes:
        a.remove()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncols=min(6, len(algorithms)),
        borderaxespad=0.0,
        labelspacing=0,
        frameon=False,
    )

    fig.savefig(
        pdir / "convergence_per_problem.pdf",
        dpi=600,
        bbox_inches="tight",
        transparent=True,
    )

    plt.clf()

    progress.update()


def custom_convergence_plot(
    conn,
    plot_dir,
    metric: str = "1.0 - nmse_train",
    metric_label: str = "$R^2$ Train",
    run_expr: str = r"format('{}.{}', fold, seed)",
    where_query: str = r"method_name NOT SIMILAR TO '.*(\*|any|all).*'",
    problem_query: str = r"format('{}.{}.{}', problem_name, template_height, linear_scaling)",
    rows=[  #
        (
            "Evaluations",
            "evaluations",
            [
                # 100_000,
                500_000,
                int(1e6),
                int(2e6),
                int(5e6),
                # int(7.5e6),
                int(1e7),
            ],
            lambda v: f"{v:.0e}".replace("e+0", "e"),
            # lambda v: f"$10^{{{int(np.log10(v))}}}$",
        ),
        (
            "Approximate Runtime",
            "total_time_seconds",
            [
                # 0.5 * 60,
                60,
                5 * 60,
                10 * 60,
                #
                20 * 60,
                #
                30 * 60,
            ],
            lambda v: f"{int(v / 60)}min" if v >= 60 else f"{int(v)}s",
        ),
    ],
    use_non_linear_scaling: bool = False,  # True,
    minor_ticks: bool = True,
    filename_suffix="",
):
    progress = tqdm(total=1)
    pdir = plot_dir / "custom"
    pdir.mkdir(parents=True, exist_ok=True)

    methods = [
        m
        for m, *_ in conn.sql(
            f"SELECT DISTINCT(method_name) AS method FROM results WHERE {where_query} ORDER by method"
        ).fetchall()
    ]

    max_evals = conn.sql("SELECT MAX(evaluations) FROM results").fetchone()[0]

    algorithms = []
    row_data = []
    for _, column, xticks, _ in rows + [
        (
            "",
            "evaluations",
            [max_evals],
            None,
        )
    ]:
        row_ticks = []
        if minor_ticks:
            row_ticks = [xticks[0]]
            i, t = 1, row_ticks[-1]
            while i < len(xticks):
                t += xticks[0]
                if t < xticks[i]:
                    row_ticks.append(t)
                else:
                    row_ticks.append(xticks[i])
                    i += 1
        else:
            row_ticks = xticks

        score_dict = {}
        for method in methods:
            data = None

            for i, value in enumerate(row_ticks):
                df = (
                    conn.execute(
                        f"""
                        SELECT
                            {run_expr} AS run,
                            {problem_query} AS problem,
                            MAX({metric}) AS value
                        FROM results
                        WHERE {column} <= $2
                            AND method_name = $1
                            AND {where_query}
                        GROUP BY all
                        ORDER BY run, problem
                        """,
                        [method, value],
                    )
                    .df()
                    .pivot(index="run", values="value", columns="problem")
                )

                if data is None:
                    data = np.empty((*df.values.shape, len(row_ticks)))

                data[:, :, i] = df.values

            if data is not None:
                alg = method2name(method)
                score_dict[alg] = data
                algorithms.append(alg)

        row_data.append((score_dict, row_ticks))

    algorithms = sorted(set(algorithms))
    keys = {m: m_order.get(m, len(m_order) + i) for i, m in enumerate(algorithms)}
    algorithms = sorted(set(algorithms), key=lambda m: keys[m])

    hues = sns.color_palette(  #
        "colorblind", n_colors=len(algorithms)
    )
    palette = {a: h for a, h in zip(algorithms, hues)}

    palette = PALETTE

    line_style = dict(marker="o", lw=2)

    nrows = len(rows)
    fig, axes = plt.subplots(
        nrows=nrows,
        # sharey=True,
        figsize=(11, 3 * nrows),
    )

    for row, (xlabel, column, xticks, fmt) in enumerate(rows):
        score_dict, row_ticks = row_data[row]
        ax = axes[row]

        iqm = lambda scores: np.array(
            [
                metrics.aggregate_iqm(scores[..., frame])
                for frame in range(scores.shape[-1])
            ]
        )
        iqm_scores, iqm_cis = rly.get_interval_estimates(
            score_dict, iqm, reps=5000
        )  # 0)

        last_iqm_scores, last_iqm_cis = rly.get_interval_estimates(
            row_data[-1][0], iqm, reps=5000
        )

        actual_xticks = (
            list(range(len(row_ticks))) if use_non_linear_scaling else row_ticks
        )

        for alg, metric_values in iqm_scores.items():
            lower, upper = iqm_cis[alg]

            ax.plot(actual_xticks, metric_values, color=palette[alg], **line_style)
            ax.fill_between(
                actual_xticks, y1=lower, y2=upper, color=palette[alg], alpha=0.2, lw=0
            )

        x_end = (
            max_evals
            if column == "evaluations"
            else ax.transData.inverted().transform(ax.transAxes.transform((1.0, 0.0)))[
                0
            ]
        )
        for alg, metric_values in iqm_scores.items():
            lower, upper = iqm_cis[alg]

            last_center = last_iqm_scores[alg]
            last_lower, last_upper = last_iqm_cis[alg]

            last_style = {**line_style}
            last_style["marker"] = None
            # last_style["lw"] = 1.0
            ax.plot(
                [actual_xticks[-1], x_end],
                [metric_values[-1], last_center[0]],
                color=palette[alg],
                ls="dotted",
                **last_style,
            )
            ax.scatter([x_end], [last_center[0]], color=palette[alg], marker="x", s=50)
            ax.fill_between(
                [actual_xticks[-1], x_end],
                y1=[lower[-1], last_lower[0]],
                y2=[upper[-1], last_upper[0]],
                color=palette[alg],
                alpha=0.2,
                lw=0,
            )

            if alg in [
                "Random",
                "$MI$",
                "$MI_{adjusted}$",
            ]:
                ax.axhline(
                    last_center[0],
                    color=palette[alg],
                    alpha=0.75,
                    linestyle="dashed",
                    zorder=0,
                )
        ax.axvline(x_end, color="black", alpha=0.5, zorder=0)  # , linestyle="dashed")

        # frames = np.array(xticks)
        # plot_utils.plot_sample_efficiency_curve(
        #     frames + 1,
        #     iqm_scores,
        #     iqm_cis,
        #     algorithms=algorithms,
        #     xlabel=xlabel,
        #     ylabel=metric_label,
        #     ax=ax,
        # )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("IQM " + metric_label)

        major_ticks = list(range(len(xticks))) if use_non_linear_scaling else xticks
        ax.set_xticks(major_ticks, [fmt(v) for v in xticks])

        ax.grid(
            visible=True,
            which="major",
            axis="both",
            lw=0.5,
            color="black",
            alpha=0.2,
        )

        sns.despine(ax=ax)

    labels, handles = [], []
    for alg in algorithms:
        h, *_ = axes.flat[0].plot([], [], color=palette[alg], lw=2)
        labels.append(alg)
        handles.append(h)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.475, -0.03),
        ncols=min(6, len(algorithms)),
        borderaxespad=0.0,
        labelspacing=0,
        frameon=False,
    )

    fig.savefig(
        pdir / f"convergence{filename_suffix}.pdf",
        dpi=600,
        bbox_inches="tight",
        transparent=True,
    )

    plt.clf()

    progress.update()


def custom_cmp_plot(
    conn,
    plot_dir,
    metric: str = "1.0 - nmse_train",
    metric_label: str = "$R^2$ Train",
    method_where_query: str = r"method_name NOT SIMILAR TO '.*(\*|any|all).*'",
    problem_query: str = r"format('{}.{}.{}', problem_name, template_height, linear_scaling)",
    max_evaluations: int | tuple[int, int] = int(1e7),  # (int(1e7), int(5e5)),
    filename_suffix="",
):
    progress = tqdm(total=1)
    pdir = plot_dir / "custom"
    pdir.mkdir(parents=True, exist_ok=True)

    if isinstance(max_evaluations, int):
        max_evaluations = [max_evaluations]

    score_dicts = []

    algorithms = []
    for evals in max_evaluations:
        score_dict, _ = rliable_score_dict(
            conn,
            run_expr="format('{}.{}', fold, run)",
            problem_query=problem_query,
            where_query=method_where_query + f"AND evaluations <= {evals}",
            normalized_value_expr=metric,
            finished_runs_only=False,
        )

        score_dict = {method2name(m): s for m, s in score_dict.items()}

        algorithms += sorted(score_dict.keys())

        score_dicts.append(score_dict)

    algorithms = sorted(set(algorithms))
    keys = {m: m_order.get(m, len(m_order) + i) for i, m in enumerate(algorithms)}
    algorithms = sorted(set(algorithms), key=lambda m: keys[m])

    fig, ax = plt.subplots(figsize=(7, 6))

    for is_upper, score_dict in enumerate(score_dicts):
        comparisons = {
            (algorithms[i], algorithms[j]): (
                score_dict[algorithms[i]],
                score_dict[algorithms[j]],
            )
            for i in range(len(algorithms))
            for j in range(i)
        }
        average_probabilities, average_prob_cis = rly.get_interval_estimates(
            comparisons, metrics.probability_of_improvement, reps=2000
        )

        P = np.empty((len(algorithms), len(algorithms)))
        M = np.triu(np.ones_like(P, dtype=np.bool_))

        A = [["" for _ in algorithms] for _ in algorithms]

        for i, a in enumerate(algorithms):
            for j, b in enumerate(algorithms):
                if j >= i:
                    continue

                P[i, j] = P[j, i] = (
                    1.0 - average_probabilities[(a, b)]
                    if is_upper
                    else average_probabilities[(a, b)]
                )

                lower, upper = average_prob_cis[(a, b)]
                if is_upper:
                    lower, upper = 1.0 - lower, 1.0 - upper

                A[j][i] = A[i][j] = (
                    f"{float(P[i, j]):.2f}\n${{}}_{{[{float(lower):.2f},{float(upper):.2f}]}}$"
                )
        A = np.array(A)

        if len(score_dicts) == 1:
            data, mask, annot, xticklabels, yticklabels = (
                P[1:, :-1],
                M[1:, :-1],
                A[1:, :-1],
                algorithms[:-1],
                algorithms[1:],
            )
        elif is_upper:
            data, mask, annot, xticklabels, yticklabels = (
                P.T,
                M.T,
                A.T,
                algorithms,
                algorithms,
            )
        else:
            data, mask, annot, xticklabels, yticklabels = (
                P,
                M,
                A,
                algorithms,
                algorithms,
            )

        print(is_upper, max_evaluations[is_upper], data, mask)

        sns.heatmap(
            data,
            mask=mask,
            annot=annot,
            fmt="s",
            # annot_kws=dict(fontsize="x-small"),
            annot_kws=dict(fontsize="small"),
            # annot=True,
            # mask=np.eye(len(algorithms), dtype=np.bool_),
            vmin=0,
            vmax=1,
            square=True,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            cbar_kws=dict(shrink=0.95, label="P(A > B)"),
            cbar=not is_upper,
            ax=ax,
        )

        if len(score_dicts) > 1:
            ax.text(
                x=0.8 if is_upper else -0.15,
                y=1.03 if is_upper else -0.1,
                s=f"{max_evaluations[is_upper]}{['\n', ' '][is_upper]}Evaluations",
                ha="center",
                va="center",
                size="medium",
                transform=ax.transAxes,
            )

    # ax.text(
    #     x=0.75,
    #     y=0.75,
    #     s=r"P(A > B)",
    #     ha="center",
    #     va="center",
    #     size="large",
    #     transform=ax.transAxes,
    # )

    # ax.text(
    #     x=0.75,
    #     y=0.75,
    #     s=r"P(B > A) = 1 - P(A > B)",
    #     ha="center",
    #     va="center",
    #     size="medium",
    #     transform=ax.transAxes,
    # )

    # ax.set_xticks(
    #     [i + 0.5 for i in range(len(algorithms) - 1)],
    #     labels=algorithms[:-1],
    #     rotation=-15,
    #     ha="left",
    #     rotation_mode="anchor",
    # )

    ax.set_ylabel("A")
    ax.set_xlabel("B")

    if len(score_dicts) > 1:
        ax.set_title("B")
        ax.text(
            x=1.03,
            y=0.5,
            s="A",
            ha="center",
            va="center",
            size="medium",
            transform=ax.transAxes,
        )

    fig.align_labels()

    fig.savefig(
        pdir / f"cmp{filename_suffix}.pdf",
        dpi=600,
        bbox_inches="tight",
        transparent=True,
    )

    plt.clf()

    progress.update()


def custom_pprof_plot(
    conn,
    plot_dir,
    metric: str = "1.0 - nmse_train",
    metric_label: str = "$R^2$ Train",
    method_where_query: str = r"method_name NOT SIMILAR TO '.*(\*|any|all).*'",
    problem_query: str = "problem_name",
    rows: list[tuple[str, str]] = [
        ("Height = 5", r"template_height::INTEGER = 5::INTEGER"),
        ("Height = 7", r"template_height::INTEGER = 7::INTEGER"),
    ],
    cols: list[tuple[str, str]] = [
        ("Without Linear Scaling", r"linear_scaling::BOOLEAN = false"),
        ("With Linear Scaling", r"linear_scaling::BOOLEAN = true"),
    ],
    zooms: dict = {  #
        (0, 1): ((0.775, 0.1), 0.12, 0.8),
        (1, 1): ((0.75, 0.15), 0.17, 0.75),
    },
    filename_suffix="",
):
    progress = tqdm(total=1)
    pdir = plot_dir / "custom"
    pdir.mkdir(parents=True, exist_ok=True)

    algorithms = []
    row_data = []
    for row_label, row_query in rows:
        col_data = []
        for col_label, col_query in cols:
            where_query = " AND ".join(
                q for q in [row_query, col_query, method_where_query] if q
            )

            score_dict, problems = rliable_score_dict(
                conn,
                run_expr="format('{}.{}', fold, run)",
                problem_query=problem_query,
                where_query=where_query,
                normalized_value_expr=metric,
            )

            score_dict = {method2name(m): s for m, s in score_dict.items()}

            algorithms += score_dict.keys()

            col_data.append(
                score_dict,
            )
        row_data.append(col_data)

    algorithms = sorted(set(algorithms))
    keys = {m: m_order.get(m, len(m_order) + i) for i, m in enumerate(algorithms)}
    algorithms = sorted(set(algorithms), key=lambda m: keys[m])

    hues = sns.color_palette(  #
        "colorblind", n_colors=len(algorithms)
    )
    palette = {a: h for a, h in zip(algorithms, hues)}

    palette = PALETTE

    nrows = len(rows)
    ncols = len(cols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        sharex=True,
        sharey=True,
        squeeze=False,
        figsize=(ncols * 7, nrows * 3),
        gridspec_kw=dict(wspace=0.05, hspace=0.1),
    )

    aucs = {a: np.empty((nrows, ncols)) for a in algorithms}

    line_style = dict(lw=1.5)
    ci_style = dict(alpha=0.15, lw=0.0)

    for row, (col_data) in enumerate(row_data):
        row_label = rows[row][0]

        for col, score_dict in enumerate(col_data):
            col_label = cols[col][0]

            ax = axes[row, col]

            auc_thresholds = np.linspace(0.0, 1.0, 50)
            auc_distributions, _ = rly.create_performance_profile(
                score_dict, auc_thresholds
            )

            thresholds = np.linspace(0.35, 0.95, 100)
            score_distributions, score_distributions_cis = (
                rly.create_performance_profile(score_dict, thresholds)
            )

            # plot_utils.plot_performance_profiles(
            #     score_distributions,
            #     thresholds,
            #     performance_profile_cis=score_distributions_cis,
            #     # use_non_linear_scaling=True,
            #     # xticks=[],
            #     colors=palette,
            #     ax=ax,
            # )

            for method, profile in score_distributions.items():
                ax.plot(thresholds, profile, color=palette[method], **line_style)
                lower_ci, upper_ci = score_distributions_cis[method]
                ax.fill_between(
                    thresholds, lower_ci, upper_ci, color=palette[method], **ci_style
                )

                aucs[method][row, col] = skm.auc(
                    auc_thresholds, auc_distributions[method]
                )

            if (row, col) in zooms:
                xy, w, h = zooms[(row, col)]

                zax = ax.inset_axes(
                    [
                        0.1,  # x0
                        0.15,  # y0
                        0.4,  # width
                        0.6,  # height
                    ]
                )

                zoom_taus = np.linspace(xy[0], xy[0] + w, 100)
                score_distributions, score_distributions_cis = (
                    rly.create_performance_profile(score_dict, zoom_taus)
                )

                zls = {**line_style}
                zls["lw"] = 1.0

                for method, profile in score_distributions.items():
                    zax.plot(zoom_taus, profile, color=palette[method], **zls)
                    lower_ci, upper_ci = score_distributions_cis[method]
                    zax.fill_between(
                        zoom_taus, lower_ci, upper_ci, color=palette[method], **ci_style
                    )

                ax.indicate_inset_zoom(zax, edgecolor="black")
                zax.tick_params(axis="both", which="major", labelsize="xx-small")

            if col == 0:
                ax.set_ylabel(row_label)
            else:
                ax.set_ylabel("")

            if row == 0:
                ax.set_title(col_label)

            # if row >= nrows - 1:
            #     ax.set_xlabel(metric_label + r" ($\tau$)")
            # else:
            #     ax.set_xlabel("")
            ax.set_xlabel("")

            sns.despine(ax=ax, offset=5)  # , left=True, bottom=True)  # , trim=True)

            # ax.tick_params("x", length=0)
            # ax.tick_params("y", pad=20, length=0)
            # ax.grid(
            #     visible=True,
            #     which="major",
            #     axis="x",
            #     lw=0.5,
            #     color="black",
            #     alpha=0.2,
            # )

    fig.supylabel(r"Fraction of runs with score $> \tau$", x=0.03)
    fig.supxlabel(metric_label + r" ($\tau$)", y=0.0)

    labels, handles = [], []
    for alg in algorithms:
        h, *_ = axes.flat[0].plot([], [], color=palette[alg], **line_style)
        labels.append(alg)  # + "\n" + str(aucs[alg]))
        handles.append(h)

        # for alg in algorithms:
        h = axes.flat[0].scatter(
            [], [], marker=r"$\mathsf{AUC}$", s=500, color="k", lw=0
        )  # , color=palette[alg])
        mat_str = "\n".join(
            " ".join(f"${{}}_{{{aucs[alg][i, j]:0.3f}}}$" for j in range(nrows))
            for i in range(nrows)
        )
        print(mat_str, aucs[alg])
        labels.append(mat_str)
        handles.append(h)

    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.125),
        ncols=min(6, len(algorithms)),
        borderaxespad=0.0,
        labelspacing=0,
        frameon=False,
    )

    fig.savefig(
        pdir / f"pprof{filename_suffix}.pdf",
        dpi=600,
        bbox_inches="tight",
        transparent=True,
    )

    plt.clf()

    progress.update()


def custom_problem_plot(conn, plot_dir):
    pdir = plot_dir / "custom"
    pdir.mkdir(parents=True, exist_ok=True)

    for group, m_where_query in tqdm(
        [  #
            (
                "_main",
                r"method_name NOT SIMILAR TO '.*(\*|any|all).*'",  # |static
            )
        ]
    ):
        problems = []
        algorithms = []
        rows = []
        for metric_label, metric in [  #
            (r"$R^2$ Train", "1.0 - nmse_train"),
            # (r"$R^2$ Test", "1.0 - nmse_test"),
        ]:
            # modifier -> info + score dict
            modifiers = []

            for h_value, h_where_query in [  #
                ("5", r"template_height::INTEGER = 5::INTEGER"),
                ("7", r"template_height::INTEGER = 7::INTEGER"),
            ]:
                for ls_value, ls_where_query in [  #
                    ("No", r"linear_scaling::BOOLEAN = false"),
                    ("Yes", r"linear_scaling::BOOLEAN = true"),
                ]:
                    where_query = " AND ".join(
                        q for q in [h_where_query, ls_where_query, m_where_query] if q
                    )

                    df = conn.execute(f"""
                        SELECT
                            problem_name AS problem,
                            method_name AS method,
                            format('{{}}.{{}}', fold, seed) AS run,
                            {metric} AS value,
                        FROM results
                        WHERE status != 'Running' AND status != 'Aborted' AND {where_query}
                    """).df()

                    df["method"] = df["method"].apply(method2name)

                    algorithms += list(df["method"].unique())

                    problems += list(df["problem"].unique())

                    modifiers.append(
                        (f"H = {h_value}{['', '\nLS'][ls_value == 'Yes']}", df)
                    )

            rows.append((metric_label, modifiers))

        problems = sorted(set(problems))

        algorithms = sorted(set(algorithms))
        keys = {m: m_order.get(m, len(m_order) + i) for i, m in enumerate(algorithms)}
        algorithms = sorted(set(algorithms), key=lambda m: keys[m])

        hues = sns.color_palette(  #
            "colorblind", n_colors=len(algorithms)
        )
        palette = {a: h for a, h in zip(algorithms, hues)}

        palette = PALETTE

        nrows = len(rows)
        ncols = len(problems)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex="col",
            sharey=False,
            squeeze=False,
            figsize=(ncols * 3, nrows * 4.5),
            gridspec_kw=dict(wspace=0.05, hspace=0.1),
        )

        for row, (row_label, modifiers) in enumerate(rows):
            for col, problem in enumerate(problems):
                ax = axes[row, col]

                q = 1e-4
                m_xlim = 1, 0

                m_axes = []

                s = 1 / len(modifiers)
                yticks, yticklabels = [], []
                for v_idx, (ytick, df) in list(enumerate(modifiers))[::-1]:
                    pdf = df[df["problem"] == problem]
                    value = len(modifiers) - v_idx

                    m_xlim = (
                        min(m_xlim[0], pdf["value"].quantile(q)),
                        max(m_xlim[1], pdf["value"].quantile(1 - q)),
                    )

                    yticks.append(value)
                    yticklabels.append(ytick)

                    # if v_idx > 0:
                    #     ax.axhline(
                    #         value + 0.5, color="black", lw=0.5, alpha=0.25, ls=":"
                    #     )

                    # hack to have a shaded background for every other row...
                    if (v_idx + 1) % 2 == 0:
                        ax.add_patch(
                            Rectangle(
                                xy=(0.0, 1.0 - v_idx * s - s),
                                width=1.0,
                                height=s,
                                color="black",
                                lw=0,
                                alpha=0.05,
                                transform=ax.transAxes,
                            )
                        )

                    m_ax = ax.inset_axes(
                        [
                            0.0,  # x0
                            1.0 - v_idx * s - s,  # y0
                            1.0,  # width
                            s,  # height
                        ],
                        transform=ax.transAxes,  # ax.transData,
                    )
                    m_axes.append(m_ax)

                    # sns.histplot(
                    #     pdf,
                    #     x="value",
                    #     hue="method",
                    #     palette=palette,
                    #     ax=m_ax,
                    #     legend=False,
                    # )
                    # sns.kdeplot(
                    #     pdf,
                    #     x="value",
                    #     hue="method",
                    #     palette=palette,
                    #     fill=True,
                    #     multiple="layer",
                    #     alpha=0.5,
                    #     linewidth=0,
                    #     ax=m_ax,
                    #     legend=False,
                    # )
                    sns.boxplot(
                        pdf,
                        x="value",
                        y="method",
                        hue="method",
                        order=algorithms,
                        palette=palette,
                        saturation=1.0,
                        # alpha=0.5,
                        linewidth=0.25,
                        flierprops=dict(marker=".", markeredgewidth=0.25),
                        fliersize=3,
                        medianprops=dict(linewidth=0.75),
                        # boxprops=dict(alpha=0.75),
                        ax=m_ax,
                        legend=False,
                    )

                    m_ax.margins(0.05)
                    m_ax.set_axis_off()

                x_range, extend = m_xlim[1] - m_xlim[0], 0.05
                m_xlim = m_xlim[0] - extend * x_range, m_xlim[1] + extend * x_range

                for m_ax in m_axes:
                    m_ax.set_xlim(*m_xlim)

                if col == 0:
                    ax.set_yticks(yticks, yticklabels, ha="center", fontsize="x-small")
                    ax.set_ylabel(row_label)
                else:
                    ax.set_ylabel("")
                    ax.set_yticks([])

                if row == 0:
                    ax.set_title(problem)

                # if row >= nrows - 1:
                #     pass
                # else:
                #     ax.set_xticks([])
                ax.set_xlim(*m_xlim)
                ax.set_xlabel("")
                ax.set_ylim(0.5, len(modifiers) + 0.5)

                sns.despine(ax=ax, left=True, bottom=True)

                ax.tick_params("x", length=0)
                ax.tick_params("y", pad=20, length=0)
                ax.grid(
                    visible=True,
                    which="major",
                    axis="x",
                    lw=0.5,
                    color="black",
                    alpha=0.2,
                )

        labels, handles = [], []
        for alg in algorithms:
            h, *_ = axes.flat[0].plot(
                [],
                [],
                color=palette[alg],
                lw=4,
                solid_capstyle="butt",
            )  # , **ci_style)
            labels.append(alg)
            handles.append(h)

        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),  # 75),
            ncols=min(6, len(algorithms)),
            borderaxespad=0.0,
            frameon=False,
        )

        fig.savefig(
            pdir / f"problem_scores{group}.pdf",
            dpi=600,
            bbox_inches="tight",
            transparent=True,
        )

        plt.clf()


def custom_interval_plot(conn, plot_dir):
    pdir = plot_dir / "custom" / "intervals"
    pdir.mkdir(parents=True, exist_ok=True)

    for group, m_where_query in tqdm(
        [  #
            (
                "_main",
                r"method_name NOT SIMILAR TO '.*(\*|any|all).*'",  # |static
            ),  # random, mi, mi_a, mi_m, node
            # (
            #     "_masking",
            #     r"method_name NOT SIMILAR TO '.*(adjusted|Random|Node).*'",
            # ),  # mi, any, all, masked
            # (
            #     "_hybrid",
            #     r"method_name NOT SIMILAR TO '.*(adjusted|Random|any|all).*'",
            # ),  # mi, node, static, hybrid
        ]
    ):
        algorithms = []
        rows = []
        for metric_label, metric in [  #
            (r"$R^2$ Train", "1.0 - nmse_train"),
            (r"$R^2$ Test", "1.0 - nmse_test"),
        ]:
            # modifier -> info + score dict
            modifiers = []

            for h_value, h_where_query in [  #
                ("5", r"template_height::INTEGER = 5::INTEGER"),
                ("7", r"template_height::INTEGER = 7::INTEGER"),
            ]:
                for ls_value, ls_where_query in [  #
                    ("No", r"linear_scaling::BOOLEAN = false"),
                    ("Yes", r"linear_scaling::BOOLEAN = true"),
                ]:
                    where_query = " AND ".join(
                        q for q in [h_where_query, ls_where_query, m_where_query] if q
                    )

                    score_dict, problems = rliable_score_dict(
                        conn,
                        run_expr="format('{}.{}', fold, run)",
                        problem_query="format('{}\nH={}\nLS={}', problem_name, template_height::STRING,IF(linear_scaling, 'Yes', 'No')::STRING)",  # operator_set::STRING,
                        # method_query="format('{} {}', method_name, init)",
                        where_query=where_query,
                        normalized_value_expr=metric,
                    )

                    score_dict = {method2name(m): s for m, s in score_dict.items()}

                    algorithms += score_dict.keys()

                    modifiers.append(
                        (
                            # h_value + " " + ls_value,
                            f"H = {h_value}{['', '\nLS']['Yes' == ls_value]}",
                            score_dict,
                        )
                    )
            rows.append((metric_label, modifiers))

        algorithms = sorted(set(algorithms))
        keys = {m: m_order.get(m, len(m_order) + i) for i, m in enumerate(algorithms)}
        algorithms = sorted(set(algorithms), key=lambda m: keys[m])

        hues = sns.color_palette(  #
            "colorblind", n_colors=len(algorithms)
        )
        palette = {a: h for a, h in zip(algorithms, hues)}

        palette = PALETTE

        aggregate_func = lambda x: np.array(
            [
                # metrics.aggregate_median(x), # is the median of means
                metrics.aggregate_iqm(x),
                # metrics.aggregate_mean(x),
                np.mean(np.median(x, axis=0, keepdims=False), axis=0),
                # metrics.aggregate_optimality_gap(x),
                # # bottom 25%
                # np.mean(np.quantile(x, 0.25, axis=0, keepdims=False), axis=0),
                # # top 25 %
                # np.mean(np.quantile(x, 0.75, axis=0, keepdims=False), axis=0)
            ]
        )
        aggregate_labels = [  # "Median of Means",
            "Interquartile Mean",  # IQM
            # "Mean",
            "Mean of Medians",
        ]

        nrows = len(rows)
        ncols = aggregate_func(np.eye(2)).shape[0]

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=True,  # "col",
            sharey=False,
            squeeze=False,
            figsize=(ncols * 7, nrows * 3),
            gridspec_kw=dict(wspace=0.05, hspace=0.1),
        )

        size = 6
        y_pad = 0.1  # padding between modifiers (defines jitter...)
        thickness = 1e-3 / 4  # ci center thickness
        ci_style = dict(
            lw=size,
            alpha=0.75,
            solid_capstyle="butt",
        )

        for row, (row_label, modifiers) in enumerate(rows):
            for col in range(ncols):
                ax = axes[row, col]

                yticks, yticklabels = [], []
                for v_idx, (ytick, score_dict) in enumerate(modifiers):
                    value = len(modifiers) - v_idx
                    # if v_idx > 0:
                    #     ax.axhline(
                    #         value + 0.5, color="black", lw=0.5, alpha=0.25, ls=":"
                    #     )

                    # hack to have a shaded background for every other row...
                    # if (v_idx + 1) % 2 == 0:
                    #     ax.axhline(value, color="black", lw=26.5, alpha=0.05)

                    if (v_idx + 1) % 2 == 0:
                        s = 1 / len(modifiers)
                        ax.add_patch(
                            Rectangle(
                                xy=(0.0, 1.0 - v_idx * s - s),
                                width=1.0,
                                height=s,
                                color="black",
                                lw=0,
                                alpha=0.05,
                                transform=ax.transAxes,
                            )
                        )

                    yticks.append(value)
                    yticklabels.append(ytick)

                    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
                        score_dict,
                        aggregate_func,
                        reps=5000,  # 0
                    )

                    if len(algorithms) > 1:
                        y_start = value + 0.5 - y_pad
                        y_end = value - 0.5 + y_pad
                        y_step = (y_start - y_end) / (len(algorithms) - 1)
                    else:
                        y_start, y_step = value, 0
                    for i, alg in enumerate(algorithms):
                        hue = palette[alg]

                        center = aggregate_scores[alg][col]
                        lower, upper = aggregate_score_cis[alg][:, col]

                        y = y_start - i * y_step
                        ax.plot([lower, upper], [y, y], color=hue, **ci_style)
                        ax.plot(
                            [center - thickness, center + thickness],
                            [y, y],
                            color="black",
                            lw=size,
                            solid_capstyle="butt",
                            zorder=10,
                        )
                        # ax.scatter(
                        #     [center],
                        #     [y],
                        #     color="black",  # hue,
                        #     marker="$|$",
                        #     s=size**2,
                        #     lw=0,
                        #     zorder=10,
                        # )  # o")

                if col == 0:
                    ax.set_yticks(yticks, yticklabels, ha="center", fontsize="x-small")
                    ax.set_ylabel(row_label)
                else:
                    ax.set_ylabel("")
                    ax.set_yticks([])

                if row == 0:
                    ax.set_title(aggregate_labels[col])

                # if row >= nrows - 1:
                #     pass
                # else:
                #     ax.set_xticks([])
                ax.set_xlabel("")

                ax.set_ylim(0.5, len(modifiers) + 0.5)
                # xmin, xmax = ax.get_xlim()
                # ax.set_xlim(max(xmin, 0.0), min(xmax, 1.0))

                sns.despine(ax=ax, left=True, bottom=True)  # , trim=True)

                # for ticklabel in ax.get_yticklabels():
                #     ticklabel.set_horizontalalignment("center")

                ax.tick_params("x", length=0)
                ax.tick_params("y", pad=20, length=0)
                ax.grid(
                    visible=True,
                    which="major",
                    axis="x",
                    lw=0.5,
                    color="black",
                    alpha=0.2,
                )

        labels, handles = [], []
        for alg in algorithms:
            h, *_ = axes.flat[0].plot([], [], color=palette[alg], **ci_style)
            labels.append(alg)
            handles.append(h)

        fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.0),
            ncols=min(6, len(algorithms)),
            borderaxespad=0.0,
            frameon=False,
        )

        fig.savefig(
            pdir / f"interval_estimates{group}.pdf",
            dpi=600,
            bbox_inches="tight",
            transparent=True,
        )

        plt.clf()


def main():
    with load_results(
        LOG_DIR,
        # enable pre-processing the .csv logs into .parquet files
        # preprocess=True,
        parquet_dir=PARQUET_DIR,
    ) as conn:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        (PLOT_DIR / "rliable").mkdir(parents=True, exist_ok=True)

        custom_convergence_plot(conn, PLOT_DIR)
        # custom_pprof_plot(conn, PLOT_DIR)

        # custom_problem_plot(conn, PLOT_DIR)
        # custom_interval_plot(conn, PLOT_DIR)

        # custom_problem_convergence_plot(conn, PLOT_DIR)

        # custom_cmp_plot(conn, PLOT_DIR)

        exit()

        for h_group, h_where_query in [  #
            ("", ""),
            ("_h5", r"template_height::INTEGER = 5::INTEGER"),
            ("_h7", r"template_height::INTEGER = 7::INTEGER"),
        ]:
            for ls_group, ls_where_query in [  #
                ("", ""),
                ("_ls", r"linear_scaling::BOOLEAN = true"),
                ("_nols", r"linear_scaling::BOOLEAN = false"),
            ]:
                for m_group, m_where_query in [  #
                    ("_all", ""),  # all
                    (
                        "_main",
                        r"method_name NOT SIMILAR TO '.*(\*|any|all|static).*'",
                    ),  # random, mi, mi_a, mi_m, node
                    (
                        "_masking",
                        r"method_name NOT SIMILAR TO '.*(adjusted|Random|Node).*'",
                    ),  # mi, any, all, masked
                    (
                        "_hybrid",
                        r"method_name NOT SIMILAR TO '.*(adjusted|Random|any|all).*'",
                    ),  # mi, node, static, hybrid
                ]:
                    group, where_query = (
                        h_group + ls_group + m_group,
                        " AND ".join(
                            q
                            for q in [h_where_query, ls_where_query, m_where_query]
                            if q
                        ),
                    )

                    score_dict, problems = rliable_score_dict(
                        conn,
                        run_expr="format('{}.{}', fold, run)",
                        problem_query="format('{}\nH={}\nLS={}', problem_name, template_height::STRING,IF(linear_scaling, 'Yes', 'No')::STRING)",  # operator_set::STRING,
                        # method_query="format('{} {}', method_name, init)",
                        where_query=where_query,
                        normalized_value_expr="1.0 - nmse_train",
                    )

                    algorithms = sorted(score_dict.keys())

                    aggregate_func = lambda x: np.array(
                        [
                            metrics.aggregate_median(x),
                            metrics.aggregate_iqm(x),
                            metrics.aggregate_mean(x),
                            # metrics.aggregate_optimality_gap(x),
                            # # bottom 25%
                            # np.mean(np.quantile(x, 0.25, axis=0, keepdims=False), axis=0),
                            # # top 25 %
                            # np.mean(np.quantile(x, 0.75, axis=0, keepdims=False), axis=0)
                        ]
                    )
                    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
                        score_dict, aggregate_func, reps=50000
                    )
                    fig, axes = plot_utils.plot_interval_estimates(
                        aggregate_scores,
                        aggregate_score_cis,
                        metric_names=[
                            "Median",
                            "IQM",
                            "Mean",
                        ],  # "Mean Worst Quartile", "Mean Best Quartile"],
                        algorithms=algorithms,
                        xlabel="Human Normalized Score",
                        row_height=0.5,
                    )

                    fig.savefig(
                        PLOT_DIR / "rliable" / f"interval_estimates{group}.pdf",
                        dpi=600,
                        bbox_inches="tight",
                        transparent=True,
                    )

                    plt.clf()

                    # thresholds = np.linspace(0.0, 1.0, 50)
                    thresholds = np.linspace(0.35, 0.95, 100)
                    score_distributions, score_distributions_cis = (
                        rly.create_performance_profile(score_dict, thresholds)
                    )

                    fig, ax = plt.subplots(ncols=1, figsize=(10, 5))
                    plot_utils.plot_performance_profiles(
                        score_distributions,
                        thresholds,
                        performance_profile_cis=score_distributions_cis,
                        # use_non_linear_scaling=True,
                        # xticks=[],
                        colors=dict(zip(algorithms, sns.color_palette("colorblind"))),
                        xlabel=r"Human Normalized Score $(\tau)$",
                        ax=ax,
                    )
                    # ax.set_yscale("log")
                    fig.legend()

                    fig.savefig(
                        PLOT_DIR / "rliable" / f"pprof{group}.pdf",
                        dpi=600,
                        bbox_inches="tight",
                        transparent=True,
                    )
                    plt.clf()

                    # score_dict, problems, generations = rliable_convergence_score_dict(
                    #     conn,
                    #     run_expr="format('{}.{}', fold, run)",
                    #     problem_query="format('{}\nH={}\nLS={}', problem_name, template_height::STRING,IF(linear_scaling, 'Yes', 'No')::STRING)",  # operator_set::STRING,
                    #     # method_query="format('{} {}', method_name, init)",
                    #     where_query=where_query,
                    #     normalized_value_expr="1.0 - nmse_train",
                    # )

                    # iqm = lambda scores: np.array([metrics.aggregate_iqm(scores[..., frame])
                    #                                for frame in range(scores.shape[-1])])
                    # iqm_scores, iqm_cis = rly.get_interval_estimates(
                    #   score_dict, iqm, reps=5000)

                    # fig, ax = plt.subplots(ncols=1, figsize=(7, 5))
                    # plot_utils.plot_sample_efficiency_curve(
                    #     generations, iqm_scores, iqm_cis, algorithms=algorithms,
                    #     xlabel=r'Number of Generations',
                    #     ylabel='$R^2$ Train', ax=ax)

                    # fig.legend(loc="lower right")

                    # fig.savefig(
                    #     PLOT_DIR / "rliable" / f"convergence{group}.pdf",
                    #     dpi=600,
                    #     bbox_inches="tight",
                    #     transparent=True,
                    # )
                    # plt.clf()

        exit()

        for split in [  #
            "train",
            "test",
        ]:
            plot_convergence_so(
                PLOT_DIR / f"convergence_{split}",
                conn,
                # y_var="1.0 - objectives[1]::DOUBLE",  # transform NMSE into R2
                y_var=f"1.0 - nmse_{split}",
                y_agg="MAX",  # higher R^2 is better
                y_label=f"$R^2$ {split.title()}",
                ymin="auto",
                ymax="auto",
                # merge folds and runs into one seaborn "unit"
                unit_query="format('{}.{}', fold, run)",
                # split up the plot into the following rows
                modifier_query="[template_height::STRING,operator_set::STRING,IF(linear_scaling, 'Yes', 'No')::STRING]",
                modifier_labels=[
                    "Template Height",
                    "Operators",
                    "Linear Scaling",
                ],
                nsamples=100,
            )


if __name__ == "__main__":
    main()
