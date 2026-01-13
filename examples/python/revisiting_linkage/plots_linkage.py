import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pygom
import seaborn as sns
from matplotlib.patches import Rectangle
from pygom import *
from rliable import library as rly
from rliable import metrics, plot_utils
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


def custom_problem_plot(conn, plot_dir):
    pdir = plot_dir / "problems"
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

        nrows = len(rows)
        ncols = len(problems)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex="col",
            sharey=False,
            squeeze=False,
            figsize=(ncols * 3, nrows * 4),
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
                        # alpha=0.5,
                        linewidth=0.25,
                        flierprops=dict(marker=".", markeredgewidth=0.25),
                        fliersize=3,
                        medianprops=dict(linewidth=0.75),
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
            bbox_to_anchor=(0.5, -0.075),
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
    pdir = plot_dir / "cmp"
    pdir.mkdir(parents=True, exist_ok=True)

    for group, m_where_query in tqdm(
        [  #
            (
                "_main",
                r"method_name NOT SIMILAR TO '.*(\*|any|all).*'",  # |static
            ),  # random, mi, mi_a, mi_m, node
            (
                "_masking",
                r"method_name NOT SIMILAR TO '.*(adjusted|Random|Node).*'",
            ),  # mi, any, all, masked
            (
                "_hybrid",
                r"method_name NOT SIMILAR TO '.*(adjusted|Random|any|all).*'",
            ),  # mi, node, static, hybrid
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
            figsize=(ncols * 6, nrows * 3),
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

        # custom_problem_plot(conn, PLOT_DIR)
        custom_interval_plot(conn, PLOT_DIR)

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
